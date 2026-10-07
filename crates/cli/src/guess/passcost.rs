use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use model::{BackendChoice, Precision, Stage, StageTimes};
use runtime::{CausalLm, SequenceCache};

use super::data::{Costs, Curve, FORMAT, Point};
use crate::CliError;
use crate::bench::record::Stat;
use crate::json_out::JsonOut;
use crate::load::load;
use crate::signal::Interrupt;
use crate::style::Style;

/// Passes of one size run for this long before any are counted: the GPU holds a power limit, so its clock
/// takes a moment to settle at what the new size draws.
const SETTLE: Duration = Duration::from_millis(150);
/// Tokens kept free past the longest context for the passes of one size.
const SPARE: usize = 512;

pub struct Options {
    /// Text whose tokens fill the context and are fed to the passes (their values do not change the time).
    pub text: PathBuf,
    pub contexts: Vec<usize>,
    pub sizes: Vec<usize>,
    pub rounds: usize,
    pub warmup: usize,
    /// Passes counted per size and round, run back to back after the settling passes.
    pub reps: usize,
    pub save: Option<PathBuf>,
    /// Also profile each pass size stage by stage and print the attention time and key/value traffic.
    pub stages: bool,
    pub precision: Precision,
}

struct Samples {
    context: usize,
    tokens: usize,
    ms: Vec<f64>,
}

fn sorted_unique(values: &[usize]) -> Vec<usize> {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    values
}

fn bad(message: &str) -> CliError {
    CliError::BadOption(message.into())
}

fn pass(
    engine: &model::Engine,
    cache: &mut model::EngineCache,
    ids: &[u32],
    at: usize,
    tokens: usize,
) -> Result<f64, CliError> {
    let chunk = ids
        .get(at..at.saturating_add(tokens))
        .ok_or_else(|| bad("the text is too short for the passes"))?;
    let start = Instant::now();
    engine.forward(chunk, cache)?;
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

/// Reads `context` tokens, runs passes of `tokens` until the GPU has settled, then times `reps` more.
fn measure(
    engine: &model::Engine,
    cache: &mut model::EngineCache,
    ids: &[u32],
    context: usize,
    tokens: usize,
    reps: usize,
) -> Result<Vec<f64>, CliError> {
    cache.clear();
    engine.forward(ids.get(..context).unwrap_or(&[]), cache)?;
    let mut at = context;
    let settled = Instant::now();
    while settled.elapsed() < SETTLE {
        pass(engine, cache, ids, at, tokens)?;
        at = at.saturating_add(tokens);
    }
    let mut counted = Vec::with_capacity(reps);
    for _ in 0..reps {
        counted.push(pass(engine, cache, ids, at, tokens)?);
        at = at.saturating_add(tokens);
    }
    Ok(counted)
}

pub fn run(
    model_dir: &Path,
    backend: BackendChoice,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    let contexts = sorted_unique(&options.contexts);
    let sizes = sorted_unique(&options.sizes);
    let (Some(&longest), true) = (contexts.last(), !sizes.is_empty()) else {
        return Err(bad("--contexts and --sizes need at least one value each"));
    };
    if options.rounds == 0 || sizes.contains(&0) {
        return Err(bad("--rounds and every size must be at least 1"));
    }
    if options.reps == 0 {
        return Err(bad("--reps must be at least 1"));
    }
    let needed = longest.saturating_add(SPARE);
    let out = options.save.as_deref().map(JsonOut::create).transpose()?;
    let text = fs::read_to_string(&options.text).map_err(|e| CliError::Input {
        path: options.text.display().to_string(),
        reason: e.to_string(),
    })?;
    let loaded = load(
        model_dir,
        Some(needed),
        needed,
        backend,
        options.precision,
        style,
        true,
    )?;
    let ids = loaded.dir.tokenizer.encode(&text)?;
    if ids.len() < needed {
        return Err(CliError::BadOption(format!(
            "the text has {} tokens, the longest context plus passes needs {needed}",
            ids.len()
        )));
    }
    let engine = &loaded.model;
    let mut cache = engine.new_cache(needed)?;
    let mut samples: Vec<Samples> = contexts
        .iter()
        .flat_map(|&context| {
            sizes.iter().map(move |&tokens| Samples {
                context,
                tokens,
                ms: Vec::new(),
            })
        })
        .collect();
    let cancel = interrupt.cancel_flag();
    let total = options.warmup.saturating_add(options.rounds);
    for round in 0..total {
        // Sizes rotate every round so none is always measured first.
        let order: Vec<usize> = sizes
            .iter()
            .cycle()
            .skip(round.checked_rem(sizes.len()).unwrap_or(0))
            .take(sizes.len())
            .copied()
            .collect();
        for &context in &contexts {
            for &tokens in &order {
                if cancel.load(Ordering::SeqCst) {
                    return Err(CliError::Cancelled);
                }
                let counted = measure(engine, &mut cache, &ids, context, tokens, options.reps)?;
                if round >= options.warmup
                    && let Some(s) = samples
                        .iter_mut()
                        .find(|s| s.context == context && s.tokens == tokens)
                {
                    s.ms.extend(counted);
                }
            }
        }
        eprint!("\rround {}/{total}", round.saturating_add(1));
    }
    eprintln!();
    let curves = contexts
        .iter()
        .map(|&context| curve(context, &samples))
        .collect::<Result<Vec<_>, _>>()?;
    print_table(&curves);
    if options.stages {
        print_stages(engine, &mut cache, &ids, &contexts, &sizes, options.reps)?;
    }
    if let (Some(out), Some(path)) = (out, &options.save) {
        out.finish(&Costs {
            format: FORMAT,
            model: loaded.name.clone(),
            quantization: options
                .precision
                .weights
                .map_or_else(|| "none".to_string(), |q| q.label()),
            backend: engine.backend_name(),
            rounds: options.rounds,
            reps: options.reps,
            curves,
        })?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn curve(context: usize, samples: &[Samples]) -> Result<Curve, CliError> {
    let points = samples
        .iter()
        .filter(|s| s.context == context)
        .map(|s| {
            Stat::of(&s.ms)
                .map(|stat| Point {
                    tokens: s.tokens,
                    median_ms: stat.median,
                    min_ms: stat.min,
                    max_ms: stat.max,
                })
                .ok_or_else(|| bad("no pass was measured"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Curve { context, points })
}

fn print_table(curves: &[Curve]) {
    for curve in curves {
        let one = curve.points.first().map_or(1.0, |p| p.median_ms);
        println!(
            "\npasses after {} tokens of context (median of runs, min..max):",
            curve.context
        );
        println!("  tokens in the pass      time         vs first     per token");
        for p in &curve.points {
            println!(
                "  {:>6}              {:>6.2} ms    {:>5.2}x       {:>6.2} ms   ({:.2}..{:.2})",
                p.tokens,
                p.median_ms,
                p.median_ms / one,
                p.median_ms / p.tokens as f64,
                p.min_ms,
                p.max_ms
            );
        }
    }
}

/// Average time of the attention stage and of the whole pass over `reps` profiled passes of `tokens` after `context`.
fn stage_times(
    engine: &model::Engine,
    cache: &mut model::EngineCache,
    ids: &[u32],
    context: usize,
    tokens: usize,
    reps: usize,
) -> Result<(f64, f64), CliError> {
    cache.clear();
    engine.forward(ids.get(..context).unwrap_or(&[]), cache)?;
    let mut at = context;
    let settled = Instant::now();
    let mut times = StageTimes::default();
    while settled.elapsed() < SETTLE {
        let chunk = ids.get(at..at.saturating_add(tokens)).unwrap_or(&[]);
        engine.forward_profiled(chunk, cache, &mut times)?;
        at = at.saturating_add(tokens);
    }
    let mut times = StageTimes::default();
    for _ in 0..reps {
        let chunk = ids.get(at..at.saturating_add(tokens)).unwrap_or(&[]);
        engine.forward_profiled(chunk, cache, &mut times)?;
        at = at.saturating_add(tokens);
    }
    let per_pass = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / reps as f64;
    Ok((
        per_pass(times.total(Stage::Attention)),
        per_pass(times.sum()),
    ))
}

fn print_stages(
    engine: &model::Engine,
    cache: &mut model::EngineCache,
    ids: &[u32],
    contexts: &[usize],
    sizes: &[usize],
    reps: usize,
) -> Result<(), CliError> {
    let cfg = engine.config();
    // Every layer reads each key and value row once per query head that scans it: f32 keys and values.
    let row_bytes = cfg.head_dim as f64 * 4.0 * 2.0;
    let unique =
        |context: usize| cfg.layers as f64 * cfg.kv_heads as f64 * context as f64 * row_bytes;
    println!("\nattention stage by pass size (profiled passes, so the totals run a little high):");
    println!(
        "  context  tokens  attention   per token   whole pass   KV read if each head scans alone   KV held   attention vs 1 token"
    );
    for &context in contexts {
        let mut one = None;
        for &tokens in sizes {
            let (attention, total) = stage_times(engine, cache, ids, context, tokens, reps)?;
            let one_ms = *one.get_or_insert(attention);
            let alone = unique(context) * (cfg.heads as f64 / cfg.kv_heads as f64) * tokens as f64;
            println!(
                "  {context:>7}  {tokens:>6}  {attention:>7.2} ms  {:>7.2} ms  {total:>8.2} ms   {:>8.2} GB              {:>8.2} GB   {:>6.2}x",
                attention / tokens as f64,
                alone / 1e9,
                unique(context) / 1e9,
                attention / one_ms
            );
        }
    }
    Ok(())
}
