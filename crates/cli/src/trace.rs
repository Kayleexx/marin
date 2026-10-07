use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use metrics::{Stage, StageTimes, TraceEntry};
use model::{BackendChoice, Engine, EngineCache, Precision};
use runtime::SequenceCache;
use serde::Serialize;

use crate::CliError;
use crate::bench::measure;
use crate::json_out::JsonOut;
use crate::load::load;
use crate::signal::Interrupt;
use crate::style::Style;

/// Which forward pass to trace: the one that reads the prompt, or the Nth one-token pass after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Prefill,
    Decode(usize),
}

impl FromStr for Pass {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        if text == "prefill" {
            return Ok(Self::Prefill);
        }
        match text.parse::<usize>() {
            Ok(0) => Err("decode passes count from 1".into()),
            Ok(n) => Ok(Self::Decode(n)),
            Err(_) => Err(format!("`{text}` is not `prefill` or a whole number")),
        }
    }
}

impl Pass {
    /// One-token passes that run before the traced one.
    fn steps(self) -> usize {
        match self {
            Self::Prefill => 0,
            Self::Decode(n) => n,
        }
    }

    fn describe(self, prompt_tokens: usize) -> String {
        match self {
            Self::Prefill => format!("the pass that reads the {prompt_tokens}-token prompt"),
            Self::Decode(n) => format!(
                "one-token pass {n} (the token at position {})",
                prompt_tokens.saturating_add(n).saturating_sub(1)
            ),
        }
    }
}

pub struct Options {
    pub prompt_tokens: usize,
    pub pass: Pass,
    pub json: Option<PathBuf>,
}

/// Columns of the per-layer table, in the order of a layer's work.
const COLUMNS: [(Stage, &str); 8] = [
    (Stage::Norm, "norm"),
    (Stage::Qkv, "qkv"),
    (Stage::Rope, "rope"),
    (Stage::KvWrite, "kv"),
    (Stage::Attention, "attn"),
    (Stage::OutProj, "proj"),
    (Stage::Mlp, "mlp"),
    (Stage::Residual, "add"),
];

type Cells = [Duration; COLUMNS.len()];

#[derive(Debug, Default, PartialEq, Eq)]
struct Table {
    embedding: Duration,
    layers: Vec<Cells>,
    final_norm: Duration,
    lm_head: Duration,
}

fn table(entries: &[TraceEntry]) -> Table {
    let mut table = Table::default();
    for entry in entries {
        match (entry.layer, entry.stage) {
            (None, Stage::Embedding) => {
                table.embedding = table.embedding.saturating_add(entry.time)
            }
            (None, Stage::Norm) => table.final_norm = table.final_norm.saturating_add(entry.time),
            (None, Stage::LmHead) => table.lm_head = table.lm_head.saturating_add(entry.time),
            (Some(layer), stage) => {
                let Some(column) = COLUMNS.iter().position(|(s, _)| *s == stage) else {
                    continue;
                };
                if layer >= table.layers.len() {
                    table
                        .layers
                        .resize(layer.saturating_add(1), Cells::default());
                }
                if let Some(cell) = table
                    .layers
                    .get_mut(layer)
                    .and_then(|row| row.get_mut(column))
                {
                    *cell = cell.saturating_add(entry.time);
                }
            }
            (None, _) => {}
        }
    }
    table
}

fn millis(time: Duration) -> f64 {
    time.as_secs_f64() * 1000.0
}

fn cell(time: Duration) -> String {
    if time.is_zero() {
        "-".into()
    } else if millis(time) < 0.005 {
        "<0.01".into()
    } else {
        format!("{:.2}", millis(time))
    }
}

fn row(label: &str, cells: &Cells) -> String {
    let values: String = cells.iter().map(|t| format!("{:>7}", cell(*t))).collect();
    format!("  {label:<7}{values}\n")
}

pub struct Header<'a> {
    pub backend: &'a str,
    pub clock: &'a str,
    pub prompt_tokens: usize,
    pub pass: Pass,
}

fn render(header: &Header, entries: &[TraceEntry], wall: Duration) -> String {
    let table = table(entries);
    let mut out = format!(
        "trace: {}\n  {} · {} prompt tokens · stage times on the {}\n\n",
        header.pass.describe(header.prompt_tokens),
        header.backend,
        header.prompt_tokens,
        header.clock
    );
    let single = |label: &str, time: Duration| format!("  {label:<14}{:>7} ms\n", cell(time));
    out.push_str(&single("embedding", table.embedding));
    let names: String = COLUMNS.iter().map(|(_, n)| format!("{n:>7}")).collect();
    out.push_str(&format!("  {:<7}{names}   (ms)\n", "layer"));
    let mut sum = Cells::default();
    for (index, cells) in table.layers.iter().enumerate() {
        out.push_str(&row(&index.to_string(), cells));
        for (total, time) in sum.iter_mut().zip(cells) {
            *total = total.saturating_add(*time);
        }
    }
    out.push_str(&row("sum", &sum));
    out.push_str(&single("final norm", table.final_norm));
    out.push_str(&single("lm head", table.lm_head));
    let timed = entries
        .iter()
        .fold(Duration::ZERO, |all, e| all.saturating_add(e.time));
    let outside = wall.saturating_sub(timed);
    let percent = if wall.is_zero() {
        0.0
    } else {
        outside.as_secs_f64() / wall.as_secs_f64() * 100.0
    };
    out.push_str(&format!(
        "\n  pass took {:.2} ms; {percent:.1}% was outside the timed stages\n",
        millis(wall)
    ));
    out.push_str(
        "\n  reading this: every layer does the same work, so the rows look alike.\n  \
         the biggest column is where this pass spends its time.\n  \
         norm = normalization · qkv = query, key and value projections · kv = kv cache write\n  \
         proj = attention output projection · add = residual add\n",
    );
    out
}

/// Runs the pass once untraced to warm up, then once traced; the traced pass is timed on its own.
fn trace_pass(
    engine: &Engine,
    cache: &mut EngineCache,
    prompt: &[u32],
    pass: Pass,
    vocab: usize,
    cancel: &AtomicBool,
) -> Result<(StageTimes, Duration), CliError> {
    cache.clear();
    let mut times = StageTimes::traced();
    let steps = pass.steps();
    if steps == 0 {
        let started = Instant::now();
        engine.forward_profiled(prompt, cache, &mut times)?;
        return Ok((times, started.elapsed()));
    }
    let mut sampler = measure::greedy()?;
    let mut logits = runtime::CausalLm::forward(engine, prompt, cache)?;
    for _ in 1..steps {
        if cancel.load(Ordering::SeqCst) {
            return Err(CliError::Cancelled);
        }
        let token = sampler.sample(&logits, vocab)?;
        logits = runtime::CausalLm::forward(engine, &[token], cache)?;
    }
    let token = sampler.sample(&logits, vocab)?;
    let started = Instant::now();
    engine.forward_profiled(&[token], cache, &mut times)?;
    Ok((times, started.elapsed()))
}

#[derive(Serialize)]
struct JsonEntry {
    layer: Option<usize>,
    stage: &'static str,
    ms: f64,
}

#[derive(Serialize)]
struct JsonTrace<'a> {
    format: u32,
    model: &'a str,
    backend: &'a str,
    clock: &'a str,
    prompt_tokens: usize,
    pass: String,
    wall_ms: f64,
    entries: Vec<JsonEntry>,
}

fn pass_name(pass: Pass) -> String {
    match pass {
        Pass::Prefill => "prefill".into(),
        Pass::Decode(n) => n.to_string(),
    }
}

pub fn run(
    model_dir: &Path,
    backend: BackendChoice,
    ctx: Option<usize>,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    if options.prompt_tokens == 0 {
        return Err(CliError::BadOption(
            "trace needs --prompt-tokens above 0".into(),
        ));
    }
    let needed = options
        .prompt_tokens
        .checked_add(options.pass.steps())
        .ok_or_else(|| CliError::BadOption("--pass is too large".into()))?;
    let ctx = match ctx {
        Some(ctx) if ctx < needed => {
            return Err(CliError::BadOption(format!(
                "--ctx {ctx} is too small: this pass needs {needed} positions ({} prompt tokens and {} generated)",
                options.prompt_tokens,
                options.pass.steps()
            )));
        }
        Some(ctx) => ctx,
        None => needed,
    };
    let json = options.json.as_deref().map(JsonOut::create).transpose()?;
    let loaded = load(
        model_dir,
        Some(ctx),
        ctx,
        backend,
        Precision::default(),
        style,
        true,
    )?;
    let engine = &loaded.model;
    let mut cache = engine.new_cache(ctx)?;
    let prompt = measure::prompt(&loaded.dir.tokenizer, options.prompt_tokens)?;
    let vocab = loaded.dir.tokenizer.vocab_size();
    let cancel = interrupt.cancel_flag();
    trace_pass(engine, &mut cache, &prompt, options.pass, vocab, cancel)?;
    let (times, wall) = trace_pass(engine, &mut cache, &prompt, options.pass, vocab, cancel)?;
    let backend_name = engine.backend_name();
    let clock = measure::clock_name(engine);
    let header = Header {
        backend: &backend_name,
        clock,
        prompt_tokens: prompt.len(),
        pass: options.pass,
    };
    std::io::stdout()
        .lock()
        .write_all(render(&header, times.trace(), wall).as_bytes())?;
    if let (Some(json), Some(path)) = (json, &options.json) {
        json.finish(&JsonTrace {
            format: 1,
            model: &loaded.name,
            backend: &backend_name,
            clock,
            prompt_tokens: prompt.len(),
            pass: pass_name(options.pass),
            wall_ms: millis(wall),
            entries: times
                .trace()
                .iter()
                .map(|e| JsonEntry {
                    layer: e.layer,
                    stage: e.stage.name(),
                    ms: millis(e.time),
                })
                .collect(),
        })?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(layer: Option<usize>, stage: Stage, micros: u64) -> TraceEntry {
        TraceEntry {
            layer,
            stage,
            time: Duration::from_micros(micros),
        }
    }

    fn sample() -> Vec<TraceEntry> {
        vec![
            entry(None, Stage::Embedding, 30),
            entry(Some(0), Stage::Norm, 10),
            entry(Some(0), Stage::Norm, 5),
            entry(Some(0), Stage::Mlp, 410),
            entry(Some(1), Stage::Mlp, 400),
            entry(Some(1), Stage::Attention, 160),
            entry(None, Stage::Norm, 8),
            entry(None, Stage::LmHead, 280),
        ]
    }

    #[test]
    fn passes_parse_and_reject_nonsense() {
        assert_eq!("prefill".parse(), Ok(Pass::Prefill));
        assert_eq!("3".parse(), Ok(Pass::Decode(3)));
        assert!("0".parse::<Pass>().unwrap_err().contains("from 1"));
        assert!("-1".parse::<Pass>().is_err());
        assert!("".parse::<Pass>().is_err());
        assert!("first".parse::<Pass>().is_err());
    }

    #[test]
    fn the_table_sums_repeated_stages_per_layer_and_keeps_the_rest_apart() {
        let table = table(&sample());
        assert_eq!(table.layers.len(), 2);
        assert_eq!(table.layers[0][0], Duration::from_micros(15));
        assert_eq!(table.layers[1][4], Duration::from_micros(160));
        assert_eq!(table.embedding, Duration::from_micros(30));
        assert_eq!(table.final_norm, Duration::from_micros(8));
        assert_eq!(table.lm_head, Duration::from_micros(280));
    }

    #[test]
    fn an_empty_trace_renders_without_rows_or_nan() {
        let header = Header {
            backend: "cpu",
            clock: "CPU wall clock",
            prompt_tokens: 4,
            pass: Pass::Prefill,
        };
        let text = render(&header, &[], Duration::ZERO);
        assert!(text.contains("0.0% was outside"));
        assert!(!text.contains("NaN"));
    }

    #[test]
    fn the_report_names_the_pass_and_shows_each_layer_and_the_outside_share() {
        let header = Header {
            backend: "cpu",
            clock: "CPU wall clock",
            prompt_tokens: 128,
            pass: Pass::Decode(1),
        };
        let text = render(&header, &sample(), Duration::from_micros(1500));
        assert!(text.contains("one-token pass 1 (the token at position 128)"));
        assert!(text.contains("  0 "), "{text}");
        assert!(text.contains("embedding        0.03 ms"), "{text}");
        assert!(text.contains("lm head          0.28 ms"), "{text}");
        assert!(
            text.contains("pass took 1.50 ms; 13.1% was outside"),
            "{text}"
        );
    }

    #[test]
    fn tiny_and_zero_times_do_not_print_as_zero_or_as_noise() {
        assert_eq!(cell(Duration::ZERO), "-");
        assert_eq!(cell(Duration::from_nanos(900)), "<0.01");
        assert_eq!(cell(Duration::from_micros(1234)), "1.23");
    }
}
