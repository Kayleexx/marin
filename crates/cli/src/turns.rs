use std::time::{Duration, Instant};

use chat::{DEFAULT_SYSTEM, Message, Role, render};
use metrics::{Sampler as GpuSampler, spread};
use model::{BackendChoice, Engine, EngineCache, QuantScheme};
use runtime::{GenerateParams, Kept, Outcome, Retained, Speculation, generate_reusing};

use crate::CliError;
use crate::bench::measure::{greedy, prompt};
use crate::load::{Loaded, load};
use crate::signal::Interrupt;
use crate::style::Style;

pub struct Options {
    /// Conversation lengths in tokens before the new message.
    pub history: Vec<usize>,
    pub reply_tokens: usize,
    pub rounds: usize,
    pub ctx: usize,
    pub energy: bool,
    pub speculate: bool,
    pub ask: String,
    pub quant: Option<QuantScheme>,
}

struct Case {
    label: String,
    before: Vec<u32>,
    timed: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Keep,
    Replay,
}

struct Sample {
    outcome: Outcome,
    tokens: Vec<u32>,
    vram: Option<u64>,
    watts: Option<f64>,
    total: Duration,
}

fn encode(loaded: &Loaded, messages: &[Message]) -> Result<Vec<u32>, CliError> {
    let specials = loaded.dir.tokenizer.special_tokens();
    let text = render(messages, &specials).map_err(|e| CliError::BadOption(e.to_string()))?;
    Ok(loaded.dir.tokenizer.encode(&text)?)
}

fn history(loaded: &Loaded, tokens: usize) -> Result<Vec<Message>, CliError> {
    let mut messages = vec![Message::new(Role::System, DEFAULT_SYSTEM)];
    if tokens == 0 {
        return Ok(messages);
    }
    let chunk = loaded
        .dir
        .tokenizer
        .decode(&prompt(&loaded.dir.tokenizer, 180)?)?;
    let mut part: usize = 1;
    while encode(
        loaded,
        &[messages.clone(), vec![Message::new(Role::User, "?")]].concat(),
    )?
    .len()
        < tokens
    {
        let user = format!("Here is part {part} of a story: {chunk}");
        messages.push(Message::new(Role::User, user));
        messages.push(Message::new(Role::Assistant, format!("Noted. {chunk}")));
        part = part.saturating_add(1);
    }
    Ok(messages)
}

fn cases(
    loaded: &Loaded,
    cache: &mut EngineCache,
    target: usize,
    ask: &str,
    params: &GenerateParams,
) -> Result<Vec<Case>, CliError> {
    let mut messages = history(loaded, target)?;
    messages.push(Message::new(
        Role::User,
        "What does a lighthouse keeper do?",
    ));
    let before = encode(loaded, &messages)?;
    let reply = run(
        &loaded.model,
        cache,
        &mut Retained::default(),
        &before,
        params,
        false,
    )?;
    let text = loaded.dir.tokenizer.decode(&reply.tokens)?;
    messages.push(Message::new(Role::Assistant, text));
    messages.push(Message::new(Role::User, ask));
    let timed = encode(loaded, &messages)?;
    // Editing the first message leaves only the system message shared.
    if let Some(first) = messages.get_mut(1) {
        first.content.insert_str(0, "(edited) ");
    }
    let edited = encode(loaded, &messages)?;
    Ok(vec![
        Case {
            label: format!("after {} tokens", before.len()),
            before: before.clone(),
            timed,
        },
        Case {
            label: format!("after {}, edited", before.len()),
            before,
            timed: edited,
        },
    ])
}

fn run(
    engine: &Engine,
    cache: &mut EngineCache,
    retained: &mut Retained,
    ids: &[u32],
    params: &GenerateParams,
    energy: bool,
) -> Result<Sample, CliError> {
    let watch = energy.then(GpuSampler::start).and_then(Result::ok);
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let mut tokens = Vec::new();
    let mut least_free: Option<u64> = None;
    let mut vram = |engine: &Engine| {
        if let Some((free, total)) = engine.gpu_memory() {
            least_free = Some(least_free.map_or(free, |f| f.min(free)));
            return Some(total);
        }
        None
    };
    let started = Instant::now();
    let outcome = generate_reusing(
        engine,
        Kept { cache, retained },
        &mut greedy()?,
        ids,
        params,
        &cancel,
        |t| {
            if tokens.is_empty() {
                vram(engine);
            }
            tokens.push(t);
        },
    )?;
    let total = started.elapsed();
    let device = vram(engine);
    let watts = watch
        .and_then(|w| w.stop().ok())
        .and_then(|s| s.power_watts)
        .map(|p| p.mean);
    Ok(Sample {
        outcome,
        tokens,
        vram: device.zip(least_free).map(|(t, f)| t.saturating_sub(f)),
        watts,
        total,
    })
}

fn timed_turn(
    loaded: &Loaded,
    cache: &mut EngineCache,
    case: &Case,
    mode: Mode,
    params: &GenerateParams,
    energy: bool,
) -> Result<Sample, CliError> {
    let mut retained = Retained::default();
    run(
        &loaded.model,
        cache,
        &mut retained,
        &case.before,
        params,
        false,
    )?;
    if mode == Mode::Replay {
        retained.forget();
    }
    run(
        &loaded.model,
        cache,
        &mut retained,
        &case.timed,
        params,
        energy,
    )
}

fn ms(d: Duration) -> String {
    format!("{:.1}", d.as_secs_f64() * 1000.0)
}

fn row(case: &Case, mode: Mode, samples: &[Sample], same: bool, speculate: bool) -> String {
    let Some(first) = samples.first() else {
        return String::new();
    };
    let o = &first.outcome;
    let ttft: Vec<Duration> = samples.iter().map(|s| s.outcome.prefill).collect();
    let pass: Vec<Duration> = samples.iter().map(|s| s.outcome.prompt_forward).collect();
    let (ttft, pass) = (spread(&ttft), spread(&pass));
    let vram = samples.iter().filter_map(|s| s.vram).max();
    let joules: Vec<f64> = samples
        .iter()
        .filter_map(|s| s.watts.map(|w| w * s.total.as_secs_f64()))
        .collect();
    let guesses = if speculate {
        let g = &o.speculation;
        format!(
            "  guessed {}/{} in {} of {} passes",
            g.accepted, g.drafted, g.guessed_passes, g.passes
        )
    } else {
        String::new()
    };
    format!(
        "{:<24} {:<6} {:>6} {:>6} {:>6}  {:>9} ({}..{})  {:>9}  {:>8}  {:>7}  {}{}",
        case.label,
        if mode == Mode::Keep { "keep" } else { "replay" },
        o.prompt_tokens,
        o.reused,
        o.prompt_tokens.saturating_sub(o.reused),
        ttft.map_or_else(String::new, |s| ms(s.p50)),
        ttft.map_or_else(String::new, |s| ms(s.min)),
        ttft.map_or_else(String::new, |s| ms(s.max)),
        pass.map_or_else(String::new, |s| ms(s.p50)),
        vram.map_or_else(|| "-".into(), |b| format!("{} MB", b / 1_000_000)),
        joules
            .get(joules.len() / 2)
            .map_or_else(|| "-".into(), |j| format!("{j:.1} J")),
        if same { "same reply" } else { "REPLY DIFFERS" },
        guesses,
    )
}

pub fn run_turns(
    model_dir: &std::path::Path,
    backend: BackendChoice,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    let loaded = load(
        model_dir,
        Some(options.ctx),
        options.ctx,
        backend,
        options.quant,
        style,
        true,
    )?;
    let mut cache = loaded.model.new_cache(options.ctx)?;
    let params = GenerateParams {
        max_tokens: options.reply_tokens,
        stop_tokens: loaded.model.config().eos_tokens.clone(),
        usable_vocab: loaded.dir.tokenizer.vocab_size(),
        speculation: options.speculate.then_some(Speculation::MEASURED),
    };
    let mut all = Vec::new();
    for target in &options.history {
        all.extend(cases(&loaded, &mut cache, *target, &options.ask, &params)?);
    }
    println!(
        "{:<24} {:<6} {:>6} {:>6} {:>6}  {:>9} (min..max)  {:>9}  {:>8}  {:>7}",
        "turn", "mode", "prompt", "kept", "read", "TTFT ms", "pass ms", "VRAM", "energy"
    );
    for case in &all {
        if case.timed.len() >= options.ctx {
            eprintln!("skipped {}: does not fit --ctx {}", case.label, options.ctx);
            continue;
        }
        let mut samples: [Vec<Sample>; 2] = [Vec::new(), Vec::new()];
        // One warmup round, then alternating order so drift hits both modes alike.
        for round in 0..=options.rounds {
            let order = if round % 2 == 0 {
                [Mode::Keep, Mode::Replay]
            } else {
                [Mode::Replay, Mode::Keep]
            };
            for mode in order {
                if interrupt
                    .cancel_flag()
                    .load(std::sync::atomic::Ordering::SeqCst)
                {
                    return Err(CliError::Cancelled);
                }
                let sample = timed_turn(&loaded, &mut cache, case, mode, &params, options.energy)?;
                if round > 0 {
                    let slot = usize::from(mode == Mode::Replay);
                    if let Some(list) = samples.get_mut(slot) {
                        list.push(sample);
                    }
                }
            }
        }
        let [keep, replay] = &samples;
        let same = keep
            .iter()
            .chain(replay)
            .all(|s| keep.first().is_some_and(|k| k.tokens == s.tokens));
        println!("{}", row(case, Mode::Keep, keep, same, options.speculate));
        println!(
            "{}",
            row(case, Mode::Replay, replay, same, options.speculate)
        );
    }
    Ok(())
}
