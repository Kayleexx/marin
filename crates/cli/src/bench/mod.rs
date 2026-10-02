mod measure;
mod record;
mod render;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use metrics::process_memory;
use model::BackendChoice;
use runtime::SequenceCache;

use self::record::{Load, Record, Results, Stat, hardware};
use crate::CliError;
use crate::load::load;
use crate::style::Style;

pub struct Options {
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub runs: usize,
    pub warmup: usize,
    pub profile: bool,
    pub json: Option<PathBuf>,
}

fn stat(values: &[f64]) -> Result<Stat, CliError> {
    Stat::of(values).ok_or_else(|| CliError::BadOption("no runs were measured".into()))
}

pub fn run(
    model_dir: &Path,
    backend: BackendChoice,
    ctx: Option<usize>,
    options: &Options,
    style: &Style,
) -> Result<(), CliError> {
    if options.runs == 0 || options.output_tokens < 2 || options.prompt_tokens == 0 {
        return Err(CliError::BadOption(
            "bench needs --prompt-tokens above 0, --tokens of at least 2 and --runs above 0".into(),
        ));
    }
    let needed = options
        .prompt_tokens
        .saturating_add(options.output_tokens)
        .saturating_add(1);
    let ctx = ctx.unwrap_or(needed);
    let rss_before = process_memory().map(|m| m.rss_bytes);
    let loaded = load(model_dir, Some(ctx), ctx, backend, style, true)?;
    let engine = &loaded.model;
    let mut cache = engine.new_cache(ctx)?;
    let gpu_growth = loaded
        .gpu_free_before
        .zip(engine.gpu_memory())
        .map(|(before, (after, _))| before.saturating_sub(after));
    let rss_growth = process_memory()
        .zip(rss_before)
        .map(|(after, before)| after.rss_bytes.saturating_sub(before));
    let prompt = measure::prompt(&loaded.dir.tokenizer, options.prompt_tokens)?;
    let vocab = loaded.dir.tokenizer.vocab_size();

    let mut runs = Vec::new();
    for round in 0..options.warmup.saturating_add(options.runs) {
        let run = measure::run_once(engine, &mut cache, &prompt, options.output_tokens, vocab)?;
        if round >= options.warmup {
            runs.push(run);
        }
    }
    let per_run = |f: &dyn Fn(&runtime::Outcome) -> Option<f64>| -> Result<Stat, CliError> {
        stat(
            &runs
                .iter()
                .filter_map(|r| f(&r.outcome))
                .collect::<Vec<_>>(),
        )
    };
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let decoded = options.output_tokens.saturating_sub(1);
    let results = Results {
        first_token_ms: per_run(&|o| Some(ms(o.prefill)))?,
        prefill_tokens_per_s: per_run(&|o| {
            crate::telemetry::rate(o.prompt_tokens, o.prompt_forward)
        })?,
        decode_tokens_per_s: per_run(&|o| crate::telemetry::rate(decoded, o.decode))?,
        token_latency_p50_ms: per_run(&|o| measure::token_latency(o).map(|l| l.0))?,
        token_latency_p95_ms: per_run(&|o| measure::token_latency(o).map(|l| l.1))?,
        sampling_ms_per_token: per_run(&|o| Some(ms(o.sampling) / o.generated.max(1) as f64))?,
        gpu: runs.last().and_then(measure::gpu_work),
        peak_rss_bytes: process_memory().map(|m| m.peak_rss_bytes),
    };
    let clock = if matches!(engine, model::Engine::Cpu(_)) {
        "CPU wall clock"
    } else {
        "GPU clock"
    };
    let profile = options
        .profile
        .then(|| {
            measure::profile(
                engine,
                &mut cache,
                &prompt,
                options.output_tokens,
                vocab,
                clock,
            )
        })
        .transpose()?;
    let record = Record {
        format: 1,
        unix_time: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        model: loaded.name.clone(),
        precision: "bf16 weights, f32 math",
        quantization: "none",
        context_length: cache.capacity(),
        batch_size: 1,
        prompt_tokens: prompt.len(),
        output_tokens: options.output_tokens,
        warmup_runs: options.warmup,
        measured_runs: options.runs,
        runtime: format!("marin {}", env!("CARGO_PKG_VERSION")),
        hardware: hardware(engine.backend_name()),
        load: Load {
            seconds: loaded.took.as_secs_f64(),
            weights_bytes: loaded.footprint.weights,
            kv_reserved_bytes: loaded.footprint.kv,
            rss_growth_bytes: rss_growth,
            gpu_growth_bytes: gpu_growth,
        },
        results,
        profile,
    };
    print!("{}", render::text(&record));
    if let Some(path) = &options.json {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&record)
                .map_err(|e| CliError::BadOption(e.to_string()))?
                + "\n",
        )?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
