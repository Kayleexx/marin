mod compare;
pub(crate) mod measure;
mod rates;
pub(crate) mod record;
pub(crate) mod render;
pub(crate) mod resources;
mod status;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use metrics::{disk_read_bytes, per_second, process_memory};
use model::{BackendChoice, Precision};
use runtime::SequenceCache;
use sampler::SamplerConfig;

pub use self::compare::run as compare;
use self::record::{
    GpuSampling, Load, Record, Results, RunRecord, SamplerSettings, Stat, Transfer, hardware,
    runtime_version,
};
use self::resources::{CpuWatch, GpuWatch};
use self::status::Status;
use crate::CliError;
use crate::json_out::JsonOut;
use crate::load::load;
use crate::signal::Interrupt;
use crate::style::Style;

pub struct Options {
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub runs: usize,
    pub warmup: usize,
    pub profile: bool,
    pub sample_gpu: bool,
    /// How tokens are chosen. The default is greedy; set it to the chat defaults to measure what chat pays.
    pub sampling: SamplerConfig,
    /// Milliseconds to wait after every token, as a slower consumer would: it shows what idle time between
    /// tokens does to the GPU.
    pub token_pause_ms: f64,
    pub json: Option<PathBuf>,
    /// Store weights quantized (INT8 on CPU, INT8/INT4 on GPU).
    pub precision: Precision,
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
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    if options.runs == 0 || options.output_tokens < 2 || options.prompt_tokens == 0 {
        return Err(CliError::BadOption(
            "bench needs --prompt-tokens above 0, --tokens of at least 2 and --runs above 0".into(),
        ));
    }
    options.sampling.validate()?;
    if !(0.0..=1000.0).contains(&options.token_pause_ms) {
        return Err(CliError::BadOption(
            "--token-pause-ms must be between 0 and 1000".into(),
        ));
    }
    let token_pause = Duration::from_secs_f64(options.token_pause_ms / 1000.0);
    let json = options.json.as_deref().map(JsonOut::create).transpose()?;
    let needed = options
        .prompt_tokens
        .saturating_add(options.output_tokens)
        .saturating_add(1);
    let ctx = ctx.unwrap_or(needed);
    let rss_before = process_memory().map(|m| m.rss_bytes);
    let disk_before = disk_read_bytes();
    let loaded = load(
        model_dir,
        Some(ctx),
        ctx,
        backend,
        options.precision,
        style,
        true,
    )?;
    let cancel = interrupt.cancel_flag();
    if cancel.load(Ordering::SeqCst) {
        return Err(CliError::Cancelled);
    }
    let engine = &loaded.model;
    let mut cache = engine.new_cache(ctx)?;
    let gpu_growth = loaded
        .gpu_free_before
        .zip(engine.gpu_memory())
        .map(|(before, (after, _))| before.saturating_sub(after));
    let rss_growth = process_memory()
        .zip(rss_before)
        .map(|(after, before)| after.rss_bytes.saturating_sub(before));
    let disk_read = disk_read_bytes()
        .zip(disk_before)
        .map(|(after, before)| after.saturating_sub(before));
    let gpu_upload = engine.traffic().map(|t| Transfer {
        bytes: t.h2d_bytes,
        seconds: t.h2d_time.as_secs_f64(),
        gb_per_s: per_second(t.h2d_bytes, t.h2d_time).map(|rate| rate / 1e9),
    });
    let prompt = measure::prompt(&loaded.dir.tokenizer, options.prompt_tokens)?;
    let vocab = loaded.dir.tokenizer.vocab_size();

    let plan = measure::Generation {
        output: options.output_tokens,
        usable_vocab: vocab,
        sampling: options.sampling,
        pause: token_pause,
    };
    let status = Status::new();
    let gpu_watch = GpuWatch::start(options.sample_gpu, engine);
    let cpu_watch = CpuWatch::start();
    let mut runs = Vec::new();
    for round in 0..options.warmup.saturating_add(options.runs) {
        match round.checked_sub(options.warmup) {
            None => status.show(&format!(
                "warmup {}/{}",
                round.saturating_add(1),
                options.warmup
            )),
            Some(n) => status.show(&format!("run {}/{}", n.saturating_add(1), options.runs)),
        }
        let run = measure::run_once(engine, &mut cache, &prompt, &plan, cancel)?;
        if round >= options.warmup {
            runs.push(run);
        }
    }
    let cpu_busy_cores = cpu_watch.busy_cores();
    let loop_seconds = cpu_watch.elapsed().as_secs_f64();
    let gpu_sampling = gpu_watch.finish();
    let generated_tokens = options
        .warmup
        .saturating_add(options.runs)
        .saturating_mul(options.output_tokens);
    let energy_per_token = match &gpu_sampling {
        GpuSampling::Sampled {
            power_watts: Some(power),
            ..
        } if generated_tokens > 0 => Some(power.mean * loop_seconds / generated_tokens as f64),
        _ => None,
    };
    status.clear();
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
    let weights = loaded.footprint.weights;
    let gpu_peak = engine
        .gpu_memory()
        .zip(runs.iter().filter_map(|r| r.gpu_free_min).min())
        .map(|((_, total), least_free)| total.saturating_sub(least_free));
    let results = Results {
        first_token_ms: per_run(&|o| Some(ms(o.prefill)))?,
        prefill_tokens_per_s: per_run(&|o| {
            crate::telemetry::rate(o.prompt_tokens, o.prompt_forward)
        })?,
        decode_tokens_per_s: per_run(&|o| crate::telemetry::rate(decoded, o.decode))?,
        token_latency_p50_ms: per_run(&|o| measure::token_latency(o).map(|l| l.0))?,
        token_latency_p95_ms: per_run(&|o| measure::token_latency(o).map(|l| l.1))?,
        sampling_ms_per_token: per_run(&|o| Some(ms(o.sampling) / o.generated.max(1) as f64))?,
        weight_bandwidth_gb_per_s: per_run(&|o| {
            let bytes = weights.checked_mul(u64::try_from(decoded).ok()?)?;
            per_second(bytes, o.decode).map(|rate| rate / 1e9)
        })?,
        cpu_busy_cores,
        gpu_device_used_peak_bytes: gpu_peak,
        gpu_sampling,
        gpu_energy_per_token_joules: energy_per_token,
        gpu: runs.last().and_then(measure::gpu_work),
        peak_rss_bytes: process_memory().map(|m| m.peak_rss_bytes),
    };
    let clock = measure::clock_name(engine);
    let profile = options
        .profile
        .then(|| {
            measure::profile(
                engine,
                &mut cache,
                &prompt,
                options.output_tokens,
                vocab,
                &options.sampling,
                clock,
            )
        })
        .transpose()?;
    let record = Record {
        format: 2,
        unix_time: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        model: loaded.name.clone(),
        precision: model::PRECISION,
        quantization: options
            .precision
            .weights
            .map_or_else(|| "none".to_string(), |q| q.label()),
        context_length: cache.capacity(),
        batch_size: model::BATCH_SIZE,
        prompt_tokens: prompt.len(),
        output_tokens: options.output_tokens,
        warmup_runs: options.warmup,
        measured_runs: options.runs,
        sampler: SamplerSettings {
            temperature: options.sampling.temperature,
            top_k: options.sampling.top_k,
            top_p: options.sampling.top_p,
            seed: options.sampling.seed,
        },
        token_pause_ms: options.token_pause_ms,
        runtime: runtime_version(),
        hardware: hardware(engine.backend_name(), engine.gpu_info()),
        load: Load {
            seconds: loaded.took.as_secs_f64(),
            weights_bytes: loaded.footprint.weights,
            kv_reserved_bytes: loaded.footprint.kv_window,
            kv_held_pages: cache
                .kv_held()
                .and_then(|(pages, _)| u64::try_from(pages).ok()),
            kv_held_bytes: u64::try_from(cache.kv_bytes_held()).ok(),
            rss_growth_bytes: rss_growth,
            disk_read_bytes: disk_read,
            gpu_upload,
            gpu_device_growth_bytes: gpu_growth,
        },
        results,
        runs: runs.iter().map(|r| run_record(&r.outcome)).collect(),
        profile,
    };
    std::io::stdout()
        .lock()
        .write_all(render::text(&record).as_bytes())?;
    if let (Some(json), Some(path)) = (json, &options.json) {
        json.finish(&record)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn run_record(o: &runtime::Outcome) -> RunRecord {
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    RunRecord {
        first_token_ms: ms(o.prefill),
        prompt_forward_ms: ms(o.prompt_forward),
        decode_ms: ms(o.decode),
        total_ms: ms(o.prefill.saturating_add(o.decode)),
        sampling_ms: ms(o.sampling),
        generated: o.generated,
        token_gaps_ms: o.gaps.iter().map(|g| ms(*g)).collect(),
    }
}

/// The "where the time goes" tables for `prompt`: one prompt pass and, if `output` is above 1, one-token passes.
pub fn profile_report(
    engine: &model::Engine,
    cache: &mut model::EngineCache,
    prompt: &[u32],
    output: usize,
    vocab: usize,
) -> Result<String, CliError> {
    let clock = measure::clock_name(engine);
    let profile = measure::profile(
        engine,
        cache,
        prompt,
        output,
        vocab,
        &measure::GREEDY,
        clock,
    )?;
    Ok(render::profile_section(&profile))
}
