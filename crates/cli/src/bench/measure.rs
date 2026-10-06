use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use loader::Tokenizer;
use metrics::{Stage, StageTimes, spread};
use model::{Engine, EngineCache, Traffic};
use model::{QuantScheme, Qwen3Config};
use runtime::{GenerateParams, Outcome, SequenceCache, StopReason, generate};
use sampler::{Sampler, SamplerConfig};

use super::rates;
use super::record::{GpuPhaseWork, GpuWork, Phase, Profile, StageRow};
use crate::CliError;

/// Fixed English text, repeated as needed, so every benchmark feeds the model the same prompt.
const TEXT: &str = "The lighthouse keeper climbed the spiral stairs each evening, trimmed the lamp, \
and watched the ships cross the dark water while the wind pulled at the old shutters. ";

pub(crate) fn prompt(tokenizer: &Tokenizer, tokens: usize) -> Result<Vec<u32>, CliError> {
    let mut ids = Vec::new();
    while ids.len() < tokens {
        let more = tokenizer.encode(TEXT)?;
        if more.is_empty() {
            return Err(CliError::BadOption(
                "the tokenizer produced no tokens".into(),
            ));
        }
        ids.extend(more);
    }
    ids.truncate(tokens);
    Ok(ids)
}

pub(crate) const GREEDY: SamplerConfig = SamplerConfig {
    temperature: 0.0,
    top_k: 0,
    top_p: 1.0,
    seed: 0,
};

pub(crate) fn greedy() -> Result<Sampler, CliError> {
    sampler(&GREEDY)
}

pub(crate) fn sampler(config: &SamplerConfig) -> Result<Sampler, CliError> {
    Ok(Sampler::new(*config)?)
}

pub struct Run {
    pub outcome: Outcome,
    /// GPU traffic of the prompt pass, and of every pass after it. `None` on the CPU.
    pub prefill_traffic: Option<Traffic>,
    pub decode_traffic: Option<Traffic>,
    /// Least free GPU memory seen between tokens; `None` on the CPU.
    pub gpu_free_min: Option<u64>,
}

/// What one benchmark generation produces and how its tokens are chosen and consumed.
pub struct Generation {
    /// Exactly this many tokens, with no stop token.
    pub output: usize,
    pub usable_vocab: usize,
    pub sampling: SamplerConfig,
    /// Wait after every token, standing in for a slower consumer.
    pub pause: Duration,
}

/// One generation, timed by the generate loop.
pub fn run_once(
    engine: &Engine,
    cache: &mut EngineCache,
    prompt: &[u32],
    plan: &Generation,
    cancel: &AtomicBool,
) -> Result<Run, CliError> {
    let (sampling, pause) = (&plan.sampling, plan.pause);
    let params = GenerateParams {
        max_tokens: plan.output,
        stop_tokens: Vec::new(),
        usable_vocab: plan.usable_vocab,
        speculation: None,
    };
    let before = engine.traffic();
    // The first token is delivered after the prompt pass and before any decode pass.
    let mut at_first_token = None;
    let mut gpu_free_min: Option<u64> = None;
    let outcome = generate(
        engine,
        cache,
        &mut sampler(sampling)?,
        prompt,
        &params,
        cancel,
        |_| {
            if at_first_token.is_none() {
                at_first_token = engine.traffic();
            }
            if let Some((free, _)) = engine.gpu_memory() {
                gpu_free_min = Some(gpu_free_min.map_or(free, |least| least.min(free)));
            }
            // Time spent here belongs to the consumer, not the engine, so the timings leave it out.
            if !pause.is_zero() {
                std::thread::sleep(pause);
            }
        },
    )?;
    if outcome.stop == StopReason::Cancelled {
        return Err(CliError::Cancelled);
    }
    let after = engine.traffic();
    let delta = |later: Option<Traffic>, earlier: Option<Traffic>| {
        later.zip(earlier).map(|(l, e)| l.since(&e))
    };
    Ok(Run {
        outcome,
        prefill_traffic: delta(at_first_token, before),
        decode_traffic: delta(after, at_first_token),
        gpu_free_min,
    })
}

fn gpu_phase(traffic: Traffic, passes: usize) -> Option<GpuPhaseWork> {
    let passes_f = passes as f64;
    (passes > 0).then(|| GpuPhaseWork {
        passes,
        launches_per_pass: traffic.launches as f64 / passes_f,
        allocations_per_pass: traffic.allocations as f64 / passes_f,
        upload_bytes_per_pass: traffic.h2d_bytes as f64 / passes_f,
        upload_ms_per_pass: traffic.h2d_time.as_secs_f64() * 1000.0 / passes_f,
        download_bytes_per_pass: traffic.d2h_bytes as f64 / passes_f,
        download_ms_per_pass: traffic.d2h_time.as_secs_f64() * 1000.0 / passes_f,
        wait_for_gpu_ms_per_pass: traffic.sync_wait.as_secs_f64() * 1000.0 / passes_f,
    })
}

/// What the GPU did per forward pass, with the prompt pass kept apart from the one-token passes.
pub fn gpu_work(run: &Run) -> Option<GpuWork> {
    let decode_passes = run.outcome.generated.saturating_sub(1);
    Some(GpuWork {
        prefill: gpu_phase(run.prefill_traffic?, 1)?,
        decode: gpu_phase(run.decode_traffic?, decode_passes)?,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Prefill,
    Decode,
}

fn phase(
    tokens: usize,
    wall: Duration,
    times: &StageTimes,
    (cfg, quant): (&Qwen3Config, Option<QuantScheme>),
    kind: Kind,
) -> Phase {
    let wall_ms = wall.as_secs_f64() * 1000.0;
    let stages = Stage::ALL
        .iter()
        .map(|stage| {
            let total_ms = times.total(*stage).as_secs_f64() * 1000.0;
            let work = rates::stage_work(cfg, *stage, quant);
            let per_pass_ms = total_ms / tokens.max(1) as f64;
            StageRow {
                stage: stage.name(),
                total_ms,
                calls: times.calls(*stage),
                gb_per_s: work
                    .filter(|_| kind == Kind::Decode)
                    .and_then(|w| rates::gb_per_s(&w, per_pass_ms)),
                tflop_per_s: work
                    .filter(|_| kind == Kind::Prefill)
                    .and_then(|w| rates::tflop_per_s(&w, tokens, total_ms)),
                share_of_wall_percent: if wall_ms > 0.0 {
                    total_ms / wall_ms * 100.0
                } else {
                    0.0
                },
            }
        })
        .collect();
    Phase {
        tokens,
        wall_ms,
        stages,
        outside_stages_ms: wall_ms - times.sum().as_secs_f64() * 1000.0,
    }
}

pub fn clock_name(engine: &Engine) -> &'static str {
    if matches!(engine, Engine::Cpu(_)) {
        "CPU wall clock"
    } else {
        "GPU clock"
    }
}

/// One pass with every stage timed: slower than a plain run, so its job is the shares, not the speed.
pub fn profile(
    engine: &Engine,
    cache: &mut EngineCache,
    prompt: &[u32],
    output: usize,
    usable_vocab: usize,
    sampling: &SamplerConfig,
    clock: &'static str,
) -> Result<Profile, CliError> {
    let mut sampler = sampler(sampling)?;
    let config = engine.config();
    let quant = engine.quantization();
    cache.clear();
    let (mut prefill, mut decode) = (StageTimes::default(), StageTimes::default());
    let started = Instant::now();
    let mut logits = engine.forward_profiled(prompt, cache, &mut prefill)?;
    let prefill_wall = started.elapsed();
    let steps = output.saturating_sub(1);
    let mut decode_wall = Duration::ZERO;
    for _ in 0..steps {
        let token = sampler.sample(&logits, usable_vocab)?;
        let started = Instant::now();
        logits = engine.forward_profiled(&[token], cache, &mut decode)?;
        decode_wall = decode_wall.saturating_add(started.elapsed());
    }
    Ok(Profile {
        clock,
        prefill: phase(
            prompt.len(),
            prefill_wall,
            &prefill,
            (config, quant),
            Kind::Prefill,
        ),
        decode: phase(steps, decode_wall, &decode, (config, quant), Kind::Decode),
    })
}

pub fn token_latency(outcome: &Outcome) -> Option<(f64, f64)> {
    let s = spread(&outcome.gaps)?;
    Some((s.p50.as_secs_f64() * 1000.0, s.p95.as_secs_f64() * 1000.0))
}

#[cfg(test)]
mod tests {
    use runtime::StopReason;

    use super::*;

    fn run(generated: usize, prefill: Traffic, decode: Traffic) -> Run {
        Run {
            outcome: Outcome {
                prompt_tokens: 8,
                reused: 0,
                generated,
                stop: StopReason::MaxTokens,
                prefill: Duration::ZERO,
                decode: Duration::ZERO,
                prompt_forward: Duration::ZERO,
                sampling: Duration::ZERO,
                gaps: Vec::new(),
                speculation: Default::default(),
            },
            prefill_traffic: Some(prefill),
            decode_traffic: Some(decode),
            gpu_free_min: None,
        }
    }

    fn launches(launches: u64) -> Traffic {
        Traffic {
            launches,
            ..Traffic::default()
        }
    }

    #[test]
    fn gpu_work_keeps_the_prompt_pass_apart_from_one_token_passes() {
        let work = gpu_work(&run(5, launches(1000), launches(400))).unwrap();
        assert_eq!(
            (work.prefill.passes, work.prefill.launches_per_pass),
            (1, 1000.0)
        );
        assert_eq!(
            (work.decode.passes, work.decode.launches_per_pass),
            (4, 100.0)
        );
    }

    #[test]
    fn gpu_work_needs_a_decode_pass_and_gpu_traffic() {
        assert!(gpu_work(&run(1, launches(1000), launches(0))).is_none());
        let mut cpu = run(5, launches(1), launches(1));
        cpu.decode_traffic = None;
        assert!(gpu_work(&cpu).is_none());
    }
}
