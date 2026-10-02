use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use loader::Tokenizer;
use metrics::{Stage, StageTimes, spread};
use model::{Engine, EngineCache, Traffic};
use runtime::{GenerateParams, Outcome, SequenceCache, generate};
use sampler::{Sampler, SamplerConfig};

use super::record::{GpuWork, Phase, Profile, StageRow};
use crate::CliError;

/// Fixed English text, repeated as needed, so every benchmark feeds the model the same prompt.
const TEXT: &str = "The lighthouse keeper climbed the spiral stairs each evening, trimmed the lamp, \
and watched the ships cross the dark water while the wind pulled at the old shutters. ";

pub fn prompt(tokenizer: &Tokenizer, tokens: usize) -> Result<Vec<u32>, CliError> {
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

fn greedy() -> Result<Sampler, CliError> {
    Ok(Sampler::new(SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 0,
    })?)
}

pub struct Run {
    pub outcome: Outcome,
    pub traffic: Option<Traffic>,
}

/// One greedy generation of exactly `output` tokens (no stop token), timed by the generate loop.
pub fn run_once(
    engine: &Engine,
    cache: &mut EngineCache,
    prompt: &[u32],
    output: usize,
    usable_vocab: usize,
) -> Result<Run, CliError> {
    let params = GenerateParams {
        max_tokens: output,
        stop_tokens: Vec::new(),
        usable_vocab,
    };
    let before = engine.traffic();
    let outcome = generate(
        engine,
        cache,
        &mut greedy()?,
        prompt,
        &params,
        &AtomicBool::new(false),
        |_| {},
    )?;
    let traffic = engine
        .traffic()
        .zip(before)
        .map(|(after, before)| after.since(&before));
    Ok(Run { outcome, traffic })
}

/// What the GPU did per forward pass of a run (one prompt pass plus one pass per further token).
pub fn gpu_work(run: &Run) -> Option<GpuWork> {
    let traffic = run.traffic?;
    let passes = run.outcome.generated.max(1) as f64;
    Some(GpuWork {
        launches_per_token: traffic.launches as f64 / passes,
        allocations_per_token: traffic.allocations as f64 / passes,
        download_bytes_per_token: traffic.d2h_bytes as f64 / passes,
        download_ms_per_token: traffic.d2h_time.as_secs_f64() * 1000.0 / passes,
        wait_for_gpu_ms_per_token: traffic.sync_wait.as_secs_f64() * 1000.0 / passes,
    })
}

fn phase(tokens: usize, wall: Duration, times: &StageTimes) -> Phase {
    let wall_ms = wall.as_secs_f64() * 1000.0;
    let stages = Stage::ALL
        .iter()
        .map(|stage| {
            let total_ms = times.total(*stage).as_secs_f64() * 1000.0;
            StageRow {
                stage: stage.name(),
                total_ms,
                calls: times.calls(*stage),
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

/// One pass with every stage timed: slower than a plain run, so its job is the shares, not the speed.
pub fn profile(
    engine: &Engine,
    cache: &mut EngineCache,
    prompt: &[u32],
    output: usize,
    usable_vocab: usize,
    clock: &'static str,
) -> Result<Profile, CliError> {
    let mut sampler = greedy()?;
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
        prefill: phase(prompt.len(), prefill_wall, &prefill),
        decode: phase(steps, decode_wall, &decode),
    })
}

pub fn token_latency(outcome: &Outcome) -> Option<(f64, f64)> {
    let s = spread(&outcome.gaps)?;
    Some((s.p50.as_secs_f64() * 1000.0, s.p95.as_secs_f64() * 1000.0))
}
