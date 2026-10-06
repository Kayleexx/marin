use std::time::Duration;

use serde::{Deserialize, Serialize};

use runtime::Outcome;

pub const FORMAT: u32 = 1;

/// One decode of one prompt in one mode.
#[derive(Serialize, Deserialize, Clone)]
pub struct RunRecord {
    pub class: String,
    pub name: String,
    pub mode: String,
    pub round: usize,
    pub rep: usize,
    pub prompt_tokens: usize,
    pub ttft_ms: f64,
    /// Tokens generated after the first, over `decode_ms`.
    pub tokens: usize,
    pub decode_ms: f64,
    pub passes: usize,
    pub guessed: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Time spent choosing tokens on the CPU over the whole reply.
    #[serde(default)]
    pub sampling_ms: f64,
    pub guesser_ms: f64,
    pub rollback_ms: f64,
    /// Time between one generated token and the next; tokens settled by one check of guesses arrive together.
    pub gap_p50_ms: f64,
    pub gap_p95_ms: f64,
    /// The tokens equal those of plain decoding of the same prompt.
    pub same_as_plain: bool,
    /// Mean GPU power and clock over the block of runs this one was in, and its energy per generated token.
    pub power_w: Option<f64>,
    pub clock_mhz: Option<f64>,
    pub joules_per_token: Option<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct Saved {
    pub format: u32,
    pub model: String,
    pub quantization: String,
    pub backend: String,
    pub max_tokens: usize,
    pub runs: Vec<RunRecord>,
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let index = (sorted.len().saturating_sub(1) as f64 * q).round() as usize;
    sorted.get(index).copied().unwrap_or(0.0)
}

#[derive(Clone)]
pub struct Block {
    pub power_w: Option<f64>,
    pub clock_mhz: Option<f64>,
    pub joules_per_token: Option<f64>,
}

/// Which prompt and run a decode belongs to.
pub struct Meta<'a> {
    pub class: &'a str,
    pub name: &'a str,
    pub round: usize,
    pub rep: usize,
    pub prompt_tokens: usize,
}

pub fn record(
    meta: &Meta<'_>,
    mode: &str,
    outcome: &Outcome,
    block: &Block,
    same_as_plain: bool,
) -> RunRecord {
    let mut gaps: Vec<f64> = outcome.gaps.iter().map(|d| ms(*d)).collect();
    gaps.sort_by(f64::total_cmp);
    let stats = &outcome.speculation;
    RunRecord {
        class: meta.class.into(),
        name: meta.name.into(),
        mode: mode.into(),
        round: meta.round,
        rep: meta.rep,
        prompt_tokens: meta.prompt_tokens,
        ttft_ms: ms(outcome.prefill),
        tokens: outcome.generated.saturating_sub(1),
        decode_ms: ms(outcome.decode),
        passes: stats.passes,
        guessed: stats.guessed_passes,
        drafted: stats.drafted,
        accepted: stats.accepted,
        sampling_ms: ms(outcome.sampling),
        guesser_ms: ms(stats.guesser),
        rollback_ms: ms(stats.rollback),
        gap_p50_ms: percentile(&gaps, 0.5),
        gap_p95_ms: percentile(&gaps, 0.95),
        same_as_plain,
        power_w: block.power_w,
        clock_mhz: block.clock_mhz,
        joules_per_token: block.joules_per_token,
    }
}
