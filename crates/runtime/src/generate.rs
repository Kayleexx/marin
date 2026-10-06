use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use sampler::{Sampler, SamplerError};
use thiserror::Error;

use crate::speculate::{Drafter, Speculation, SpeculationStats};
use crate::{CausalLm, RollbackError, SequenceCache};

#[derive(Debug, Error)]
pub enum GenerateError<E: std::error::Error + 'static> {
    #[error("the prompt is empty")]
    EmptyPrompt,
    #[error("the prompt has {prompt} tokens but the context only holds {capacity}")]
    PromptTooLong { prompt: usize, capacity: usize },
    #[error("guessing ahead needs a match length and a guess length of at least 1")]
    BadSpeculation,
    #[error(transparent)]
    Model(E),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error(transparent)]
    Rollback(#[from] RollbackError),
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    /// Ids at or above this are never sampled (model padding the tokenizer cannot decode).
    pub usable_vocab: usize,
    /// Guess ahead by copying earlier text and check the guesses in one pass. The tokens are the ones plain
    /// decoding gives. Ignored when the model cannot check several tokens in one pass.
    pub speculation: Option<Speculation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    EndOfText,
    ContextFull,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub stop: StopReason,
    /// Time until the first token was ready: prompt processing plus the first sample.
    pub prefill: Duration,
    /// Time spent producing every token after the first.
    pub decode: Duration,
    /// The forward pass over the prompt alone, without sampling.
    pub prompt_forward: Duration,
    /// Time spent choosing tokens (sampling), over the whole reply.
    pub sampling: Duration,
    /// Time from each generated token to the next: the inter-token latency of every decode step. Tokens settled by
    /// one check of guesses arrive together, so most of their gaps are near zero.
    pub gaps: Vec<Duration>,
    pub speculation: SpeculationStats,
}

/// The logits after each token of one pass, one row per token.
struct Rows {
    values: Vec<f32>,
    count: usize,
}

impl Rows {
    fn row(&self, index: usize) -> &[f32] {
        let width = self.values.len().checked_div(self.count).unwrap_or(0);
        self.values
            .get(index.saturating_mul(width)..index.saturating_add(1).saturating_mul(width))
            .unwrap_or(&[])
    }
}

/// Runs a fresh sequence: prefill the prompt, then sample one token at a time, or, with `params.speculation` and a
/// model that can check several tokens in one pass, several at a time.
/// `on_token` sees each generated token except the stop token.
pub fn generate<M: CausalLm>(
    model: &M,
    kv: &mut M::Cache,
    sampler: &mut Sampler,
    prompt: &[u32],
    params: &GenerateParams,
    cancel: &AtomicBool,
    mut on_token: impl FnMut(u32),
) -> Result<Outcome, GenerateError<M::Error>> {
    if prompt.is_empty() {
        return Err(GenerateError::EmptyPrompt);
    }
    if prompt.len() > kv.capacity() {
        return Err(GenerateError::PromptTooLong {
            prompt: prompt.len(),
            capacity: kv.capacity(),
        });
    }
    if params
        .speculation
        .is_some_and(|s| s.match_len == 0 || s.max_draft == 0)
    {
        return Err(GenerateError::BadSpeculation);
    }
    kv.clear();
    let mut outcome = Outcome {
        prompt_tokens: prompt.len(),
        generated: 0,
        stop: StopReason::MaxTokens,
        prefill: Duration::ZERO,
        decode: Duration::ZERO,
        prompt_forward: Duration::ZERO,
        sampling: Duration::ZERO,
        gaps: Vec::new(),
        speculation: SpeculationStats::default(),
    };
    if params.max_tokens == 0 {
        return Ok(outcome);
    }
    let start = Instant::now();
    let logits = model.forward(prompt, kv).map_err(GenerateError::Model)?;
    outcome.prompt_forward = start.elapsed();
    let mut drafter = params
        .speculation
        .filter(|_| model.max_rows() >= 2)
        .map(|rule| Drafter::new(rule, model.max_rows()));
    if let Some(drafter) = drafter.as_mut() {
        prompt
            .iter()
            .for_each(|t| drafter.push(*t, &mut outcome.speculation));
    }
    let mut rows = Rows {
        values: logits,
        count: 1,
    };
    // Guessed tokens fed with the last token in this pass, and the positions known good once row `j` is accepted:
    // `valid_base + j`.
    let mut draft: Vec<u32> = Vec::new();
    let mut valid_base = kv.used();
    // Time spent inside `on_token` (printing, say) is the caller's, not the engine's.
    let mut consumer = Duration::ZERO;
    let mut first_token_at = None;
    let mut previous_token_at: Option<Duration> = None;
    let mut last = 0;
    loop {
        let mut consumed = 0;
        let mut finished = false;
        for index in 0..rows.count {
            consumed = index;
            if cancel.load(Ordering::Relaxed) {
                outcome.stop = StopReason::Cancelled;
                finished = true;
                break;
            }
            let sampling = Instant::now();
            let token = sampler.sample(rows.row(index), params.usable_vocab)?;
            outcome.sampling = outcome.sampling.saturating_add(sampling.elapsed());
            if params.stop_tokens.contains(&token) {
                outcome.stop = StopReason::EndOfText;
                finished = true;
                break;
            }
            let now = start.elapsed().saturating_sub(consumer);
            let delivered = Instant::now();
            on_token(token);
            consumer = consumer.saturating_add(delivered.elapsed());
            outcome.generated = outcome.generated.saturating_add(1);
            first_token_at.get_or_insert(now);
            if let Some(previous) = previous_token_at {
                outcome.gaps.push(now.saturating_sub(previous));
            }
            previous_token_at = Some(now);
            last = token;
            if let Some(drafter) = drafter.as_mut() {
                drafter.push(token, &mut outcome.speculation);
            }
            if outcome.generated >= params.max_tokens {
                finished = true;
                break;
            }
            if valid_base.saturating_add(index) >= kv.capacity() {
                outcome.stop = StopReason::ContextFull;
                finished = true;
                break;
            }
            if draft.get(index) != Some(&token) {
                break;
            }
        }
        if rows.count > 1 {
            let rolling_back = Instant::now();
            kv.truncate(valid_base.saturating_add(consumed))?;
            let spent = rolling_back.elapsed();
            outcome.speculation.rollback = outcome.speculation.rollback.saturating_add(spent);
            outcome.speculation.accepted = outcome.speculation.accepted.saturating_add(consumed);
        }
        if finished {
            break;
        }
        let room = kv.capacity().saturating_sub(kv.used());
        let remaining = params.max_tokens.saturating_sub(outcome.generated);
        draft = match drafter.as_ref() {
            Some(drafter) => drafter.draft(
                room.saturating_sub(1).min(remaining.saturating_sub(1)),
                &mut outcome.speculation,
            ),
            None => Vec::new(),
        };
        outcome.speculation.passes = outcome.speculation.passes.saturating_add(1);
        if draft.is_empty() {
            let logits = model.forward(&[last], kv).map_err(GenerateError::Model)?;
            rows = Rows {
                values: logits,
                count: 1,
            };
            valid_base = kv.used();
        } else {
            let mut fed = vec![last];
            fed.extend(&draft);
            let before = kv.used();
            let values = model.forward_rows(&fed, kv).map_err(GenerateError::Model)?;
            rows = Rows {
                values,
                count: fed.len(),
            };
            valid_base = before.saturating_add(1);
            outcome.speculation.guessed_passes =
                outcome.speculation.guessed_passes.saturating_add(1);
            outcome.speculation.drafted = outcome.speculation.drafted.saturating_add(draft.len());
        }
    }
    let total = start.elapsed().saturating_sub(consumer);
    outcome.prefill = first_token_at.unwrap_or(total);
    outcome.decode = total.saturating_sub(outcome.prefill);
    Ok(outcome)
}
