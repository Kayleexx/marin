use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use sampler::{Sampler, SamplerError};
use thiserror::Error;

use crate::{CausalLm, SequenceCache};

#[derive(Debug, Error)]
pub enum GenerateError<E: std::error::Error + 'static> {
    #[error("the prompt is empty")]
    EmptyPrompt,
    #[error("the prompt has {prompt} tokens but the context only holds {capacity}")]
    PromptTooLong { prompt: usize, capacity: usize },
    #[error(transparent)]
    Model(E),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    /// Ids at or above this are never sampled (model padding the tokenizer cannot decode).
    pub usable_vocab: usize,
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
    /// Time from each generated token to the next: the inter-token latency of every decode step.
    pub gaps: Vec<Duration>,
}

/// Runs a fresh sequence: prefill the prompt, then sample one token at a time.
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
    };
    if params.max_tokens == 0 {
        return Ok(outcome);
    }
    let start = Instant::now();
    let mut logits = model.forward(prompt, kv).map_err(GenerateError::Model)?;
    outcome.prompt_forward = start.elapsed();
    let mut first_token_at = None;
    let mut previous_token_at: Option<Duration> = None;
    loop {
        if cancel.load(Ordering::Relaxed) {
            outcome.stop = StopReason::Cancelled;
            break;
        }
        let sampling = Instant::now();
        let token = sampler.sample(&logits, params.usable_vocab)?;
        outcome.sampling = outcome.sampling.saturating_add(sampling.elapsed());
        if params.stop_tokens.contains(&token) {
            outcome.stop = StopReason::EndOfText;
            break;
        }
        on_token(token);
        outcome.generated = outcome.generated.saturating_add(1);
        let now = start.elapsed();
        first_token_at.get_or_insert(now);
        if let Some(previous) = previous_token_at {
            outcome.gaps.push(now.saturating_sub(previous));
        }
        previous_token_at = Some(now);
        if outcome.generated >= params.max_tokens {
            break;
        }
        if kv.used() >= kv.capacity() {
            outcome.stop = StopReason::ContextFull;
            break;
        }
        logits = model.forward(&[token], kv).map_err(GenerateError::Model)?;
    }
    let total = start.elapsed();
    outcome.prefill = first_token_at.unwrap_or(total);
    outcome.decode = total.saturating_sub(outcome.prefill);
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use kv::KvCache;
    use sampler::SamplerConfig;

    use super::*;

    /// Predicts `last token + 1` and counts how many tokens it was fed.
    struct CountingModel {
        vocab: usize,
        fed: Cell<usize>,
        fail: bool,
    }

    #[derive(Debug, Error)]
    #[error("model failed")]
    struct Failure;

    impl CausalLm for CountingModel {
        type Error = Failure;
        type Cache = KvCache;

        fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
            self.fed.set(self.fed.get().saturating_add(tokens.len()));
            if self.fail {
                return Err(Failure);
            }
            kv.advance(tokens.len()).map_err(|_| Failure)?;
            let next = tokens
                .last()
                .map_or(0, |t| (*t as usize).saturating_add(1))
                .checked_rem(self.vocab)
                .unwrap_or(0);
            let mut logits = vec![0.0; self.vocab];
            logits[next] = 10.0;
            Ok(logits)
        }
    }

    fn setup(capacity: usize) -> (CountingModel, KvCache, Sampler) {
        let model = CountingModel {
            vocab: 10,
            fed: Cell::new(0),
            fail: false,
        };
        let kv = KvCache::new(1, 1, 1, capacity).unwrap();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        })
        .unwrap();
        (model, kv, sampler)
    }

    fn params(max_tokens: usize, stop: &[u32]) -> GenerateParams {
        GenerateParams {
            max_tokens,
            stop_tokens: stop.to_vec(),
            usable_vocab: 10,
        }
    }

    fn run(
        capacity: usize,
        prompt: &[u32],
        params: &GenerateParams,
        cancel: &AtomicBool,
    ) -> (Result<Outcome, GenerateError<Failure>>, Vec<u32>, usize) {
        let (model, mut kv, mut sampler) = setup(capacity);
        let mut seen = Vec::new();
        let result = generate(&model, &mut kv, &mut sampler, prompt, params, cancel, |t| {
            seen.push(t)
        });
        (result, seen, model.fed.get())
    }

    #[test]
    fn stops_at_max_tokens() {
        let (outcome, seen, _) = run(32, &[1, 2], &params(3, &[]), &AtomicBool::new(false));
        let outcome = outcome.unwrap();
        assert_eq!(seen, [3, 4, 5]);
        assert_eq!(outcome.stop, StopReason::MaxTokens);
        assert_eq!(outcome.generated, 3);
        assert_eq!(outcome.prompt_tokens, 2);
    }

    #[test]
    fn stop_token_ends_generation_and_is_not_emitted() {
        let (outcome, seen, _) = run(32, &[1], &params(10, &[4]), &AtomicBool::new(false));
        assert_eq!(outcome.unwrap().stop, StopReason::EndOfText);
        assert_eq!(seen, [2, 3]);
    }

    #[test]
    fn stops_when_context_is_full() {
        let (outcome, seen, _) = run(4, &[1, 2, 3], &params(10, &[]), &AtomicBool::new(false));
        assert_eq!(outcome.unwrap().stop, StopReason::ContextFull);
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn prompt_that_exactly_fills_context_still_yields_one_token() {
        let (outcome, seen, _) = run(3, &[1, 2, 3], &params(10, &[]), &AtomicBool::new(false));
        assert_eq!(outcome.unwrap().stop, StopReason::ContextFull);
        assert_eq!(seen, [4]);
    }

    #[test]
    fn rejects_empty_and_oversized_prompts() {
        let cancel = AtomicBool::new(false);
        let (empty, _, _) = run(4, &[], &params(1, &[]), &cancel);
        assert!(matches!(empty, Err(GenerateError::EmptyPrompt)));
        let (long, _, _) = run(2, &[1, 2, 3], &params(1, &[]), &cancel);
        assert!(matches!(
            long,
            Err(GenerateError::PromptTooLong {
                prompt: 3,
                capacity: 2
            })
        ));
    }

    #[test]
    fn zero_max_tokens_does_no_work() {
        let (outcome, seen, fed) = run(8, &[1], &params(0, &[]), &AtomicBool::new(false));
        assert_eq!(outcome.unwrap().generated, 0);
        assert!(seen.is_empty());
        assert_eq!(fed, 0);
    }

    #[test]
    fn cancellation_is_reported() {
        let (outcome, seen, _) = run(8, &[1], &params(5, &[]), &AtomicBool::new(true));
        assert_eq!(outcome.unwrap().stop, StopReason::Cancelled);
        assert!(seen.is_empty());
    }

    #[test]
    fn model_errors_surface() {
        let (mut model, mut kv, mut sampler) = setup(8);
        model.fail = true;
        let result = generate(
            &model,
            &mut kv,
            &mut sampler,
            &[1],
            &params(5, &[]),
            &AtomicBool::new(false),
            |_| {},
        );
        assert!(matches!(result, Err(GenerateError::Model(Failure))));
    }
}
