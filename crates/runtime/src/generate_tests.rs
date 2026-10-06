use std::cell::Cell;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use kv::KvCache;
use sampler::{Sampler, SamplerConfig};
use thiserror::Error;

use crate::CausalLm;
use crate::generate::{GenerateError, GenerateParams, Outcome, StopReason, generate};

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
        speculation: None,
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
fn slow_token_consumer_does_not_inflate_timings() {
    let pause = Duration::from_millis(40);
    let (model, mut kv, mut sampler) = setup(32);
    let outcome = generate(
        &model,
        &mut kv,
        &mut sampler,
        &[1],
        &params(4, &[]),
        &AtomicBool::new(false),
        |_| std::thread::sleep(pause),
    )
    .unwrap();
    let half = pause / 2;
    assert!(outcome.prefill < half, "first token {:?}", outcome.prefill);
    assert!(outcome.decode < half, "decode {:?}", outcome.decode);
    assert_eq!(outcome.gaps.len(), 3);
    assert!(outcome.gaps.iter().all(|gap| *gap < half));
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
