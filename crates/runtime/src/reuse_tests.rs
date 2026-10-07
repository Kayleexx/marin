//! Keeping the cache between runs must give exactly what reading the whole prompt from scratch gives, and the record
//! of what the cache holds must always be the truth. The model here stores each token id as its key and predicts from
//! every earlier position it reads back, so a kept position that does not match the prompt, or a stale one left
//! behind, changes the tokens.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use kv::KvCache;
use sampler::{Sampler, SamplerConfig};
use thiserror::Error;

use crate::generate::{GenerateError, GenerateParams, Outcome, StopReason, generate};
use crate::{CausalLm, Kept, Retained, SequenceCache, Speculation, generate_reusing};

const VOCAB: usize = 7;

#[derive(Debug, Error)]
#[error("model failed")]
struct Failure;

#[derive(Default)]
struct WholeHistory {
    /// Tokens fed over the model's life.
    fed: Cell<usize>,
    /// Fail the pass with this number (counting from 1), after committing its tokens, like a long prompt whose last
    /// piece fails after the earlier pieces were written.
    fail_on: Option<usize>,
    passes: Cell<usize>,
    /// Break the contract: read the tokens without committing them to the cache.
    skip_commit: bool,
}

impl WholeHistory {
    fn feed(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
        self.fed.set(self.fed.get().saturating_add(tokens.len()));
        self.passes.set(self.passes.get().saturating_add(1));
        let start = kv.len();
        let keys: Vec<f32> = tokens.iter().map(|t| *t as f32).collect();
        kv.write(0, &keys, &keys).map_err(|_| Failure)?;
        if !self.skip_commit {
            kv.advance(tokens.len()).map_err(|_| Failure)?;
        }
        if self.fail_on == Some(self.passes.get()) {
            return Err(Failure);
        }
        let end = start.saturating_add(tokens.len());
        let (held, _) = kv.read(0, end).map_err(|_| Failure)?;
        let mut rows = Vec::new();
        for position in start..end {
            let mixed = held[..=position].iter().fold(position, |h, v| {
                h.wrapping_mul(31).wrapping_add(*v as usize).wrapping_add(1)
            });
            let next = mixed % VOCAB;
            let mut logits = vec![0.0; VOCAB];
            logits[next] = 3.0;
            logits[next.wrapping_add(1) % VOCAB] = 2.5;
            rows.extend(logits);
        }
        Ok(rows)
    }
}

impl CausalLm for WholeHistory {
    type Error = Failure;
    type Cache = KvCache;

    fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
        let rows = self.feed(tokens, kv)?;
        Ok(rows[rows.len().saturating_sub(VOCAB)..].to_vec())
    }

    fn max_rows(&self) -> usize {
        8
    }

    fn forward_rows(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
        self.feed(tokens, kv)
    }
}

fn params(max_tokens: usize, stop: &[u32], guess: bool) -> GenerateParams {
    GenerateParams {
        max_tokens,
        stop_tokens: stop.to_vec(),
        usable_vocab: VOCAB,
        speculation: guess.then_some(Speculation {
            match_len: 1,
            max_draft: 7,
        }),
    }
}

fn sampler(temperature: f32) -> Sampler {
    Sampler::new(SamplerConfig {
        temperature,
        top_k: 0,
        top_p: 1.0,
        seed: 5,
    })
    .unwrap()
}

/// The ids the cache really holds.
fn held(kv: &KvCache) -> Vec<u32> {
    kv.read(0, kv.len())
        .unwrap()
        .0
        .iter()
        .map(|v| *v as u32)
        .collect()
}

struct Turn {
    outcome: Outcome,
    tokens: Vec<u32>,
}

/// One reply on a long-lived cache, cancelled after `cancel_after` tokens when given.
fn turn(
    model: &WholeHistory,
    kv: &mut KvCache,
    retained: &mut Retained,
    prompt: &[u32],
    params: &GenerateParams,
    cancel_after: Option<usize>,
) -> Result<Turn, GenerateError<Failure>> {
    let cancel = AtomicBool::new(false);
    let mut tokens = Vec::new();
    let outcome = generate_reusing(
        model,
        Kept {
            cache: kv,
            retained,
        },
        &mut sampler(0.8),
        prompt,
        params,
        &cancel,
        |t| {
            tokens.push(t);
            if cancel_after == Some(tokens.len()) {
                cancel.store(true, Ordering::Relaxed);
            }
        },
    )?;
    assert_eq!(retained.tokens(), held(kv), "the record is the cache");
    Ok(Turn { outcome, tokens })
}

/// The same reply read from scratch on a new cache.
fn fresh(prompt: &[u32], params: &GenerateParams, capacity: usize) -> Vec<u32> {
    let mut kv = KvCache::new(1, 1, 1, capacity).unwrap();
    let mut tokens = Vec::new();
    generate(
        &WholeHistory::default(),
        &mut kv,
        &mut sampler(0.8),
        prompt,
        params,
        &AtomicBool::new(false),
        |t| tokens.push(t),
    )
    .unwrap();
    tokens
}

fn cache(capacity: usize) -> (WholeHistory, KvCache, Retained) {
    let kv = KvCache::new(1, 1, 1, capacity).unwrap();
    (WholeHistory::default(), kv, Retained::default())
}

/// A chat-like run: each prompt is the last prompt, the reply and a new message, and every reply must equal a
/// fresh one while only the new part is read.
#[test]
fn a_growing_conversation_reads_only_what_is_new_and_answers_as_a_fresh_run_would() {
    for guess in [false, true] {
        let (model, mut kv, mut retained) = cache(400);
        let p = params(12, &[], guess);
        let mut prompt = vec![1, 2, 3];
        for round in 0..6_u32 {
            let fed_before = model.fed.get();
            let kept = retained.len();
            let reply = turn(&model, &mut kv, &mut retained, &prompt, &p, None).unwrap();
            assert_eq!(reply.tokens, fresh(&prompt, &p, 400), "round {round}");
            assert_eq!(reply.outcome.reused, kept, "round {round}");
            assert_eq!(reply.outcome.prompt_tokens, prompt.len());
            if !guess {
                let decoded = reply.tokens.len() - 1;
                assert_eq!(model.fed.get() - fed_before, prompt.len() - kept + decoded);
            }
            // The cache holds the prompt and every reply token but the last, which was never fed.
            let mut expected = prompt.clone();
            expected.extend(&reply.tokens[..reply.tokens.len() - 1]);
            assert_eq!(retained.tokens(), expected);
            prompt.extend(&reply.tokens);
            prompt.extend([round % 5, 6]);
        }
    }
}

#[test]
fn a_prompt_that_differs_keeps_only_the_shared_start() {
    for guess in [false, true] {
        let (model, mut kv, mut retained) = cache(200);
        let p = params(8, &[], guess);
        let first = [1, 2, 3, 4, 5, 6, 1, 2];
        turn(&model, &mut kv, &mut retained, &first, &p, None).unwrap();
        // (prompt, positions that may be kept)
        let cases: [(&[u32], usize); 5] = [
            (&[1, 2, 3, 0, 0, 4], 3),          // diverges in the middle
            (&[6, 2, 3, 4], 0),                // diverges at the first token
            (&[1, 2, 3, 4, 5], 4), // shorter: all but the last, which must be read again
            (&[1, 2, 3, 4, 5, 6, 1, 2], 7), // the same prompt again
            (&[1, 2, 3, 4, 5, 6, 1, 2, 2], 8), // a little longer
        ];
        for (prompt, keep) in cases {
            // Restart from the first prompt each time, so `keep` is measured against it.
            turn(&model, &mut kv, &mut retained, &first, &p, None).unwrap();
            let reply = turn(&model, &mut kv, &mut retained, prompt, &p, None).unwrap();
            assert_eq!(reply.outcome.reused, keep, "{prompt:?}");
            assert_eq!(reply.tokens, fresh(prompt, &p, 200), "{prompt:?}");
        }
    }
}

#[test]
fn a_stop_token_is_never_held_and_the_reply_before_it_is() {
    let (model, mut kv, mut retained) = cache(200);
    let prompt = [1, 2, 3];
    let p = params(50, &[4], false);
    let reply = turn(&model, &mut kv, &mut retained, &prompt, &p, None).unwrap();
    assert_eq!(reply.outcome.stop, StopReason::EndOfText);
    let mut expected = prompt.to_vec();
    expected.extend(&reply.tokens);
    assert_eq!(
        retained.tokens(),
        expected,
        "every delivered token was fed back"
    );
    assert!(!retained.tokens()[prompt.len()..].contains(&4));
}

#[test]
fn a_cancelled_reply_keeps_what_was_fed_and_the_next_one_is_still_exact() {
    for guess in [false, true] {
        let (model, mut kv, mut retained) = cache(200);
        let p = params(30, &[], guess);
        let prompt = vec![2, 2, 5, 1];
        let cut = turn(&model, &mut kv, &mut retained, &prompt, &p, Some(3)).unwrap();
        assert_eq!(cut.outcome.stop, StopReason::Cancelled);
        assert_eq!(retained.tokens(), held(&kv));
        assert!(retained.tokens().starts_with(&prompt));
        let mut next = prompt.clone();
        next.extend(&cut.tokens);
        next.extend([3, 3]);
        let reply = turn(&model, &mut kv, &mut retained, &next, &p, None).unwrap();
        assert!(reply.outcome.reused >= prompt.len());
        assert_eq!(reply.tokens, fresh(&next, &p, 200));
    }
}

#[test]
fn a_failed_reply_leaves_nothing_trusted() {
    let (mut model, mut kv, mut retained) = cache(200);
    let p = params(6, &[], false);
    let prompt = [1, 2, 3, 4];
    turn(&model, &mut kv, &mut retained, &prompt, &p, None).unwrap();
    // The next prompt pass commits its tokens and then fails.
    model.fail_on = Some(model.passes.get() + 1);
    let longer = [1, 2, 3, 4, 0, 5, 5];
    let failed = turn(&model, &mut kv, &mut retained, &longer, &p, None);
    assert!(matches!(failed, Err(GenerateError::Model(Failure))));
    assert!(retained.is_empty());
    assert!(kv.used() > 0, "the failed pass did write to the cache");
    model.fail_on = None;
    let reply = turn(&model, &mut kv, &mut retained, &longer, &p, None).unwrap();
    assert_eq!(reply.outcome.reused, 0, "read from scratch");
    assert_eq!(reply.tokens, fresh(&longer, &p, 200));
}

#[test]
fn a_cache_changed_behind_the_record_is_not_trusted() {
    let (model, mut kv, mut retained) = cache(200);
    let p = params(4, &[], false);
    let prompt = [1, 2, 3, 4, 5];
    turn(&model, &mut kv, &mut retained, &prompt, &p, None).unwrap();
    kv.truncate(2).unwrap();
    let reply = turn(&model, &mut kv, &mut retained, &prompt, &p, None).unwrap();
    assert_eq!(reply.outcome.reused, 0);
    assert_eq!(reply.tokens, fresh(&prompt, &p, 200));
}

#[test]
fn the_context_limit_holds_for_kept_and_new_positions_alike() {
    let capacity = 12;
    let (model, mut kv, mut retained) = cache(capacity);
    let p = params(50, &[], false);
    let first = [1, 2, 3, 4, 5, 6];
    let full = turn(&model, &mut kv, &mut retained, &first, &p, None).unwrap();
    assert_eq!(full.outcome.stop, StopReason::ContextFull);
    assert_eq!(retained.len(), capacity);
    // A prompt that fills the window exactly, built on the cache.
    let mut exact = first.to_vec();
    exact.extend(&full.tokens);
    exact.truncate(capacity);
    let reply = turn(&model, &mut kv, &mut retained, &exact, &p, None).unwrap();
    assert_eq!(reply.outcome.reused, capacity - 1);
    assert_eq!(reply.tokens, fresh(&exact, &p, capacity));
    // Too long: refused before anything changes, so the record still stands.
    let before = retained.clone();
    let mut over = exact.clone();
    over.push(1);
    let refused = turn(&model, &mut kv, &mut retained, &over, &p, None);
    assert!(matches!(refused, Err(GenerateError::PromptTooLong { .. })));
    assert_eq!(retained, before);
    assert_eq!(held(&kv), before.tokens());
}

#[test]
fn zero_max_tokens_keeps_the_shared_start_and_reads_nothing() {
    let (model, mut kv, mut retained) = cache(64);
    turn(
        &model,
        &mut kv,
        &mut retained,
        &[1, 2, 3],
        &params(3, &[], false),
        None,
    )
    .unwrap();
    let fed = model.fed.get();
    let none = turn(
        &model,
        &mut kv,
        &mut retained,
        &[1, 2, 6],
        &params(0, &[], false),
        None,
    )
    .unwrap();
    assert_eq!(none.outcome.generated, 0);
    assert_eq!(model.fed.get(), fed);
    assert_eq!(retained.tokens(), [1, 2]);
}

#[test]
fn plain_generate_always_reads_the_whole_prompt() {
    let (model, mut kv, _) = cache(64);
    let p = params(3, &[], false);
    for _ in 0..2 {
        let fed = model.fed.get();
        let outcome = generate(
            &model,
            &mut kv,
            &mut sampler(0.0),
            &[1, 2, 3],
            &p,
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();
        assert_eq!(outcome.reused, 0);
        assert_eq!(model.fed.get() - fed, 3 + 2);
    }
}

#[test]
fn a_model_that_does_not_commit_what_it_reads_leaves_nothing_trusted() {
    let (mut model, mut kv, mut retained) = cache(64);
    model.skip_commit = true;
    let cancel = AtomicBool::new(false);
    generate_reusing(
        &model,
        Kept {
            cache: &mut kv,
            retained: &mut retained,
        },
        &mut sampler(0.0),
        &[1, 2, 3],
        &params(3, &[], false),
        &cancel,
        |_| {},
    )
    .unwrap();
    assert!(retained.is_empty());
}
