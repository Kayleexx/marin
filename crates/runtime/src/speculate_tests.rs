//! Guessing ahead must give exactly the tokens, stop reason and cache that plain decoding gives, however often the
//! guesses are right or wrong. The model here reads its history back out of the cache, so a rollback that leaves a
//! rejected token behind, or removes an accepted one, changes what it predicts.

use std::sync::atomic::{AtomicBool, Ordering};

use kv::KvCache;
use sampler::{Sampler, SamplerConfig};
use thiserror::Error;

use crate::generate::{GenerateError, GenerateParams, Outcome, StopReason, generate};
use crate::{CausalLm, SequenceCache, Speculation};

const VOCAB: usize = 7;

#[derive(Debug, Error)]
#[error("model failed")]
struct Failure;

/// Predicts the next token from the two before it and the position, all read from the cache.
struct HistoryModel {
    rows: usize,
    native_rows: bool,
    fail_rows: bool,
}

impl HistoryModel {
    fn new(rows: usize, native_rows: bool) -> Self {
        Self {
            rows,
            native_rows,
            fail_rows: false,
        }
    }

    fn feed(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
        let start = kv.len();
        let keys: Vec<f32> = tokens.iter().map(|t| *t as f32).collect();
        kv.write(0, &keys, &keys).map_err(|_| Failure)?;
        kv.advance(tokens.len()).map_err(|_| Failure)?;
        let (held, _) = kv.read(0, kv.len()).map_err(|_| Failure)?;
        let mut rows = Vec::new();
        for position in start..kv.len() {
            let before = position
                .checked_sub(1)
                .and_then(|p| held.get(p))
                .map_or(0, |v| *v as usize);
            let this = held.get(position).map_or(0, |v| *v as usize);
            let mixed = [
                before.wrapping_mul(5),
                this.wrapping_mul(3),
                position.wrapping_rem(3).wrapping_mul(2),
                1,
            ];
            let next = mixed.iter().fold(0_usize, |sum, v| sum.wrapping_add(*v)) % VOCAB;
            let mut logits = vec![0.0; VOCAB];
            logits[next] = 3.0;
            logits[next.wrapping_add(1) % VOCAB] = 2.5;
            rows.extend(logits);
        }
        Ok(rows)
    }
}

impl CausalLm for HistoryModel {
    type Error = Failure;
    type Cache = KvCache;

    fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
        let rows = self.feed(tokens, kv)?;
        Ok(rows
            .get(rows.len().saturating_sub(VOCAB)..)
            .unwrap_or(&[])
            .to_vec())
    }

    fn max_rows(&self) -> usize {
        self.rows
    }

    fn forward_rows(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, Failure> {
        if self.fail_rows {
            return Err(Failure);
        }
        if self.native_rows {
            self.feed(tokens, kv)
        } else {
            let mut rows = Vec::new();
            for token in tokens {
                rows.extend(self.forward(&[*token], kv)?);
            }
            Ok(rows)
        }
    }
}

struct Case {
    prompt: Vec<u32>,
    max_tokens: usize,
    capacity: usize,
    stop: Vec<u32>,
    temperature: f32,
}

struct Run {
    outcome: Outcome,
    tokens: Vec<u32>,
    used: usize,
}

fn run(model: &HistoryModel, case: &Case, speculation: Option<Speculation>) -> Run {
    let mut kv = KvCache::new(1, 1, 1, case.capacity).unwrap();
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: case.temperature,
        top_k: 0,
        top_p: 1.0,
        seed: 11,
    })
    .unwrap();
    let params = GenerateParams {
        max_tokens: case.max_tokens,
        stop_tokens: case.stop.clone(),
        usable_vocab: VOCAB,
        speculation,
    };
    let mut tokens = Vec::new();
    let outcome = generate(
        model,
        &mut kv,
        &mut sampler,
        &case.prompt,
        &params,
        &AtomicBool::new(false),
        |t| tokens.push(t),
    )
    .unwrap();
    Run {
        outcome,
        tokens,
        used: kv.used(),
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for prompt in [
        vec![1],
        vec![2, 4, 1, 3],
        vec![5, 5, 5, 5, 5, 5, 5, 5, 1, 2],
    ] {
        for max_tokens in [1, 2, 7, 60] {
            for (capacity, stop) in [
                (200, vec![]),
                (200, vec![3]),
                (prompt.len().saturating_add(9), vec![]),
                (prompt.len().saturating_add(1), vec![]),
            ] {
                for temperature in [0.0, 0.8] {
                    cases.push(Case {
                        prompt: prompt.clone(),
                        max_tokens,
                        capacity,
                        stop: stop.clone(),
                        temperature,
                    });
                }
            }
        }
    }
    cases
}

#[test]
fn guessing_gives_the_tokens_stop_and_cache_of_plain_decoding() {
    let (mut guessed_passes, mut rejected, mut accepted) = (0, 0, 0);
    for native in [true, false] {
        for case in cases() {
            let model = HistoryModel::new(8, native);
            let plain = run(&model, &case, None);
            for match_len in 1..=3 {
                for max_draft in [1, 3, 7] {
                    let rule = Speculation {
                        match_len,
                        max_draft,
                    };
                    let guessed = run(&model, &case, Some(rule));
                    let label = format!(
                        "prompt {:?} max {} cap {} stop {:?} t {} rule {rule:?} native {native}",
                        case.prompt, case.max_tokens, case.capacity, case.stop, case.temperature
                    );
                    assert_eq!(guessed.tokens, plain.tokens, "{label}");
                    assert_eq!(guessed.outcome.stop, plain.outcome.stop, "{label}");
                    assert_eq!(
                        guessed.outcome.generated, plain.outcome.generated,
                        "{label}"
                    );
                    assert_eq!(guessed.used, plain.used, "{label}: cache length");
                    let stats = &guessed.outcome.speculation;
                    assert!(stats.accepted <= stats.drafted, "{label}");
                    assert!(stats.passes <= plain.outcome.speculation.passes, "{label}");
                    guessed_passes += stats.guessed_passes;
                    accepted += stats.accepted;
                    rejected += stats.drafted - stats.accepted;
                }
            }
        }
    }
    assert!(
        guessed_passes > 100 && accepted > 100 && rejected > 100,
        "{guessed_passes} {accepted} {rejected}"
    );
}

#[test]
fn guessing_cuts_passes_when_the_text_repeats() {
    let model = HistoryModel::new(8, true);
    let case = Case {
        prompt: vec![1, 2, 3, 4],
        max_tokens: 200,
        capacity: 400,
        stop: vec![],
        temperature: 0.0,
    };
    let plain = run(&model, &case, None);
    let guessed = run(
        &model,
        &case,
        Some(Speculation {
            match_len: 3,
            max_draft: 7,
        }),
    );
    assert_eq!(guessed.tokens, plain.tokens);
    assert!(guessed.outcome.speculation.passes * 2 < plain.outcome.speculation.passes);
}

#[test]
fn a_model_that_cannot_check_several_tokens_decodes_plainly() {
    let model = HistoryModel::new(0, false);
    let case = Case {
        prompt: vec![1, 2, 3],
        max_tokens: 30,
        capacity: 100,
        stop: vec![],
        temperature: 0.0,
    };
    let plain = run(&model, &case, None);
    let asked = run(&model, &case, Some(Speculation::MEASURED));
    assert_eq!(asked.tokens, plain.tokens);
    assert_eq!(asked.outcome.speculation.guessed_passes, 0);
    assert_eq!(
        asked.outcome.speculation.passes,
        plain.outcome.speculation.passes
    );
}

#[test]
fn cancelling_in_the_middle_of_a_burst_stops_at_the_same_token() {
    let model = HistoryModel::new(8, true);
    let case = Case {
        prompt: vec![1, 2, 3, 4],
        max_tokens: 100,
        capacity: 300,
        stop: vec![],
        temperature: 0.0,
    };
    let stopped_at = |rule: Option<Speculation>| {
        let mut kv = KvCache::new(1, 1, 1, case.capacity).unwrap();
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        })
        .unwrap();
        let params = GenerateParams {
            max_tokens: case.max_tokens,
            stop_tokens: vec![],
            usable_vocab: VOCAB,
            speculation: rule,
        };
        let cancel = AtomicBool::new(false);
        let mut tokens = Vec::new();
        let outcome = generate(
            &model,
            &mut kv,
            &mut sampler,
            &case.prompt,
            &params,
            &cancel,
            |t| {
                tokens.push(t);
                if tokens.len() == 23 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        )
        .unwrap();
        (tokens, outcome.stop, kv.used())
    };
    let plain = stopped_at(None);
    assert_eq!(plain.1, StopReason::Cancelled);
    assert_eq!(
        stopped_at(Some(Speculation {
            match_len: 3,
            max_draft: 7
        })),
        plain
    );
}

#[test]
fn bad_rules_and_model_failures_are_errors() {
    let case = Case {
        prompt: vec![1, 2, 3, 4],
        max_tokens: 50,
        capacity: 300,
        stop: vec![],
        temperature: 0.0,
    };
    let try_run = |model: &HistoryModel, rule: Speculation| {
        let mut kv = KvCache::new(1, 1, 1, case.capacity).unwrap();
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        })
        .unwrap();
        let params = GenerateParams {
            max_tokens: case.max_tokens,
            stop_tokens: vec![],
            usable_vocab: VOCAB,
            speculation: Some(rule),
        };
        generate(
            model,
            &mut kv,
            &mut sampler,
            &case.prompt,
            &params,
            &AtomicBool::new(false),
            |_| {},
        )
    };
    let model = HistoryModel::new(8, true);
    for rule in [
        Speculation {
            match_len: 0,
            max_draft: 7,
        },
        Speculation {
            match_len: 3,
            max_draft: 0,
        },
    ] {
        assert!(matches!(
            try_run(&model, rule),
            Err(GenerateError::BadSpeculation)
        ));
    }
    let mut failing = HistoryModel::new(8, true);
    failing.fail_rows = true;
    assert!(matches!(
        try_run(
            &failing,
            Speculation {
                match_len: 3,
                max_draft: 7
            }
        ),
        Err(GenerateError::Model(Failure))
    ));
}
