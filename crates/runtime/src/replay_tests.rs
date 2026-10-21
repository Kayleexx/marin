//! A cache that cannot be cut back in place reads its whole history again, and the outcome says so instead of
//! counting those tokens as kept.

use std::sync::atomic::AtomicBool;

use kv::KvCache;
use sampler::{Sampler, SamplerConfig};

use crate::generate::{GenerateParams, StopReason};
use crate::{CausalLm, Kept, Retained, RollbackError, SequenceCache, generate_reusing};

/// A cache whose cut-back, like a recurrent state's, is a restart.
struct Recurrent {
    kv: KvCache,
    replay: bool,
    cuts_back_in_place: bool,
}

impl SequenceCache for Recurrent {
    fn used(&self) -> usize {
        self.kv.len()
    }

    fn capacity(&self) -> usize {
        self.kv.capacity()
    }

    fn clear(&mut self) {
        self.kv.clear();
        self.replay = false;
    }

    fn truncate(&mut self, len: usize) -> Result<(), RollbackError> {
        if len != self.kv.len() && !self.cuts_back_in_place {
            self.replay = true;
        }
        SequenceCache::truncate(&mut self.kv, len)
    }

    fn replays(&self) -> bool {
        self.replay
    }
}

struct Model;

impl CausalLm for Model {
    type Error = std::convert::Infallible;
    type Cache = Recurrent;

    fn forward(&self, tokens: &[u32], cache: &mut Recurrent) -> Result<Vec<f32>, Self::Error> {
        let keys: Vec<f32> = tokens.iter().map(|t| *t as f32).collect();
        if cache.kv.write(0, &keys, &keys).is_ok() {
            let _ = cache.kv.advance(tokens.len());
        }
        cache.replay = false;
        Ok(vec![0.0, 1.0, 0.5])
    }
}

fn run(cache: &mut Recurrent, retained: &mut Retained, prompt: &[u32]) -> crate::Outcome {
    let params = GenerateParams {
        max_tokens: 2,
        stop_tokens: Vec::new(),
        usable_vocab: 3,
        speculation: None,
    };
    let sampler = &mut Sampler::new(SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 0,
    })
    .unwrap();
    generate_reusing(
        &Model,
        Kept { cache, retained },
        sampler,
        prompt,
        &params,
        &AtomicBool::new(false),
        |_| {},
    )
    .unwrap()
}

fn cache(cuts_back_in_place: bool) -> Recurrent {
    Recurrent {
        kv: KvCache::new(1, 1, 1, 64).unwrap(),
        replay: false,
        cuts_back_in_place,
    }
}

#[test]
fn tokens_a_cache_had_to_read_again_are_not_counted_as_kept() {
    for (in_place, reused, replayed) in [(true, 6, 0), (false, 0, 6)] {
        let (mut kv, mut retained) = (cache(in_place), Retained::default());
        let first = run(&mut kv, &mut retained, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!((first.reused, first.replayed), (0, 0));
        assert_eq!(first.stop, StopReason::MaxTokens);
        let second = run(&mut kv, &mut retained, &[1, 2, 3, 4, 5, 6, 9, 9]);
        assert_eq!(
            (second.reused, second.replayed),
            (reused, replayed),
            "cuts back in place: {in_place}"
        );
        let mut extended = retained.tokens().to_vec();
        extended.extend([3, 3]);
        let third = run(&mut kv, &mut retained, &extended);
        assert_eq!(third.replayed, 0, "extending the history is not a replay");
        assert_eq!(third.reused, extended.len().saturating_sub(2));
    }
}
