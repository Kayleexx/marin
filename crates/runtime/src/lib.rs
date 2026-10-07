#![forbid(unsafe_code)]

mod generate;
#[cfg(test)]
mod generate_tests;
mod ngram;
mod reuse;
#[cfg(test)]
mod reuse_tests;
mod speculate;
#[cfg(test)]
mod speculate_tests;

pub use generate::{
    GenerateError, GenerateParams, Outcome, StopReason, generate, generate_reusing,
};
pub use reuse::{Kept, Retained};
pub use speculate::{Speculation, SpeculationStats};

use kv::KvCache;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
#[error("cannot roll back to {len} positions: only {held} are held")]
pub struct RollbackError {
    pub len: usize,
    pub held: usize,
}

/// What generation needs to know about a model's attention memory, wherever it lives.
pub trait SequenceCache {
    fn used(&self) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);
    /// Forgets every position from `len` on, so the next pass writes there; the positions below stay as they are.
    fn truncate(&mut self, len: usize) -> Result<(), RollbackError>;
}

impl SequenceCache for KvCache {
    fn used(&self) -> usize {
        KvCache::len(self)
    }

    fn capacity(&self) -> usize {
        KvCache::capacity(self)
    }

    fn clear(&mut self) {
        KvCache::clear(self);
    }

    fn truncate(&mut self, len: usize) -> Result<(), RollbackError> {
        KvCache::truncate(self, len).map_err(|_| RollbackError {
            len,
            held: KvCache::len(self),
        })
    }
}

pub trait CausalLm {
    type Error: std::error::Error + Send + Sync + 'static;
    type Cache: SequenceCache;

    /// Feeds `tokens` after the positions already in `cache`, commits them,
    /// and returns the logits that predict the token following the last one.
    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Vec<f32>, Self::Error>;

    /// Most tokens `forward_rows` takes in one pass. 0 (the default) means the model has no fast way to check
    /// several tokens at once, and generation never guesses ahead.
    fn max_rows(&self) -> usize {
        0
    }

    /// Like `forward`, but returns the logits after every token fed (one row of vocabulary size each, in order),
    /// so guessed tokens can be checked against what the model would choose. The default feeds the tokens one at a
    /// time; models that can check them in one pass should do so and report a `max_rows` of at least 2.
    fn forward_rows(
        &self,
        tokens: &[u32],
        cache: &mut Self::Cache,
    ) -> Result<Vec<f32>, Self::Error> {
        let mut rows = Vec::new();
        for token in tokens {
            rows.extend(self.forward(&[*token], cache)?);
        }
        Ok(rows)
    }
}
