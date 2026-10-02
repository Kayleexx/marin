#![forbid(unsafe_code)]

mod generate;

pub use generate::{GenerateError, GenerateParams, Outcome, StopReason, generate};

use kv::KvCache;

/// What generation needs to know about a model's attention memory, wherever it lives.
pub trait SequenceCache {
    fn used(&self) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);
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
}

pub trait CausalLm {
    type Error: std::error::Error + Send + Sync + 'static;
    type Cache: SequenceCache;

    /// Feeds `tokens` after the positions already in `cache`, commits them,
    /// and returns the logits that predict the token following the last one.
    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Vec<f32>, Self::Error>;
}
