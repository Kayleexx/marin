use runtime::{CausalLm, RollbackError, SequenceCache};

use super::engine::{Engine, EngineCache, wrong_backend};
use crate::ModelError;

impl SequenceCache for EngineCache {
    fn used(&self) -> usize {
        match self {
            Self::Cpu(c) => SequenceCache::used(c),
            Self::Qwen35(c) => c.used(),
            Self::Qwen35Cuda(c) => c.used(),
            Self::GraniteCuda(c) => c.used(),
            Self::Cuda(c) => c.used(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Cpu(c) => SequenceCache::capacity(c),
            Self::Qwen35(c) => c.capacity(),
            Self::Qwen35Cuda(c) => c.capacity(),
            Self::GraniteCuda(c) => c.capacity(),
            Self::Cuda(c) => c.capacity(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Cpu(c) => SequenceCache::clear(c),
            Self::Qwen35(c) => c.clear(),
            Self::Qwen35Cuda(c) => c.clear(),
            Self::GraniteCuda(c) => c.clear(),
            Self::Cuda(c) => c.clear(),
        }
    }

    fn truncate(&mut self, len: usize) -> Result<(), RollbackError> {
        match self {
            Self::Cpu(c) => SequenceCache::truncate(c, len),
            Self::Qwen35(c) => c.truncate(len),
            Self::Qwen35Cuda(c) => c.truncate(len),
            Self::GraniteCuda(c) => c.truncate(len),
            Self::Cuda(c) => c.truncate(len),
        }
    }

    fn replays(&self) -> bool {
        match self {
            Self::Qwen35(c) => c.replays(),
            Self::Qwen35Cuda(c) => c.replays(),
            Self::Cpu(_) | Self::GraniteCuda(_) | Self::Cuda(_) => false,
        }
    }
}

impl CausalLm for Engine {
    type Error = ModelError;
    type Cache = EngineCache;

    fn forward(&self, tokens: &[u32], cache: &mut EngineCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }

    /// The Qwen3 GPU checks up to 8 tokens in one pass and the Qwen3.5 GPU up to 4; the CPU has no such pass, so it
    /// never guesses ahead.
    fn max_rows(&self) -> usize {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) | Self::GraniteCuda(_) => 0,
            Self::Cuda(m) => m.max_rows(),
            Self::Qwen35Cuda(_) => crate::qwen35::MAX_ROWS,
        }
    }

    fn speculation(&self) -> runtime::Speculation {
        match self {
            Self::Qwen35Cuda(_) => runtime::Speculation::SHORT_PASS,
            _ => runtime::Speculation::MEASURED,
        }
    }

    fn forward_rows(
        &self,
        tokens: &[u32],
        cache: &mut EngineCache,
    ) -> Result<Vec<f32>, ModelError> {
        match (self, cache) {
            (Self::Cuda(m), EngineCache::Cuda(kv)) => m.forward_rows(tokens, kv),
            (Self::Cpu(m), EngineCache::Cpu(kv)) => m.forward_rows(tokens, kv),
            (Self::Granite(m), EngineCache::Cpu(kv)) => m.forward_rows(tokens, kv),
            (Self::GraniteCuda(m), EngineCache::GraniteCuda(kv)) => m.forward_rows(tokens, kv),
            (Self::Qwen35(m), EngineCache::Qwen35(kv)) => m.forward_rows(tokens, kv),
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(kv)) => m.forward_rows(tokens, kv),
            _ => Err(wrong_backend()),
        }
    }
}
