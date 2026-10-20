//! Copying a Qwen3.5 GPU cache out of and into an engine, block by block, for checkpoints. Only that cache can do it.

use super::engine::{Engine, EngineCache};
use crate::ModelError;
use crate::qwen35::StateLayout;

fn unsupported() -> ModelError {
    ModelError::Unsupported("keeping the cache needs Qwen3.5 on the GPU".into())
}

impl Engine {
    /// What a checkpoint of this engine's cache is made of, or `None` if it cannot make one.
    pub fn state_layout(&self) -> Option<StateLayout> {
        match self {
            Self::Qwen35Cuda(m) => m.state_layout().ok(),
            _ => None,
        }
    }

    /// The tokens the cache has read, if it holds a state that can be kept.
    pub fn kept_tokens<'a>(&self, cache: &'a EngineCache) -> Result<&'a [u32], ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(c)) => m.kept_tokens(c),
            _ => Err(unsupported()),
        }
    }

    pub fn export_state_delta(
        &self,
        cache: &EngineCache,
        index: usize,
    ) -> Result<Vec<u8>, ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(c)) => m.export_delta(c, index),
            _ => Err(unsupported()),
        }
    }

    pub fn export_state_page(
        &self,
        cache: &EngineCache,
        page: usize,
        rows: usize,
    ) -> Result<Vec<u8>, ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(c)) => m.export_page(c, page, rows),
            _ => Err(unsupported()),
        }
    }

    /// An empty cache to fill with `import_state_delta` and `import_state_page`, then commit with
    /// `finish_state_import`. Dropping it abandons the import.
    pub fn begin_state_import(
        &self,
        capacity: usize,
        tokens: usize,
    ) -> Result<EngineCache, ModelError> {
        match self {
            Self::Qwen35Cuda(m) => Ok(EngineCache::Qwen35Cuda(Box::new(
                m.begin_import(capacity, tokens)?,
            ))),
            _ => Err(unsupported()),
        }
    }

    pub fn import_state_delta(
        &self,
        cache: &mut EngineCache,
        index: usize,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(c)) => m.import_delta(c, index, bytes),
            _ => Err(unsupported()),
        }
    }

    pub fn import_state_page(
        &self,
        cache: &mut EngineCache,
        page: usize,
        rows: usize,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(c)) => {
                m.import_page(c, page, rows, bytes)
            }
            _ => Err(unsupported()),
        }
    }

    pub fn finish_state_import(
        &self,
        cache: &mut EngineCache,
        tokens: &[u32],
    ) -> Result<(), ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(c)) => m.finish_import(c, tokens),
            _ => Err(unsupported()),
        }
    }
}
