//! Copying cache pages out of and into an engine's cache, for snapshots. Only the GPU cache can do it.

use cuda::KvFormat;

use super::engine::{Engine, EngineCache};
use crate::ModelError;

/// What a snapshot of this engine's cache is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub format: KvFormat,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

fn unsupported() -> ModelError {
    ModelError::Unsupported("saving the cache needs the GPU backend".into())
}

impl Engine {
    pub fn kv_layout(&self) -> Option<KvLayout> {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) | Self::Qwen35Cuda(_) => None,
            Self::Cuda(m) => {
                let config = m.config();
                Some(KvLayout {
                    format: m.kv_format(),
                    layers: config.layers,
                    kv_heads: config.kv_heads,
                    head_dim: config.head_dim,
                })
            }
        }
    }

    pub fn export_kv_page(
        &self,
        cache: &EngineCache,
        page: usize,
        rows: usize,
    ) -> Result<Vec<u8>, ModelError> {
        match (self, cache) {
            (Self::Cuda(m), EngineCache::Cuda(c)) => Ok(c.kv().export_page(m.gpu(), page, rows)?),
            _ => Err(unsupported()),
        }
    }

    /// Maps pages for `tokens` positions of an empty cache. Fill them with `import_kv_page`, then call
    /// `finish_kv_import`. If anything fails in between, `clear` the cache.
    pub fn begin_kv_import(
        &self,
        cache: &mut EngineCache,
        tokens: usize,
    ) -> Result<(), ModelError> {
        match (self, cache) {
            (Self::Cuda(m), EngineCache::Cuda(c)) => {
                Ok(c.kv_mut().reserve_empty(m.gpu(), tokens)?)
            }
            _ => Err(unsupported()),
        }
    }

    pub fn import_kv_page(
        &self,
        cache: &mut EngineCache,
        page: usize,
        rows: usize,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        match (self, cache) {
            (Self::Cuda(m), EngineCache::Cuda(c)) => {
                Ok(c.kv_mut().import_page(m.gpu(), page, rows, bytes)?)
            }
            _ => Err(unsupported()),
        }
    }

    pub fn finish_kv_import(
        &self,
        cache: &mut EngineCache,
        tokens: usize,
    ) -> Result<(), ModelError> {
        match cache {
            EngineCache::Cuda(c) => Ok(c.kv_mut().advance(tokens)?),
            EngineCache::Cpu(_) | EngineCache::Qwen35(_) | EngineCache::Qwen35Cuda(_) => {
                Err(unsupported())
            }
        }
    }

    /// Bytes `export_kv_page` returns for a page of `rows` positions.
    pub fn kv_page_payload_bytes(&self, cache: &EngineCache, rows: usize) -> Option<usize> {
        match cache {
            EngineCache::Cuda(c) => Some(c.kv().page_payload_bytes(rows)),
            EngineCache::Cpu(_) | EngineCache::Qwen35(_) | EngineCache::Qwen35Cuda(_) => None,
        }
    }
}
