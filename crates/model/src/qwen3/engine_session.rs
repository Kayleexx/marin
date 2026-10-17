use super::{Engine, EngineCache, GPU_SLACK_BYTES, wrong_backend};
use crate::ModelError;

impl Engine {
    /// Snapshot a Qwen3.5 GPU history boundary while leaving the generation cache writable.
    pub fn session_checkpoint(
        &self,
        cache: &EngineCache,
    ) -> Result<Option<EngineCache>, ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(kv)) => {
                let free = u64::try_from(m.gpu().info()?.free_bytes).unwrap_or(0);
                let copy = u64::try_from(kv.checkpoint_bytes()).unwrap_or(u64::MAX);
                if free < copy.saturating_add(GPU_SLACK_BYTES) {
                    return Ok(None);
                }
                Ok(Some(EngineCache::Qwen35Cuda(Box::new(
                    m.checkpoint_cache(kv)?,
                ))))
            }
            (Self::Qwen35Cuda(_), _) => Err(wrong_backend()),
            _ => Ok(None),
        }
    }
}
