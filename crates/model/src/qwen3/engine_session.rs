use super::{Engine, EngineCache, GPU_SLACK_BYTES, wrong_backend};
use crate::ModelError;
use crate::qwen35::Mark;
use kv::KvCache;
use runtime::CausalLm;

impl EngineCache {
    /// An empty stand-in to leave behind while the real cache is lent out; it holds no memory.
    pub fn vacant() -> Option<Self> {
        KvCache::new(0, 1, 1, 1).ok().map(Self::Cpu)
    }
}

impl Engine {
    /// Prepares the calling thread to use and free this model's GPU memory. Does nothing on the CPU.
    pub fn bind_to_thread(&self) -> Result<(), ModelError> {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) => Ok(()),
            Self::GraniteCuda(m) => Ok(m.gpu().bind_to_thread()?),
            Self::Cuda(m) => Ok(m.gpu().bind_to_thread()?),
            Self::Qwen35Cuda(m) => Ok(m.gpu().bind_to_thread()?),
        }
    }

    /// GPU memory that is free right now, for the engines that keep sessions there.
    pub fn gpu_free_bytes(&self) -> Option<u64> {
        match self {
            Self::Qwen35Cuda(m) => m
                .gpu()
                .info()
                .ok()
                .and_then(|i| u64::try_from(i.free_bytes).ok()),
            _ => None,
        }
    }

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

    /// A point of a Qwen3.5 GPU cache that `rewind` returns to, taking only the DeltaNet state (a few tens of MB) and no
    /// second copy of the keys and values. The state goes on the GPU when it has room to spare and into host memory
    /// when it has not, so a GPU short of memory still gets a mark. `None` for any other engine.
    pub fn mark(&self, cache: &EngineCache) -> Result<Option<Mark>, ModelError> {
        match (self, cache) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(kv)) => {
                let free = u64::try_from(m.gpu().info()?.free_bytes).unwrap_or(0);
                let bytes = u64::try_from(kv.mark_bytes()).unwrap_or(u64::MAX);
                if free < bytes.saturating_add(GPU_SLACK_BYTES) {
                    return Ok(m.mark_cache_in_host(kv).ok().map(Mark::Gpu));
                }
                Ok(Some(Mark::Gpu(m.mark_cache(kv)?)))
            }
            (Self::Qwen35(m), EngineCache::Qwen35(kv)) => Ok(Some(Mark::Cpu(m.mark_cache(kv)?))),
            (Self::Qwen35Cuda(_) | Self::Qwen35(_), _) => Err(wrong_backend()),
            _ => Ok(None),
        }
    }

    /// Returns `cache` to `mark`. Refused if the history was cut back or restarted since the mark was taken.
    pub fn rewind(&self, cache: &mut EngineCache, mark: Mark) -> Result<(), ModelError> {
        match (self, cache, mark) {
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(kv), Mark::Gpu(mark)) => {
                m.rewind_cache(kv, mark)
            }
            (Self::Qwen35(m), EngineCache::Qwen35(kv), Mark::Cpu(mark)) => m.rewind_cache(kv, mark),
            _ => Err(wrong_backend()),
        }
    }

    /// How many sequences `decode_batch` serves in one pass; 1 where batching is not supported.
    pub fn batch_limit(&self) -> usize {
        match self {
            Self::Qwen35Cuda(_) => crate::qwen35::MAX_BATCH,
            _ => 1,
        }
    }

    /// One decode step for each sequence in a single pass, with the logits each `forward` would give. If the pass
    /// cannot run as a batch, every sequence is stepped alone, so a failure is reported only for the sequence it
    /// belongs to.
    pub fn decode_batch(
        &self,
        steps: Vec<(u32, EngineCache)>,
    ) -> Vec<(EngineCache, Result<Vec<f32>, ModelError>)> {
        let Self::Qwen35Cuda(model) = self else {
            return alone(self, steps);
        };
        if steps
            .iter()
            .any(|(_, cache)| !matches!(cache, EngineCache::Qwen35Cuda(_)))
        {
            return alone(self, steps);
        }
        let (tokens, mut caches): (Vec<u32>, Vec<_>) = steps
            .into_iter()
            .filter_map(|(token, cache)| match cache {
                EngineCache::Qwen35Cuda(gpu) => Some((token, gpu)),
                _ => None,
            })
            .unzip();
        let mut refs: Vec<&mut crate::qwen35::GpuCache> =
            caches.iter_mut().map(|c| &mut **c).collect();
        let batched = model.forward_batch(&tokens, &mut refs);
        drop(refs);
        match batched {
            Ok(rows) => caches
                .into_iter()
                .zip(rows)
                .map(|(cache, row)| (EngineCache::Qwen35Cuda(cache), Ok(row)))
                .collect(),
            Err(_) => alone(
                self,
                tokens
                    .into_iter()
                    .zip(caches)
                    .map(|(token, cache)| (token, EngineCache::Qwen35Cuda(cache)))
                    .collect(),
            ),
        }
    }
}

fn alone(
    engine: &Engine,
    steps: Vec<(u32, EngineCache)>,
) -> Vec<(EngineCache, Result<Vec<f32>, ModelError>)> {
    steps
        .into_iter()
        .map(|(token, mut cache)| {
            let logits = engine.forward(&[token], &mut cache);
            (cache, logits)
        })
        .collect()
}
