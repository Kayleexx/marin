//! One decode step for several independent sequences in a single pass: the projections, norms and MLP run over
//! one row per sequence, so each weight is read once; the convolution, recurrence, positions, KV writes and
//! attention run per sequence on that sequence's own state.
use super::Qwen35Gpu;
use super::gpu::Workspace;
use super::gpu_cache::GpuCache;
use super::gpu_forward::{Rows, delta_state, full_attention, pass_chunk};
use super::gpu_weights::{Delta, Full};
use super::math::{invalid, size};
use crate::ModelError;
use crate::qwen3::gpu_probe::GpuProbe;
use cuda::Gpu;
use runtime::{CausalLm, SequenceCache};

/// The widest batch the multi-token projection kernels cover.
pub const MAX_BATCH: usize = 4;

pub(super) fn delta_each(
    gpu: &Gpu,
    (w, c): (&Delta, &super::Config),
    (ws, scratch): (&mut Workspace, &mut Workspace),
    (caches, index): (&mut [&mut GpuCache], usize),
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    let (channels, value) = (c.channels()?, c.value_dim()?);
    for (i, cache) in caches.iter_mut().enumerate() {
        gpu.copy_range_into(&ws.qkv, size(i, channels)?, &mut scratch.qkv)?;
        gpu.copy_range_into(&ws.z, size(i, value)?, &mut scratch.z)?;
        gpu.copy_range_into(&ws.a, size(i, c.linear_num_value_heads)?, &mut scratch.a)?;
        gpu.copy_range_into(&ws.b, size(i, c.linear_num_value_heads)?, &mut scratch.b)?;
        let state = cache
            .delta
            .get_mut(index)
            .and_then(Option::as_mut)
            .ok_or_else(|| invalid("missing GPU Delta state"))?;
        delta_state(gpu, (w, c), scratch, state, 1, probe)?;
        gpu.copy_into_range(&scratch.core, &mut ws.core, size(i, value)?)?;
    }
    Ok(())
}

pub(super) fn full_each(
    gpu: &Gpu,
    (w, c): (&Full, &super::Config),
    (ws, scratch): (&mut Workspace, &mut Workspace),
    caches: &mut [&mut GpuCache],
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    let query = size(c.num_attention_heads, c.head_dim)?;
    let kv = size(c.num_key_value_heads, c.head_dim)?;
    for (i, cache) in caches.iter_mut().enumerate() {
        gpu.copy_range_into(&ws.q, size(i, query)?, &mut scratch.q)?;
        gpu.copy_range_into(&ws.k, size(i, kv)?, &mut scratch.k)?;
        gpu.copy_range_into(&ws.v, size(i, kv)?, &mut scratch.v)?;
        let start = cache.kv.len();
        full_attention(gpu, (w, c), scratch, &mut cache.kv, (1, start), probe)?;
        gpu.copy_into_range(&scratch.context, &mut ws.context, size(i, query)?)?;
    }
    Ok(())
}

impl Qwen35Gpu {
    pub(super) fn batch_pass(
        &self,
        tokens: &[u32],
        caches: &mut [&mut GpuCache],
    ) -> Result<Vec<f32>, ModelError> {
        let outcome = self.run_batch(tokens, caches);
        if outcome.is_err() {
            for cache in caches.iter_mut() {
                cache.replay = true;
            }
        }
        outcome
    }

    fn run_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut GpuCache],
    ) -> Result<Vec<f32>, ModelError> {
        let mut probe = GpuProbe::new(&self.gpu, None);
        let mut rows = Rows::Batch {
            caches: caches.iter_mut().map(|cache| &mut **cache).collect(),
            scratch: Box::new(Workspace::new(&self.gpu, &self.config, (1, 1))?),
        };
        let flat = pass_chunk(self, &mut rows, tokens, &mut probe, None)?;
        probe.finish()?;
        Ok(flat)
    }

    /// Feeds `tokens[i]` after the positions in `caches[i]` and returns each sequence's logits, exactly as one
    /// `forward` per sequence would. A cache that has to replay its history is
    /// served by its own `forward` instead.
    pub fn forward_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut GpuCache],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        if tokens.is_empty() || tokens.len() != caches.len() || tokens.len() > MAX_BATCH {
            return Err(invalid("a batch needs one token per cache, at most four"));
        }
        if caches.len() == 1 || caches.iter().any(|cache| cache.replay) {
            return tokens
                .iter()
                .zip(caches.iter_mut())
                .map(|(token, cache)| self.forward(&[*token], cache))
                .collect();
        }
        for (token, cache) in tokens.iter().zip(caches.iter()) {
            if cache.owner != self.owner() {
                return Err(invalid("cache belongs to a different Qwen3.5 GPU model"));
            }
            if cache.used() >= cache.capacity {
                return Err(kv::KvError::ContextFull {
                    needed: cache.used().saturating_add(1),
                    capacity: cache.capacity,
                }
                .into());
            }
            if *token as usize >= self.config().vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token: *token,
                    vocab: self.config().vocab_size,
                });
            }
        }
        for cache in caches.iter_mut() {
            cache
                .tokens
                .try_reserve(1)
                .map_err(|e| invalid(format!("GPU token ledger: {e}")))?;
        }
        let flat = self.batch_pass(tokens, caches)?;
        for (cache, token) in caches.iter_mut().zip(tokens) {
            cache.tokens.push(*token);
        }
        let vocab = self.config().vocab_size;
        Ok(flat.chunks(vocab).map(<[f32]>::to_vec).collect())
    }
}
