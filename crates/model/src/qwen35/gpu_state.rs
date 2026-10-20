//! A Qwen3.5 GPU cache as host bytes and back, block by block, for checkpoints that outlive the process. A block is one
//! DeltaNet layer's state or one page of attention keys and values; the caller stores, hashes and checks them.

use super::gpu::Qwen35Gpu;
use super::gpu_cache::GpuCache;
use super::gpu_weights::Mixer;
use super::math::{invalid, size};
use crate::ModelError;

/// What a checkpoint of this model's cache is made of. Two caches may be exchanged only if these are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateLayout {
    pub delta_layers: usize,
    pub conv_values: usize,
    pub recurrent_values: usize,
    pub full_layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl StateLayout {
    /// Bytes of the block of a page holding `rows` positions of keys and values (stored as f32) in every full layer.
    pub fn page_bytes(&self, rows: usize) -> usize {
        self.kv_heads
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
            .saturating_mul(rows)
            .saturating_mul(self.full_layers)
            .saturating_mul(2)
    }

    /// Bytes of one DeltaNet layer's block.
    pub fn delta_bytes(&self) -> usize {
        self.conv_values
            .saturating_add(self.recurrent_values)
            .saturating_mul(4)
    }
}

fn little_endian(values: &[f32], out: &mut Vec<u8>) {
    out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
}

impl Qwen35Gpu {
    pub fn state_layout(&self) -> Result<StateLayout, ModelError> {
        let c = &self.config.block;
        let delta_layers = self
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::Delta(_)))
            .count();
        Ok(StateLayout {
            delta_layers,
            conv_values: size(c.channels()?, c.linear_conv_kernel_dim)?,
            recurrent_values: size(
                size(c.linear_num_value_heads, c.linear_key_head_dim)?,
                c.linear_value_head_dim,
            )?,
            full_layers: self.layers.len().saturating_sub(delta_layers),
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
        })
    }

    fn keepable(&self, cache: &GpuCache) -> Result<(), ModelError> {
        if cache.owner != self.owner() {
            return Err(invalid("cache belongs to a different Qwen3.5 GPU model"));
        }
        if cache.replay || cache.tokens.is_empty() || cache.kv.len() != cache.tokens.len() {
            return Err(invalid("the cache holds no state that can be kept"));
        }
        Ok(())
    }

    /// The tokens a cache has read, if it holds a state that can be kept.
    pub fn kept_tokens<'a>(&self, cache: &'a GpuCache) -> Result<&'a [u32], ModelError> {
        self.keepable(cache)?;
        Ok(&cache.tokens)
    }

    /// DeltaNet layer `index`: its convolution state, then its recurrent state, as little-endian f32.
    pub fn export_delta(&self, cache: &GpuCache, index: usize) -> Result<Vec<u8>, ModelError> {
        self.keepable(cache)?;
        let state = cache
            .delta
            .iter()
            .flatten()
            .nth(index)
            .ok_or_else(|| invalid("no such DeltaNet layer"))?;
        let mut out = Vec::with_capacity(self.state_layout()?.delta_bytes());
        for buffer in [&state.conv, &state.recurrent] {
            little_endian(&self.gpu.download(buffer)?, &mut out);
        }
        Ok(out)
    }

    pub fn export_page(
        &self,
        cache: &GpuCache,
        page: usize,
        rows: usize,
    ) -> Result<Vec<u8>, ModelError> {
        self.keepable(cache)?;
        Ok(cache.kv.export_page(&self.gpu, page, rows)?)
    }

    /// An empty cache of `capacity` positions with room mapped for `tokens`, to be filled by `import_delta` and
    /// `import_page` and committed by `finish_import`. Drop it if anything fails in between.
    pub fn begin_import(&self, capacity: usize, tokens: usize) -> Result<GpuCache, ModelError> {
        if tokens == 0 || tokens > capacity {
            return Err(invalid("a checkpoint does not fit this context"));
        }
        let mut cache = self.new_cache(capacity)?;
        cache.kv.reserve_empty(&self.gpu, tokens)?;
        Ok(cache)
    }

    pub fn import_delta(
        &self,
        cache: &mut GpuCache,
        index: usize,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        let layout = self.state_layout()?;
        if bytes.len() != layout.delta_bytes() {
            return Err(invalid("DeltaNet block has the wrong size"));
        }
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .filter_map(|b| b.try_into().ok().map(f32::from_le_bytes))
            .collect();
        let (conv, recurrent) = values.split_at(layout.conv_values);
        let state = cache
            .delta
            .iter_mut()
            .flatten()
            .nth(index)
            .ok_or_else(|| invalid("no such DeltaNet layer"))?;
        state.conv = self.gpu.upload(conv)?;
        state.recurrent = self.gpu.upload(recurrent)?;
        Ok(())
    }

    pub fn import_page(
        &self,
        cache: &mut GpuCache,
        page: usize,
        rows: usize,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        Ok(cache.kv.import_page(&self.gpu, page, rows, bytes)?)
    }

    /// Commits an import: the cache now holds exactly `tokens`.
    pub fn finish_import(&self, cache: &mut GpuCache, tokens: &[u32]) -> Result<(), ModelError> {
        cache.kv.advance(tokens.len())?;
        cache.tokens = tokens.to_vec();
        cache.replay = false;
        Ok(self.gpu.synchronize()?)
    }
}
