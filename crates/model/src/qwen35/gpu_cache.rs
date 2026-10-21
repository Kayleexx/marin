use super::TextConfig;
use super::gpu_weights::{Layer, Mixer};
use super::math::{invalid, size};
use crate::ModelError;
use cuda::{Buffer, DeviceKv, Gpu, KvFormat};
use runtime::{RollbackError, SequenceCache};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CACHE: AtomicU64 = AtomicU64::new(1);

pub(super) struct DeltaState {
    pub conv: Buffer,
    pub recurrent: Buffer,
}

pub struct GpuCache {
    pub(super) delta: Vec<Option<DeltaState>>,
    pub(super) kv: DeviceKv,
    pub(super) tokens: Vec<u32>,
    pub(super) capacity: usize,
    pub(super) replay: bool,
    pub(super) owner: u64,
    /// Counts the times the history was cut back or started again, so a `Mark` taken before cannot be applied after.
    pub(super) epoch: u64,
    /// Tells this cache from every other, so a `Mark` is only ever put back into the cache it came from.
    pub(super) id: u64,
}

impl GpuCache {
    pub(super) fn new(
        gpu: &Gpu,
        config: &TextConfig,
        layers: &[Layer],
        capacity: usize,
        owner: u64,
    ) -> Result<Self, ModelError> {
        if capacity == 0 || capacity > config.summary().max_positions {
            return Err(invalid("invalid Qwen3.5 GPU context"));
        }
        let full = layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::Full(_)))
            .count();
        let c = &config.block;
        Ok(Self {
            delta: Self::fresh_state(gpu, config, layers)?,
            kv: DeviceKv::new(
                gpu,
                full,
                (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
                capacity,
                KvFormat::F32,
            )?,
            tokens: Vec::new(),
            capacity,
            replay: false,
            owner,
            epoch: 0,
            id: NEXT_CACHE.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Zeroed DeltaNet state for every linear-attention layer.
    fn fresh_state(
        gpu: &Gpu,
        config: &TextConfig,
        layers: &[Layer],
    ) -> Result<Vec<Option<DeltaState>>, ModelError> {
        let c = &config.block;
        let mut delta = Vec::new();
        delta
            .try_reserve_exact(layers.len())
            .map_err(|e| invalid(format!("GPU cache allocation: {e}")))?;
        for layer in layers {
            delta.push(match &layer.mixer {
                Mixer::Delta(_) => Some(DeltaState {
                    conv: gpu.zeros(size(c.channels()?, c.linear_conv_kernel_dim)?)?,
                    recurrent: gpu.zeros(size(
                        size(c.linear_num_value_heads, c.linear_key_head_dim)?,
                        c.linear_value_head_dim,
                    )?)?,
                }),
                Mixer::Full(_) => None,
            });
        }
        Ok(delta)
    }

    /// Empties the state for a pass that reads the whole history again.
    pub(super) fn reset(
        &mut self,
        gpu: &Gpu,
        config: &TextConfig,
        layers: &[Layer],
    ) -> Result<(), ModelError> {
        self.delta = Self::fresh_state(gpu, config, layers)?;
        self.kv.clear();
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    /// A full, independent copy of the cache, for a retained session or a shared prefix.
    pub(super) fn duplicate(
        &self,
        gpu: &Gpu,
        config: &TextConfig,
        layers: &[Layer],
    ) -> Result<Self, ModelError> {
        if self.replay {
            return Self::new(gpu, config, layers, self.capacity, self.owner);
        }
        let mut delta = Vec::new();
        delta
            .try_reserve_exact(self.delta.len())
            .map_err(|e| invalid(format!("GPU state staging: {e}")))?;
        for state in &self.delta {
            delta.push(match state {
                Some(s) => Some(DeltaState {
                    conv: gpu.copy_buffer(&s.conv)?,
                    recurrent: gpu.copy_buffer(&s.recurrent)?,
                }),
                None => None,
            });
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(self.tokens.len())
            .map_err(|e| invalid(format!("GPU token staging: {e}")))?;
        tokens.extend_from_slice(&self.tokens);
        Ok(Self {
            delta,
            kv: self.kv.clone_prefix(gpu)?,
            tokens,
            capacity: self.capacity,
            replay: false,
            owner: self.owner,
            epoch: 0,
            id: NEXT_CACHE.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub fn bytes_held(&self) -> usize {
        self.delta
            .iter()
            .flatten()
            .fold(self.kv.bytes_held(), |n, s| {
                n.saturating_add(
                    s.conv
                        .len()
                        .saturating_add(s.recurrent.len())
                        .saturating_mul(4),
                )
            })
    }

    pub fn checkpoint_bytes(&self) -> usize {
        self.bytes_held().saturating_add(self.kv.auxiliary_bytes())
    }

    pub fn kv_held(&self) -> (usize, usize) {
        (self.kv.pages_held(), self.bytes_held())
    }
}

impl SequenceCache for GpuCache {
    fn used(&self) -> usize {
        self.tokens.len()
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn clear(&mut self) {
        self.tokens.clear();
        self.replay = true;
        self.epoch = self.epoch.wrapping_add(1);
    }
    fn replays(&self) -> bool {
        self.replay
    }
    fn truncate(&mut self, len: usize) -> Result<(), RollbackError> {
        if len > self.tokens.len() {
            return Err(RollbackError {
                len,
                held: self.tokens.len(),
            });
        }
        if len != self.tokens.len() {
            self.tokens.truncate(len);
            self.replay = true;
            self.epoch = self.epoch.wrapping_add(1);
        }
        Ok(())
    }
}
