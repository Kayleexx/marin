use super::math::{copy, invalid, size, zeros};
use super::{Config, LayerKind};
use crate::ModelError;
use kv::KvCache;

pub(super) enum Storage {
    Delta { conv: Vec<f32>, recurrent: Vec<f32> },
    Full(KvCache),
}
/// One independent block/session. Delta state deliberately has no KV truncation contract.
pub struct State {
    pub(super) config: Config,
    pub(super) kind: LayerKind,
    pub(super) storage: Storage,
    pub(super) capacity: usize,
    pub(super) len: usize,
}
impl State {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn reset(&mut self) {
        match &mut self.storage {
            Storage::Delta { conv, recurrent } => {
                conv.fill(0.0);
                recurrent.fill(0.0);
            }
            Storage::Full(kv) => kv.clear(),
        }
        self.len = 0;
    }
    pub fn recurrent(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::Delta { recurrent, .. } => Some(recurrent),
            _ => None,
        }
    }
    pub fn convolution(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::Delta { conv, .. } => Some(conv),
            _ => None,
        }
    }
    pub fn attention_kv(&self) -> Result<(&[f32], &[f32]), ModelError> {
        match &self.storage {
            Storage::Full(kv) => Ok(kv.read(0, self.len)?),
            _ => Err(invalid("Delta state has no attention KV")),
        }
    }
    pub(super) fn new(c: &Config, kind: LayerKind, capacity: usize) -> Result<Self, ModelError> {
        let storage = match kind {
            LayerKind::LinearAttention => Storage::Delta {
                conv: zeros(size(c.channels()?, c.linear_conv_kernel_dim)?)?,
                recurrent: zeros(size(
                    size(c.linear_num_value_heads, c.linear_key_head_dim)?,
                    c.linear_value_head_dim,
                )?)?,
            },
            LayerKind::FullAttention => Storage::Full(KvCache::new(
                1,
                c.num_key_value_heads,
                c.head_dim,
                capacity,
            )?),
        };
        Ok(Self {
            config: c.clone(),
            kind,
            storage,
            capacity,
            len: 0,
        })
    }
    pub(super) fn staged(&self, end: usize) -> Result<Self, ModelError> {
        let storage = match &self.storage {
            Storage::Delta { conv, recurrent } => Storage::Delta {
                conv: copy(conv)?,
                recurrent: copy(recurrent)?,
            },
            Storage::Full(kv) => {
                let mut next = KvCache::new(
                    1,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.capacity,
                )?;
                next.reserve(end)?;
                if self.len > 0 {
                    let (k, v) = kv.read(0, self.len)?;
                    next.write(0, k, v)?;
                    next.advance(self.len)?;
                }
                Storage::Full(next)
            }
        };
        Ok(Self {
            config: self.config.clone(),
            kind: self.kind,
            storage,
            capacity: self.capacity,
            len: self.len,
        })
    }
}

/// Additional live allocation admission. The caller's explicit budget excludes already owned weights/state.
pub(super) fn admit(needed: usize, budget: usize) -> Result<(), ModelError> {
    if needed > budget {
        return Err(invalid(format!(
            "Qwen3.5 reference needs {needed} additional bytes, budget is {budget}"
        )));
    }
    let needed = u64::try_from(needed).map_err(|_| invalid("memory size overflow"))?;
    let guarded = needed
        .checked_add(crate::CPU_SLACK_BYTES)
        .ok_or_else(|| invalid("memory size overflow"))?;
    if let Some(available) = metrics::cpu_available_bytes().map_err(ModelError::CpuMemoryInfo)?
        && guarded > available
    {
        return Err(ModelError::CpuMemory {
            needed: guarded,
            available,
        });
    }
    Ok(())
}
