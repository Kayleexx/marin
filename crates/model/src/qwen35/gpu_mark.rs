use super::gpu_cache::{DeltaState, GpuCache};
use super::math::invalid;
use crate::ModelError;
use cuda::Gpu;
use runtime::SequenceCache;

struct HostState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

enum Saved {
    Device(Vec<Option<DeltaState>>),
    Host(Vec<Option<HostState>>),
}

/// A point in the history of a cache that it can be returned to exactly, without a second copy of the keys and values:
/// those are only ever appended past the committed length, so cutting them back is enough. What a pass does change in
/// place is the DeltaNet state, and that is what a mark holds, on the GPU or, when the GPU has no room, in host memory.
pub struct Mark {
    saved: Saved,
    positions: usize,
    tokens: usize,
    epoch: u64,
    cache: u64,
}

fn reserve<T>(len: usize) -> Result<Vec<T>, ModelError> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|e| invalid(format!("state mark: {e}")))?;
    Ok(out)
}

impl GpuCache {
    fn marked(&self, saved: Saved) -> Mark {
        Mark {
            saved,
            positions: self.kv.len(),
            tokens: self.tokens.len(),
            epoch: self.epoch,
            cache: self.id,
        }
    }

    /// A mark of the committed state kept on the GPU: a copy of the DeltaNet state and the lengths of the history.
    pub(super) fn mark(&self, gpu: &Gpu) -> Result<Mark, ModelError> {
        let mut delta = reserve(self.delta.len())?;
        for state in &self.delta {
            delta.push(match state {
                Some(s) => Some(DeltaState {
                    conv: gpu.copy_buffer(&s.conv)?,
                    recurrent: gpu.copy_buffer(&s.recurrent)?,
                }),
                None => None,
            });
        }
        Ok(self.marked(Saved::Device(delta)))
    }

    /// The same mark with the DeltaNet state in host memory, which needs no free GPU memory now or when it is put back.
    pub(super) fn mark_in_host(&self, gpu: &Gpu) -> Result<Mark, ModelError> {
        let mut host = reserve(self.delta.len())?;
        for state in &self.delta {
            host.push(match state {
                Some(s) => Some(HostState {
                    conv: gpu.download(&s.conv)?,
                    recurrent: gpu.download(&s.recurrent)?,
                }),
                None => None,
            });
        }
        Ok(self.marked(Saved::Host(host)))
    }

    /// Returns to `mark`: the keys and values past its length are forgotten and its DeltaNet state put back. Refused if
    /// the history was cut back or restarted since, or the cache is not the one the mark came from. If the state
    /// cannot be put back the cache is left to be read again from the start, never partly restored.
    pub(super) fn rewind(&mut self, gpu: &Gpu, mark: Mark) -> Result<(), ModelError> {
        let holds = mark.cache == self.id
            && mark.epoch == self.epoch
            && !self.replay
            && self.kv.len() >= mark.positions
            && self.tokens.len() >= mark.tokens;
        if !holds {
            return Err(invalid("the cache no longer extends the marked history"));
        }
        self.kv.truncate(mark.positions)?;
        self.tokens.truncate(mark.tokens);
        match mark.saved {
            Saved::Device(delta) => self.delta = delta,
            Saved::Host(host) => {
                if let Err(error) = self.restore(gpu, &host) {
                    self.clear();
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    fn restore(&mut self, gpu: &Gpu, host: &[Option<HostState>]) -> Result<(), ModelError> {
        if host.len() != self.delta.len() {
            return Err(invalid("the marked state does not fit this cache"));
        }
        for (live, kept) in self.delta.iter_mut().zip(host) {
            if let (Some(live), Some(kept)) = (live, kept) {
                gpu.upload_into(&kept.conv, &mut live.conv)?;
                gpu.upload_into(&kept.recurrent, &mut live.recurrent)?;
            }
        }
        Ok(())
    }

    /// GPU memory a device mark of this cache takes: the DeltaNet state, whatever the history.
    pub fn mark_bytes(&self) -> usize {
        self.delta.iter().flatten().fold(0_usize, |n, s| {
            n.saturating_add(
                s.conv
                    .len()
                    .saturating_add(s.recurrent.len())
                    .saturating_mul(4),
            )
        })
    }
}

#[cfg(test)]
impl Mark {
    /// Makes a host mark that no longer fits the cache it came from.
    pub(super) fn drop_a_layer(&mut self) {
        if let Saved::Host(host) = &mut self.saved {
            host.pop();
        }
    }
}
