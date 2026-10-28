use super::math::{copy, invalid};
use super::state::Storage;
use super::{Qwen35, State};
use crate::ModelError;
use runtime::{RollbackError, SequenceCache};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CACHE: AtomicU64 = AtomicU64::new(1);

pub struct Cache {
    pub(super) states: Vec<State>,
    pub(super) tokens: Vec<u32>,
    pub(super) capacity: usize,
    pub(super) replay: bool,
    pub(super) owner: u64,
    /// Counts the times the history was cut back or started again, so a `CpuMark` taken before cannot be applied after.
    epoch: u64,
    /// Tells this cache from every other, so a `CpuMark` is only ever put back into the cache it came from.
    id: u64,
}

/// The recurrent state and the length of the history at one moment. Attention keys are only appended past the
/// committed length, so cutting them back is enough; what a pass changes in place is the DeltaNet state, kept here.
pub struct CpuMark {
    saved: Vec<Saved>,
    tokens: usize,
    epoch: u64,
    cache: u64,
}
pub(super) enum Undo {
    Replaced(Vec<State>),
    Marked(Vec<Saved>),
}
pub(super) enum Saved {
    Delta {
        conv: Vec<f32>,
        recurrent: Vec<f32>,
        len: usize,
    },
    Full {
        len: usize,
    },
}
impl Cache {
    pub(super) fn new(states: Vec<State>, capacity: usize, owner: u64) -> Self {
        Self {
            states,
            tokens: Vec::new(),
            capacity,
            replay: false,
            owner,
            epoch: 0,
            id: NEXT_CACHE.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// A mark of the committed history. A cache that is to be read again holds state a mark must not copy.
    pub(super) fn mark(&self) -> Result<CpuMark, ModelError> {
        if self.replay {
            return Err(invalid("a cache that is to be read again cannot be marked"));
        }
        let saved = self.save()?;
        Ok(CpuMark {
            saved,
            tokens: self.tokens.len(),
            epoch: self.epoch,
            cache: self.id,
        })
    }

    /// Returns to `mark`, refused if the history was cut back or restarted since, or if the cache is not the one the
    /// mark came from. If the state cannot be put back the cache is left to be read again, never partly restored.
    pub(super) fn rewind(&mut self, mark: CpuMark) -> Result<(), ModelError> {
        if mark.cache != self.id
            || mark.epoch != self.epoch
            || self.replay
            || self.tokens.len() < mark.tokens
        {
            return Err(invalid("the cache no longer extends the marked history"));
        }
        self.undo(Undo::Marked(mark.saved));
        if self.replay {
            self.clear();
            return Err(invalid("the marked state could not be put back"));
        }
        self.tokens.truncate(mark.tokens);
        Ok(())
    }

    fn save(&self) -> Result<Vec<Saved>, ModelError> {
        let mut saved = Vec::new();
        saved
            .try_reserve_exact(self.states.len())
            .map_err(|e| invalid(e.to_string()))?;
        for state in &self.states {
            saved.push(match &state.storage {
                Storage::Delta { conv, recurrent } => Saved::Delta {
                    conv: copy(conv)?,
                    recurrent: copy(recurrent)?,
                    len: state.len,
                },
                Storage::Full(_) => Saved::Full { len: state.len },
            });
        }
        Ok(saved)
    }

    /// Readies a pass over the cache in place. Recurrent state is copied first and attention keys are only appended
    /// past the committed length, so `undo` can put everything back. A replay starts from fresh state instead.
    pub(super) fn begin(&mut self, model: &Qwen35, end: usize) -> Result<Undo, ModelError> {
        self.tokens
            .try_reserve(end.saturating_sub(self.tokens.len()))
            .map_err(|e| invalid(e.to_string()))?;
        if self.replay {
            let mut fresh = Vec::new();
            fresh
                .try_reserve_exact(self.states.len())
                .map_err(|e| invalid(e.to_string()))?;
            for block in &model.blocks {
                fresh.push(block.new_state(self.capacity, usize::MAX)?);
            }
            return Ok(Undo::Replaced(std::mem::replace(&mut self.states, fresh)));
        }
        for state in &mut self.states {
            if let Storage::Full(kv) = &mut state.storage {
                kv.reserve(end)?;
            }
        }
        let saved = self.save()?;
        Ok(Undo::Marked(saved))
    }
    /// Puts back what a failed pass changed. Keys written past the old length are simply forgotten.
    pub(super) fn undo(&mut self, undo: Undo) {
        match undo {
            Undo::Replaced(old) => self.states = old,
            Undo::Marked(saved) => {
                for (state, saved) in self.states.iter_mut().zip(saved) {
                    match (&mut state.storage, saved) {
                        (
                            Storage::Delta { conv, recurrent },
                            Saved::Delta {
                                conv: c,
                                recurrent: r,
                                len,
                            },
                        ) => {
                            *conv = c;
                            *recurrent = r;
                            state.len = len;
                        }
                        (Storage::Full(kv), Saved::Full { len }) => {
                            if kv.truncate(len).is_ok() {
                                state.len = len;
                            } else {
                                self.replay = true;
                            }
                        }
                        _ => self.replay = true,
                    }
                }
            }
        }
    }
    pub(super) fn commit(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.replay = false;
    }
    pub fn bytes_in_use(&self) -> usize {
        self.states.iter().fold(0_usize, |n, s| {
            n.saturating_add(match &s.storage {
                super::state::Storage::Delta { conv, recurrent } => {
                    conv.len().saturating_add(recurrent.len()).saturating_mul(4)
                }
                super::state::Storage::Full(kv) => kv.bytes_initialized(),
            })
        })
    }
    pub fn bytes_held(&self) -> usize {
        self.states.iter().fold(0_usize, |n, s| {
            n.saturating_add(match &s.storage {
                super::state::Storage::Delta { conv, recurrent } => conv
                    .capacity()
                    .saturating_add(recurrent.capacity())
                    .saturating_mul(4),
                super::state::Storage::Full(kv) => kv.bytes_held(),
            })
        })
    }
}
impl SequenceCache for Cache {
    fn used(&self) -> usize {
        self.tokens.len()
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        for s in &mut self.states {
            s.reset();
        }
        self.tokens.clear();
        self.replay = false;
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
            self.epoch = self.epoch.wrapping_add(1);
            self.tokens.truncate(len);
            for s in &mut self.states {
                s.reset();
            }
            self.replay = len > 0;
        }
        Ok(())
    }
}
