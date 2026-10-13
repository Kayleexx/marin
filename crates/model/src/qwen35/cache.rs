use super::math::invalid;
use super::{Qwen35, State};
use crate::ModelError;
use runtime::{RollbackError, SequenceCache};

pub struct Cache {
    pub(super) states: Vec<State>,
    pub(super) tokens: Vec<u32>,
    pub(super) capacity: usize,
    pub(super) replay: bool,
    pub(super) owner: u64,
}
impl Cache {
    pub(super) fn stage(&self, model: &Qwen35, end: usize) -> Result<Self, ModelError> {
        let mut states = Vec::new();
        states
            .try_reserve_exact(self.states.len())
            .map_err(|e| invalid(e.to_string()))?;
        for (block, state) in model.blocks.iter().zip(&self.states) {
            states.push(if self.replay {
                block.new_state(self.capacity, usize::MAX)?
            } else {
                state.staged(end)?
            });
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(end)
            .map_err(|e| invalid(e.to_string()))?;
        tokens.extend_from_slice(&self.tokens);
        Ok(Self {
            states,
            tokens,
            capacity: self.capacity,
            replay: false,
            owner: self.owner,
        })
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
        for s in &mut self.states {
            s.reset();
        }
        self.tokens.clear();
        self.replay = false;
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
            for s in &mut self.states {
                s.reset();
            }
            self.replay = len > 0;
        }
        Ok(())
    }
}
