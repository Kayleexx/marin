//! Dense Qwen3.5 text backbone on CPU, with FP32 math and explicit hybrid state.
mod cache;
mod config;
mod delta;
mod full;
mod gpu;
mod gpu_budget;
mod gpu_cache;
mod gpu_forward;
mod gpu_weights;
pub(crate) mod layout;
mod math;
mod model;
mod model_config;
mod read;
mod state;
mod weights;
pub use cache::Cache;
pub use gpu::Qwen35Gpu;
pub use gpu_budget::bytes as gpu_budget_bytes;
pub use gpu_cache::GpuCache;
pub use model::{Qwen35, load_bytes, weight_bytes};
pub use model_config::{TextConfig, is_qwen35};

use crate::ModelError;
pub use config::{Config, LayerKind, Rope};
use math::{copy, invalid, size};
pub use state::State;
use state::{Storage, admit};
use std::collections::BTreeMap;
use weights::{Common, Mixer};

// Vector bookkeeping and allocator overhead for the reference pass.
const ALLOCATION_SLACK: usize = 1024;

pub type Observer<'a> = Option<&'a mut dyn FnMut(&str, &[f32])>;
pub(super) fn notify(observe: &mut Observer<'_>, name: &str, data: &[f32]) {
    if let Some(callback) = observe {
        callback(name, data);
    }
}

pub struct ReferenceBlock {
    config: Config,
    kind: LayerKind,
    common: Common,
    mixer: Mixer,
}
impl ReferenceBlock {
    /// Exact FP32 weights with official names relative to `model.language_model.layers.N.`.
    /// Caller owns/admitted weight memory; malformed and extra tensors are rejected.
    pub fn new(
        config: Config,
        kind: LayerKind,
        weights: BTreeMap<String, Vec<f32>>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        let (common, mixer) = weights::load(&config, kind, weights)?;
        Ok(Self {
            config,
            kind,
            common,
            mixer,
        })
    }
    /// Budget is additional allocation, excluding caller-owned weights. No full-context KV reservation.
    pub fn new_state(&self, capacity: usize, budget: usize) -> Result<State, ModelError> {
        if capacity == 0 {
            return Err(invalid("zero context capacity"));
        }
        let needed = self.state_floats(0)?;
        admit(allocation_bytes(needed)?, budget)?;
        State::new(&self.config, self.kind, capacity)
    }
    fn state_floats(&self, end: usize) -> Result<usize, ModelError> {
        let c = &self.config;
        match self.kind {
            LayerKind::LinearAttention => size(c.channels()?, c.linear_conv_kernel_dim)?
                .checked_add(size(
                    size(c.linear_num_value_heads, c.linear_key_head_dim)?,
                    c.linear_value_head_dim,
                )?)
                .ok_or_else(|| invalid("state overflow")),
            // Match growable KV's 256-position window (conservatively includes growth slack).
            LayerKind::FullAttention => {
                let positions = if end == 0 {
                    0
                } else {
                    end.checked_next_multiple_of(256)
                        .and_then(usize::checked_next_power_of_two)
                        .ok_or_else(|| invalid("KV window overflow"))?
                };
                size(
                    size(size(c.num_key_value_heads, c.head_dim)?, positions)?,
                    2,
                )
            }
        }
    }
    /// Upper bound for transaction copy, output and reference workspace. Existing state remains live.
    pub fn additional_bytes(&self, tokens: usize, end: usize) -> Result<usize, ModelError> {
        let c = &self.config;
        let width = [
            c.hidden_size,
            c.intermediate_size,
            c.channels()?,
            size(c.num_attention_heads, c.head_dim)?,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let floats = self
            .state_floats(end)?
            .checked_add(size(size(width, 32)?, tokens)?)
            .and_then(|n| {
                n.checked_add(
                    width
                        .checked_mul(128)?
                        .checked_mul(tensor::threads().min(width))?,
                )
            })
            .and_then(|n| n.checked_add(tokens.checked_mul(c.hidden_size)?))
            .ok_or_else(|| invalid("workspace overflow"))?;
        allocation_bytes(floats)
    }
    /// Atomic block pass: commit recurrent/conv or KV state only after all tokens succeed.
    /// Dense operations batch the whole input; recurrence remains sequential.
    pub fn forward(
        &self,
        input: &[f32],
        state: &mut State,
        budget: usize,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        if state.config != *c || state.kind != self.kind {
            return Err(invalid("incompatible block state"));
        }
        if input.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if !input.len().is_multiple_of(c.hidden_size) {
            return Err(invalid("ragged block input"));
        }
        tensor::ensure_finite(input, "Qwen3.5 block input")?;
        let tokens = input
            .len()
            .checked_div(c.hidden_size)
            .ok_or_else(|| invalid("zero hidden size"))?;
        let end = state
            .len
            .checked_add(tokens)
            .ok_or_else(|| invalid("context overflow"))?;
        if end > state.capacity {
            return Err(kv::KvError::ContextFull {
                needed: end,
                capacity: state.capacity,
            }
            .into());
        }
        admit(self.additional_bytes(tokens, end)?, budget)?;
        let mut staged = state.staged(end)?;
        let output = self.forward_in_place(input, &mut staged, observe)?;
        *state = staged;
        Ok(output)
    }
    pub(super) fn forward_in_place(
        &self,
        input: &[f32],
        state: &mut State,
        mut observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let end = state
            .len
            .checked_add(input.len().checked_div(c.hidden_size).unwrap_or(0))
            .ok_or_else(|| invalid("context overflow"))?;
        // Small passes retain GEMV: avoid GEMM packing/thread startup on decode and short prompts.
        let width = if !self.common.gate.is_matrix()
            || input.len().checked_div(c.hidden_size).unwrap_or(0) < 16
        {
            c.hidden_size
        } else {
            input.len()
        };
        let mut output = math::zeros(input.len())?;
        for (input, dst) in input
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
        {
            dst.copy_from_slice(&self.forward_batch(input, state, &mut observe)?);
        }
        state.len = end;
        Ok(output)
    }
    fn forward_batch(
        &self,
        input: &[f32],
        state: &mut State,
        observe: &mut Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let mut norm = copy(input)?;
        tensor::rms_norm_rows(&mut norm, &self.common.input_norm, c.rms_norm_eps)?;
        notify(observe, "input_norm", &norm);
        let mut output = match (&self.mixer, &mut state.storage) {
            (Mixer::Delta(w), Storage::Delta { conv, recurrent }) => {
                delta::forward(w, c, &norm, conv, recurrent, observe)?
            }
            (Mixer::Full(w), Storage::Full(kv)) => full::forward(w, c, &norm, kv, observe)?,
            _ => return Err(invalid("incompatible mixer/state")),
        };
        notify(observe, "mixer", &output);
        tensor::add_in_place(&mut output, input)?;
        notify(observe, "residual", &output);
        norm.copy_from_slice(&output);
        tensor::rms_norm_rows(&mut norm, &self.common.post_norm, c.rms_norm_eps)?;
        notify(observe, "post_norm", &norm);
        let mut gate = self.common.gate.apply(&norm)?;
        let up = self.common.up.apply(&norm)?;
        notify(observe, "mlp_gate", &gate);
        notify(observe, "mlp_up", &up);
        tensor::silu_mul(&mut gate, &up)?;
        drop(up);
        notify(observe, "mlp_act", &gate);
        let mlp = self.common.down.apply(&gate)?;
        notify(observe, "mlp", &mlp);
        tensor::add_in_place(&mut output, &mlp)?;
        tensor::ensure_finite(&output, "Qwen3.5 block output")?;
        notify(observe, "output", &output);
        Ok(output)
    }
}

fn allocation_bytes(floats: usize) -> Result<usize, ModelError> {
    size(floats, size_of::<f32>())?
        .checked_add(ALLOCATION_SLACK)
        .ok_or_else(|| invalid("allocation budget overflow"))
}

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod model_tests;
