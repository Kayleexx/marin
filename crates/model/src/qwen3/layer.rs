use kv::KvCache;
use metrics::{Probe, Stage};
use tensor::{Matrix, add_in_place, rms_norm_rows};

use super::attention::{attend, rope_heads};
use super::config::Qwen3Config;
use super::mlp::{MlpBuffers, mlp};
use super::weights::LayerWeights;
use crate::ModelError;

/// Buffers one forward pass reuses for every layer, so a prompt does not allocate and fault in about 35 MB of
/// fresh memory per layer (that cost 15% of the time and 2 s of system time at 8 threads).
#[derive(Default)]
pub(crate) struct Buffers {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    context: Vec<f32>,
    attended: Vec<f32>,
    mlp: MlpBuffers,
}

fn project(
    weights: &Matrix,
    x: &[f32],
    tokens: usize,
    out: &mut Vec<f32>,
) -> Result<(), ModelError> {
    let size = tokens
        .checked_mul(weights.rows())
        .ok_or_else(|| ModelError::InvalidConfig("projection size overflows".into()))?;
    out.resize(size, 0.0);
    Ok(weights.matmul(x, out)?)
}

impl LayerWeights<Matrix, Vec<f32>> {
    /// Applies attention and MLP blocks (with residuals) to `hidden`, `[token][hidden_size]`.
    pub(crate) fn forward(
        &self,
        cfg: &Qwen3Config,
        layer: usize,
        hidden: &mut [f32],
        start: usize,
        (kv, buffers): (&mut KvCache, &mut Buffers),
        probe: &mut Probe<'_>,
    ) -> Result<(), ModelError> {
        let tokens = hidden.len().checked_div(cfg.hidden_size).unwrap_or(0);
        let Buffers {
            normed,
            q,
            k,
            v,
            context,
            attended,
            mlp: mlp_buffers,
        } = buffers;

        normed.clear();
        normed.extend_from_slice(hidden);
        probe.time(Stage::Norm, || {
            rms_norm_rows(normed, &self.input_norm, cfg.rms_eps)
        })?;
        probe.time(Stage::Qkv, || {
            project(&self.q_proj, normed, tokens, q)?;
            project(&self.k_proj, normed, tokens, k)?;
            project(&self.v_proj, normed, tokens, v)
        })?;
        probe.time(Stage::Norm, || {
            rms_norm_rows(q, &self.q_norm, cfg.rms_eps)?;
            rms_norm_rows(k, &self.k_norm, cfg.rms_eps)
        })?;
        probe.time(Stage::Rope, || {
            rope_heads(q, cfg.q_dim()?, cfg, start)?;
            rope_heads(k, cfg.kv_dim()?, cfg, start)
        })?;

        probe.time(Stage::KvWrite, || kv.write(layer, k, v))?;
        let visible = start.saturating_add(tokens);
        let (keys, values) = kv.read(layer, visible)?;
        context.resize(q.len(), 0.0);
        probe.time(Stage::Attention, || {
            attend(cfg, q, keys, values, start, context)
        })?;
        probe.time(Stage::OutProj, || {
            project(&self.o_proj, context, tokens, attended)
        })?;
        probe.time(Stage::Residual, || add_in_place(hidden, attended))?;

        normed.clear();
        normed.extend_from_slice(hidden);
        probe.time(Stage::Norm, || {
            rms_norm_rows(normed, &self.post_norm, cfg.rms_eps)
        })?;
        probe.time(Stage::Mlp, || {
            mlp(
                normed,
                tokens,
                (&self.gate_proj, &self.up_proj, &self.down_proj),
                mlp_buffers,
            )
        })?;
        probe.time(Stage::Residual, || add_in_place(hidden, &mlp_buffers.out))?;
        Ok(())
    }
}
