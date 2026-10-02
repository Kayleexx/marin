use kv::KvCache;
use metrics::{Probe, Stage};
use tensor::{Matrix, add_in_place, rms_norm_rows};

use super::attention::{attend, rope_heads};
use super::config::Qwen3Config;
use super::mlp::mlp;
use super::weights::LayerWeights;
use crate::ModelError;

fn project(weights: &Matrix, x: &[f32], tokens: usize) -> Result<Vec<f32>, ModelError> {
    let size = tokens
        .checked_mul(weights.rows())
        .ok_or_else(|| ModelError::InvalidConfig("projection size overflows".into()))?;
    let mut out = vec![0.0; size];
    weights.matmul(x, &mut out)?;
    Ok(out)
}

impl LayerWeights<Matrix, Vec<f32>> {
    /// Applies attention and MLP blocks (with residuals) to `hidden`, `[token][hidden_size]`.
    pub(crate) fn forward(
        &self,
        cfg: &Qwen3Config,
        layer: usize,
        hidden: &mut [f32],
        start: usize,
        kv: &mut KvCache,
        probe: &mut Probe<'_>,
    ) -> Result<(), ModelError> {
        let tokens = hidden.len().checked_div(cfg.hidden_size).unwrap_or(0);

        let mut normed = hidden.to_vec();
        probe.time(Stage::Norm, || {
            rms_norm_rows(&mut normed, &self.input_norm, cfg.rms_eps)
        })?;
        let (mut q, mut k, v) = probe.time(Stage::Qkv, || {
            Ok::<_, ModelError>((
                project(&self.q_proj, &normed, tokens)?,
                project(&self.k_proj, &normed, tokens)?,
                project(&self.v_proj, &normed, tokens)?,
            ))
        })?;
        probe.time(Stage::Norm, || {
            rms_norm_rows(&mut q, &self.q_norm, cfg.rms_eps)?;
            rms_norm_rows(&mut k, &self.k_norm, cfg.rms_eps)
        })?;
        probe.time(Stage::Rope, || {
            rope_heads(&mut q, cfg.q_dim()?, cfg, start)?;
            rope_heads(&mut k, cfg.kv_dim()?, cfg, start)
        })?;

        probe.time(Stage::KvWrite, || kv.write(layer, &k, &v))?;
        let visible = start.saturating_add(tokens);
        let (keys, values) = kv.read(layer, visible)?;
        let mut context = vec![0.0; q.len()];
        probe.time(Stage::Attention, || {
            attend(cfg, &q, keys, values, start, &mut context)
        })?;
        let attended = probe.time(Stage::OutProj, || project(&self.o_proj, &context, tokens))?;
        probe.time(Stage::Residual, || add_in_place(hidden, &attended))?;

        let mut normed = hidden.to_vec();
        probe.time(Stage::Norm, || {
            rms_norm_rows(&mut normed, &self.post_norm, cfg.rms_eps)
        })?;
        let ff = probe.time(Stage::Mlp, || {
            mlp(
                &normed,
                tokens,
                &self.gate_proj,
                &self.up_proj,
                &self.down_proj,
            )
        })?;
        probe.time(Stage::Residual, || add_in_place(hidden, &ff))?;
        Ok(())
    }
}
