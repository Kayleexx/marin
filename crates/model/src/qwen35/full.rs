use super::math::{sigmoid, size, slice, zeros};
use super::weights::Full;
use super::{Config, Observer, notify};
use crate::ModelError;
use kv::KvCache;

pub(super) fn forward(
    w: &Full,
    c: &Config,
    x: &[f32],
    kv: &mut KvCache,
    observe: &mut Observer<'_>,
) -> Result<Vec<f32>, ModelError> {
    let projected = w.q.apply(x)?;
    let mut k = w.k.apply(x)?;
    let v = w.v.apply(x)?;
    notify(observe, "q_gate", &projected);
    notify(observe, "k_proj", &k);
    notify(observe, "v_proj", &v);
    let q_dim = size(c.num_attention_heads, c.head_dim)?;
    let tokens = x.len().checked_div(c.hidden_size).unwrap_or(0);
    let mut q = zeros(size(tokens, q_dim)?)?;
    let mut gate = zeros(q.len())?;
    for ((src, q), gate) in projected
        .chunks_exact(size(c.head_dim, 2)?)
        .zip(q.chunks_exact_mut(c.head_dim))
        .zip(gate.chunks_exact_mut(c.head_dim))
    {
        q.copy_from_slice(slice(src, 0, c.head_dim)?);
        gate.copy_from_slice(slice(src, c.head_dim, c.head_dim)?);
    }
    tensor::rms_norm_rows(&mut q, &w.q_norm, c.rms_norm_eps)?;
    tensor::rms_norm_rows(&mut k, &w.k_norm, c.rms_norm_eps)?;
    notify(observe, "q_norm", &q);
    notify(observe, "k_norm", &k);
    for (data, width) in [
        (&mut q, q_dim),
        (&mut k, size(c.num_key_value_heads, c.head_dim)?),
    ] {
        for (t, row) in data.chunks_exact_mut(width).enumerate() {
            for head in row.chunks_exact_mut(c.head_dim) {
                let rotary = head
                    .get_mut(..c.rotary())
                    .ok_or_else(|| super::math::invalid("rotary shape"))?;
                tensor::rope_tokens(
                    rotary,
                    c.rotary(),
                    c.rotary(),
                    kv.len()
                        .checked_add(t)
                        .ok_or_else(|| super::math::invalid("position overflow"))?,
                    c.rope_parameters.rope_theta,
                )?;
            }
        }
    }
    notify(observe, "q_rope", &q);
    notify(observe, "k_rope", &k);
    let start = kv.len();
    kv.write(0, &k, &v)?;
    kv.advance(tokens)?;
    let (keys, values) = kv.read(0, kv.len())?;
    let mut core = zeros(q.len())?;
    tensor::attention(
        &tensor::AttentionShape {
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
        },
        &q,
        keys,
        values,
        start,
        &mut core,
    )?;
    notify(observe, "core", &core);
    for (v, gate) in core.iter_mut().zip(gate) {
        *v *= sigmoid(gate);
    }
    notify(observe, "gated", &core);
    w.out.apply(&core)
}
