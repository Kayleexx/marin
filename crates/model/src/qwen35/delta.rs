use super::math::{copy, invalid, sigmoid, silu, size, slice, softplus, zeros};
use super::weights::Delta;
use super::{Config, Observer, notify};
use crate::ModelError;

pub(super) fn forward(
    w: &Delta,
    c: &Config,
    x: &[f32],
    conv: &mut [f32],
    state: &mut [f32],
    observe: &mut Observer<'_>,
) -> Result<Vec<f32>, ModelError> {
    let qkv = w.qkv.apply(x)?;
    let z = w.z.apply(x)?;
    let a = w.a.apply(x)?;
    let b = w.b.apply(x)?;
    for (name, data) in [("qkv", &qkv), ("z", &z), ("a", &a), ("b", &b)] {
        notify(observe, name, data);
    }
    let mut cores = zeros(z.len())?;
    for ((((qkv, z), a), b), out) in qkv
        .chunks_exact(c.channels()?)
        .zip(z.chunks_exact(c.value_dim()?))
        .zip(a.chunks_exact(c.linear_num_value_heads))
        .zip(b.chunks_exact(c.linear_num_value_heads))
        .zip(cores.chunks_exact_mut(c.value_dim()?))
    {
        let core = recurrent(w, c, (qkv, z, a, b), conv, state, observe)?;
        out.copy_from_slice(&core);
    }
    w.out.apply(&cores)
}

fn recurrent(
    w: &Delta,
    c: &Config,
    projections: (&[f32], &[f32], &[f32], &[f32]),
    conv: &mut [f32],
    state: &mut [f32],
    observe: &mut Observer<'_>,
) -> Result<Vec<f32>, ModelError> {
    let (qkv, z, a, b) = projections;
    let mut mixed = zeros(qkv.len())?;
    let kernel = c.linear_conv_kernel_dim;
    for (((history, weight), input), out) in conv
        .chunks_exact_mut(kernel)
        .zip(w.conv.chunks_exact(kernel))
        .zip(qkv)
        .zip(&mut mixed)
    {
        history.rotate_left(1);
        *history
            .last_mut()
            .ok_or_else(|| invalid("empty convolution"))? = *input;
        *out = silu(tensor::dot(history, weight));
    }
    notify(observe, "conv", &mixed);
    let key_dim = c.key_dim()?;
    let query = slice(&mixed, 0, key_dim)?;
    let key = slice(&mixed, key_dim, key_dim)?;
    let value = slice(&mixed, size(key_dim, 2)?, c.value_dim()?)?;
    notify(observe, "query", query);
    notify(observe, "key", key);
    notify(observe, "value", value);
    let mut beta = zeros(b.len())?;
    for (dst, b) in beta.iter_mut().zip(b) {
        *dst = sigmoid(*b);
    }
    let mut g = zeros(a.len())?;
    for (((dst, a), dt), log) in g.iter_mut().zip(a).zip(&w.dt_bias).zip(&w.a_log) {
        *dst = -log.exp() * softplus(a + dt);
    }
    tensor::ensure_finite(&g, "Delta decay")?;
    notify(observe, "beta", &beta);
    notify(observe, "g", &g);
    let kd = c.linear_key_head_dim;
    let vd = c.linear_value_head_dim;
    let group = c
        .linear_num_value_heads
        .checked_div(c.linear_num_key_heads)
        .ok_or_else(|| invalid("Delta head ratio"))?;
    let mut core = zeros(c.value_dim()?)?;
    for (h, ((s, out), (g, beta))) in state
        .chunks_exact_mut(size(kd, vd)?)
        .zip(core.chunks_exact_mut(vd))
        .zip(g.iter().zip(&beta))
        .enumerate()
    {
        let head = h
            .checked_div(group)
            .ok_or_else(|| invalid("Delta head ratio"))?;
        let mut q = copy(slice(query, size(head, kd)?, kd)?)?;
        let mut k = copy(slice(key, size(head, kd)?, kd)?)?;
        normalize(&mut q);
        normalize(&mut k);
        for v in &mut q {
            *v /= (kd as f32).sqrt();
        }
        for v in s.iter_mut() {
            *v *= g.exp();
        }
        let v = slice(value, size(h, vd)?, vd)?;
        let mut delta = zeros(vd)?;
        for (row, k) in s.chunks_exact(vd).zip(&k) {
            for (pred, s) in delta.iter_mut().zip(row) {
                *pred += s * k;
            }
        }
        for (d, v) in delta.iter_mut().zip(v) {
            *d = (v - *d) * beta;
        }
        for ((row, k), q) in s.chunks_exact_mut(vd).zip(&k).zip(&q) {
            for ((s, d), out) in row.iter_mut().zip(&delta).zip(out.iter_mut()) {
                *s += k * d;
                *out += *s * q;
            }
        }
    }
    tensor::ensure_finite(state, "Delta recurrent state")?;
    notify(observe, "core", &core);
    tensor::rms_norm_rows(&mut core, &w.norm, c.rms_norm_eps)?;
    for (v, z) in core.iter_mut().zip(z) {
        *v *= silu(*z);
    }
    notify(observe, "gated", &core);
    Ok(core)
}
fn normalize(x: &mut [f32]) {
    let inverse = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    for v in x {
        *v *= inverse;
    }
}
