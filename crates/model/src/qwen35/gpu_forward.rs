use super::Qwen35Gpu;
use super::gpu::Workspace;
use super::gpu_cache::GpuCache;
use super::gpu_weights::{Delta, Full, Mixer};
use super::math::{invalid, size};
use crate::qwen3::gpu_probe::GpuProbe;
use crate::{ModelError, Observer};
use cuda::{DeltaDims, DeltaInputs, DeviceKv, Gpu};
use metrics::Stage;

#[cfg(test)]
thread_local! {
    /// Layers still to run before a test's injected failure; `None` means no failure is armed.
    pub(super) static FAIL_AFTER_LAYERS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Whether the layer about to run is the one a test asked to fail.
#[cfg(test)]
fn injected_failure() -> bool {
    FAIL_AFTER_LAYERS.with(|armed| match armed.get() {
        Some(0) => {
            armed.set(None);
            true
        }
        Some(left) => {
            armed.set(Some(left.saturating_sub(1)));
            false
        }
        None => false,
    })
}

/// The sequences one pass serves: the chunk of a single sequence, or one token each of several.
pub(super) enum Rows<'a> {
    One(&'a mut GpuCache),
    Batch {
        caches: Vec<&'a mut GpuCache>,
        /// Single-row buffers the per-sequence stages work in.
        scratch: Box<Workspace>,
    },
}

pub(super) fn pass_chunk(
    model: &Qwen35Gpu,
    rows: &mut Rows<'_>,
    tokens: &[u32],
    probe: &mut GpuProbe<'_>,
    mut observe: crate::Observer<'_>,
    each: bool,
) -> Result<Vec<f32>, ModelError> {
    let (gpu, c, count) = (&model.gpu, &model.config.block, tokens.len());
    let logit_rows = match rows {
        Rows::One(_) if each => count,
        Rows::One(_) => 1,
        Rows::Batch { caches, .. } => caches.len(),
    };
    let mut ws = Workspace::new(gpu, &model.config, (count, logit_rows))?;
    probe.stage(Stage::Embedding, || {
        Ok(model
            .embed
            .embed(gpu, tokens, &mut ws.ids, &mut ws.hidden)?)
    })?;
    watch(&mut observe, gpu, 0, &ws.hidden, c.hidden_size)?;
    for (index, layer) in model.layers.iter().enumerate() {
        #[cfg(test)]
        if injected_failure() {
            return Err(invalid("injected failure"));
        }
        probe.layer(Some(index));
        probe.stage(Stage::Norm, || {
            Ok(gpu.rms_norm_into(
                &ws.hidden,
                &layer.common.input_norm,
                &mut ws.normed,
                c.rms_norm_eps,
            )?)
        })?;
        match &layer.mixer {
            Mixer::Delta(w) => delta(gpu, w, c, &mut ws, (rows, index), count, probe)?,
            Mixer::Full(w) => full(gpu, w, c, &mut ws, rows, count, probe)?,
        }
        probe.stage(Stage::Residual, || {
            Ok(gpu.add(&mut ws.hidden, &ws.attended)?)
        })?;
        probe.stage(Stage::Norm, || {
            Ok(gpu.rms_norm_into(
                &ws.hidden,
                &layer.common.post_norm,
                &mut ws.normed,
                c.rms_norm_eps,
            )?)
        })?;
        probe.stage(Stage::Mlp, || {
            layer
                .common
                .gate
                .matmul_precise(gpu, &ws.normed, count, &mut ws.gate)?;
            layer
                .common
                .up
                .matmul_precise(gpu, &ws.normed, count, &mut ws.up)?;
            gpu.silu_mul(&mut ws.gate, &ws.up)?;
            Ok(layer
                .common
                .down
                .matmul_precise(gpu, &ws.gate, count, &mut ws.ff)?)
        })?;
        probe.stage(Stage::Residual, || Ok(gpu.add(&mut ws.hidden, &ws.ff)?))?;
        watch(
            &mut observe,
            gpu,
            index.saturating_add(1),
            &ws.hidden,
            c.hidden_size,
        )?;
    }
    probe.layer(None);
    match rows {
        Rows::One(_) if !each => {
            probe.stage(Stage::Norm, || {
                gpu.copy_range_into(
                    &ws.hidden,
                    size(count.saturating_sub(1), c.hidden_size)?,
                    &mut ws.last,
                )?;
                Ok(gpu.rms_norm(&mut ws.last, &model.norm, c.rms_norm_eps)?)
            })?;
            probe.stage(Stage::LmHead, || {
                Ok(model
                    .embed
                    .matmul_precise(gpu, &ws.last, 1, &mut ws.logits)?)
            })?;
        }
        Rows::One(_) | Rows::Batch { .. } => {
            probe.stage(Stage::Norm, || {
                Ok(gpu.rms_norm(&mut ws.hidden, &model.norm, c.rms_norm_eps)?)
            })?;
            probe.stage(Stage::LmHead, || {
                Ok(model
                    .embed
                    .matmul_precise(gpu, &ws.hidden, count, &mut ws.logits)?)
            })?;
        }
    }
    let logits = gpu.download(&ws.logits)?;
    tensor::ensure_finite(&logits, "Qwen3.5 CUDA logits")?;
    match rows {
        Rows::One(cache) => cache.kv.advance(count)?,
        Rows::Batch { caches, .. } => {
            for cache in caches.iter_mut() {
                cache.kv.advance(1)?;
            }
        }
    }
    Ok(logits)
}

fn delta_dims(c: &super::Config) -> DeltaDims {
    DeltaDims {
        key_heads: c.linear_num_key_heads,
        value_heads: c.linear_num_value_heads,
        key_dim: c.linear_key_head_dim,
        value_dim: c.linear_value_head_dim,
        kernel: c.linear_conv_kernel_dim,
    }
}

fn delta(
    gpu: &Gpu,
    w: &Delta,
    c: &super::Config,
    ws: &mut Workspace,
    (rows, index): (&mut Rows<'_>, usize),
    count: usize,
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    probe.stage(Stage::Qkv, || {
        w.qkv.matmul_precise(gpu, &ws.normed, count, &mut ws.qkv)?;
        w.z.matmul_precise(gpu, &ws.normed, count, &mut ws.z)?;
        w.a.matmul_precise(gpu, &ws.normed, count, &mut ws.a)?;
        Ok(w.b.matmul_precise(gpu, &ws.normed, count, &mut ws.b)?)
    })?;
    match rows {
        Rows::One(cache) => {
            let state = cache
                .delta
                .get_mut(index)
                .and_then(Option::as_mut)
                .ok_or_else(|| invalid("missing GPU Delta state"))?;
            delta_state(gpu, (w, c), ws, state, count, probe)?;
        }
        Rows::Batch { caches, scratch } => {
            super::gpu_batch::delta_each(
                gpu,
                (w, c),
                (ws, scratch),
                (caches.as_mut_slice(), index),
                probe,
            )?;
        }
    }
    probe.stage(Stage::OutProj, || {
        Ok(w.out
            .matmul_precise(gpu, &ws.core, count, &mut ws.attended)?)
    })?;
    Ok(())
}

/// The convolution and recurrence over the rows in `ws`, advancing one sequence's state.
pub(super) fn delta_state(
    gpu: &Gpu,
    (w, c): (&Delta, &super::Config),
    ws: &mut Workspace,
    state: &mut super::gpu_cache::DeltaState,
    count: usize,
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    let dims = delta_dims(c);
    probe.stage(Stage::Convolution, || {
        Ok(gpu.q35_convolution(
            &ws.qkv,
            &w.conv,
            &mut state.conv,
            &mut ws.mixed,
            dims,
            count,
        )?)
    })?;
    probe.stage(Stage::Recurrence, || {
        Ok(gpu.q35_recurrent(
            DeltaInputs {
                mixed: &ws.mixed,
                z: &ws.z,
                a: &ws.a,
                b: &ws.b,
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
                norm: &w.norm,
            },
            &mut state.recurrent,
            &mut ws.core,
            dims,
            count,
            c.rms_norm_eps,
        )?)
    })
}

fn full(
    gpu: &Gpu,
    w: &Full,
    c: &super::Config,
    ws: &mut Workspace,
    rows: &mut Rows<'_>,
    count: usize,
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    probe.stage(Stage::Qkv, || {
        w.q.matmul_precise(gpu, &ws.normed, count, &mut ws.projected)?;
        w.k.matmul_precise(gpu, &ws.normed, count, &mut ws.k)?;
        w.v.matmul_precise(gpu, &ws.normed, count, &mut ws.v)?;
        Ok(gpu.q35_split_query(
            &ws.projected,
            &mut ws.q,
            &mut ws.q_gate,
            c.num_attention_heads,
            c.head_dim,
        )?)
    })?;
    probe.stage(Stage::Norm, || {
        gpu.rms_norm(&mut ws.q, &w.q_norm, c.rms_norm_eps)?;
        Ok(gpu.rms_norm(&mut ws.k, &w.k_norm, c.rms_norm_eps)?)
    })?;
    match rows {
        Rows::One(cache) => {
            let start = cache.kv.len();
            full_attention(gpu, (w, c), ws, &mut cache.kv, (count, start), probe)?;
        }
        Rows::Batch { caches, scratch } => {
            super::gpu_batch::full_each(gpu, (w, c), (ws, scratch), caches.as_mut_slice(), probe)?;
        }
    }
    probe.stage(Stage::Attention, || {
        Ok(gpu.q35_gate_attention(&mut ws.context, &ws.q_gate)?)
    })?;
    probe.stage(Stage::OutProj, || {
        Ok(w.out
            .matmul_precise(gpu, &ws.context, count, &mut ws.attended)?)
    })?;
    Ok(())
}

/// Position encoding, cache write and attention for the rows in `ws` against one sequence's cache.
pub(super) fn full_attention(
    gpu: &Gpu,
    (w, c): (&Full, &super::Config),
    ws: &mut Workspace,
    kv: &mut DeviceKv,
    (count, start): (usize, usize),
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    probe.stage(Stage::Rope, || {
        gpu.q35_partial_rope(
            &mut ws.q,
            c.num_attention_heads,
            c.head_dim,
            c.rotary(),
            start,
            c.rope_parameters.rope_theta,
        )?;
        Ok(gpu.q35_partial_rope(
            &mut ws.k,
            c.num_key_value_heads,
            c.head_dim,
            c.rotary(),
            start,
            c.rope_parameters.rope_theta,
        )?)
    })?;
    probe.stage(Stage::KvWrite, || Ok(kv.write(gpu, w.index, &ws.k, &ws.v)?))?;
    probe.stage(Stage::Attention, || {
        Ok(gpu.attention(
            &ws.q,
            kv,
            w.index,
            c.num_attention_heads,
            count,
            &mut ws.context,
        )?)
    })
}

fn watch(
    observe: &mut Observer<'_>,
    gpu: &Gpu,
    index: usize,
    hidden: &cuda::Buffer,
    _width: usize,
) -> Result<(), ModelError> {
    if let Some(callback) = observe {
        callback(index, &gpu.download(hidden)?);
    }
    Ok(())
}
