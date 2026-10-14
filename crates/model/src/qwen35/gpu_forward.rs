use super::Qwen35Gpu;
use super::gpu::Workspace;
use super::gpu_cache::GpuCache;
use super::gpu_weights::{Delta, Full, Mixer};
use super::math::{invalid, size};
use crate::qwen3::gpu_probe::GpuProbe;
use crate::{ModelError, Observer};
use cuda::{DeltaDims, DeltaInputs, Gpu};
use metrics::{Stage, StageTimes};
use runtime::SequenceCache;

const CHUNK: usize = 512;

pub(super) fn forward(
    model: &Qwen35Gpu,
    tokens: &[u32],
    cache: &mut GpuCache,
    mut observe: crate::Observer<'_>,
    times: Option<&mut StageTimes>,
) -> Result<Vec<f32>, ModelError> {
    if tokens.is_empty() {
        return Err(ModelError::NoTokens);
    }
    if cache.owner != model.owner() {
        return Err(invalid("cache belongs to a different Qwen3.5 GPU model"));
    }
    let end = cache
        .used()
        .checked_add(tokens.len())
        .ok_or_else(|| invalid("context overflow"))?;
    if end > cache.capacity {
        return Err(kv::KvError::ContextFull {
            needed: end,
            capacity: cache.capacity,
        }
        .into());
    }
    for id in tokens {
        if *id as usize >= model.config().vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token: *id,
                vocab: model.config().vocab_size,
            });
        }
    }
    let mut probe = GpuProbe::new(&model.gpu, times);
    let mut staged = probe.stage(Stage::StateCopy, || {
        cache.stage(&model.gpu, &model.config, &model.layers)
    })?;
    let mut input = Vec::new();
    input
        .try_reserve_exact(if cache.replay { end } else { tokens.len() })
        .map_err(|e| invalid(format!("GPU input allocation: {e}")))?;
    if cache.replay {
        input.extend_from_slice(&cache.tokens);
    }
    input.extend_from_slice(tokens);
    let mut logits = Vec::new();
    let chunk_count = input.len().div_ceil(CHUNK);
    for (i, chunk) in input.chunks(CHUNK).enumerate() {
        logits = pass_chunk(
            model,
            &mut staged,
            chunk,
            &mut probe,
            if i.saturating_add(1) == chunk_count {
                observe.take()
            } else {
                None
            },
        )?;
    }
    probe.finish()?;
    staged
        .tokens
        .try_reserve(tokens.len())
        .map_err(|e| invalid(format!("GPU token ledger: {e}")))?;
    if cache.replay {
        staged.tokens.extend_from_slice(&cache.tokens);
    }
    staged.tokens.extend_from_slice(tokens);
    *cache = staged;
    Ok(logits)
}

fn pass_chunk(
    model: &Qwen35Gpu,
    cache: &mut GpuCache,
    tokens: &[u32],
    probe: &mut GpuProbe<'_>,
    mut observe: crate::Observer<'_>,
) -> Result<Vec<f32>, ModelError> {
    let (gpu, c, count) = (&model.gpu, &model.config.block, tokens.len());
    let mut ws = Workspace::new(gpu, &model.config, count)?;
    probe.stage(Stage::Embedding, || {
        Ok(model
            .embed
            .embed(gpu, tokens, &mut ws.ids, &mut ws.hidden)?)
    })?;
    watch(&mut observe, gpu, 0, &ws.hidden, c.hidden_size)?;
    let start = cache.kv.len();
    for (index, layer) in model.layers.iter().enumerate() {
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
            Mixer::Delta(w) => {
                let state = cache
                    .delta
                    .get_mut(index)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| invalid("missing GPU Delta state"))?;
                delta(gpu, w, c, &mut ws, state, count, probe)?;
            }
            Mixer::Full(w) => full(gpu, w, c, &mut ws, &mut cache.kv, (count, start), probe)?,
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
    let logits = gpu.download(&ws.logits)?;
    tensor::ensure_finite(&logits, "Qwen3.5 CUDA logits")?;
    cache.kv.advance(count)?;
    Ok(logits)
}

fn delta(
    gpu: &Gpu,
    w: &Delta,
    c: &super::Config,
    ws: &mut Workspace,
    state: &mut super::gpu_cache::DeltaState,
    count: usize,
    probe: &mut GpuProbe<'_>,
) -> Result<(), ModelError> {
    let dims = DeltaDims {
        key_heads: c.linear_num_key_heads,
        value_heads: c.linear_num_value_heads,
        key_dim: c.linear_key_head_dim,
        value_dim: c.linear_value_head_dim,
        kernel: c.linear_conv_kernel_dim,
    };
    probe.stage(Stage::Qkv, || {
        w.qkv.matmul_precise(gpu, &ws.normed, count, &mut ws.qkv)?;
        w.z.matmul_precise(gpu, &ws.normed, count, &mut ws.z)?;
        w.a.matmul_precise(gpu, &ws.normed, count, &mut ws.a)?;
        Ok(w.b.matmul_precise(gpu, &ws.normed, count, &mut ws.b)?)
    })?;
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
    })?;
    probe.stage(Stage::OutProj, || {
        Ok(w.out
            .matmul_precise(gpu, &ws.core, count, &mut ws.attended)?)
    })?;
    Ok(())
}

fn full(
    gpu: &Gpu,
    w: &Full,
    c: &super::Config,
    ws: &mut Workspace,
    kv: &mut cuda::DeviceKv,
    (count, start): (usize, usize),
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
        gpu.attention(
            &ws.q,
            kv,
            w.index,
            c.num_attention_heads,
            count,
            &mut ws.context,
        )?;
        Ok(gpu.q35_gate_attention(&mut ws.context, &ws.q_gate)?)
    })?;
    probe.stage(Stage::OutProj, || {
        Ok(w.out
            .matmul_precise(gpu, &ws.context, count, &mut ws.attended)?)
    })?;
    Ok(())
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
