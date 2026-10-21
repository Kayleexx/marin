//! One forward pass over a Qwen3.5 GPU cache, made undoable. The pass runs in the cache itself: its keys and values are
//! only appended past the committed length, so cutting them back undoes them, and the DeltaNet state, which the kernels
//! change in place, is marked first and put back on failure. Nothing the cache holds is copied except that state.

use super::Qwen35Gpu;
use super::gpu_cache::GpuCache;
use super::gpu_forward::{Rows, pass_chunk};
use super::math::invalid;
use crate::ModelError;
use crate::qwen3::gpu_probe::GpuProbe;
use metrics::{Stage, StageTimes};
use runtime::SequenceCache;

/// Tokens one pass over a prompt takes at a time.
const CHUNK: usize = 512;

pub(super) fn forward(
    model: &Qwen35Gpu,
    tokens: &[u32],
    cache: &mut GpuCache,
    observe: crate::Observer<'_>,
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
    cache
        .tokens
        .try_reserve(tokens.len())
        .map_err(|e| invalid(format!("GPU token ledger: {e}")))?;
    let mut probe = GpuProbe::new(&model.gpu, times);
    let replay = cache.replay;
    // The pass runs in the cache itself. Keys and values are only appended past the committed length, so a failed
    // pass is undone by cutting them back; the DeltaNet state changes in place, so a mark of it is kept meanwhile.
    let mark = probe.stage(Stage::StateCopy, || {
        if replay {
            cache.reset(&model.gpu, &model.config, &model.layers)?;
            Ok(None)
        } else {
            cache.mark(&model.gpu).map(Some)
        }
    })?;
    let mut input = Vec::new();
    input
        .try_reserve_exact(if replay { end } else { tokens.len() })
        .map_err(|e| invalid(format!("GPU input allocation: {e}")))?;
    if replay {
        input.extend_from_slice(&cache.tokens);
    }
    input.extend_from_slice(tokens);
    match run_chunks(model, cache, &input, probe, observe) {
        Ok(logits) => {
            cache.tokens.extend_from_slice(tokens);
            cache.replay = false;
            Ok(logits)
        }
        Err(error) => {
            // Without a mark the state was only being rebuilt: it stays to be rebuilt.
            if mark.is_none_or(|mark| cache.rewind(mark).is_err()) {
                cache.kv.clear();
                cache.replay = true;
                cache.epoch = cache.epoch.wrapping_add(1);
            }
            Err(error)
        }
    }
}

fn run_chunks(
    model: &Qwen35Gpu,
    cache: &mut GpuCache,
    input: &[u32],
    mut probe: GpuProbe<'_>,
    mut observe: crate::Observer<'_>,
) -> Result<Vec<f32>, ModelError> {
    let mut logits = Vec::new();
    let chunk_count = input.len().div_ceil(CHUNK);
    for (i, chunk) in input.chunks(CHUNK).enumerate() {
        logits = pass_chunk(
            model,
            &mut Rows::One(cache),
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
    Ok(logits)
}
