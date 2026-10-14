use super::math::{invalid, size};
use super::{LayerKind, TextConfig};
use crate::ModelError;
use cuda::{DeviceKv, KvFormat};

/// Extra GPU bytes beyond retained weights, including staged state and a bounded pass.
pub fn bytes(config: &TextConfig, ctx: usize) -> Result<(u64, u64), ModelError> {
    let c = &config.block;
    let full = config
        .layer_types
        .iter()
        .filter(|kind| **kind == LayerKind::FullAttention)
        .count();
    let fixed = config.state_bytes(0)?;
    let first = DeviceKv::startup_bytes(
        full,
        (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
        ctx,
        KvFormat::F32,
    )
    .ok_or_else(|| invalid("GPU KV budget overflow"))?;
    let window = config
        .state_bytes(ctx)?
        .checked_sub(fixed)
        .ok_or_else(|| invalid("GPU KV budget underflow"))?;
    let q = size(c.num_attention_heads, c.head_dim)?;
    let kv = size(c.num_key_value_heads, c.head_dim)?;
    let widths = [
        c.hidden_size,
        c.hidden_size,
        c.channels()?,
        c.value_dim()?,
        c.linear_num_value_heads,
        c.linear_num_value_heads,
        c.channels()?,
        c.value_dim()?,
        size(q, 2)?,
        q,
        q,
        kv,
        kv,
        q,
        c.hidden_size,
        c.intermediate_size,
        c.intermediate_size,
        c.hidden_size,
    ];
    let per_token = widths
        .iter()
        .try_fold(0_usize, |n, w| n.checked_add(*w))
        .ok_or_else(|| invalid("GPU workspace overflow"))?;
    let workspace = size(per_token, ctx.min(512))?
        .checked_add(c.hidden_size)
        .and_then(|n| n.checked_add(config.summary().vocab_size))
        .and_then(|n| size(n, 4).ok())
        .ok_or_else(|| invalid("GPU workspace overflow"))?;
    let overhead = size(fixed, 2)?
        .checked_add(workspace)
        .ok_or_else(|| invalid("GPU state budget overflow"))?;
    let startup = overhead
        .checked_add(size(first, 2)?)
        .ok_or_else(|| invalid("GPU startup budget overflow"))?;
    let whole = overhead
        .checked_add(size(window.max(first), 2)?)
        .and_then(|n| n.checked_add(first))
        .ok_or_else(|| invalid("GPU window budget overflow"))?;
    Ok((startup as u64, whole.max(startup) as u64))
}
