use super::{
    config::Qwen3Config,
    weights::{expected_tensors, validate},
};
use crate::ModelError;
use loader::Weights;
use tensor::QuantScheme;

/// Bytes of the tensors this model actually loads (shards may hold extras such as an unused `lm_head`). With a
/// quantization scheme the projection matrices count as they are stored once quantized.
pub(crate) fn loaded_bytes(
    weights: &Weights,
    cfg: &Qwen3Config,
    quant: Option<QuantScheme>,
) -> Result<u64, ModelError> {
    let expected = expected_tensors(cfg)?;
    let mut read = validate(weights, &expected)?;
    if let Some(group) = weights.packed_group() {
        if quant != Some(QuantScheme::new(tensor::QuantBits::Int8, group)?) {
            return Err(ModelError::Unsupported(
                "packed artifact requires --quantize int8 and its original --quant-group".into(),
            ));
        }
        return Ok(read);
    }
    // Norms are retained as FP32 even when their source is BF16.
    for want in expected.iter().filter(|w| w.shape.len() == 1) {
        read = read.saturating_add(weights.info(&want.name).map_or(0, |i| i.byte_len()));
    }
    let Some(scheme) = quant else {
        return Ok(read);
    };
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let (q, kv) = (cfg.q_dim()?, cfg.kv_dim()?);
    let (mut original, mut quantized) = (0_u64, 0_u64);
    for (rows, cols) in [(q, h), (kv, h), (kv, h), (h, q), (i, h), (i, h), (h, i)] {
        if cols.checked_rem(scheme.group) != Some(0) {
            return Err(ModelError::Unsupported(format!(
                "a projection matrix has {cols} columns, which is not a multiple of the quantization group size {}",
                scheme.group
            )));
        }
        let (Some(bf16), Some(packed)) = (
            rows.checked_mul(cols)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| u64::try_from(n).ok()),
            scheme.matrix_bytes(rows, cols),
        ) else {
            return Err(ModelError::InvalidConfig("weight size overflows".into()));
        };
        original = original.saturating_add(bf16);
        quantized = quantized.saturating_add(packed);
    }
    let layers = u64::try_from(cfg.layers).unwrap_or(u64::MAX);
    let mut saved = original.saturating_sub(quantized).saturating_mul(layers);
    let table = scheme.table();
    let tables = if cfg.tie_embeddings { 1 } else { 2 };
    let (Some(bf16), Some(packed)) = (
        cfg.vocab_size
            .checked_mul(h)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| u64::try_from(n).ok()),
        table.matrix_bytes(cfg.vocab_size, h),
    ) else {
        return Err(ModelError::InvalidConfig("weight size overflows".into()));
    };
    saved = saved.saturating_add(bf16.saturating_sub(packed).saturating_mul(tables));
    Ok(read.saturating_sub(saved))
}
