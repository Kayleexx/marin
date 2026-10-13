use super::math::{invalid, size};
use super::{LayerKind, TextConfig};
use crate::ModelError;
use loader::{Dtype, Weights};

pub(super) fn quantized(name: &str) -> bool {
    name.ends_with("embed_tokens.weight")
        || ((name.ends_with(".weight") && name.contains("_proj"))
            && !name.ends_with("in_proj_a.weight")
            && !name.ends_with("in_proj_b.weight"))
}

pub(crate) fn tensors(config: &TextConfig) -> Result<Vec<(String, Vec<usize>, bool)>, ModelError> {
    let c = &config.block;
    let summary = config.summary();
    let mut all = vec![
        (
            "model.language_model.embed_tokens.weight".into(),
            vec![summary.vocab_size, c.hidden_size],
            true,
        ),
        (
            "model.language_model.norm.weight".into(),
            vec![c.hidden_size],
            false,
        ),
    ];
    for (i, kind) in config.layer_types.iter().enumerate() {
        let mut rows = vec![
            ("input_layernorm.weight", vec![c.hidden_size]),
            ("post_attention_layernorm.weight", vec![c.hidden_size]),
            (
                "mlp.gate_proj.weight",
                vec![c.intermediate_size, c.hidden_size],
            ),
            (
                "mlp.up_proj.weight",
                vec![c.intermediate_size, c.hidden_size],
            ),
            (
                "mlp.down_proj.weight",
                vec![c.hidden_size, c.intermediate_size],
            ),
        ];
        rows.extend(match kind {
            LayerKind::LinearAttention => vec![
                (
                    "linear_attn.in_proj_qkv.weight",
                    vec![c.channels()?, c.hidden_size],
                ),
                (
                    "linear_attn.in_proj_z.weight",
                    vec![c.value_dim()?, c.hidden_size],
                ),
                (
                    "linear_attn.in_proj_a.weight",
                    vec![c.linear_num_value_heads, c.hidden_size],
                ),
                (
                    "linear_attn.in_proj_b.weight",
                    vec![c.linear_num_value_heads, c.hidden_size],
                ),
                (
                    "linear_attn.out_proj.weight",
                    vec![c.hidden_size, c.value_dim()?],
                ),
                (
                    "linear_attn.conv1d.weight",
                    vec![c.channels()?, 1, c.linear_conv_kernel_dim],
                ),
                ("linear_attn.norm.weight", vec![c.linear_value_head_dim]),
                ("linear_attn.A_log", vec![c.linear_num_value_heads]),
                ("linear_attn.dt_bias", vec![c.linear_num_value_heads]),
            ],
            LayerKind::FullAttention => {
                let q = size(c.num_attention_heads, c.head_dim)?;
                let kv = size(c.num_key_value_heads, c.head_dim)?;
                vec![
                    ("self_attn.q_proj.weight", vec![size(q, 2)?, c.hidden_size]),
                    ("self_attn.k_proj.weight", vec![kv, c.hidden_size]),
                    ("self_attn.v_proj.weight", vec![kv, c.hidden_size]),
                    ("self_attn.o_proj.weight", vec![c.hidden_size, q]),
                    ("self_attn.q_norm.weight", vec![c.head_dim]),
                    ("self_attn.k_norm.weight", vec![c.head_dim]),
                ]
            }
        });
        for (name, shape) in rows {
            all.push((
                format!("model.language_model.layers.{i}.{name}"),
                shape,
                quantized(name),
            ));
        }
    }
    Ok(all)
}

pub(crate) fn validate(weights: &Weights, config: &TextConfig) -> Result<(), ModelError> {
    let mut count = 0_usize;
    for (name, shape, quant) in tensors(config)? {
        let info = weights
            .info(&name)
            .ok_or_else(|| invalid(format!("missing {name}")))?;
        if info.shape != shape {
            return Err(invalid(format!("wrong shape: {name}")));
        }
        count = count.saturating_add(1);
        if let Some(group) = weights.packed_group() {
            let dtype = if quant { Dtype::I8 } else { Dtype::F32 };
            if info.dtype != dtype {
                return Err(invalid(format!("wrong packed dtype: {name}")));
            }
            if quant {
                let [rows, cols] = shape.as_slice() else {
                    return Err(invalid("Q8 matrix shape"));
                };
                let scale_name = format!("{name}.scales");
                if cols.checked_rem(group) != Some(0)
                    || !weights.info(&scale_name).is_some_and(|s| {
                        s.dtype == Dtype::F32
                            && s.shape == [*rows, cols.checked_div(group).unwrap_or(0)]
                    })
                {
                    return Err(invalid(format!("invalid scales/group: {name}")));
                }
                count = count.saturating_add(1);
            }
        } else if !matches!(info.dtype, Dtype::Bf16 | Dtype::F32) {
            return Err(invalid(format!("wrong source dtype: {name}")));
        }
    }
    if weights
        .names()
        .filter(|n| weights.packed_group().is_some() || n.starts_with("model.language_model."))
        .count()
        != count
    {
        return Err(invalid("unexpected text tensors"));
    }
    Ok(())
}
