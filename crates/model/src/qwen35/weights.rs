use super::math::{Linear, size};
use super::read::Reader;
use super::{Config, LayerKind};
use crate::ModelError;
use std::collections::BTreeMap;

pub(super) struct Common {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}
pub(super) struct Delta {
    pub qkv: Linear,
    pub z: Linear,
    pub a: Linear,
    pub b: Linear,
    pub out: Linear,
    pub conv: Vec<f32>,
    pub norm: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
}
pub(super) struct Full {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}
pub(super) enum Mixer {
    Delta(Delta),
    Full(Full),
}

pub(super) fn load(
    c: &Config,
    kind: LayerKind,
    data: BTreeMap<String, Vec<f32>>,
) -> Result<(Common, Mixer), ModelError> {
    load_reader(c, kind, Reader::Fixture(data))
}
pub(super) fn load_reader(
    c: &Config,
    kind: LayerKind,
    mut data: Reader<'_>,
) -> Result<(Common, Mixer), ModelError> {
    let common = Common {
        input_norm: data.norm("input_layernorm.weight", c.hidden_size)?,
        post_norm: data.norm("post_attention_layernorm.weight", c.hidden_size)?,
        gate: data.matrix("mlp.gate_proj.weight", c.intermediate_size, c.hidden_size)?,
        up: data.matrix("mlp.up_proj.weight", c.intermediate_size, c.hidden_size)?,
        down: data.matrix("mlp.down_proj.weight", c.hidden_size, c.intermediate_size)?,
    };
    let mixer = match kind {
        LayerKind::LinearAttention => Mixer::Delta(Delta {
            qkv: data.matrix(
                "linear_attn.in_proj_qkv.weight",
                c.channels()?,
                c.hidden_size,
            )?,
            z: data.matrix(
                "linear_attn.in_proj_z.weight",
                c.value_dim()?,
                c.hidden_size,
            )?,
            a: data.matrix(
                "linear_attn.in_proj_a.weight",
                c.linear_num_value_heads,
                c.hidden_size,
            )?,
            b: data.matrix(
                "linear_attn.in_proj_b.weight",
                c.linear_num_value_heads,
                c.hidden_size,
            )?,
            out: data.matrix("linear_attn.out_proj.weight", c.hidden_size, c.value_dim()?)?,
            conv: data.vector_shape(
                "linear_attn.conv1d.weight",
                &[c.channels()?, 1, c.linear_conv_kernel_dim],
            )?,
            norm: data.vector("linear_attn.norm.weight", c.linear_value_head_dim)?,
            a_log: data.vector("linear_attn.A_log", c.linear_num_value_heads)?,
            dt_bias: data.vector("linear_attn.dt_bias", c.linear_num_value_heads)?,
        }),
        LayerKind::FullAttention => {
            let q_dim = size(c.num_attention_heads, c.head_dim)?;
            let kv_dim = size(c.num_key_value_heads, c.head_dim)?;
            Mixer::Full(Full {
                q: data.matrix("self_attn.q_proj.weight", size(q_dim, 2)?, c.hidden_size)?,
                k: data.matrix("self_attn.k_proj.weight", kv_dim, c.hidden_size)?,
                v: data.matrix("self_attn.v_proj.weight", kv_dim, c.hidden_size)?,
                out: data.matrix("self_attn.o_proj.weight", c.hidden_size, q_dim)?,
                q_norm: data.norm("self_attn.q_norm.weight", c.head_dim)?,
                k_norm: data.norm("self_attn.k_norm.weight", c.head_dim)?,
            })
        }
    };
    data.finish()?;
    Ok((common, mixer))
}
