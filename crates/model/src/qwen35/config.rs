use super::math::{invalid, size};
use crate::ModelError;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub hidden_act: String,
    pub attention_bias: bool,
    #[serde(default = "gated")]
    pub attn_output_gate: bool,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: Rope,
}
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Rope {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

impl Config {
    pub(super) fn validate(&self) -> Result<(), ModelError> {
        let dims = [
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
        ];
        if self.hidden_act != "silu"
            || self.attention_bias
            || !self.attn_output_gate
            || self.rope_parameters.rope_type != "default"
            || dims.contains(&0)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
            || !self.rope_parameters.partial_rotary_factor.is_finite()
            || self.rope_parameters.partial_rotary_factor <= 0.0
            || self.rope_parameters.partial_rotary_factor > 1.0
        {
            return Err(invalid("unsupported or invalid Qwen3.5 reference config"));
        }
        let rotary = self.rotary();
        if rotary == 0 || !rotary.is_multiple_of(2) {
            return Err(invalid("invalid partial rotary dimension"));
        }
        self.channels()?;
        size(self.num_attention_heads, self.head_dim)?;
        size(self.num_key_value_heads, self.head_dim)?;
        Ok(())
    }
    pub(super) fn rotary(&self) -> usize {
        (self.head_dim as f64 * f64::from(self.rope_parameters.partial_rotary_factor)) as usize
    }
    pub(super) fn key_dim(&self) -> Result<usize, ModelError> {
        size(self.linear_num_key_heads, self.linear_key_head_dim)
    }
    pub(super) fn value_dim(&self) -> Result<usize, ModelError> {
        size(self.linear_num_value_heads, self.linear_value_head_dim)
    }
    pub(super) fn channels(&self) -> Result<usize, ModelError> {
        size(self.key_dim()?, 2)?
            .checked_add(self.value_dim()?)
            .ok_or_else(|| invalid("QKV overflow"))
    }
}

fn gated() -> bool {
    true
}
