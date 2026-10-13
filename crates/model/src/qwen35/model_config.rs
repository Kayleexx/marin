use super::math::{invalid, size};
use super::{Config, LayerKind};
use crate::{ModelError, Qwen3Config};
use serde::Deserialize;

#[derive(Deserialize)]
struct Root {
    model_type: String,
    text_config: TextConfig,
}
#[derive(Deserialize)]
pub struct TextConfig {
    #[serde(flatten)]
    pub block: Config,
    model_type: String,
    pub layer_types: Vec<LayerKind>,
    num_hidden_layers: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    tie_word_embeddings: bool,
    eos_token_id: u32,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
}
pub fn is_qwen35(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| {
            v.get("model_type")
                .and_then(|s| s.as_str())
                .map(str::to_owned)
        })
        .is_some_and(|s| s == "qwen3_5")
}
impl TextConfig {
    pub fn parse(json: &str, generation: Option<&str>) -> Result<Self, ModelError> {
        let root: Root = serde_json::from_str(json)?;
        let mut c = root.text_config;
        if root.model_type != "qwen3_5" || c.model_type != "qwen3_5_text" {
            return Err(invalid("expected dense Qwen3.5 text config"));
        }
        c.block.validate()?;
        if c.num_hidden_layers == 0
            || c.layer_types.len() != c.num_hidden_layers
            || c.vocab_size == 0
            || c.max_position_embeddings == 0
            || !c.tie_word_embeddings
            || !c.mlp_only_layers.is_empty()
        {
            return Err(invalid("invalid/unsupported Qwen3.5 layout or untied head"));
        }
        if let Some(generation) = generation {
            let v: serde_json::Value = serde_json::from_str(generation)?;
            if let Some(eos) = v.get("eos_token_id").and_then(|v| v.as_u64()) {
                c.eos_token_id = u32::try_from(eos).map_err(|_| invalid("EOS overflows"))?;
            }
        }
        if usize::try_from(c.eos_token_id).unwrap_or(usize::MAX) >= c.vocab_size {
            return Err(invalid("EOS outside vocabulary"));
        }
        Ok(c)
    }
    /// Common CLI/generation dimensions only; hybrid execution never uses Qwen3's layer or memory formulas.
    pub fn summary(&self) -> Qwen3Config {
        let b = &self.block;
        Qwen3Config {
            hidden_size: b.hidden_size,
            intermediate_size: b.intermediate_size,
            layers: self.layer_types.len(),
            heads: b.num_attention_heads,
            kv_heads: b.num_key_value_heads,
            head_dim: b.head_dim,
            vocab_size: self.vocab_size,
            max_positions: self.max_position_embeddings,
            rms_eps: b.rms_norm_eps,
            rope_theta: b.rope_parameters.rope_theta,
            tie_embeddings: true,
            eos_tokens: vec![self.eos_token_id],
            eos_from_generation_config: false,
        }
    }
    pub fn state_bytes(&self, positions: usize) -> Result<usize, ModelError> {
        let b = &self.block;
        self.layer_types.iter().try_fold(0_usize, |total, kind| {
            let n = match kind {
                LayerKind::LinearAttention => size(b.channels()?, b.linear_conv_kernel_dim)?
                    .checked_add(size(
                        size(b.linear_num_value_heads, b.linear_key_head_dim)?,
                        b.linear_value_head_dim,
                    )?)
                    .ok_or_else(|| invalid("recurrent memory overflow"))?,
                LayerKind::FullAttention => size(
                    size(size(b.num_key_value_heads, b.head_dim)?, positions)?,
                    2,
                )?,
            };
            total
                .checked_add(size(n, 4)?)
                .ok_or_else(|| invalid("state memory overflow"))
        })
    }
    pub fn workspace_bytes(&self, tokens: usize, positions: usize) -> Result<usize, ModelError> {
        let b = &self.block;
        let width = b
            .channels()?
            .max(b.intermediate_size)
            .max(size(b.num_attention_heads, b.head_dim)?);
        // Eight widest FP32 rows per token cover live activations; GEMM panels are bounded.
        let packing = size(
            size(b.num_key_value_heads, b.head_dim)?,
            positions
                .checked_next_multiple_of(64)
                .ok_or_else(|| invalid("packing overflow"))?,
        )?;
        let panels = if tokens >= 16 {
            size(size(width, 512)?, tensor::threads().min(width))?
        } else {
            size(width, 128)?
        };
        panels
            .checked_add(size(size(tokens, width)?, 32)?)
            .and_then(|n| n.checked_add(self.vocab_size.checked_mul(4)?))
            .and_then(|n| n.checked_add(tokens.checked_mul(8)?))
            .and_then(|n| n.checked_add(packing.checked_mul(8)?))
            .ok_or_else(|| invalid("workspace memory overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_workspace_covers_activations_panels_and_retained_prefix_packing() {
        let c = TextConfig::parse(
            include_str!("../../../../tests/fixtures/tiny-qwen35/config.json"),
            None,
        )
        .unwrap();
        let width = c
            .block
            .channels()
            .unwrap()
            .max(c.block.intermediate_size)
            .max(c.block.num_attention_heads * c.block.head_dim);
        let minimum = 512 * width * 32 + width * 512 * tensor::threads().min(width);
        assert!(c.workspace_bytes(512, 512).unwrap() >= minimum);
        assert!(c.workspace_bytes(16, 1024).unwrap() > c.workspace_bytes(16, 16).unwrap());
        assert!(c.workspace_bytes(usize::MAX, 1).is_err());
        assert!(c.workspace_bytes(1, usize::MAX).is_err());
    }
}
