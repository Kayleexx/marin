use serde::Deserialize;

use crate::{ModelError, Qwen3Config};

#[derive(Debug, Deserialize)]
pub struct Config {
    model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
    pub eos_token_id: u32,
    attention_bias: bool,
    attention_dropout: f32,
    hidden_act: String,
    tie_word_embeddings: bool,
    rope_scaling: Option<serde_json::Value>,
}

pub fn is_granite(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|value| {
            value
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|kind| kind == "granitemoe")
}

impl Config {
    pub fn parse(json: &str) -> Result<Self, ModelError> {
        let config: Self = serde_json::from_str(json)?;
        if config.model_type != "granitemoe" {
            return Err(ModelError::UnsupportedModelType {
                found: config.model_type,
            });
        }
        if config.attention_bias
            || config.attention_dropout != 0.0
            || config.hidden_act != "silu"
            || !config.tie_word_embeddings
            || config.rope_scaling.as_ref().is_some_and(|v| !v.is_null())
        {
            return Err(ModelError::Unsupported(
                "Granite requires bias-free full attention, SiLU experts and tied embeddings"
                    .into(),
            ));
        }
        let positive = [
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.num_local_experts,
            config.num_experts_per_tok,
            config.vocab_size,
            config.max_position_embeddings,
        ];
        if positive.contains(&0)
            || config.num_experts_per_tok > config.num_local_experts
            || !config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            || !config.head_dim().is_multiple_of(2)
            || usize::try_from(config.eos_token_id).unwrap_or(usize::MAX) >= config.vocab_size
        {
            return Err(ModelError::InvalidConfig(
                "invalid Granite dimensions, routing or EOS".into(),
            ));
        }
        let scales = [
            config.rms_norm_eps,
            config.rope_theta,
            config.attention_multiplier,
            config.embedding_multiplier,
            config.residual_multiplier,
            config.logits_scaling,
        ];
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(ModelError::InvalidConfig(
                "Granite scales must be positive finite numbers".into(),
            ));
        }
        config
            .num_local_experts
            .checked_mul(
                config.intermediate_size.checked_mul(2).ok_or_else(|| {
                    ModelError::InvalidConfig("Granite expert size overflows".into())
                })?,
            )
            .and_then(|n| n.checked_mul(config.hidden_size))
            .and_then(|n| n.checked_mul(config.num_hidden_layers))
            .ok_or_else(|| ModelError::InvalidConfig("Granite weight size overflows".into()))?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size
            .checked_div(self.num_attention_heads)
            .unwrap_or(0)
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads.saturating_mul(self.head_dim())
    }

    pub fn summary(&self) -> Qwen3Config {
        Qwen3Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            layers: self.num_hidden_layers,
            heads: self.num_attention_heads,
            kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            vocab_size: self.vocab_size,
            max_positions: self.max_position_embeddings,
            rms_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            tie_embeddings: true,
            eos_tokens: vec![self.eos_token_id],
            eos_from_generation_config: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_config_and_invalid_routing() {
        let raw = include_str!("../../../../tests/fixtures/granite-config.json");
        let config = Config::parse(raw).unwrap();
        assert_eq!(
            (config.num_hidden_layers, config.num_local_experts),
            (24, 32)
        );
        assert_eq!((config.head_dim(), config.kv_dim()), (64, 512));
        let bad = raw.replace("\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 33");
        assert!(Config::parse(&bad).is_err());
    }
}
