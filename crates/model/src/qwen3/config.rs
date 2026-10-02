use serde::Deserialize;

use crate::ModelError;

const SUPPORTED_ACTIVATION: &str = "silu";

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrMany {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: Option<f32>,
    rope_parameters: Option<RopeParameters>,
    hidden_act: Option<String>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    use_sliding_window: bool,
    rope_scaling: Option<serde_json::Value>,
    eos_token_id: Option<OneOrMany>,
}

/// Newer transformers releases nest the rope settings here instead of at the top level.
#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: Option<f32>,
    rope_type: Option<String>,
}

#[derive(Deserialize)]
struct RawGeneration {
    eos_token_id: Option<OneOrMany>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub tie_embeddings: bool,
    pub eos_tokens: Vec<u32>,
    /// True when the stop tokens came from `generation_config.json` rather than `config.json`.
    pub eos_from_generation_config: bool,
}

impl Qwen3Config {
    pub fn parse(config_json: &str, generation_json: Option<&str>) -> Result<Self, ModelError> {
        let raw: RawConfig = serde_json::from_str(config_json)?;
        if raw.model_type != "qwen3" {
            return Err(ModelError::UnsupportedModelType {
                found: raw.model_type,
            });
        }
        reject_unsupported(&raw)?;
        let rope_theta = raw
            .rope_theta
            .or(raw.rope_parameters.as_ref().and_then(|p| p.rope_theta))
            .ok_or_else(|| ModelError::InvalidConfig("rope_theta is missing".into()))?;
        let heads = raw.num_attention_heads;
        let kv_heads = raw.num_key_value_heads.unwrap_or(heads);
        let head_dim = match raw.head_dim {
            Some(dim) => dim,
            None => raw.hidden_size.checked_div(heads).unwrap_or(0),
        };
        let generation_eos = match generation_json {
            Some(text) => serde_json::from_str::<RawGeneration>(text)?.eos_token_id,
            None => None,
        };
        let eos_from_generation_config = generation_eos.is_some();
        let eos_tokens = generation_eos
            .or(raw.eos_token_id)
            .map(OneOrMany::into_vec)
            .unwrap_or_default();
        let config = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            layers: raw.num_hidden_layers,
            heads,
            kv_heads,
            head_dim,
            vocab_size: raw.vocab_size,
            max_positions: raw.max_position_embeddings,
            rms_eps: raw.rms_norm_eps,
            rope_theta,
            tie_embeddings: raw.tie_word_embeddings,
            eos_tokens,
            eos_from_generation_config,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn q_dim(&self) -> Result<usize, ModelError> {
        self.heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| ModelError::InvalidConfig("attention size overflows".into()))
    }

    pub fn kv_dim(&self) -> Result<usize, ModelError> {
        self.kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| ModelError::InvalidConfig("key/value size overflows".into()))
    }

    /// Total weight count implied by the config, or None if it overflows.
    pub fn parameter_count(&self) -> Option<u64> {
        let h = u64::try_from(self.hidden_size).ok()?;
        let i = u64::try_from(self.intermediate_size).ok()?;
        let hd = u64::try_from(self.head_dim).ok()?;
        let q = u64::try_from(self.q_dim().ok()?).ok()?;
        let kv = u64::try_from(self.kv_dim().ok()?).ok()?;
        let v = u64::try_from(self.vocab_size).ok()?;
        let layers = u64::try_from(self.layers).ok()?;
        let attention = h.checked_mul(q.checked_add(kv.checked_mul(2)?)?.checked_add(q)?)?;
        let mlp = h.checked_mul(i)?.checked_mul(3)?;
        let norms = h.checked_mul(2)?.checked_add(hd.checked_mul(2)?)?;
        let per_layer = attention.checked_add(mlp)?.checked_add(norms)?;
        let embeddings = v.checked_mul(h)?;
        let head = if self.tie_embeddings { 0 } else { embeddings };
        per_layer
            .checked_mul(layers)?
            .checked_add(embeddings)?
            .checked_add(head)?
            .checked_add(h)
    }

    fn validate(&self) -> Result<(), ModelError> {
        let positive = [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.layers),
            ("num_attention_heads", self.heads),
            ("num_key_value_heads", self.kv_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_positions),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(ModelError::InvalidConfig(format!("{name} must be above 0")));
            }
        }
        if self.heads.checked_rem(self.kv_heads) != Some(0) {
            return Err(ModelError::InvalidConfig(format!(
                "{} attention heads cannot be shared evenly by {} key/value heads",
                self.heads, self.kv_heads
            )));
        }
        if self.head_dim.checked_rem(2) != Some(0) {
            return Err(ModelError::InvalidConfig(format!(
                "head_dim {} must be even for rotary embeddings",
                self.head_dim
            )));
        }
        if !self.rms_eps.is_finite() || !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(ModelError::InvalidConfig(
                "rms_norm_eps and rope_theta must be finite numbers".into(),
            ));
        }
        self.q_dim()?;
        self.kv_dim()?;
        Ok(())
    }
}

fn reject_unsupported(raw: &RawConfig) -> Result<(), ModelError> {
    let unsupported = |what: &str| Err(ModelError::Unsupported(what.to_string()));
    if raw.attention_bias {
        return unsupported("attention_bias=true");
    }
    if raw.use_sliding_window {
        return unsupported("sliding window attention");
    }
    if raw.rope_scaling.as_ref().is_some_and(|v| !v.is_null()) {
        return unsupported("rope_scaling (long-context rope variants)");
    }
    let rope_type = raw
        .rope_parameters
        .as_ref()
        .and_then(|p| p.rope_type.as_deref());
    if let Some(kind) = rope_type.filter(|kind| *kind != "default") {
        return Err(ModelError::Unsupported(format!("rope type `{kind}`")));
    }
    match raw.hidden_act.as_deref() {
        None | Some(SUPPORTED_ACTIVATION) => Ok(()),
        Some(other) => Err(ModelError::Unsupported(format!(
            "activation `{other}` (only silu)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_1_7B: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "bos_token_id": 151643,
        "eos_token_id": 151643, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
        "intermediate_size": 6144, "max_position_embeddings": 32768, "model_type": "qwen3",
        "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000,
        "sliding_window": null, "tie_word_embeddings": true, "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    fn with(field: &str, value: &str) -> String {
        QWEN3_1_7B.replacen(field, value, 1)
    }

    #[test]
    fn parses_the_real_qwen3_1_7b_config() {
        let config = Qwen3Config::parse(QWEN3_1_7B, None).unwrap();
        assert_eq!((config.layers, config.heads, config.kv_heads), (28, 16, 8));
        assert_eq!(config.q_dim().unwrap(), 2048);
        assert_eq!(config.kv_dim().unwrap(), 1024);
        assert_eq!(config.eos_tokens, [151643]);
        assert!(!config.eos_from_generation_config);
        assert!(config.tie_embeddings);
        assert_eq!(config.parameter_count(), Some(1_720_574_976));
    }

    #[test]
    fn newer_configs_nest_rope_theta_in_rope_parameters() {
        let nested = with(
            "\"rope_theta\": 1000000",
            "\"rope_parameters\": {\"rope_theta\": 500000.0, \"rope_type\": \"default\"}",
        );
        assert_eq!(
            Qwen3Config::parse(&nested, None).unwrap().rope_theta,
            500_000.0
        );
        let yarn = nested.replace("\"default\"", "\"yarn\"");
        assert!(matches!(
            Qwen3Config::parse(&yarn, None),
            Err(ModelError::Unsupported(_))
        ));
        let missing = with("\"rope_theta\": 1000000,", "");
        assert!(matches!(
            Qwen3Config::parse(&missing, None),
            Err(ModelError::InvalidConfig(_))
        ));
    }

    #[test]
    fn generation_config_eos_wins_and_accepts_lists() {
        let generation = r#"{"eos_token_id": [1, 2]}"#;
        let config = Qwen3Config::parse(QWEN3_1_7B, Some(generation)).unwrap();
        assert_eq!(config.eos_tokens, [1, 2]);
        assert!(config.eos_from_generation_config);
    }

    #[test]
    fn other_model_types_are_rejected_by_name() {
        let err = Qwen3Config::parse(&with("\"qwen3\"", "\"llama\""), None).unwrap_err();
        assert!(err.to_string().contains("llama"));
    }

    #[test]
    fn unsupported_variants_are_refused_not_ignored() {
        let cases = [
            with("\"attention_bias\": false", "\"attention_bias\": true"),
            with(
                "\"use_sliding_window\": false",
                "\"use_sliding_window\": true",
            ),
            with(
                "\"rope_scaling\": null",
                "\"rope_scaling\": {\"type\": \"yarn\"}",
            ),
            with("\"silu\"", "\"gelu\""),
        ];
        for text in cases {
            assert!(matches!(
                Qwen3Config::parse(&text, None),
                Err(ModelError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn nonsensical_dimensions_are_rejected() {
        let zero = with("\"hidden_size\": 2048", "\"hidden_size\": 0");
        assert!(Qwen3Config::parse(&zero, None).is_err());
        let uneven = with("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 5");
        assert!(Qwen3Config::parse(&uneven, None).is_err());
        let odd = with("\"head_dim\": 128", "\"head_dim\": 127");
        assert!(Qwen3Config::parse(&odd, None).is_err());
    }

    #[test]
    fn missing_fields_and_bad_json_are_config_errors() {
        assert!(matches!(
            Qwen3Config::parse("{}", None),
            Err(ModelError::Config(_))
        ));
        assert!(Qwen3Config::parse("not json", None).is_err());
    }
}
