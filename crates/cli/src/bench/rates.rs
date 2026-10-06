use metrics::Stage;
use model::{QuantScheme, Qwen3Config};

/// What a stage that reads weights must move and compute: the yardstick its time is judged against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageWork {
    /// Weight bytes the stage reads once per pass, over every layer, as stored (bf16 unless quantized).
    pub weight_bytes: f64,
    /// Floating-point operations per token it runs on; a multiply-add counts as two.
    pub flops_per_token: f64,
    /// Whether every token of a pass goes through it. The LM head only sees the last one.
    pub every_token: bool,
}

/// `None` for stages that read no weights (norms, rope, attention, residual adds).
pub fn stage_work(
    cfg: &Qwen3Config,
    stage: Stage,
    quant: Option<QuantScheme>,
) -> Option<StageWork> {
    let (layers, hidden) = (cfg.layers as f64, cfg.hidden_size as f64);
    let (q_dim, kv_dim) = (cfg.q_dim().ok()? as f64, cfg.kv_dim().ok()? as f64);
    let weights = match stage {
        Stage::Qkv => layers * hidden * (q_dim + 2.0 * kv_dim),
        Stage::OutProj => layers * q_dim * hidden,
        Stage::Mlp => layers * 3.0 * hidden * cfg.intermediate_size as f64,
        Stage::LmHead => cfg.vocab_size as f64 * hidden,
        _ => return None,
    };
    let stored = quant.map_or(2.0, |scheme| match stage {
        Stage::LmHead => scheme.table().bytes_per_weight(),
        _ => scheme.bytes_per_weight(),
    });
    Some(StageWork {
        weight_bytes: weights * stored,
        flops_per_token: weights * 2.0,
        every_token: stage != Stage::LmHead,
    })
}

/// Weight bytes read per second by a stage that took `ms_per_pass` for one pass, in GB/s.
pub fn gb_per_s(work: &StageWork, ms_per_pass: f64) -> Option<f64> {
    (ms_per_pass > 0.0).then(|| work.weight_bytes / (ms_per_pass / 1000.0) / 1e9)
}

/// Math done per second by a stage that took `ms` for a pass over `tokens` tokens, in TFLOP/s.
pub fn tflop_per_s(work: &StageWork, tokens: usize, ms: f64) -> Option<f64> {
    let tokens = if work.every_token { tokens as f64 } else { 1.0 };
    (ms > 0.0).then(|| work.flops_per_token * tokens / (ms / 1000.0) / 1e12)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-1.7B's shape.
    fn qwen3_1_7b() -> Qwen3Config {
        let json = r#"{"model_type":"qwen3","hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":28,
            "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"vocab_size":151936,
            "max_position_embeddings":40960,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
            "tie_word_embeddings":true,"hidden_act":"silu","eos_token_id":151645}"#;
        Qwen3Config::parse(json, None).unwrap()
    }

    #[test]
    fn the_weight_stages_add_up_to_the_weights_a_decode_step_reads() {
        let cfg = qwen3_1_7b();
        let bytes = |s| stage_work(&cfg, s, None).unwrap().weight_bytes;
        assert_eq!(bytes(Stage::Mlp), 28.0 * 3.0 * 2048.0 * 6144.0 * 2.0);
        assert_eq!(bytes(Stage::LmHead), 151_936.0 * 2048.0 * 2.0);
        let total: f64 = [Stage::Qkv, Stage::OutProj, Stage::Mlp, Stage::LmHead]
            .into_iter()
            .map(bytes)
            .sum();
        assert!((total - 3.441e9).abs() < 2e6, "{total}");
        assert!(stage_work(&cfg, Stage::Attention, None).is_none());
        assert!(stage_work(&cfg, Stage::Norm, None).is_none());
    }

    #[test]
    fn rates_are_derived_from_time_and_refuse_zero() {
        let cfg = qwen3_1_7b();
        let head = stage_work(&cfg, Stage::LmHead, None).unwrap();
        let rate = gb_per_s(&head, 1.78).unwrap();
        assert!((rate - 349.7).abs() < 1.0, "{rate}");
        assert_eq!(gb_per_s(&head, 0.0), None);
        let mlp = stage_work(&cfg, Stage::Mlp, None).unwrap();
        let flops = tflop_per_s(&mlp, 128, 244.4).unwrap();
        assert!((flops - 1.107).abs() < 0.01, "{flops}");
        let one_token_head = tflop_per_s(&head, 128, 1.7).unwrap();
        assert!(
            (one_token_head - 0.366).abs() < 0.005,
            "the head counts one token however long the prompt: {one_token_head}"
        );
        assert_eq!(tflop_per_s(&mlp, 128, 0.0), None);
    }

    #[test]
    fn quantized_stages_are_judged_by_the_bytes_they_actually_read() {
        let cfg = qwen3_1_7b();
        let int4 = QuantScheme::new(model::QuantBits::Int4, 128).unwrap();
        let bf16 = stage_work(&cfg, Stage::Mlp, None).unwrap();
        let packed = stage_work(&cfg, Stage::Mlp, Some(int4)).unwrap();
        assert!((packed.weight_bytes / bf16.weight_bytes - (0.5 + 4.0 / 128.0) / 2.0).abs() < 1e-9);
        assert_eq!(
            packed.flops_per_token, bf16.flops_per_token,
            "the math is the same"
        );
        let head = stage_work(&cfg, Stage::LmHead, Some(int4)).unwrap();
        let bf16_head = stage_work(&cfg, Stage::LmHead, None).unwrap();
        assert!(
            (head.weight_bytes / bf16_head.weight_bytes - (1.0 + 4.0 / 128.0) / 2.0).abs() < 1e-9,
            "int8 head"
        );
        assert_eq!(stage_work(&cfg, Stage::Attention, Some(int4)), None);
    }
}
