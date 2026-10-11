use tensor::{AttentionShape, attention, rope_tokens};

use super::config::Qwen3Config;
use crate::ModelError;

/// Rotates every head of every token in `x` (`[token][head][head_dim]`) by its position.
pub(crate) fn rope_heads(
    x: &mut [f32],
    token_dim: usize,
    cfg: &Qwen3Config,
    start: usize,
) -> Result<(), ModelError> {
    Ok(rope_tokens(
        x,
        token_dim,
        cfg.head_dim,
        start,
        cfg.rope_theta,
    )?)
}

/// Causal grouped-query attention for `queries.len() / q_dim` new tokens.
/// `keys`/`values` hold every position so far, including these tokens.
pub(crate) fn attend(
    cfg: &Qwen3Config,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    start: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let shape = AttentionShape {
        heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        head_dim: cfg.head_dim,
    };
    Ok(attention(&shape, queries, keys, values, start, out)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(heads: usize, kv_heads: usize, head_dim: usize) -> Qwen3Config {
        Qwen3Config {
            hidden_size: heads.saturating_mul(head_dim),
            intermediate_size: 4,
            layers: 1,
            heads,
            kv_heads,
            head_dim,
            vocab_size: 4,
            max_positions: 16,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_embeddings: true,
            eos_tokens: vec![],
            eos_from_generation_config: false,
        }
    }

    #[test]
    fn single_token_attends_only_to_itself() {
        let cfg = config(1, 1, 2);
        let mut out = [0.0; 2];
        attend(&cfg, &[1.0, 0.0], &[5.0, 5.0], &[3.0, 4.0], 0, &mut out).unwrap();
        assert_eq!(out, [3.0, 4.0]);
    }

    #[test]
    fn equal_scores_average_the_values_and_the_future_is_masked() {
        let cfg = config(1, 1, 2);
        let queries = [0.0, 0.0, 0.0, 0.0];
        let keys = [1.0, 1.0, 2.0, 2.0];
        let values = [2.0, 0.0, 4.0, 8.0];
        let mut out = [0.0; 4];
        attend(&cfg, &queries, &keys, &values, 0, &mut out).unwrap();
        // token 0 sees only position 0; token 1 averages positions 0 and 1
        assert_eq!(out, [2.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn grouped_heads_share_one_key_value_head() {
        let cfg = config(2, 1, 2);
        let mut out = [0.0; 4];
        attend(
            &cfg,
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 1.0],
            &[7.0, 9.0],
            0,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [7.0, 9.0, 7.0, 9.0]);
    }

    #[test]
    fn decode_uses_cached_positions_before_the_new_token() {
        let cfg = config(1, 1, 2);
        let keys = [0.0, 0.0, 0.0, 0.0];
        let values = [2.0, 2.0, 6.0, 6.0];
        let mut out = [0.0; 2];
        attend(&cfg, &[0.0, 0.0], &keys, &values, 1, &mut out).unwrap();
        assert_eq!(out, [4.0, 4.0]);
    }

    #[test]
    fn rope_heads_leaves_position_zero_unchanged_and_rotates_later_ones() {
        let cfg = config(1, 1, 2);
        let mut x = [1.0, 0.0, 1.0, 0.0];
        rope_heads(&mut x, 2, &cfg, 0).unwrap();
        assert_eq!(&x[..2], [1.0, 0.0]);
        assert!((x[2] - 0.540_302_3).abs() < 1e-6 && (x[3] - 0.841_470_98).abs() < 1e-6);
    }
}
