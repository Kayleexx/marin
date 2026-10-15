use crate::TensorError;

const LANES: usize = 8;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0_f64; LANES];
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let tail: f64 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    for (ca, cb) in a_chunks.zip(b_chunks) {
        for ((sum, x), y) in acc.iter_mut().zip(ca).zip(cb) {
            *sum += f64::from(*x) * f64::from(*y);
        }
    }
    (acc.iter().sum::<f64>() + tail) as f32
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn add_in_place(a: &mut [f32], b: &[f32]) -> Result<(), TensorError> {
    if a.len() != b.len() {
        return Err(TensorError::Shape {
            what: "add operands",
            expected: a.len(),
            actual: b.len(),
        });
    }
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
    Ok(())
}

/// Rotary embedding of every head of every `token_dim`-long token row of `x`; token `t` sits at position
/// `start + t`. Element `i` of a head pairs with `i + size / 2`. The angles are computed once per token and
/// shared by all its heads.
pub fn rope_tokens(
    x: &mut [f32],
    token_dim: usize,
    head_size: usize,
    start: usize,
    theta: f32,
) -> Result<(), TensorError> {
    if head_size == 0 || head_size.checked_rem(2) != Some(0) {
        return Err(TensorError::OddHeadSize {
            what: "rope",
            size: head_size,
        });
    }
    if token_dim.checked_rem(head_size) != Some(0)
        || x.len().checked_rem(token_dim.max(1)) != Some(0)
    {
        return Err(TensorError::Shape {
            what: "rope input",
            expected: token_dim,
            actual: x.len(),
        });
    }
    let half = head_size.checked_div(2).unwrap_or(0);
    let frequencies: Vec<f32> = (0..half)
        .map(|i| theta.powf(-(i as f32 * 2.0 / head_size as f32)))
        .collect();
    let mut angles = vec![(0.0_f32, 0.0_f32); half];
    for (t, token) in x.chunks_exact_mut(token_dim.max(1)).enumerate() {
        let position = start.saturating_add(t) as f32;
        for (angle, frequency) in angles.iter_mut().zip(&frequencies) {
            let (sin, cos) = (position * frequency).sin_cos();
            *angle = (cos, sin);
        }
        for head in token.chunks_exact_mut(head_size) {
            let (first, second) = head.split_at_mut(half);
            for ((a, b), (cos, sin)) in first.iter_mut().zip(second.iter_mut()).zip(&angles) {
                let (x, y) = (*a, *b);
                *a = x * cos - y * sin;
                *b = y * cos + x * sin;
            }
        }
    }
    Ok(())
}

pub fn ensure_finite(values: &[f32], what: &'static str) -> Result<(), TensorError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(TensorError::NotFinite { what })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn dot_handles_lane_tail() {
        let a: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        assert_eq!(dot(&a, &a), 385.0);
    }

    #[test]
    fn softmax_matches_hand_computation() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert_close(&x, &[0.090_030_57, 0.244_728_47, 0.665_240_96]);
    }

    #[test]
    fn softmax_survives_large_values() {
        let mut x = [1000.0, 1000.0];
        softmax(&mut x);
        assert_close(&x, &[0.5, 0.5]);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut head = [1.0, 2.0, 3.0, 4.0];
        rope_tokens(&mut head, 4, 4, 0, 10_000.0).unwrap();
        assert_close(&head, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotates_the_pair_by_the_position_angle() {
        let mut head = [1.0, 0.0];
        rope_tokens(&mut head, 2, 2, 1, 10_000.0).unwrap();
        assert_close(&head, &[0.540_302_3, 0.841_470_98]);
    }

    #[test]
    fn rope_tokens_equals_the_per_head_formula_bit_for_bit() {
        let (heads, size, theta) = (3, 8, 10_000.0_f32);
        let input: Vec<f32> = (0..4 * heads * size)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let mut actual = input.clone();
        rope_tokens(&mut actual, heads * size, size, 5, theta).unwrap();
        let mut expected = input;
        for (t, token) in expected.chunks_exact_mut(heads * size).enumerate() {
            for head in token.chunks_exact_mut(size) {
                let (first, second) = head.split_at_mut(size / 2);
                for (i, (a, b)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
                    let exponent = i as f32 * 2.0 / size as f32;
                    let (sin, cos) = ((5 + t) as f32 * theta.powf(-exponent)).sin_cos();
                    (*a, *b) = (*a * cos - *b * sin, *b * cos + *a * sin);
                }
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn rope_rejects_odd_head() {
        assert!(rope_tokens(&mut [1.0, 2.0, 3.0], 3, 3, 1, 10_000.0).is_err());
    }

    #[test]
    fn add_checks_lengths() {
        let mut a = [1.0, 2.0];
        add_in_place(&mut a, &[1.0, 1.0]).unwrap();
        assert_eq!(a, [2.0, 3.0]);
        assert!(add_in_place(&mut a, &[1.0]).is_err());
    }

    #[test]
    fn finite_check_flags_nan_and_infinity() {
        assert!(ensure_finite(&[1.0, 2.0], "x").is_ok());
        assert!(ensure_finite(&[1.0, f32::NAN], "x").is_err());
        assert!(ensure_finite(&[f32::INFINITY], "x").is_err());
    }
}
