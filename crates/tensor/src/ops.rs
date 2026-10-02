use crate::TensorError;

const LANES: usize = 8;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0_f32; LANES];
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (ca, cb) in a_chunks.zip(b_chunks) {
        for ((sum, x), y) in acc.iter_mut().zip(ca).zip(cb) {
            *sum += x * y;
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// RMSNorm over each `weight.len()`-sized row of `x`, in place.
pub fn rms_norm_rows(x: &mut [f32], weight: &[f32], eps: f32) -> Result<(), TensorError> {
    if weight.is_empty() || x.len().checked_rem(weight.len()) != Some(0) {
        return Err(TensorError::Shape {
            what: "rms_norm input",
            expected: weight.len(),
            actual: x.len(),
        });
    }
    for row in x.chunks_exact_mut(weight.len()) {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = *v * scale * w;
        }
    }
    Ok(())
}

/// `gate = silu(gate) * up`.
pub fn silu_mul(gate: &mut [f32], up: &[f32]) -> Result<(), TensorError> {
    if gate.len() != up.len() {
        return Err(TensorError::Shape {
            what: "silu_mul operands",
            expected: gate.len(),
            actual: up.len(),
        });
    }
    for (g, u) in gate.iter_mut().zip(up) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
    Ok(())
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

/// Rotary embedding on one head vector, pairing element `i` with `i + size/2`.
pub fn rope(head: &mut [f32], position: usize, theta: f32) -> Result<(), TensorError> {
    let size = head.len();
    if size == 0 || size.checked_rem(2) != Some(0) {
        return Err(TensorError::OddHeadSize { what: "rope", size });
    }
    let (first, second) = head.split_at_mut(size.checked_div(2).unwrap_or(0));
    for (i, (a, b)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
        let exponent = i as f32 * 2.0 / size as f32;
        let angle = position as f32 * theta.powf(-exponent);
        let (sin, cos) = angle.sin_cos();
        let (x, y) = (*a, *b);
        *a = x * cos - y * sin;
        *b = y * cos + x * sin;
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
    fn rms_norm_matches_hand_computation() {
        let mut x = [3.0, 4.0];
        rms_norm_rows(&mut x, &[1.0, 1.0], 0.0).unwrap();
        assert_close(&x, &[0.848_528_1, 1.131_370_9]);
    }

    #[test]
    fn rms_norm_normalizes_every_row() {
        let mut x = [3.0, 4.0, 6.0, 8.0];
        rms_norm_rows(&mut x, &[1.0, 1.0], 0.0).unwrap();
        assert_close(&x, &[0.848_528_1, 1.131_370_9, 0.848_528_1, 1.131_370_9]);
    }

    #[test]
    fn rms_norm_rejects_ragged_input() {
        let mut x = [1.0, 2.0, 3.0];
        assert!(rms_norm_rows(&mut x, &[1.0, 1.0], 1e-6).is_err());
    }

    #[test]
    fn silu_mul_matches_hand_computation() {
        let mut gate = [0.0, 1.0];
        silu_mul(&mut gate, &[2.0, 2.0]).unwrap();
        assert_close(&gate, &[0.0, 1.462_117_2]);
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
        rope(&mut head, 0, 10_000.0).unwrap();
        assert_close(&head, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotates_the_pair_by_the_position_angle() {
        let mut head = [1.0, 0.0];
        rope(&mut head, 1, 10_000.0).unwrap();
        assert_close(&head, &[0.540_302_3, 0.841_470_98]);
    }

    #[test]
    fn rope_rejects_odd_head() {
        assert!(rope(&mut [1.0, 2.0, 3.0], 1, 10_000.0).is_err());
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
