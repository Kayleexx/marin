use pulp::{Simd, WithSimd};

use crate::TensorError;
use crate::dispatch::accelerated;

/// RMSNorm over each `weight.len()`-sized row of `x`, in place.
pub fn rms_norm_rows(x: &mut [f32], weight: &[f32], eps: f32) -> Result<(), TensorError> {
    if weight.is_empty() || x.len().checked_rem(weight.len()) != Some(0) {
        return Err(TensorError::Shape {
            what: "rms_norm input",
            expected: weight.len(),
            actual: x.len(),
        });
    }
    if let Err(RmsNorm { x, weight, eps }) = accelerated(RmsNorm { x, weight, eps }) {
        for row in x.chunks_exact_mut(weight.len()) {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for (v, w) in row.iter_mut().zip(weight) {
                *v = *v * scale * w;
            }
        }
    }
    Ok(())
}

struct RmsNorm<'a> {
    x: &'a mut [f32],
    weight: &'a [f32],
    eps: f32,
}

impl WithSimd for RmsNorm<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let RmsNorm { x, weight, eps } = self;
        let (weight_vectors, weight_tail) = S::as_simd_f32s(weight);
        let weight_rest = simd.partial_load_f32s(weight_tail);
        for row in x.chunks_exact_mut(weight.len()) {
            let (vectors, tail) = S::as_mut_simd_f32s(row);
            let mut sum = simd.splat_f32s(0.0);
            for v in vectors.iter() {
                sum = simd.mul_add_e_f32s(*v, *v, sum);
            }
            let rest = simd.partial_load_f32s(tail);
            sum = simd.mul_add_e_f32s(rest, rest, sum);
            let mean_sq = simd.reduce_sum_f32s(sum) / weight.len() as f32;
            let scale = simd.splat_f32s(1.0 / (mean_sq + eps).sqrt());
            for (v, w) in vectors.iter_mut().zip(weight_vectors) {
                *v = simd.mul_f32s(simd.mul_f32s(*v, scale), *w);
            }
            simd.partial_store_f32s(tail, simd.mul_f32s(simd.mul_f32s(rest, scale), weight_rest));
        }
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
    fn rms_norm_matches_the_scalar_formula_on_awkward_widths() {
        for width in [1, 2, 7, 8, 9, 31, 64, 100, 1024] {
            let weight: Vec<f32> = (0..width)
                .map(|i| 0.5 + (i as f32 * 0.13).sin().abs())
                .collect();
            let input: Vec<f32> = (0..3 * width)
                .map(|i| (i as f32 * 0.71).cos() * 4.0)
                .collect();
            let mut actual = input.clone();
            rms_norm_rows(&mut actual, &weight, 1e-6).unwrap();
            for (row, out) in input.chunks_exact(width).zip(actual.chunks_exact(width)) {
                let mean_sq = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / width as f64;
                let scale = 1.0 / (mean_sq + 1e-6).sqrt();
                for ((v, w), o) in row.iter().zip(&weight).zip(out) {
                    let expected = f64::from(*v) * scale * f64::from(*w);
                    assert!(
                        (f64::from(*o) - expected).abs() <= 1e-5 * expected.abs().max(1e-2),
                        "width {width}"
                    );
                }
            }
        }
    }
}
