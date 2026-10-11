use pulp::{Simd, WithSimd};

use crate::TensorError;
use crate::dispatch::accelerated;
use crate::kernel::exp;

/// `gate = silu(gate) * up`, element by element.
pub fn silu_mul(gate: &mut [f32], up: &[f32]) -> Result<(), TensorError> {
    if gate.len() != up.len() {
        return Err(TensorError::Shape {
            what: "silu_mul operands",
            expected: gate.len(),
            actual: up.len(),
        });
    }
    if let Err(Silu { gate, up }) = accelerated(Silu { gate, up }) {
        for (g, u) in gate.iter_mut().zip(up) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
    }
    Ok(())
}

struct Silu<'a> {
    gate: &'a mut [f32],
    up: &'a [f32],
}

impl WithSimd for Silu<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Silu { gate, up } = self;
        let (up_vectors, up_tail) = S::as_simd_f32s(up);
        let (vectors, tail) = S::as_mut_simd_f32s(gate);
        for (g, u) in vectors.iter_mut().zip(up_vectors) {
            *g = silu(simd, *g, *u);
        }
        let rest = silu(
            simd,
            simd.partial_load_f32s(tail),
            simd.partial_load_f32s(up_tail),
        );
        simd.partial_store_f32s(tail, rest);
    }
}

#[inline(always)]
fn silu<S: Simd>(simd: S, gate: S::f32s, up: S::f32s) -> S::f32s {
    let e = exp(simd, simd.sub_f32s(simd.splat_f32s(0.0), gate));
    let sigmoid_input = simd.add_f32s(simd.splat_f32s(1.0), e);
    simd.mul_f32s(simd.div_f32s(gate, sigmoid_input), up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_mul_matches_hand_computation() {
        let mut gate = [0.0, 1.0];
        silu_mul(&mut gate, &[2.0, 2.0]).unwrap();
        assert!((gate[0]).abs() < 1e-6 && (gate[1] - 1.462_117_2).abs() < 1e-5);
    }

    #[test]
    fn silu_mul_matches_the_libm_formula_on_awkward_lengths_and_extremes() {
        for len in [0, 1, 7, 8, 9, 31, 64, 100] {
            let gate: Vec<f32> = (0..len).map(|i| (i as f32 - 50.0) * 0.9).collect();
            let up: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).cos() * 3.0).collect();
            let mut actual = gate.clone();
            silu_mul(&mut actual, &up).unwrap();
            for ((a, g), u) in actual.iter().zip(&gate).zip(&up) {
                let expected = g / (1.0 + (-g).exp()) * u;
                let tolerance = 1e-5 * expected.abs().max(1e-3);
                assert!(
                    (a - expected).abs() <= tolerance,
                    "silu({g})*{u}: {a} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn silu_mul_rejects_mismatched_lengths() {
        assert!(silu_mul(&mut [1.0, 2.0], &[1.0]).is_err());
    }
}
