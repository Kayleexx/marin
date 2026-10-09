//! Shared pieces of the matmul kernels. Weights are bf16 pairs packed in `u32` (element `2i` in the low half,
//! `2i + 1` in the high half); activations are split into their even and odd elements, so a dot product is
//! `low * even + high * odd` and needs no bf16 conversion.

use pulp::Simd;

pub(crate) struct Operands<'a> {
    pub weights: &'a [u32],
    pub even: &'a [f32],
    pub odd: &'a [f32],
    pub out: &'a mut [f32],
    pub rows: usize,
    pub pairs: usize,
}

/// The low and high bf16 of each pair, as f32.
#[inline(always)]
pub(crate) fn halves<S: Simd>(simd: S, pairs: S::u32s) -> (S::f32s, S::f32s) {
    let high_mask = simd.splat_u32s(0xffff_0000);
    let sixteen = simd.splat_u32s(16);
    (
        simd.transmute_f32s_u32s(simd.wrapping_dyn_shl_u32s(pairs, sixteen)),
        simd.transmute_f32s_u32s(simd.and_u32s(pairs, high_mask)),
    )
}

/// One accumulation step: `acc + low * even + high * odd`.
#[inline(always)]
pub(crate) fn step<S: Simd>(
    simd: S,
    (low, high): (S::f32s, S::f32s),
    (even, odd): (S::f32s, S::f32s),
    acc: S::f32s,
) -> S::f32s {
    simd.mul_add_e_f32s(high, odd, simd.mul_add_e_f32s(low, even, acc))
}
