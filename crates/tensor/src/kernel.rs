//! Shared pieces of the matmul kernels. Weights are bf16 pairs packed in `u32` (element `2i` in the low half,
//! `2i + 1` in the high half); activations are split into their even and odd elements, so a dot product is
//! `low * even + high * odd` and needs no bf16 conversion.

use pulp::Simd;

pub(crate) struct Operands<'a> {
    pub weights: &'a [u32],
    pub x: &'a [f32],
    pub out: &'a mut [f32],
    pub rows: usize,
    pub cols: usize,
    pub pairs: usize,
}

/// The even and odd elements of every `cols`-long token row; an odd row gets a zero for its missing last odd.
pub(crate) fn split_pairs(x: &[f32], cols: usize, even: &mut Vec<f32>, odd: &mut Vec<f32>) {
    even.clear();
    odd.clear();
    for token in x.chunks_exact(cols.max(1)) {
        for pair in token.chunks(2) {
            even.push(pair.first().copied().unwrap_or(0.0));
            odd.push(pair.get(1).copied().unwrap_or(0.0));
        }
    }
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
