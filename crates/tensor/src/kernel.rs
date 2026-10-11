//! Shared pieces of the matmul kernels. Weights are bf16 pairs packed in `u32` (element `2i` in the low half,
//! `2i + 1` in the high half); activations are split into their even and odd elements, so a dot product is
//! `low * even + high * odd` and needs no bf16 conversion.

use pulp::Simd;

/// A product of some weight rows with `x`: `out` holds one slice per token, each as long as `weights` has rows.
pub(crate) struct Operands<'a, 'b> {
    pub weights: &'a [u32],
    pub x: &'a [f32],
    pub out: &'a mut [&'b mut [f32]],
    pub cols: usize,
    pub pairs: usize,
}

/// The even and odd elements of every `cols`-long token row; an odd row gets a zero for its missing last odd.
#[inline(always)]
pub(crate) fn split_pairs(x: &[f32], cols: usize, even: &mut Vec<f32>, odd: &mut Vec<f32>) {
    let pairs = cols.div_ceil(2);
    let tokens = x.len().checked_div(cols.max(1)).unwrap_or(0);
    let total = tokens.saturating_mul(pairs);
    even.clear();
    even.resize(total, 0.0);
    odd.clear();
    odd.resize(total, 0.0);
    let rows = x
        .chunks_exact(cols.max(1))
        .zip(even.chunks_exact_mut(pairs.max(1)))
        .zip(odd.chunks_exact_mut(pairs.max(1)));
    for ((token, even_row), odd_row) in rows {
        let halves = token
            .chunks_exact(2)
            .zip(even_row.iter_mut())
            .zip(odd_row.iter_mut());
        for ((pair, e), o) in halves {
            if let [a, b] = pair {
                *e = *a;
                *o = *b;
            }
        }
        if let (Some(last), Some(slot)) = (
            token.chunks_exact(2).remainder().first(),
            even_row.get_mut(cols / 2),
        ) {
            *slot = *last;
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

/// `e^x` in every lane, within a few f32 ulps of `f32::exp` (the Cephes polynomial). `x` is clamped to
/// `[-87.3, 88]`, so a very negative input gives a tiny positive number, not zero.
#[inline(always)]
pub(crate) fn exp<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
    let splat = |v: f32| simd.splat_f32s(v);
    let x = simd.min_f32s(simd.max_f32s(x, splat(-87.3)), splat(88.0));
    let magic = splat(12_582_912.0);
    let rounded = simd.mul_add_e_f32s(x, splat(std::f32::consts::LOG2_E), magic);
    let n = simd.sub_f32s(rounded, magic);
    let r = simd.negate_mul_add_e_f32s(n, splat(0.693_359_4), x);
    let r = simd.negate_mul_add_e_f32s(n, splat(-2.121_944_4e-4), r);
    let mut y = splat(1.987_569_1e-4);
    for coefficient in [
        1.398_199_9e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        0.5,
    ] {
        y = simd.mul_add_e_f32s(y, r, splat(coefficient));
    }
    let y = simd.mul_add_e_f32s(y, simd.mul_f32s(r, r), simd.add_f32s(r, splat(1.0)));
    let bias = simd.splat_u32s(127_u32.wrapping_sub(0x4B40_0000));
    let exponent = simd.add_u32s(simd.transmute_u32s_f32s(rounded), bias);
    let power = simd.transmute_f32s_u32s(simd.wrapping_dyn_shl_u32s(exponent, simd.splat_u32s(23)));
    simd.mul_f32s(y, power)
}
