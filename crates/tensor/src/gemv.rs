use pulp::{Simd, WithSimd};

use crate::kernel::{Operands, halves, split_pairs};

/// One token against every weight row, one row at a time with a low-half and a high-half accumulator. Reading
/// several rows at once was slower: with 1024 columns the rows sit 2 KiB apart and the streams interfere
/// (16 to 19 GB/s against 24 to 28 for one row).
pub(crate) struct Gemv<'a, 'b>(pub Operands<'a, 'b>);

impl WithSimd for Gemv<'_, '_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Operands {
            weights,
            x,
            out,
            cols,
            pairs,
        } = self.0;
        let Some(out) = out.first_mut() else { return };
        let (mut even, mut odd) = (Vec::new(), Vec::new());
        split_pairs(x, cols, &mut even, &mut odd);
        let (even_head, even_tail) = S::as_simd_f32s(&even);
        let (odd_head, odd_tail) = S::as_simd_f32s(&odd);
        let tail_x = (
            simd.partial_load_f32s(even_tail),
            simd.partial_load_f32s(odd_tail),
        );
        let zero = simd.splat_f32s(0.0);
        for (row, slot) in weights.chunks_exact(pairs).zip(out.iter_mut()) {
            let (head, tail) = S::as_simd_u32s(row);
            let mut sum = (zero, zero);
            for ((v, e), o) in head.iter().zip(even_head).zip(odd_head) {
                let (low, high) = halves(simd, *v);
                sum.0 = simd.mul_add_e_f32s(low, *e, sum.0);
                sum.1 = simd.mul_add_e_f32s(high, *o, sum.1);
            }
            let (low, high) = halves(simd, simd.partial_load_u32s(tail));
            sum.0 = simd.mul_add_e_f32s(low, tail_x.0, sum.0);
            sum.1 = simd.mul_add_e_f32s(high, tail_x.1, sum.1);
            *slot = simd.reduce_sum_f32s(sum.0) + simd.reduce_sum_f32s(sum.1);
        }
    }
}
