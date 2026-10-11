use pulp::{Simd, WithSimd};

pub(crate) struct Q8Gemv<'a, 'b> {
    pub words: &'a [u32],
    pub scales: &'a [f32],
    pub x: &'a [f32],
    pub out: &'a mut [&'b mut [f32]],
    pub cols: usize,
    pub group: usize,
}

impl WithSimd for Q8Gemv<'_, '_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            words,
            scales,
            x,
            out,
            cols,
            group,
        } = self;
        let Some(output) = out.first_mut() else {
            return;
        };
        let zero = simd.splat_f32s(0.0);
        let mut packed = Vec::new();
        let mut planes: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; S::F32_LANES]);
        for chunk in x.chunks(S::F32_LANES.saturating_mul(4)) {
            for plane in &mut planes {
                plane.fill(0.0);
            }
            for (i, values) in chunk.chunks_exact(4).enumerate() {
                for (plane, value) in planes.iter_mut().zip(values) {
                    if let Some(slot) = plane.get_mut(i) {
                        *slot = *value;
                    }
                }
            }
            packed.push(planes.each_ref().map(|p| simd.partial_load_f32s(p)));
        }
        let group_vecs = (group / 4).div_ceil(S::F32_LANES);
        let row_groups = cols.checked_div(group).unwrap_or(1);
        for ((row, scales), slot) in words
            .chunks_exact(cols / 4)
            .zip(scales.chunks_exact(row_groups))
            .zip(output.iter_mut())
        {
            let mut total = zero;
            for ((codes, scale), inputs) in row
                .chunks_exact(group / 4)
                .zip(scales)
                .zip(packed.chunks_exact(group_vecs))
            {
                let (head, _) = S::as_simd_u32s(codes);
                let mut a = zero;
                let mut b = zero;
                let mut c = zero;
                let mut d = zero;
                for (w, [x0, x1, x2, x3]) in head.iter().zip(inputs) {
                    a = simd.mul_add_e_f32s(byte(simd, *w, 0), *x0, a);
                    b = simd.mul_add_e_f32s(byte(simd, *w, 8), *x1, b);
                    c = simd.mul_add_e_f32s(byte(simd, *w, 16), *x2, c);
                    d = simd.mul_add_e_f32s(byte(simd, *w, 24), *x3, d);
                }
                let sum = simd.add_f32s(simd.add_f32s(a, b), simd.add_f32s(c, d));
                total = simd.mul_add_e_f32s(sum, simd.splat_f32s(*scale), total);
            }
            *slot = simd.reduce_sum_f32s(total);
        }
    }
}

// Exact signed-byte to f32 conversion using the mantissa of 2^23.
#[inline(always)]
fn byte<S: Simd>(simd: S, word: S::u32s, shift: u32) -> S::f32s {
    let shifted = simd.wrapping_dyn_shr_u32s(word, simd.splat_u32s(shift));
    let code = simd.xor_u32s(
        simd.and_u32s(shifted, simd.splat_u32s(255)),
        simd.splat_u32s(128),
    );
    let bits = simd.or_u32s(code, simd.splat_u32s(0x4b00_0000));
    simd.sub_f32s(simd.transmute_f32s_u32s(bits), simd.splat_f32s(8_388_736.0))
}
