use std::ops::Range;

use pulp::{Simd, WithSimd};

use crate::gemm::{MR, NR, TB, Tile, store};

const KC: usize = 512;

pub(crate) struct Q8Gemm<'a, 'b> {
    pub codes: &'a [u8],
    pub scales: &'a [f32],
    pub x: &'a [f32],
    pub out: &'a mut [&'b mut [f32]],
    pub cols: usize,
    pub group: usize,
}

impl WithSimd for Q8Gemm<'_, '_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            codes,
            scales,
            x,
            out,
            cols,
            group,
        } = self;
        let zero = simd.splat_f32s(0.0);
        let vecs = KC.div_ceil(S::F32_LANES);
        let mut weights = vec![zero; vecs.saturating_mul(MR)];
        let panel_len = vecs.saturating_mul(NR);
        let mut inputs = vec![zero; vecs.saturating_mul(TB)];
        let mut scratch = vec![0.0; KC];
        let groups = cols.checked_div(group).unwrap_or(1);
        for (input_block, output_block) in x.chunks(TB.saturating_mul(cols)).zip(out.chunks_mut(TB))
        {
            for k0 in (0..cols).step_by(KC) {
                let k = k0..cols.min(k0.saturating_add(KC));
                let count = k.len().div_ceil(S::F32_LANES);
                for (tokens, panel) in input_block
                    .chunks(NR.saturating_mul(cols))
                    .zip(inputs.chunks_exact_mut(panel_len))
                {
                    panel.fill(zero);
                    for (t, token) in tokens.chunks_exact(cols).enumerate() {
                        pack(simd, panel, token.get(k.clone()).unwrap_or_default(), NR, t);
                    }
                }
                for (r0, rows) in codes.chunks(MR.saturating_mul(cols)).enumerate() {
                    weights.fill(zero);
                    let start = r0.saturating_mul(MR).saturating_mul(groups);
                    let scales = scales.get(start..).unwrap_or_default();
                    for (r, (row, scales)) in rows
                        .chunks_exact(cols)
                        .zip(scales.chunks_exact(groups))
                        .enumerate()
                    {
                        let decoded = scratch.get_mut(..k.len()).unwrap_or_default();
                        decode_panel(row, scales, group, &k, decoded);
                        pack(simd, &mut weights, decoded, MR, r);
                    }
                    let w = weights.get(..count.saturating_mul(MR)).unwrap_or_default();
                    for (panel, outputs) in inputs
                        .chunks_exact(panel_len)
                        .zip(output_block.chunks_mut(NR))
                    {
                        let x = panel.get(..count.saturating_mul(NR)).unwrap_or_default();
                        let tile = multiply(simd, w, x);
                        store(simd, &tile, outputs, r0.saturating_mul(MR), k0 == 0);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn pack<S: Simd>(simd: S, panel: &mut [S::f32s], values: &[f32], stride: usize, column: usize) {
    let (head, tail) = S::as_simd_f32s(values);
    let mut slots = panel.chunks_exact_mut(stride).map(|v| v.get_mut(column));
    for (value, slot) in head.iter().zip(slots.by_ref()) {
        if let Some(slot) = slot {
            *slot = *value;
        }
    }
    if let (Some(Some(slot)), false) = (slots.next(), tail.is_empty()) {
        *slot = simd.partial_load_f32s(tail);
    }
}

#[inline(always)]
fn decode_panel(row: &[u8], scales: &[f32], group: usize, k: &Range<usize>, out: &mut [f32]) {
    let mut offset = k.start;
    let mut source = row.get(k.clone()).unwrap_or_default();
    let mut remaining = out;
    while !source.is_empty() {
        let take = source
            .len()
            .min(group.saturating_sub(offset.checked_rem(group).unwrap_or(0)));
        let (codes, tail) = source.split_at(take);
        let (values, rest) = remaining.split_at_mut(take);
        let scale = scales
            .get(offset.checked_div(group).unwrap_or(0))
            .copied()
            .unwrap_or(0.0);
        for (value, code) in values.iter_mut().zip(codes) {
            *value = f32::from(*code as i8) * scale;
        }
        offset = offset.saturating_add(take);
        remaining = rest;
        source = tail;
    }
}

#[inline(always)]
fn multiply<S: Simd>(simd: S, weights: &[S::f32s], tokens: &[S::f32s]) -> Tile<S> {
    let mut acc: Tile<S> = [[simd.splat_f32s(0.0); NR]; MR];
    for (w, x) in weights.chunks_exact(MR).zip(tokens.chunks_exact(NR)) {
        let [x0, x1] = x else { continue };
        for ([a, b], weight) in acc.iter_mut().zip(w) {
            *a = simd.mul_add_e_f32s(*weight, *x0, *a);
            *b = simd.mul_add_e_f32s(*weight, *x1, *b);
        }
    }
    acc
}
