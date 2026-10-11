use std::ops::Range;

use pulp::{Simd, WithSimd};

use crate::kernel::{Operands, halves, split_pairs, step};

/// Weight rows and tokens in one register tile: 8 accumulators plus the 4 weight rows and 2 activation pairs
/// stay in the 16 vector registers (4x3 and 6x2 spill and run at half the speed).
pub(crate) const MR: usize = 4;
pub(crate) const NR: usize = 2;
/// Pairs along the shared dimension per pass, and tokens per block: sized so the packed activations stay in L2.
const KC: usize = 256;
pub(crate) const TB: usize = 64;

/// Many tokens against every weight row. Weights and activations are packed into panels laid out in the order
/// the micro-kernel reads them, then multiplied tile by tile.
pub(crate) struct Gemm<'a, 'b>(pub Operands<'a, 'b>);

pub(crate) type Tile<S> = [[<S as Simd>::f32s; NR]; MR];

impl WithSimd for Gemm<'_, '_> {
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
        let lanes = size_of::<S::f32s>()
            .checked_div(size_of::<f32>())
            .unwrap_or(1)
            .max(1);
        let kvecs = KC.div_ceil(lanes);
        let mut weight_panel = vec![simd.splat_u32s(0); kvecs.saturating_mul(MR)];
        let panel_len = kvecs.saturating_mul(2 * NR);
        let mut token_panels = vec![simd.splat_f32s(0.0); panel_len.saturating_mul(TB / NR)];
        let (mut even, mut odd) = (Vec::new(), Vec::new());
        let blocks = x.chunks(TB.saturating_mul(cols)).zip(out.chunks_mut(TB));
        for (block_x, block_out) in blocks {
            split_pairs(block_x, cols, &mut even, &mut odd);
            for k0 in (0..pairs).step_by(KC) {
                let k = k0..pairs.min(k0.saturating_add(KC));
                let vecs = k.len().div_ceil(lanes);
                let group_stride = NR.saturating_mul(pairs);
                let groups = even.chunks(group_stride).zip(odd.chunks(group_stride));
                for (group, panel) in groups.zip(token_panels.chunks_exact_mut(panel_len)) {
                    pack_tokens(simd, panel, group, pairs, &k);
                }
                for (g, rows_group) in weights.chunks(MR.saturating_mul(pairs)).enumerate() {
                    pack_weights(simd, &mut weight_panel, rows_group, pairs, &k);
                    let packed_w = weight_panel.get(..vecs.saturating_mul(MR)).unwrap_or(&[]);
                    let outs = block_out.chunks_mut(NR);
                    for (panel, tokens_out) in token_panels.chunks_exact(panel_len).zip(outs) {
                        let packed_x = panel.get(..vecs.saturating_mul(2 * NR)).unwrap_or(&[]);
                        let tile = multiply::<S>(simd, packed_w, packed_x);
                        let first_row = g.saturating_mul(MR);
                        store(simd, &tile, tokens_out, first_row, k.start == 0);
                    }
                }
            }
        }
    }
}

/// `[k vector][even of token 0, odd of token 0, even of token 1, ...]`, zero past the last token and past `k`.
#[inline(always)]
fn pack_tokens<S: Simd>(
    simd: S,
    panel: &mut [S::f32s],
    (even, odd): (&[f32], &[f32]),
    pairs: usize,
    k: &Range<usize>,
) {
    panel.fill(simd.splat_f32s(0.0));
    let tokens = even.chunks_exact(pairs).zip(odd.chunks_exact(pairs));
    for (t, (token_even, token_odd)) in tokens.enumerate() {
        for (plane, token) in [token_even, token_odd].into_iter().enumerate() {
            let Some(segment) = token.get(k.clone()) else {
                continue;
            };
            let column = t.saturating_mul(2).saturating_add(plane);
            let (head, tail) = S::as_simd_f32s(segment);
            let mut slots = panel.chunks_exact_mut(2 * NR).map(|v| v.get_mut(column));
            for (value, slot) in head.iter().zip(slots.by_ref()) {
                if let Some(slot) = slot {
                    *slot = *value;
                }
            }
            if let (Some(Some(slot)), false) = (slots.next(), tail.is_empty()) {
                *slot = simd.partial_load_f32s(tail);
            }
        }
    }
}

/// `[k vector][row 0, row 1, ...]`, zero past the last row and past `k`.
#[inline(always)]
fn pack_weights<S: Simd>(
    simd: S,
    panel: &mut [S::u32s],
    group: &[u32],
    pairs: usize,
    k: &Range<usize>,
) {
    panel.fill(simd.splat_u32s(0));
    for (r, row) in group.chunks_exact(pairs).enumerate() {
        let Some(segment) = row.get(k.clone()) else {
            continue;
        };
        let (head, tail) = S::as_simd_u32s(segment);
        let mut slots = panel.chunks_exact_mut(MR).map(|v| v.get_mut(r));
        for (value, slot) in head.iter().zip(slots.by_ref()) {
            if let Some(slot) = slot {
                *slot = *value;
            }
        }
        if let (Some(Some(slot)), false) = (slots.next(), tail.is_empty()) {
            *slot = simd.partial_load_u32s(tail);
        }
    }
}

#[inline(always)]
fn multiply<S: Simd>(simd: S, weights: &[S::u32s], tokens: &[S::f32s]) -> Tile<S> {
    let zero = simd.splat_f32s(0.0);
    let mut acc: Tile<S> = [[zero; NR]; MR];
    for (w, x) in weights.chunks_exact(MR).zip(tokens.chunks_exact(2 * NR)) {
        let mut rows = [(zero, zero); MR];
        for (half, v) in rows.iter_mut().zip(w) {
            *half = halves(simd, *v);
        }
        for (t, pair) in x.chunks_exact(2).enumerate() {
            let [even, odd] = pair else { continue };
            for (sums, half) in acc.iter_mut().zip(&rows) {
                if let Some(sum) = sums.get_mut(t) {
                    *sum = step(simd, *half, (*even, *odd), *sum);
                }
            }
        }
    }
    acc
}

/// Adds the tile's horizontal sums to the outputs of up to `NR` tokens; the first k block overwrites them.
#[inline(always)]
pub(crate) fn store<S: Simd>(
    simd: S,
    tile: &Tile<S>,
    tokens_out: &mut [&mut [f32]],
    first_row: usize,
    overwrite: bool,
) {
    for (t, token_out) in tokens_out.iter_mut().enumerate() {
        let slots = token_out.get_mut(first_row..).unwrap_or_default();
        for (slot, sums) in slots.iter_mut().zip(tile) {
            let Some(sum) = sums.get(t) else { continue };
            let sum = simd.reduce_sum_f32s(*sum);
            *slot = if overwrite { sum } else { *slot + sum };
        }
    }
}
