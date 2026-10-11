//! Attention with one (token, head) row per vector lane. Each lane holds a query, so a score is a plain
//! multiply-add of a broadcast key element with a transposed query vector: no horizontal sums, and every key and
//! value row is read once for all 8 rows. The keys and values of a kv head are packed once into the order the
//! loops stream them, and positions are processed in tiles that stay in L1.

use std::ops::Range;

use pulp::{Simd, WithSimd};

use crate::attention::AttentionShape;
use crate::kernel::exp;

/// Positions per tile of values: 64 rows of 512 bytes fit in L1 together with the accumulators.
const TILE: usize = 64;
/// Keys scored at once, each with its own accumulator.
const KEYS: usize = 4;
/// Output columns accumulated at once.
const COLUMNS: usize = 8;
/// Added to the scores of positions a row must not see.
const MASKED: f32 = -1.0e30;

/// The kv heads `kv_range` of an attention call: `out` has one slice per token, holding only those heads' rows.
pub(crate) struct Attend<'a, 'b> {
    pub queries: &'a [f32],
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub out: &'a mut [&'b mut [f32]],
    pub shape: AttentionShape,
    pub start: usize,
    pub kv_range: Range<usize>,
}

/// One kv head's keys as `[position][head_dim]` and values as `[tile][column chunk][position][COLUMNS]`, every
/// tile padded to `TILE` positions so a block that uses fewer positions reads the same layout.
#[derive(Default)]
struct Packed {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// The `head_dim` values of one kv head in each of the first `positions` rows of `all`.
fn kv_rows<'a>(
    all: &'a [f32],
    shape: &AttentionShape,
    kv_head: usize,
    positions: usize,
) -> Vec<&'a [f32]> {
    let column = kv_head.saturating_mul(shape.head_dim);
    all.chunks_exact(shape.kv_dim().max(1))
        .take(positions)
        .map(|row| {
            row.get(column..column.saturating_add(shape.head_dim))
                .unwrap_or_default()
        })
        .collect()
}

impl Packed {
    fn fill(&mut self, operands: &Attend<'_, '_>, kv_head: usize, positions: usize) {
        let shape = &operands.shape;
        self.keys.clear();
        for key in kv_rows(operands.keys, shape, kv_head, positions) {
            self.keys.extend_from_slice(key);
        }
        self.values.clear();
        let values = kv_rows(operands.values, shape, kv_head, positions);
        for tile in values.chunks(TILE) {
            for c in 0..shape.head_dim.checked_div(COLUMNS).unwrap_or(0) {
                let from = c.saturating_mul(COLUMNS);
                for row in tile {
                    let piece = row.get(from..from.saturating_add(COLUMNS));
                    self.values.extend_from_slice(piece.unwrap_or_default());
                }
                let padding = TILE.saturating_sub(tile.len()).saturating_mul(COLUMNS);
                self.values
                    .resize(self.values.len().saturating_add(padding), 0.0);
            }
        }
    }
}

impl WithSimd for Attend<'_, '_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let shape = self.shape;
        let (d, q_dim, group) = (shape.head_dim, shape.q_dim(), shape.group());
        let lanes = size_of::<S::f32s>()
            .checked_div(size_of::<f32>())
            .unwrap_or(1);
        let per_block = lanes.checked_div(group).unwrap_or(1).max(1);
        let scale = simd.splat_f32s(1.0 / (d as f32).sqrt());
        let total = self
            .start
            .saturating_add(self.queries.len().checked_div(q_dim).unwrap_or(0));
        let mut packed = Packed::default();
        let mut transposed = vec![0.0_f32; d.saturating_mul(lanes)];
        let mut weights = Vec::new();
        let mut sums = vec![0.0_f32; d.saturating_mul(lanes)];
        for kv_head in self.kv_range.clone() {
            packed.fill(&self, kv_head, total);
            let step = per_block.saturating_mul(q_dim);
            let blocks = self
                .queries
                .chunks(step)
                .zip(self.out.chunks_mut(per_block));
            for (b, (block_q, block_out)) in blocks.enumerate() {
                let first = self.start.saturating_add(b.saturating_mul(per_block));
                let tokens = block_q.len().checked_div(q_dim).unwrap_or(0);
                transpose(block_q, &shape, kv_head, lanes, &mut transposed);
                let (queries_t, _) = S::as_simd_f32s(&transposed);
                weights.clear();
                weights.resize(first.saturating_add(tokens), simd.splat_f32s(0.0));
                score(simd, &packed.keys, d, queries_t, scale, &mut weights);
                let masks = tail_masks::<S>(lanes, group, tokens);
                softmax(simd, &mut weights, &masks, first);
                weighted_values(simd, &packed.values, d, &weights, &mut sums);
                let local = kv_head.saturating_sub(self.kv_range.start);
                scatter(&sums, lanes, &shape, local, block_out);
            }
        }
    }
}

/// `[head_dim][lane]`: lane `token * group + g` holds the query of head `kv_head * group + g` of that token.
fn transpose(
    block_q: &[f32],
    shape: &AttentionShape,
    kv_head: usize,
    lanes: usize,
    out: &mut [f32],
) {
    let (d, group) = (shape.head_dim, shape.group());
    out.fill(0.0);
    for (t, token) in block_q.chunks_exact(shape.q_dim()).enumerate() {
        for g in 0..group {
            let lane = t.saturating_mul(group).saturating_add(g);
            let head = kv_head.saturating_mul(group).saturating_add(g);
            let row = token.chunks_exact(d).nth(head).unwrap_or_default();
            for (column, q) in out.chunks_exact_mut(lanes).zip(row) {
                if let Some(slot) = column.get_mut(lane) {
                    *slot = *q;
                }
            }
        }
    }
}

/// `scores[p]` holds, in each lane, the scaled dot product of that lane's query with the key at position `p`.
#[inline(always)]
fn score<S: Simd>(
    simd: S,
    keys: &[f32],
    d: usize,
    queries: &[S::f32s],
    scale: S::f32s,
    scores: &mut [S::f32s],
) {
    let zero = simd.splat_f32s(0.0);
    let slabs = keys.chunks(KEYS.saturating_mul(d).max(1));
    for (group, slab) in scores.chunks_mut(KEYS).zip(slabs) {
        let mut rows = slab.chunks_exact(d.max(1));
        let k0 = rows.next().unwrap_or_default();
        let (k1, k2, k3) = (
            rows.next().unwrap_or(k0),
            rows.next().unwrap_or(k0),
            rows.next().unwrap_or(k0),
        );
        let mut acc = [zero; KEYS];
        let columns = queries.iter().zip(k0).zip(k1).zip(k2).zip(k3);
        for ((((q, a), b), c), e) in columns {
            for (sum, k) in acc.iter_mut().zip([a, b, c, e]) {
                *sum = simd.mul_add_e_f32s(*q, simd.splat_f32s(*k), *sum);
            }
        }
        for (slot, sum) in group.iter_mut().zip(acc) {
            *slot = simd.mul_f32s(sum, scale);
        }
    }
}

/// For each of the last `tokens - 1` positions, a lane is 1.0 if its token may see that position and 0.0 if not.
/// Lanes past the real rows stay 1.0.
fn tail_masks<S: Simd>(lanes: usize, group: usize, tokens: usize) -> Vec<S::f32s> {
    let count = tokens.saturating_sub(1);
    let mut flat = vec![1.0_f32; count.saturating_mul(lanes)];
    for (j, mask) in flat.chunks_exact_mut(lanes).enumerate() {
        for (lane, slot) in mask
            .iter_mut()
            .enumerate()
            .take(tokens.saturating_mul(group))
        {
            if lane.checked_div(group).unwrap_or(0) <= j {
                *slot = 0.0;
            }
        }
    }
    let (vectors, _) = S::as_simd_f32s(&flat);
    vectors.to_vec()
}

/// Softmax of every lane over its visible positions (`first + token + 1`), in place, as weights.
#[inline(always)]
fn softmax<S: Simd>(simd: S, scores: &mut [S::f32s], masks: &[S::f32s], first: usize) {
    let bias = |mask: &S::f32s| {
        simd.mul_add_e_f32s(*mask, simd.splat_f32s(-MASKED), simd.splat_f32s(MASKED))
    };
    let (head, tail) = scores.split_at_mut(first.saturating_add(1).min(scores.len()));
    let mut max = simd.splat_f32s(f32::NEG_INFINITY);
    for s in head.iter() {
        max = simd.max_f32s(max, *s);
    }
    for (s, mask) in tail.iter().zip(masks) {
        max = simd.max_f32s(max, simd.add_f32s(*s, bias(mask)));
    }
    let mut sum = simd.splat_f32s(0.0);
    for s in head.iter_mut() {
        *s = exp(simd, simd.sub_f32s(*s, max));
        sum = simd.add_f32s(sum, *s);
    }
    for (s, mask) in tail.iter_mut().zip(masks) {
        *s = simd.mul_f32s(exp(simd, simd.sub_f32s(*s, max)), *mask);
        sum = simd.add_f32s(sum, *s);
    }
    for s in head.iter_mut().chain(tail.iter_mut()) {
        *s = simd.div_f32s(*s, sum);
    }
}

/// `sums[column]` holds, in each lane, the weighted sum of that column of the values over all positions.
#[inline(always)]
fn weighted_values<S: Simd>(
    simd: S,
    values: &[f32],
    d: usize,
    weights: &[S::f32s],
    sums: &mut [f32],
) {
    sums.fill(0.0);
    let lanes = size_of::<S::f32s>()
        .checked_div(size_of::<f32>())
        .unwrap_or(1);
    let tiles = weights
        .chunks(TILE)
        .zip(values.chunks(TILE.saturating_mul(d).max(1)));
    for (tile_weights, tile_values) in tiles {
        let chunk_len = TILE.saturating_mul(COLUMNS);
        let sums_chunks = sums.chunks_mut(COLUMNS.saturating_mul(lanes));
        for (columns, packed) in sums_chunks.zip(tile_values.chunks(chunk_len)) {
            let (vectors, _) = S::as_mut_simd_f32s(columns);
            let mut acc = [simd.splat_f32s(0.0); COLUMNS];
            for (a, v) in acc.iter_mut().zip(vectors.iter()) {
                *a = *v;
            }
            for (w, x) in tile_weights.iter().zip(packed.chunks_exact(COLUMNS)) {
                for (a, x) in acc.iter_mut().zip(x) {
                    *a = simd.mul_add_e_f32s(*w, simd.splat_f32s(*x), *a);
                }
            }
            for (v, a) in vectors.iter_mut().zip(acc) {
                *v = a;
            }
        }
    }
}

/// Writes each lane's column sums to its (token, head) row of the output; `local` counts kv heads from the first
/// one of the output slices.
fn scatter(
    sums: &[f32],
    lanes: usize,
    shape: &AttentionShape,
    local: usize,
    block_out: &mut [&mut [f32]],
) {
    let group = shape.group();
    for (t, token_out) in block_out.iter_mut().enumerate() {
        for g in 0..group {
            let lane = t.saturating_mul(group).saturating_add(g);
            let head = local.saturating_mul(group).saturating_add(g);
            let Some(row) = token_out.chunks_exact_mut(shape.head_dim).nth(head) else {
                continue;
            };
            for (dst, column) in row.iter_mut().zip(sums.chunks_exact(lanes)) {
                if let Some(v) = column.get(lane) {
                    *dst = *v;
                }
            }
        }
    }
}
