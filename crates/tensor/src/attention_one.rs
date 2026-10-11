//! Attention for a single new token (decode). The keys and then the values are streamed once, row by row, with
//! every query head scored against its kv head as the row passes; a score is a vector dot product with four
//! independent accumulators.

use pulp::{Simd, WithSimd};

use crate::attention::AttentionShape;
use crate::kernel::exp;

pub(crate) struct AttendOne<'a> {
    pub query: &'a [f32],
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub out: &'a mut [f32],
    pub shape: AttentionShape,
    pub positions: usize,
}

impl WithSimd for AttendOne<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let AttendOne {
            query,
            keys,
            values,
            out,
            shape,
            positions,
        } = self;
        let (d, group, kv_dim) = (shape.head_dim, shape.group(), shape.kv_dim());
        let scale = 1.0 / (d as f32).sqrt();
        let queries: Vec<&[S::f32s]> = query
            .chunks_exact(d)
            .map(|q| S::as_simd_f32s(q).0)
            .collect();
        let mut scores = vec![0.0_f32; shape.heads.saturating_mul(positions)];
        let mut rows: Vec<&mut [f32]> = scores.chunks_mut(positions.max(1)).collect();
        for (p, row) in keys.chunks_exact(kv_dim).take(positions).enumerate() {
            let heads = queries.iter().zip(rows.iter_mut()).enumerate();
            for (h, (q, scores)) in heads {
                let kv_head = h.checked_div(group).unwrap_or(0);
                let key = row.chunks_exact(d).nth(kv_head).unwrap_or_default();
                if let Some(slot) = scores.get_mut(p) {
                    *slot = dot(simd, q, S::as_simd_f32s(key).0) * scale;
                }
            }
        }
        for row in rows.iter_mut() {
            softmax(simd, row);
        }
        out.fill(0.0);
        let (out_vectors, _) = S::as_mut_simd_f32s(out);
        let lanes = size_of::<S::f32s>()
            .checked_div(size_of::<f32>())
            .unwrap_or(1);
        let per_head = d.checked_div(lanes).unwrap_or(1).max(1);
        let mut accumulators: Vec<&mut [S::f32s]> = out_vectors.chunks_mut(per_head).collect();
        for (p, row) in values.chunks_exact(kv_dim).take(positions).enumerate() {
            let heads = rows.iter().zip(accumulators.iter_mut()).enumerate();
            for (h, (weights, acc)) in heads {
                let kv_head = h.checked_div(group).unwrap_or(0);
                let value = row.chunks_exact(d).nth(kv_head).unwrap_or_default();
                let weight = simd.splat_f32s(weights.get(p).copied().unwrap_or(0.0));
                for (a, x) in acc.iter_mut().zip(S::as_simd_f32s(value).0) {
                    *a = simd.mul_add_e_f32s(weight, *x, *a);
                }
            }
        }
    }
}

#[inline(always)]
fn dot<S: Simd>(simd: S, a: &[S::f32s], b: &[S::f32s]) -> f32 {
    let mut acc = [simd.splat_f32s(0.0); 4];
    let (quads_a, rest_a) = a.as_chunks::<4>();
    let (quads_b, rest_b) = b.as_chunks::<4>();
    for (x, y) in quads_a.iter().zip(quads_b) {
        for ((sum, p), q) in acc.iter_mut().zip(x).zip(y) {
            *sum = simd.mul_add_e_f32s(*p, *q, *sum);
        }
    }
    for ((sum, p), q) in acc.iter_mut().zip(rest_a).zip(rest_b) {
        *sum = simd.mul_add_e_f32s(*p, *q, *sum);
    }
    let [s0, s1, s2, s3] = acc;
    simd.reduce_sum_f32s(simd.add_f32s(simd.add_f32s(s0, s1), simd.add_f32s(s2, s3)))
}

/// Softmax in place; the ragged end goes through the same vector `exp` as a padded vector.
#[inline(always)]
fn softmax<S: Simd>(simd: S, scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let offset = simd.splat_f32s(max);
    let (vectors, tail) = S::as_mut_simd_f32s(scores);
    let mut sum = simd.splat_f32s(0.0);
    for v in vectors.iter_mut() {
        *v = exp(simd, simd.sub_f32s(*v, offset));
        sum = simd.add_f32s(sum, *v);
    }
    let rest = exp(simd, simd.sub_f32s(simd.partial_load_f32s(tail), offset));
    simd.partial_store_f32s(tail, rest);
    let total = simd.reduce_sum_f32s(sum) + tail.iter().sum::<f32>();
    let divisor = simd.splat_f32s(total);
    for v in vectors.iter_mut() {
        *v = simd.div_f32s(*v, divisor);
    }
    for v in tail.iter_mut() {
        *v /= total;
    }
}
