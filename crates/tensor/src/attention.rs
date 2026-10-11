use crate::attention_one::AttendOne;
use crate::attention_simd::Attend;
use crate::dispatch::{accelerated, available};
use crate::threads;
use crate::{TensorError, dot, softmax};

/// Heads of a grouped-query attention layer; each key/value head serves `heads / kv_heads` query heads.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionShape {
    pub(crate) fn group(&self) -> usize {
        self.heads.checked_div(self.kv_heads).unwrap_or(1)
    }

    pub(crate) fn q_dim(&self) -> usize {
        self.heads.saturating_mul(self.head_dim)
    }

    pub(crate) fn kv_dim(&self) -> usize {
        self.kv_heads.saturating_mul(self.head_dim)
    }

    /// The prefill kernel puts one (token, head) row in each of 8 lanes, so it needs a group size that divides 8;
    /// both kernels need heads of whole 8-float vectors.
    fn vectorizable(&self, tokens: usize) -> bool {
        self.head_dim.is_multiple_of(8)
            && (tokens == 1 || 8_usize.checked_rem(self.group()) == Some(0))
    }
}

/// Causal grouped-query attention for the new tokens in `queries` (`[token][head][head_dim]`) against
/// `keys`/`values` (`[position][kv_head][head_dim]`), which hold every position so far including these tokens.
/// `start` is the position of the first new token.
pub fn attention(
    shape: &AttentionShape,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    start: usize,
    out: &mut [f32],
) -> Result<(), TensorError> {
    let (q_dim, kv_dim) = (shape.q_dim(), shape.kv_dim());
    if q_dim == 0
        || kv_dim == 0
        || shape.kv_heads == 0
        || !shape.heads.is_multiple_of(shape.kv_heads)
    {
        return Err(TensorError::Shape {
            what: "attention heads",
            expected: shape.heads,
            actual: shape.kv_heads,
        });
    }
    let tokens = queries.len().checked_div(q_dim).unwrap_or(0);
    let visible = start.saturating_add(tokens);
    check(
        "attention queries",
        tokens.saturating_mul(q_dim),
        queries.len(),
    )?;
    check("attention output", queries.len(), out.len())?;
    let held = visible.saturating_mul(kv_dim);
    if keys.len() < held || values.len() < held {
        return Err(TensorError::Shape {
            what: "attention keys and values",
            expected: held,
            actual: keys.len().min(values.len()),
        });
    }
    if !shape.vectorizable(tokens) || !available() {
        return attend_scalar(shape, queries, keys, values, start, out);
    }
    if tokens == 1 {
        let one = AttendOne {
            query: queries,
            keys,
            values,
            out,
            shape: *shape,
            positions: visible,
        };
        let _ = accelerated(one);
        return Ok(());
    }
    let work = tokens
        .saturating_mul(visible)
        .saturating_mul(q_dim)
        .saturating_mul(2);
    let ranges = threads::ranges(shape.kv_heads, threads::workers(work, shape.kv_heads), 1);
    let width = shape.group().saturating_mul(shape.head_dim);
    let lens: Vec<usize> = ranges
        .iter()
        .map(|r| r.len().saturating_mul(width))
        .collect();
    let parts = threads::split_columns(out, q_dim, &lens);
    threads::run_jobs(
        ranges.into_iter().zip(parts).collect(),
        |(kv_range, mut outs)| {
            let _ = accelerated(Attend {
                queries,
                keys,
                values,
                out: &mut outs,
                shape: *shape,
                start,
                kv_range,
            });
        },
    );
    Ok(())
}

fn check(what: &'static str, expected: usize, actual: usize) -> Result<(), TensorError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TensorError::Shape {
            what,
            expected,
            actual,
        })
    }
}

fn head_of(row: &[f32], index: usize, size: usize) -> Result<&[f32], TensorError> {
    row.chunks_exact(size).nth(index).ok_or(TensorError::Shape {
        what: "attention head",
        expected: row.len(),
        actual: index.saturating_mul(size),
    })
}

/// One (token, head) at a time: a dot per cached position, a softmax, then a weighted sum of the values.
fn attend_scalar(
    shape: &AttentionShape,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    start: usize,
    out: &mut [f32],
) -> Result<(), TensorError> {
    let (q_dim, kv_dim, group) = (shape.q_dim(), shape.kv_dim(), shape.group());
    let scale = 1.0 / (shape.head_dim as f32).sqrt();
    let mut scores = Vec::new();
    let tokens = queries.chunks_exact(q_dim).zip(out.chunks_exact_mut(q_dim));
    for (t, (query, output)) in tokens.enumerate() {
        let visible = start.saturating_add(t).saturating_add(1);
        let heads = query
            .chunks_exact(shape.head_dim)
            .zip(output.chunks_exact_mut(shape.head_dim));
        for (h, (q, o)) in heads.enumerate() {
            let kv_head = h.checked_div(group).unwrap_or(0);
            scores.clear();
            for position in keys.chunks_exact(kv_dim).take(visible) {
                scores.push(dot(q, head_of(position, kv_head, shape.head_dim)?) * scale);
            }
            softmax(&mut scores);
            o.fill(0.0);
            for (weight, position) in scores.iter().zip(values.chunks_exact(kv_dim)) {
                let v = head_of(position, kv_head, shape.head_dim)?;
                for (acc, x) in o.iter_mut().zip(v) {
                    *acc += weight * x;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
