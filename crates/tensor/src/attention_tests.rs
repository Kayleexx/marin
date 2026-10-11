//! Attention against an f64 reference, on the vector path and the scalar loop, over awkward shapes.

use super::*;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

fn reference(shape: &AttentionShape, q: &[f32], k: &[f32], v: &[f32], start: usize) -> Vec<f64> {
    let (q_dim, kv_dim, group) = (shape.q_dim(), shape.kv_dim(), shape.group());
    let d = shape.head_dim;
    let mut out = Vec::new();
    for (t, token) in q.chunks_exact(q_dim).enumerate() {
        for (h, query) in token.chunks_exact(d).enumerate() {
            let kv = h.checked_div(group).unwrap_or(0);
            let head = |all: &'_ [f32], p: usize| -> Vec<f64> {
                all.chunks_exact(kv_dim)
                    .nth(p)
                    .map(|row| row.chunks_exact(d).nth(kv).unwrap_or_default())
                    .unwrap_or_default()
                    .iter()
                    .map(|x| f64::from(*x))
                    .collect()
            };
            let visible = start.saturating_add(t).saturating_add(1);
            let scores: Vec<f64> = (0..visible)
                .map(|p| {
                    let dot: f64 = query
                        .iter()
                        .zip(head(k, p))
                        .map(|(a, b)| f64::from(*a) * b)
                        .sum();
                    dot / (d as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = exps.iter().sum();
            let mut row = vec![0.0; d];
            for (p, e) in exps.iter().enumerate() {
                for (acc, x) in row.iter_mut().zip(head(v, p)) {
                    *acc += e / total * x;
                }
            }
            out.extend(row);
        }
    }
    out
}

fn worst(shape: &AttentionShape, tokens: usize, start: usize, seed: u64, scalar: bool) -> f64 {
    let mut rng = Lcg(seed);
    let visible = start.saturating_add(tokens);
    let held = visible.saturating_mul(shape.kv_dim());
    let q: Vec<f32> = (0..tokens.saturating_mul(shape.q_dim()))
        .map(|_| rng.next())
        .collect();
    let k: Vec<f32> = (0..held).map(|_| rng.next() * 3.0).collect();
    let v: Vec<f32> = (0..held).map(|_| rng.next()).collect();
    let mut out = vec![f32::NAN; q.len()];
    if scalar {
        attend_scalar(shape, &q, &k, &v, start, &mut out).unwrap();
    } else {
        attention(shape, &q, &k, &v, start, &mut out).unwrap();
    }
    let expected = reference(shape, &q, &k, &v, start);
    out.iter()
        .zip(&expected)
        .map(|(a, e)| (f64::from(*a) - e).abs())
        .fold(0.0, f64::max)
}

#[test]
fn attention_matches_an_f64_reference_on_both_paths() {
    let shapes = [
        (16, 8, 128),
        (16, 16, 64),
        (8, 2, 8),
        (8, 1, 16),
        (2, 1, 10),
        (3, 3, 6),
    ];
    for (heads, kv_heads, head_dim) in shapes {
        let shape = AttentionShape {
            heads,
            kv_heads,
            head_dim,
        };
        for tokens in [1, 2, 3, 5, 8, 17, 66] {
            for start in [0, 1, 63, 64, 130] {
                for scalar in [false, true] {
                    let error = worst(&shape, tokens, start, 11, scalar);
                    assert!(
                        error < 1e-5,
                        "{heads}/{kv_heads}x{head_dim}, {tokens} tokens after {start}, scalar {scalar}: error {error:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn attention_rejects_bad_shapes() {
    let shape = AttentionShape {
        heads: 4,
        kv_heads: 3,
        head_dim: 8,
    };
    assert!(
        attention(
            &shape,
            &[0.0; 32],
            &[0.0; 24],
            &[0.0; 24],
            0,
            &mut [0.0; 32]
        )
        .is_err()
    );
    let shape = AttentionShape {
        heads: 2,
        kv_heads: 1,
        head_dim: 8,
    };
    assert!(attention(&shape, &[0.0; 16], &[0.0; 4], &[0.0; 4], 0, &mut [0.0; 16]).is_err());
    assert!(attention(&shape, &[0.0; 16], &[0.0; 8], &[0.0; 8], 0, &mut [0.0; 8]).is_err());
}

#[test]
fn attention_is_identical_for_every_thread_count() {
    for (heads, kv_heads, head_dim) in [(16, 8, 128), (8, 2, 8), (6, 3, 16)] {
        let shape = AttentionShape {
            heads,
            kv_heads,
            head_dim,
        };
        for tokens in [2, 5, 17, 66] {
            for start in [0, 63] {
                let mut rng = Lcg(5);
                let held = (start + tokens) * shape.kv_dim();
                let q: Vec<f32> = (0..tokens * shape.q_dim()).map(|_| rng.next()).collect();
                let k: Vec<f32> = (0..held).map(|_| rng.next()).collect();
                let v: Vec<f32> = (0..held).map(|_| rng.next()).collect();
                let run = |threads| {
                    crate::set_threads(threads);
                    let mut out = vec![0.0; q.len()];
                    attention(&shape, &q, &k, &v, start, &mut out).unwrap();
                    out
                };
                let single = run(1);
                for threads in [2, 3, 4, 8, 16] {
                    assert_eq!(
                        run(threads),
                        single,
                        "{heads}/{kv_heads}x{head_dim}, {tokens} tokens, {threads} threads"
                    );
                }
                crate::set_threads(0);
            }
        }
    }
}
