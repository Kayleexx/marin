//! Compares every GPU kernel with the CPU implementation (or a plain f64 reference).
//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

use cuda::{CudaError, DeviceKv, Gpu};
use tensor::Matrix;

fn gpu() -> Result<Option<Gpu>, String> {
    match Gpu::new() {
        Ok(gpu) => Ok(Some(gpu)),
        Err(CudaError::Unavailable(why)) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("skipping GPU test: {why}");
            Ok(None)
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: gpu {a} vs cpu {e}"
        );
    }
}

#[test]
fn buffers_round_trip_and_ranges_are_checked() {
    let Some(gpu) = gpu().unwrap() else { return };
    let data = values(10, 1);
    let buffer = gpu.upload(&data).unwrap();
    assert_eq!(gpu.download(&buffer).unwrap(), data);
    assert_eq!(gpu.download_range(&buffer, 2, 3).unwrap(), data[2..5]);
    assert!(gpu.download_range(&buffer, 8, 3).is_err());
    assert_eq!(
        gpu.download(&gpu.duplicate(&buffer).unwrap()).unwrap(),
        data
    );
    assert!(gpu.upload_matrix(2, 3, &[0; 5]).is_err());
}

#[test]
fn matmul_matches_the_cpu_for_aligned_and_odd_shapes_and_token_tiles() {
    let Some(gpu) = gpu().unwrap() else { return };
    for (rows, cols, tokens) in [
        (5, 64, 1),
        (70, 64, 8),
        (9, 13, 11),
        (33, 24, 17),
        (4, 256, 3),
    ] {
        let weights = bf16_bits(&values(rows * cols, 7));
        let x = values(tokens * cols, 9);
        let mut expected = vec![0.0; tokens * rows];
        Matrix::new(rows, cols, weights.clone())
            .unwrap()
            .matmul(&x, &mut expected)
            .unwrap();
        let device = gpu.upload_matrix(rows, cols, &weights).unwrap();
        let input = gpu.upload(&x).unwrap();
        let mut out = gpu.zeros(tokens * rows).unwrap();
        gpu.matmul(&device, &input, tokens, &mut out).unwrap();
        assert_close(&gpu.download(&out).unwrap(), &expected, 1e-4);
    }
}

#[test]
fn matmul_rejects_wrongly_sized_buffers_instead_of_launching() {
    let Some(gpu) = gpu().unwrap() else { return };
    let w = gpu.upload_matrix(2, 8, &[0; 16]).unwrap();
    let x = gpu.upload(&[0.0; 8]).unwrap();
    let mut small = gpu.zeros(1).unwrap();
    assert!(gpu.matmul(&w, &x, 1, &mut small).is_err());
    let mut out = gpu.zeros(2).unwrap();
    assert!(gpu.matmul(&w, &x, 2, &mut out).is_err());
}

#[test]
fn rms_norm_matches_the_cpu_including_dims_that_are_not_a_warp_multiple() {
    let Some(gpu) = gpu().unwrap() else { return };
    for dim in [64, 17, 128, 2048] {
        let x = values(3 * dim, 3);
        let weight = values(dim, 4);
        let mut expected = x.clone();
        tensor::rms_norm_rows(&mut expected, &weight, 1e-6).unwrap();
        let mut device = gpu.upload(&x).unwrap();
        gpu.rms_norm(&mut device, &gpu.upload(&weight).unwrap(), 1e-6)
            .unwrap();
        assert_close(&gpu.download(&device).unwrap(), &expected, 1e-5);
    }
    let mut ragged = gpu.zeros(10).unwrap();
    assert!(
        gpu.rms_norm(&mut ragged, &gpu.zeros(4).unwrap(), 1e-6)
            .is_err()
    );
}

#[test]
fn rope_matches_the_cpu() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (heads, head_dim, tokens, start) = (3, 16, 4, 5);
    let x = values(tokens * heads * head_dim, 5);
    let mut expected = x.clone();
    for (t, token) in expected.chunks_exact_mut(heads * head_dim).enumerate() {
        for head in token.chunks_exact_mut(head_dim) {
            tensor::rope(head, start + t, 10_000.0).unwrap();
        }
    }
    let mut device = gpu.upload(&x).unwrap();
    gpu.rope(&mut device, heads, head_dim, start, 10_000.0)
        .unwrap();
    assert_close(&gpu.download(&device).unwrap(), &expected, 1e-4);
    assert!(gpu.rope(&mut device, heads, 15, start, 10_000.0).is_err());
}

#[test]
fn silu_mul_add_and_embed_match_the_cpu() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (a, b) = (values(1000, 11), values(1000, 12));

    let mut expected = a.clone();
    tensor::silu_mul(&mut expected, &b).unwrap();
    let mut gate = gpu.upload(&a).unwrap();
    gpu.silu_mul(&mut gate, &gpu.upload(&b).unwrap()).unwrap();
    assert_close(&gpu.download(&gate).unwrap(), &expected, 1e-5);

    let mut expected = a.clone();
    tensor::add_in_place(&mut expected, &b).unwrap();
    let mut sum = gpu.upload(&a).unwrap();
    gpu.add(&mut sum, &gpu.upload(&b).unwrap()).unwrap();
    assert_close(&gpu.download(&sum).unwrap(), &expected, 1e-6);
    assert!(gpu.add(&mut sum, &gpu.zeros(3).unwrap()).is_err());

    let table_values = values(10 * 16, 13);
    let bits = bf16_bits(&table_values);
    let table = gpu.upload_matrix(10, 16, &bits).unwrap();
    let mut out = gpu.zeros(3 * 16).unwrap();
    gpu.embed(&table, &[4, 0, 9], &mut out).unwrap();
    let cpu = Matrix::new(10, 16, bits).unwrap();
    let mut row = vec![0.0; 16];
    let mut expected = Vec::new();
    for id in [4, 0, 9] {
        cpu.row_f32(id, &mut row).unwrap();
        expected.extend_from_slice(&row);
    }
    assert_eq!(gpu.download(&out).unwrap(), expected);
    assert!(
        gpu.embed(&table, &[10], &mut gpu.zeros(16).unwrap())
            .is_err()
    );
}

/// Plain f64 causal grouped-query attention for the last `tokens` positions.
fn reference_attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    (heads, kv_heads, head_dim, start, tokens): (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let kv_stride = kv_heads.saturating_mul(head_dim);
    let group = heads.checked_div(kv_heads).unwrap_or(1);
    let q_dim = heads.saturating_mul(head_dim);
    let mut out = Vec::new();
    for (t, q_token) in q.chunks_exact(q_dim).enumerate().take(tokens) {
        let visible = start.saturating_add(t).saturating_add(1);
        for (h, query) in q_token.chunks_exact(head_dim).enumerate() {
            let kv_head = h.checked_div(group).unwrap_or(0);
            let head_rows = |data: &'_ [f32]| -> Vec<Vec<f32>> {
                data.chunks_exact(kv_stride)
                    .take(visible)
                    .map(|row| {
                        row.chunks_exact(head_dim)
                            .nth(kv_head)
                            .unwrap_or(&[])
                            .to_vec()
                    })
                    .collect()
            };
            let (head_keys, head_values) = (head_rows(keys), head_rows(values));
            let scores: Vec<f64> = head_keys
                .iter()
                .map(|key| {
                    let dot: f64 = query.iter().zip(key).map(|(a, b)| f64::from(a * b)).sum();
                    dot / (head_dim as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..head_dim {
                let acc: f64 = weights
                    .iter()
                    .zip(&head_values)
                    .map(|(w, v)| w * f64::from(v.get(d).copied().unwrap_or(0.0)))
                    .sum();
                out.push((acc / total) as f32);
            }
        }
    }
    out
}

#[test]
fn attention_matches_a_reference_across_cache_chunks_and_context_lengths() {
    let Some(gpu) = gpu().unwrap() else { return };
    for (head_dim, history, tokens) in [
        (16, 0, 1),
        (16, 97, 3),
        (16, 5, 9),
        (64, 40, 4),
        (128, 70, 2),
    ] {
        let (heads, kv_heads) = (4, 2);
        let stride = kv_heads * head_dim;
        let total = history + tokens;
        let keys = values(total * stride, 21);
        let vals = values(total * stride, 22);
        let q = values(tokens * heads * head_dim, 23);

        let mut kv = DeviceKv::new(&gpu, 2, kv_heads, head_dim, total + 3).unwrap();
        if history > 0 {
            let k = gpu.upload(&keys[..history * stride]).unwrap();
            let v = gpu.upload(&vals[..history * stride]).unwrap();
            kv.write(&gpu, 1, &k, &v).unwrap();
            kv.advance(history).unwrap();
        }
        let k = gpu.upload(&keys[history * stride..]).unwrap();
        let v = gpu.upload(&vals[history * stride..]).unwrap();
        kv.write(&gpu, 1, &k, &v).unwrap();

        let q_device = gpu.upload(&q).unwrap();
        let mut out = gpu.zeros(q.len()).unwrap();
        gpu.attention(&q_device, &kv, 1, heads, tokens, &mut out)
            .unwrap();
        let expected = reference_attention(
            &q,
            &keys,
            &vals,
            (heads, kv_heads, head_dim, history, tokens),
        );
        assert_close(&gpu.download(&out).unwrap(), &expected, 1e-4);
    }
}

#[test]
fn kv_cache_and_attention_reject_invalid_requests() {
    let Some(gpu) = gpu().unwrap() else { return };
    let mut kv = DeviceKv::new(&gpu, 1, 2, 16, 4).unwrap();
    let chunk = gpu.zeros(5 * 32).unwrap();
    assert!(kv.write(&gpu, 0, &chunk, &chunk).is_err(), "over capacity");
    let two = gpu.zeros(2 * 32).unwrap();
    assert!(kv.write(&gpu, 3, &two, &two).is_err(), "bad layer");
    assert!(
        kv.write(&gpu, 0, &gpu.zeros(33).unwrap(), &gpu.zeros(33).unwrap())
            .is_err()
    );
    assert!(kv.advance(5).is_err());
    kv.write(&gpu, 0, &two, &two).unwrap();
    kv.advance(2).unwrap();
    assert_eq!(kv.len(), 2);
    kv.clear();
    assert!(kv.is_empty());

    let q = gpu.zeros(5 * 4 * 16).unwrap();
    let mut out = gpu.zeros(5 * 4 * 16).unwrap();
    assert!(
        gpu.attention(&q, &kv, 0, 4, 5, &mut out).is_err(),
        "more tokens than capacity"
    );
    assert!(
        gpu.attention(&q, &kv, 7, 4, 1, &mut out).is_err(),
        "bad layer"
    );
    assert!(
        gpu.attention(&q, &kv, 0, 3, 1, &mut out).is_err(),
        "heads not a multiple of kv heads"
    );
}

#[test]
fn reports_the_device() {
    let Some(gpu) = gpu().unwrap() else { return };
    let info = gpu.info().unwrap();
    assert!(!info.name.is_empty() && info.total_bytes >= info.free_bytes);
    eprintln!(
        "{} (sm_{}{}), {} MB free",
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        info.free_bytes / 1_000_000
    );
}
