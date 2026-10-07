//! Compares the attention kernels and the key/value cache with a plain f64 reference.
//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod attention_reference;
mod common;

use attention_reference::reference_attention;
use common::{assert_close, gpu, values};
use cuda::{CudaError, DeviceKv, Gpu};

#[test]
fn attention_matches_a_reference_across_cache_chunks_and_context_lengths() {
    let Some(gpu) = gpu().unwrap() else { return };
    for (head_dim, history, tokens) in [
        (16, 0, 1),
        (16, 97, 3),
        (16, 5, 9),
        (64, 40, 4),
        (128, 70, 2),
        // More than 8 tokens takes the general kernel; these cross its 128-position blocks at awkward counts and use
        // head sizes that are not a multiple of 4.
        (128, 0, 40),
        (128, 300, 12),
        (64, 127, 16),
        (16, 129, 9),
        (16, 255, 11),
        (30, 130, 9),
        (18, 260, 10),
        (10, 130, 9),
        (13, 70, 9),
    ] {
        let (heads, kv_heads) = (4, 2);
        let stride = kv_heads * head_dim;
        let total = history + tokens;
        let keys = values(total * stride, 21);
        let vals = values(total * stride, 22);
        let q = values(tokens * heads * head_dim, 23);

        let mut kv = DeviceKv::new(&gpu, 2, (heads, kv_heads, head_dim), total + 3).unwrap();
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
        gpu.attention(&q_device, &mut kv, 1, heads, tokens, &mut out)
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

/// `tokens` new tokens after `history` cached positions, against the f64 reference.
fn few_token_case(
    gpu: &Gpu,
    (heads, kv_heads, head_dim): (usize, usize, usize),
    (history, tokens): (usize, usize),
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let stride = kv_heads.saturating_mul(head_dim);
    let total = history.saturating_add(tokens);
    let keys = values(total.saturating_mul(stride), 41);
    let vals = values(total.saturating_mul(stride), 42);
    let q = values(tokens.saturating_mul(heads).saturating_mul(head_dim), 43);
    let mut kv = DeviceKv::new(gpu, 1, (heads, kv_heads, head_dim), total.saturating_add(7))?;
    let split = history.saturating_mul(stride);
    if history > 0 {
        let k = gpu.upload(keys.get(..split).unwrap_or(&[]))?;
        let v = gpu.upload(vals.get(..split).unwrap_or(&[]))?;
        kv.write(gpu, 0, &k, &v)?;
        kv.advance(history)?;
    }
    let k = gpu.upload(keys.get(split..).unwrap_or(&[]))?;
    let v = gpu.upload(vals.get(split..).unwrap_or(&[]))?;
    kv.write(gpu, 0, &k, &v)?;
    let q_device = gpu.upload(&q)?;
    let mut out = gpu.zeros(q.len())?;
    gpu.attention(&q_device, &mut kv, 0, heads, tokens, &mut out)?;
    let expected = reference_attention(
        &q,
        &keys,
        &vals,
        (heads, kv_heads, head_dim, history, tokens),
    );
    Ok((gpu.download(&out)?, expected))
}

#[test]
fn few_token_attention_matches_a_reference_for_every_size_context_and_head_layout() {
    let Some(gpu) = gpu().unwrap() else { return };
    for layout in [(16, 8, 128), (8, 2, 64), (4, 4, 16)] {
        for history in [0, 1, 63, 64, 65, 300, 1500] {
            for tokens in 2..=8 {
                let (actual, expected) = few_token_case(&gpu, layout, (history, tokens)).unwrap();
                assert_close(&actual, &expected, 1e-4);
            }
        }
    }
}

/// One decode step of attention (the newest token) over `context` cached positions, against the f64 reference.
fn decode_case(
    gpu: &Gpu,
    (heads, kv_heads, head_dim): (usize, usize, usize),
    context: usize,
    q_scale: f32,
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let stride = kv_heads.saturating_mul(head_dim);
    let len = context.saturating_mul(stride);
    let keys = values(len, 31);
    let vals = values(len, 32);
    let q: Vec<f32> = values(heads.saturating_mul(head_dim), 33)
        .iter()
        .map(|v| v * q_scale)
        .collect();
    let history = context.saturating_sub(1);
    let split = history.saturating_mul(stride);
    let mut kv = DeviceKv::new(
        gpu,
        1,
        (heads, kv_heads, head_dim),
        context.saturating_add(5),
    )?;
    let (old_keys, new_keys) = keys.split_at(split);
    let (old_vals, new_vals) = vals.split_at(split);
    if history > 0 {
        let k = gpu.upload(old_keys)?;
        let v = gpu.upload(old_vals)?;
        kv.write(gpu, 0, &k, &v)?;
        kv.advance(history)?;
    }
    let k = gpu.upload(new_keys)?;
    let v = gpu.upload(new_vals)?;
    kv.write(gpu, 0, &k, &v)?;
    let q_device = gpu.upload(&q)?;
    let mut out = gpu.zeros(q.len())?;
    gpu.attention(&q_device, &mut kv, 0, heads, 1, &mut out)?;
    let expected = reference_attention(&q, &keys, &vals, (heads, kv_heads, head_dim, history, 1));
    Ok((gpu.download(&out)?, expected))
}

#[test]
fn decode_attention_matches_the_reference_for_every_context_and_head_layout() {
    let Some(gpu) = gpu().unwrap() else { return };
    // (9, 1) puts more query heads on a key head than the split kernel allows, so it takes the general kernel.
    let layouts = [
        (2, 2, 16),
        (4, 2, 64),
        (16, 8, 128),
        (8, 2, 128),
        (32, 4, 96),
        (8, 1, 256),
        (9, 1, 32),
    ];
    let contexts = [
        1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 700, 2049,
    ];
    for layout in layouts {
        for context in contexts {
            let (actual, expected) = decode_case(&gpu, layout, context, 1.0).unwrap();
            assert_close(&actual, &expected, 1e-4);
        }
    }
}

#[test]
fn decode_attention_stays_exact_when_scores_are_large_and_when_one_position_dominates() {
    let Some(gpu) = gpu().unwrap() else { return };
    for context in [40, 300, 1000] {
        for q_scale in [8.0, 40.0] {
            let (actual, expected) = decode_case(&gpu, (16, 8, 128), context, q_scale).unwrap();
            assert!(
                actual.iter().all(|v| v.is_finite()),
                "scale {q_scale}, context {context}"
            );
            assert_close(&actual, &expected, 1e-3);
        }
    }
}

#[test]
fn decode_attention_reuses_one_scratch_across_growing_contexts() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (heads, kv_heads, head_dim, steps) = (16, 8, 128, 300);
    let stride = kv_heads * head_dim;
    let keys = values(steps * stride, 41);
    let vals = values(steps * stride, 42);
    let mut kv = DeviceKv::new(&gpu, 1, (heads, kv_heads, head_dim), steps).unwrap();
    for step in 0..steps {
        let row = step * stride..(step + 1) * stride;
        kv.write(
            &gpu,
            0,
            &gpu.upload(&keys[row.clone()]).unwrap(),
            &gpu.upload(&vals[row]).unwrap(),
        )
        .unwrap();
        let q = values(heads * head_dim, 100 + step as u64);
        let mut out = gpu.zeros(q.len()).unwrap();
        gpu.attention(&gpu.upload(&q).unwrap(), &mut kv, 0, heads, 1, &mut out)
            .unwrap();
        let expected = reference_attention(&q, &keys, &vals, (heads, kv_heads, head_dim, step, 1));
        assert_close(&gpu.download(&out).unwrap(), &expected, 1e-4);
        kv.advance(1).unwrap();
    }
}

#[test]
fn decode_attention_gives_the_same_bits_every_time() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (first, _) = decode_case(&gpu, (16, 8, 128), 777, 1.0).unwrap();
    let (second, _) = decode_case(&gpu, (16, 8, 128), 777, 1.0).unwrap();
    assert_eq!(first, second);
}

#[test]
fn kv_cache_and_attention_reject_invalid_requests() {
    let Some(gpu) = gpu().unwrap() else { return };
    let mut kv = DeviceKv::new(&gpu, 1, (4, 2, 16), 4).unwrap();
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
        gpu.attention(&q, &mut kv, 0, 4, 5, &mut out).is_err(),
        "more tokens than capacity"
    );
    assert!(
        gpu.attention(&q, &mut kv, 7, 4, 1, &mut out).is_err(),
        "bad layer"
    );
    assert!(
        gpu.attention(&q, &mut kv, 0, 3, 1, &mut out).is_err(),
        "heads not a multiple of kv heads"
    );
    assert!(
        gpu.attention(&q, &mut kv, 0, 2, 1, &mut out).is_err(),
        "heads differ from the cache's"
    );
}
