//! Growing the cache when the GPU has no room for the next slab must fail cleanly. The slab here is more than half the
//! GPU, so the first one fits and a second never can, whatever else is running.
//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod attention_reference;
mod common;

use std::error::Error;

use attention_reference::reference_attention;
use common::{assert_close, gpu, values};
use cuda::{CudaError, DeviceKv, KvFormat, PAGE_TOKENS, SLAB_PAGES};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const DIMS: (usize, usize, usize) = (16, 8, 128);

#[test]
fn failing_to_allocate_the_next_slab_leaves_the_cache_valid() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let (heads, kv_heads, head_dim) = DIMS;
    let stride = kv_heads * head_dim;
    let info = gpu.info()?;
    let per_layer = 2 * SLAB_PAGES * PAGE_TOKENS * stride * size_of::<f32>();
    let layers = (info.total_bytes / 100 * 55).div_ceil(per_layer);
    let mut kv = DeviceKv::new(
        &gpu,
        layers,
        DIMS,
        4 * SLAB_PAGES * PAGE_TOKENS,
        KvFormat::F32,
    )?;
    let slab = kv.slab_bytes();
    if info.free_bytes < slab + slab / 10 {
        eprintln!("skipping: the GPU has less than one slab ({slab} bytes) free");
        return Ok(());
    }

    let tokens = SLAB_PAGES * PAGE_TOKENS;
    let (keys, vals) = (values(tokens * stride, 10), values(tokens * stride, 50));
    kv.write(&gpu, 0, &gpu.upload(&keys)?, &gpu.upload(&vals)?)?;
    kv.advance(tokens)?;
    assert_eq!(kv.bytes_held(), slab);

    let state = |kv: &DeviceKv| {
        (
            kv.len(),
            kv.pages_in_use(),
            kv.pages_held(),
            kv.bytes_held(),
            kv.page_ids().to_vec(),
        )
    };
    let before = state(&kv);
    let one = gpu.upload(&values(stride, 99))?;
    for attempt in 1..=2 {
        let failure = kv.write(&gpu, 0, &one, &one).unwrap_err();
        assert!(matches!(failure, CudaError::KvMemory { .. }), "{failure}");
        assert_eq!(
            state(&kv),
            before,
            "attempt {attempt}: nothing advanced, nothing half-installed, nothing leaked"
        );
    }
    let query = values(heads * head_dim, 3);
    let q = gpu.upload(&query)?;
    let mut out = gpu.zeros(heads * head_dim)?;
    assert!(
        gpu.attention(&q, &mut kv, 0, heads, 1, &mut out).is_err(),
        "attention refuses the page that was never mapped"
    );

    // What was already cached still reads correctly: redo the last position and attend over all of them.
    kv.truncate(tokens - 1)?;
    let (k_last, v_last) = (values(stride, 7), values(stride, 8));
    kv.write(&gpu, 0, &gpu.upload(&k_last)?, &gpu.upload(&v_last)?)?;
    gpu.attention(&q, &mut kv, 0, heads, 1, &mut out)?;
    let (mut all_keys, mut all_vals) = (keys.clone(), vals.clone());
    all_keys.truncate((tokens - 1) * stride);
    all_vals.truncate((tokens - 1) * stride);
    all_keys.extend(&k_last);
    all_vals.extend(&v_last);
    let expected = reference_attention(
        &query,
        &all_keys,
        &all_vals,
        (heads, kv_heads, head_dim, tokens - 1, 1),
    );
    assert_close(&gpu.download(&out)?, &expected, 1e-4);

    // And it can be emptied and refilled inside the slab it has.
    kv.clear();
    kv.write(&gpu, 0, &gpu.upload(&keys)?, &gpu.upload(&vals)?)?;
    kv.advance(tokens)?;
    assert_eq!(kv.bytes_held(), slab);
    Ok(())
}
