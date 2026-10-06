use cuda::{CudaError, Gpu};

pub fn gpu() -> Result<Option<Gpu>, String> {
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
pub fn values(count: usize, seed: u64) -> Vec<f32> {
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

pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: gpu {a} vs cpu {e}"
        );
    }
}
