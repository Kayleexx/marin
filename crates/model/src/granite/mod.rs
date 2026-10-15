mod batch;
mod config;
mod gpu;
mod gpu_weights;
mod layer;
mod model;
mod weights;

pub use config::{Config, is_granite};
pub use gpu::{GpuCache, GraniteGpu, workspace_bytes as gpu_workspace_bytes};
pub use layer::RouteObserver;
pub use model::Granite;
pub use weights::{resident_bytes, weight_bytes};

pub fn gpu_kv_budget(config: &Config, context: usize) -> Result<(u64, u64), crate::ModelError> {
    let summary = config.summary();
    let startup = crate::qwen3::kv_startup_bytes(&summary, context, cuda::KvFormat::F32)
        .ok_or_else(|| {
            crate::ModelError::InvalidConfig("Granite CUDA KV budget overflows".into())
        })?;
    let slab = cuda::KvFormat::F32
        .row_bytes(summary.kv_heads, summary.head_dim)
        .and_then(|row| row.checked_mul(cuda::PAGE_TOKENS))
        .and_then(|bytes| bytes.checked_mul(cuda::SLAB_PAGES))
        .and_then(|bytes| bytes.checked_mul(summary.layers.checked_mul(2)?))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            crate::ModelError::InvalidConfig("Granite CUDA KV budget overflows".into())
        })?;
    let slabs = cuda::pages_for(context).div_ceil(cuda::SLAB_PAGES).max(1);
    let additional = u64::try_from(slabs.saturating_sub(1))
        .ok()
        .and_then(|count| count.checked_mul(slab))
        .ok_or_else(|| {
            crate::ModelError::InvalidConfig("Granite CUDA KV budget overflows".into())
        })?;
    let window = startup.checked_add(additional).ok_or_else(|| {
        crate::ModelError::InvalidConfig("Granite CUDA KV budget overflows".into())
    })?;
    Ok((startup, window))
}

pub(crate) fn export_tensors(config: &Config) -> Vec<(String, Vec<usize>, bool)> {
    weights::expected(config)
        .into_iter()
        .map(|(name, shape)| {
            let codes = weights::quantized(&name, &shape);
            (name, shape, codes)
        })
        .collect()
}

#[cfg(test)]
mod gpu_budget_tests {
    use super::*;

    #[test]
    fn gpu_window_rounds_up_to_complete_kv_slabs() {
        let config = Config::parse(include_str!(
            "../../../../tests/fixtures/granite-config.json"
        ))
        .unwrap();
        let (first, within_slab) = gpu_kv_budget(&config, 512).unwrap();
        let (second, next_slab) = gpu_kv_budget(&config, 513).unwrap();
        assert_eq!(first, within_slab);
        assert!(next_slab > second);
        assert!(next_slab > within_slab);
    }
}
