mod attention;
mod config;
mod cpu_memory;
mod engine;
mod export;
mod fit;
mod gpu;
pub(crate) mod gpu_probe;
mod kv_state;
mod layer;
mod mlp;
mod model;
pub(crate) mod weight;
mod weight_bytes;
mod weights;
pub use export::export_int8;
mod workspace;

pub use config::Qwen3Config;
pub use cpu_memory::{CPU_SLACK_BYTES, cpu_workspace_bytes};

/// Receives the last token's hidden state before layer 0 (index 0) and after each layer.
pub type Observer<'a> = Option<&'a mut dyn FnMut(usize, &[f32])>;

pub use engine::{Backend, BackendChoice, Engine, EngineCache, Selected};
pub use gpu::{GpuCache, Qwen3Gpu, kv_startup_bytes, kv_window_bytes};
pub use kv_state::KvLayout;
pub use model::Qwen3;

/// Checks the weights against the config and returns how many bytes they take in memory once loaded, quantized
/// by `quant` if given.
pub fn weight_bytes(
    weights: &loader::Weights,
    config: &Qwen3Config,
    quant: Option<tensor::QuantScheme>,
) -> Result<u64, crate::ModelError> {
    weight_bytes::loaded_bytes(weights, config, quant)
}
