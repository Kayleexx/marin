mod attention;
mod config;
mod engine;
mod gpu;
mod gpu_probe;
mod layer;
mod mlp;
mod model;
mod weights;

pub use config::Qwen3Config;

/// Receives the last token's hidden state before layer 0 (index 0) and after each layer.
pub type Observer<'a> = Option<&'a mut dyn FnMut(usize, &[f32])>;

pub use engine::{Backend, BackendChoice, Engine, EngineCache, Selected};
pub use gpu::{GpuCache, Qwen3Gpu};
pub use model::Qwen3;

/// Checks the weights against the config and returns how many bytes loading them reads.
pub fn weight_bytes(
    weights: &loader::Weights,
    config: &Qwen3Config,
) -> Result<u64, crate::ModelError> {
    weights::loaded_bytes(weights, config)
}
