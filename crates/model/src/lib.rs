#![forbid(unsafe_code)]

mod error;
mod precision;
mod qwen3;

/// What the engine does with the numbers today, before any quantization. Recorded by benchmarks, so change them with
/// the engine.
pub const PRECISION: &str = "bf16 weights, f32 math";
/// Sequences the generate loop runs at once.
pub const BATCH_SIZE: usize = 1;

pub use cuda::{
    CopyTimes, CudaError, Gpu, GpuInfo, KvFormat, PAGE_TOKENS, Traffic, Window, pages_for,
};
pub use error::ModelError;
pub use metrics::{Probe, Stage, StageTimes, TraceEntry};
pub use precision::Precision;
pub use qwen3::{
    Backend, BackendChoice, CPU_SLACK_BYTES, Engine, EngineCache, GpuCache, KvLayout, Observer,
    Qwen3, Qwen3Config, Qwen3Gpu, Selected, cpu_workspace_bytes, kv_startup_bytes, kv_window_bytes,
    weight_bytes,
};
pub use tensor::{DEFAULT_GROUP, QuantBits, QuantScheme};
