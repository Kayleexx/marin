//! Safe CUDA primitives for transformer inference. This is the only crate allowed `unsafe`;
//! every launch is funneled through `Gpu::launch` after the wrappers in `ops` check sizes.

mod counters;
mod error;
mod gpu;
mod kv;
mod ops;

pub use counters::{Mark, Traffic};
pub use error::CudaError;
pub use gpu::{Buffer, DeviceMatrix, Gpu, GpuInfo};
pub use kv::DeviceKv;
