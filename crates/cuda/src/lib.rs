//! Safe CUDA primitives for transformer inference. This is the only crate allowed `unsafe`;
//! every launch is funneled through `Gpu::launch` after the wrappers in `ops` check sizes.

mod attention;
mod counters;
mod error;
mod gpu;
mod kv;
mod matmul;
mod ops;
mod probe;
mod quant;

pub use counters::{Mark, Traffic};
pub use error::CudaError;
pub use gpu::{Buffer, DeviceMatrix, Gpu, GpuInfo, Tokens};
pub use kv::DeviceKv;
pub use probe::{CopyTimes, Window};
pub use quant::DeviceQuantMatrix;
