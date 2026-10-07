//! Safe CUDA primitives for transformer inference. This is the only crate allowed `unsafe`;
//! every launch is funneled through `Gpu::launch` after the wrappers in `ops` check sizes.

mod attention;
mod counters;
mod error;
mod gpu;
mod kv;
mod kv_format;
mod matmul;
mod ops;
mod pages;
#[cfg(test)]
mod pages_tests;
mod probe;
mod quant;

pub use counters::{Mark, Traffic};
pub use error::CudaError;
pub use gpu::{Buffer, DeviceMatrix, Gpu, GpuInfo, Tokens};
pub use kv::DeviceKv;
pub use kv_format::KvFormat;
pub use pages::{PAGE_TOKENS, PagePool, PageTable, PagesExhausted, SLAB_PAGES, pages_for};
pub use probe::{CopyTimes, Window};
pub use quant::DeviceQuantMatrix;
