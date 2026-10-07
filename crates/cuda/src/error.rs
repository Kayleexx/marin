use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA is not available: {0}")]
    Unavailable(String),
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("could not compile the GPU kernels: {0}")]
    Compile(#[from] CompileError),
    #[error("{what}: expected {expected}, got {actual}")]
    Shape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "the GPU could not allocate {needed} more bytes for the KV cache, which holds {held} bytes for {positions} positions: {source}"
    )]
    KvMemory {
        held: usize,
        needed: usize,
        positions: usize,
        source: DriverError,
    },
    #[error("{what}: size arithmetic overflowed")]
    Overflow { what: &'static str },
    #[error("the {what} probe computed a wrong result, so its timing cannot be trusted")]
    SelfCheck { what: &'static str },
    #[error("{what} must be between 1 and {max}, got {actual}")]
    Unsupported {
        what: &'static str,
        max: usize,
        actual: usize,
    },
}
