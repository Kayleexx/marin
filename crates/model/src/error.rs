use cuda::CudaError;
use kv::KvError;
use loader::LoaderError;
use tensor::TensorError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("config.json: {0}")]
    Config(#[from] serde_json::Error),
    #[error("this model is `{found}`; marin supports only `qwen3` so far")]
    UnsupportedModelType { found: String },
    #[error("unsupported model setting: {0}")]
    Unsupported(String),
    #[error("invalid config.json: {0}")]
    InvalidConfig(String),
    #[error("the weights do not match config.json:\n{}", .0.join("\n"))]
    InvalidWeights(Vec<String>),
    #[error("token {token} is outside the model vocabulary of {vocab}")]
    TokenOutOfRange { token: u32, vocab: usize },
    #[error("cannot run the model on zero tokens")]
    NoTokens,
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Kv(#[from] KvError),
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("the GPU has {} MB free but this needs about {} MB", free / 1_000_000, needed / 1_000_000)]
    GpuMemory { needed: u64, free: u64 },
    #[error("cannot read CPU memory limits: {0}")]
    CpuMemoryInfo(std::io::Error),
    #[error("CPU needs another {} MB but only {} MB is available", needed / 1_000_000, available / 1_000_000)]
    CpuMemory { needed: u64, available: u64 },
}

impl ModelError {
    /// Whether the pass failed for want of GPU memory; the cache is as it was before it, so it can be repeated.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::Cuda(error) if error.is_out_of_memory())
    }
}
