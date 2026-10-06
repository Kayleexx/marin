use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("{0} is not a directory")]
    NotADirectory(PathBuf),
    #[error("{dir} is missing: {}", files.join(", "))]
    MissingFiles { dir: PathBuf, files: Vec<String> },
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path} is not valid JSON: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("{path} is not a valid safetensors file: {reason}")]
    Safetensors { path: PathBuf, reason: String },
    #[error("tensor {name} is stored as {found} but only BF16 is supported")]
    UnsupportedDtype { name: String, found: String },
    #[error("tensor {0} is not in the weights")]
    UnknownTensor(String),
    #[error("tensor {0} appears in more than one shard")]
    DuplicateTensor(String),
    #[error("tokenizer {path}: {reason}")]
    Tokenizer { path: PathBuf, reason: String },
    #[error("tokenizer failed: {0}")]
    TokenizerRuntime(String),
}
