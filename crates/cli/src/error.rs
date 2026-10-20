use loader::LoaderError;
use model::{CudaError, ModelError};
use runtime::GenerateError;
use sampler::SamplerError;
use thiserror::Error;

use crate::units;

#[derive(Debug, Error)]
pub enum CliError {
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Generate(#[from] GenerateError<ModelError>),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("nothing to complete: the prompt is empty")]
    EmptyPrompt,
    #[error("the prompt is {tokens} tokens long but the context window is {ctx}")]
    PromptTooLong { tokens: usize, ctx: usize },
    #[error("--ctx {ctx} is above what this model supports ({max})")]
    ContextTooLarge { ctx: usize, max: usize },
    #[error("{0}")]
    BadOption(String),
    #[error("this needs about {} of memory but only {} is available", units::bytes(*.needed), units::bytes(*.available))]
    NotEnoughMemory { needed: u64, available: u64 },
    #[error("{gpu}; CPU also cannot run: {cpu}")]
    NoFit { gpu: String, cpu: String },
    #[error("cannot use reference file {path}: {reason}")]
    Reference { path: String, reason: String },
    #[error("no model found")]
    NoModel,
    #[error("no chat model found{}", base_note(.base))]
    NoChatModel { base: Vec<String> },
    #[error("{0} is a base model: it continues text and cannot hold a conversation")]
    NotAChatModel(String),
    #[error("several models found: {0}")]
    ChooseModel(String),
    #[error("{0} is not a model folder (it needs config.json, tokenizer.json and weights)")]
    NotAModel(String),
    #[error("no model is loaded")]
    ModelNotLoaded,
    #[error("no prompt given")]
    NoPrompt,
    #[error("cancelled")]
    Cancelled,
    #[error("cannot read {path}: {reason}")]
    Input { path: String, reason: String },
    #[error("nothing to show: no file was given and nothing is piped in")]
    NoInput,
    #[error("cannot use result file {path}: {reason}")]
    Saved { path: String, reason: String },
    #[error("these results were measured under different setups ({0})")]
    NotComparable(String),
    #[error("cannot write {path}: {reason}")]
    Output { path: String, reason: String },
    #[error(transparent)]
    Code(#[from] crate::code::CodeError),
    #[error(transparent)]
    Conversation(#[from] crate::saved::SavedError),
    #[error("port {0} is already in use")]
    PortInUse(u16),
    #[error("the model does not match the reference output")]
    VerifyFailed,
}

fn base_note(base: &[String]) -> String {
    match base {
        [] => String::new(),
        [one] => {
            format!("; {one} is a base model (it continues text and cannot hold a conversation)")
        }
        many => format!(
            "; {} are base models (they continue text and cannot hold a conversation)",
            many.join(", ")
        ),
    }
}

impl CliError {
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::Loader(LoaderError::MissingFiles { .. }) => Some(
                "download the whole model folder (config.json, tokenizer.json, model.safetensors) from Hugging Face",
            ),
            Self::ModelNotLoaded => Some("open / actions and choose Switch model"),
            Self::Loader(LoaderError::NotADirectory(_)) => {
                Some("pass the folder that contains config.json, not a file inside it")
            }
            Self::Loader(LoaderError::Safetensors { .. }) => {
                Some("the file may be a partial download; fetch it again")
            }
            Self::Model(e) if e.to_string().contains("packed artifact requires") => Some(
                "this folder holds packed 8-bit weights: add --quantize int8 (and its --quant-group), or pick it from the model list in `marin`",
            ),
            Self::Model(ModelError::UnsupportedModelType { .. }) => {
                Some("supported text models are Qwen3, Qwen3.5 and Granite 3.0")
            }
            Self::Model(ModelError::InvalidWeights(_)) => {
                Some("the download may be incomplete or mix files from different versions")
            }
            Self::Model(ModelError::Cuda(_)) => {
                Some("run with --backend cpu, or install the CUDA toolkit (it provides libnvrtc)")
            }
            Self::Model(ModelError::GpuMemory { .. }) => {
                Some("try a smaller --ctx, or run with --backend cpu")
            }
            Self::Model(ModelError::CpuMemory { .. })
            | Self::Generate(GenerateError::Model(ModelError::CpuMemory { .. })) => {
                Some("try a smaller --ctx, use --quantize int8, or close other programs")
            }
            Self::Generate(GenerateError::Model(ModelError::Cuda(CudaError::KvMemory {
                ..
            }))) => Some(
                "the GPU has no room for more of this conversation: close other programs, start a new chat, or run with --backend cpu",
            ),
            Self::Conversation(e) => e.hint(),
            Self::NoChatModel { .. } => Some(
                "install one: huggingface-cli download Qwen/Qwen3-1.7B --local-dir ~/models/Qwen3-1.7B (or use marin-run for base models)",
            ),
            Self::NotAChatModel(_) => {
                Some("use marin-run for base models, or pick a chat model such as Qwen/Qwen3-1.7B")
            }
            Self::NoModel => Some("put a model folder in ~/models, or run `marin model PATH`"),
            Self::ChooseModel(_) => Some("pick one with `marin model PATH` or --model PATH"),
            Self::NotAModel(_) => Some("download the whole model folder from Hugging Face"),
            Self::NoPrompt => Some("pass the text after `run`, or pipe it in"),
            Self::EmptyPrompt => Some("type some text for the model to continue"),
            Self::PromptTooLong { .. } => Some("shorten the prompt or raise --ctx"),
            Self::ContextTooLarge { .. } => Some("pick a smaller --ctx"),
            Self::NotEnoughMemory { .. } | Self::NoFit { .. } => Some(
                "try --quantize int8 (half the memory), a smaller --ctx, or close other programs",
            ),
            Self::Input { .. } => Some("check that the path exists and is a readable file"),
            Self::NoInput => Some("pass a file, or pipe text into `marin show`"),
            Self::Saved { .. } => Some("pass files written by `marin-run bench --json`"),
            Self::NotComparable(_) => Some("run both benchmarks again with the same options"),
            Self::Output { .. } => {
                Some("check that the folder exists and that you can write to it")
            }
            Self::Code(e) => e.hint(),
            Self::PortInUse(_) => Some(
                "another `marin serve` may be running there: use --port for a different one, or `marin code` to use it",
            ),
            Self::VerifyFailed => Some("the lines above show the first step that disagrees"),
            _ => None,
        }
    }

    pub fn is_broken_pipe(&self) -> bool {
        matches!(self, Self::Io(e) if e.kind() == std::io::ErrorKind::BrokenPipe)
    }
}
