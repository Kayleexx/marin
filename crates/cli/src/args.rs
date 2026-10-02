use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

const EXAMPLES: &str = "\
Examples:
  marin-run ./Qwen3-1.7B-Base                       chat-style prompt loop
  marin-run ./Qwen3-1.7B-Base -p \"The capital of France is\" --temp 0
  marin-run inspect ./Qwen3-1.7B-Base               explain the model and its memory needs
  marin-run verify ./Qwen3-1.7B-Base --reference ref.json";

#[derive(Parser, Debug)]
#[command(
    name = "marin-run",
    version,
    about = "Run a language model on this machine and see how it works",
    arg_required_else_help = true,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true,
    after_help = EXAMPLES
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
    #[command(flatten)]
    pub run: RunArgs,
    /// Never use colors (also honors NO_COLOR)
    #[arg(long, global = true)]
    pub no_color: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Explain a model folder in plain words and estimate its memory use
    Inspect {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        #[command(flatten)]
        ctx: CtxArg,
    },
    /// Measure load time, prefill and decode speed, memory and (with --profile) where the time goes
    Bench {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        /// Tokens in the prompt
        #[arg(long, default_value_t = 128)]
        prompt_tokens: usize,
        /// Tokens to generate (always exactly this many)
        #[arg(long, default_value_t = 64)]
        tokens: usize,
        /// Measured runs; the table shows the median
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// Runs before measuring, to warm caches and the GPU
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Also run one pass with every stage timed (RMSNorm, RoPE, attention, MLP, LM head, ...)
        #[arg(long)]
        profile: bool,
        /// Write the full result as JSON
        #[arg(long)]
        json: Option<PathBuf>,
        #[command(flatten)]
        ctx: CtxArg,
    },
    /// Compare this program's output with Hugging Face reference output
    Verify {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        /// Reference file written by scripts/gen_reference.py
        #[arg(long)]
        reference: PathBuf,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendArg {
    /// Use the GPU when CUDA works and the model fits, otherwise the CPU
    Auto,
    Cpu,
    Cuda,
}

#[derive(Args, Debug)]
pub struct CtxArg {
    /// Context window: tokens (prompt plus reply) held at once [default: 2048, or the model's maximum if smaller]
    #[arg(long)]
    pub ctx: Option<usize>,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Folder with config.json, tokenizer.json and model.safetensors
    #[arg(required = true)]
    pub model_dir: Option<PathBuf>,
    /// Where the math runs
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub backend: BackendArg,
    /// Complete this text once and exit (reply on stdout, everything else on stderr)
    #[arg(short, long)]
    pub prompt: Option<String>,
    #[command(flatten)]
    pub ctx: CtxArg,
    /// Most tokens to generate per prompt
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,
    /// Randomness: 0 always picks the likeliest token, higher is more varied
    #[arg(long, default_value_t = 0.7)]
    pub temp: f32,
    /// Only consider the K likeliest tokens (0 = off)
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,
    /// Only consider the smallest set of tokens whose probabilities add up to P
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,
    /// Random seed; the same seed repeats the same text (default: time based)
    #[arg(long)]
    pub seed: Option<u64>,
    /// After a -p reply, print where its time and memory went
    #[arg(long)]
    pub metrics: bool,
}
