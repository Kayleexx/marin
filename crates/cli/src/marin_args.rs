use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::arg_types::{ChatQuantArg, KvTypeArg};
use crate::args::BackendArg;

const EXAMPLES: &str = "\
Examples:
  marin                                 start a session with your default model
  marin run Explain what a mutex is    ask once and print only the reply
  echo \"Say hi in French\" | marin run    read the question from stdin
  marin --model ~/models/Qwen3.5-4B-Q8 --quantize int8 serve    serve local text requests
  marin code                            chat about the code in this folder (needs `marin serve` running)
  marin model ~/models/Qwen3-1.7B       choose the default chat model
  marin show notes.md                   read a file with formatting, wrapped to your terminal

Press / in a session for actions (settings, tokens, switching model). `marin` needs a chat model such as Qwen3-1.7B. Base models and debug tools live in `marin-run`.";

#[derive(Parser, Debug)]
#[command(
    name = "marin",
    version,
    about = "A private assistant that runs on your own machine",
    after_help = EXAMPLES
)]
pub struct MarinCli {
    #[command(subcommand)]
    pub command: Option<MarinCommand>,
    #[command(flatten)]
    pub options: Options,
}

#[derive(Subcommand, Debug)]
pub enum MarinCommand {
    /// Ask one question and print only the reply (question from the arguments or stdin)
    Run { prompt: Vec<String> },
    /// Serve one loaded chat model over a loopback OpenAI-compatible text API
    Serve {
        /// Local TCP port (0 asks the OS to choose an available port)
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
    /// Ask questions about a code project; the assistant reads it with search, list and read tools (needs `marin serve`)
    Code {
        /// Project folder to look at [default: the current directory]
        path: Option<PathBuf>,
        /// Port of the running `marin serve`
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Append every model request and tool call to this file as JSON lines
        #[arg(long, value_name = "FILE")]
        log: Option<PathBuf>,
    },
    /// Show the selected backend and memory budget without loading weights
    Plan,
    /// Show a file (or text piped in) in the terminal: markdown, code, json, diffs, tables and logs, nicely
    Show {
        /// File to show; `-` or nothing reads standard input as it arrives
        path: Option<PathBuf>,
        /// Treat the text as this kind instead of detecting it: markdown, text, rust, python, json, diff, csv, log, ...
        #[arg(long = "as", value_name = "KIND")]
        kind: Option<String>,
    },
    /// Show the default model, or set it to PATH
    Model { path: Option<PathBuf> },
}

#[derive(Args, Debug)]
pub struct Options {
    /// Model folder to use (it becomes the default)
    #[arg(long, global = true)]
    pub model: Option<PathBuf>,
    /// Where the math runs
    #[arg(long, value_enum, default_value_t = BackendArg::Auto, global = true)]
    pub backend: BackendArg,
    /// Context window in tokens [default: 4096, or the model's maximum if smaller]
    #[arg(long, global = true)]
    pub ctx: Option<usize>,
    /// Most tokens to generate per reply
    #[arg(long, default_value_t = 512, global = true)]
    pub max_tokens: usize,
    /// Randomness: 0 always picks the likeliest token (Qwen's recommendation for chat: 0.7)
    #[arg(long, default_value_t = 0.7, global = true)]
    pub temp: f32,
    /// Only consider the K likeliest tokens (0 = off)
    #[arg(long, default_value_t = 20, global = true)]
    pub top_k: usize,
    /// Only consider tokens whose probabilities add up to P
    #[arg(long, default_value_t = 0.8, global = true)]
    pub top_p: f32,
    /// Random seed; the same seed repeats the same text
    #[arg(long, global = true)]
    pub seed: Option<u64>,
    /// Keep the start of the conversation in this file when the session ends, and use it next time if the new
    /// conversation starts the same way: the shared start is not read again. Only the GPU can do this
    #[arg(long, global = true, value_name = "FILE")]
    pub snapshot: Option<PathBuf>,
    /// How the KV cache stores keys and values: f16 (the default on the GPU) takes half the memory and fits twice the
    /// context, and rounding can change a reply slightly; f32 is exact (the only choice on the CPU)
    #[arg(long, value_enum, global = true)]
    pub kv_type: Option<KvTypeArg>,
    /// Keep weights as 8-bit numbers on CPU or GPU: much less memory to read per token, so replies come faster,
    /// at a small cost in accuracy
    #[arg(long, value_enum, default_value_t = ChatQuantArg::None, global = true)]
    pub quantize: ChatQuantArg,
    /// Weights that share one scale when quantizing, a multiple of 32: smaller is more accurate and takes more memory
    /// [default: 128]
    #[arg(long, global = true)]
    pub quant_group: Option<usize>,
    /// Never use colors (also honors NO_COLOR)
    #[arg(long, global = true)]
    pub no_color: bool,
    /// CPU threads for prompt passes (default: physical cores capped by affinity/quota; explicit N overrides this)
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,
}
