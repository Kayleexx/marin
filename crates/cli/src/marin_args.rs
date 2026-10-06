use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::args::BackendArg;

const EXAMPLES: &str = "\
Examples:
  marin                                 start a session with your default model
  marin run Explain what a mutex is    ask once and print only the reply
  echo \"Say hi in French\" | marin run    read the question from stdin
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
    /// Never use colors (also honors NO_COLOR)
    #[arg(long, global = true)]
    pub no_color: bool,
}
