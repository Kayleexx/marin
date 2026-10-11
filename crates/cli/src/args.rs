use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

pub use crate::arg_types::*;
use crate::trace::Pass;

const EXAMPLES: &str = "\
Examples:
  marin-run ./Qwen3-1.7B-Base                       chat-style prompt loop
  marin-run ./Qwen3-1.7B-Base -p \"The capital of France is\" --temp 0
  marin-run inspect ./Qwen3-1.7B-Base               explain the model and its memory needs
  marin-run trace ./Qwen3-1.7B-Base                 time each layer of one pass
  marin-run compare before.json after.json          what changed between two bench runs
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
    /// CPU threads for prompt passes (default: one per physical core; 1 runs on a single thread)
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Explain a model folder in plain words and estimate its memory use
    Inspect {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        #[command(flatten)]
        quant: QuantArgs,
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
        /// Also watch GPU load, power, clock, temperature and whether the power limit is slowing it, through nvidia-smi
        #[arg(long)]
        sample_gpu: bool,
        /// Randomness while generating: 0 (the default) always picks the likeliest token; chat uses 0.7
        #[arg(long, default_value_t = 0.0)]
        temp: f32,
        /// Only consider the K likeliest tokens (0 = off; chat uses 20)
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Only consider tokens whose probabilities add up to P (chat uses 0.8)
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// Wait this long after every token, as a slower consumer would (to see what idle time does to the GPU)
        #[arg(long, default_value_t = 0.0)]
        token_pause_ms: f64,
        /// Random seed for sampling, so repeated runs make the same choices
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Write the full result as JSON
        #[arg(long)]
        json: Option<PathBuf>,
        #[command(flatten)]
        quant: QuantArgs,
        #[command(flatten)]
        ctx: CtxArg,
    },
    /// Measure what this GPU can do (memory speed, math speed, copies to the host) to judge the engine against
    Probe {
        /// Seconds each test runs (in half-second windows)
        #[arg(long, default_value_t = 4.0)]
        seconds: f64,
        /// Write the full result as JSON
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Compare two `bench --json` results and say what changed, and whether it may just be noise
    Compare {
        /// The earlier result
        old: PathBuf,
        /// The later result
        new: PathBuf,
    },
    /// Time every stage of one forward pass, layer by layer
    Trace {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        /// Tokens in the prompt
        #[arg(long, default_value_t = 32)]
        prompt_tokens: usize,
        /// `prefill` for the pass that reads the prompt, or N for the Nth one-token pass after it
        #[arg(long, default_value = "1")]
        pass: Pass,
        /// Write every timed call as JSON
        #[arg(long)]
        json: Option<PathBuf>,
        #[command(flatten)]
        ctx: CtxArg,
    },
    /// Score how well the model predicts a text, to see what quantization costs in accuracy
    Quality {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        /// Text file to read (the model is asked, token by token, what comes next)
        #[arg(long)]
        text: PathBuf,
        /// Tokens of the text to use: the first 64 are the prompt, the rest are scored
        #[arg(long, default_value_t = 1024)]
        tokens: usize,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        /// Write the scores to a file, to compare a later run with it
        #[arg(long)]
        save: Option<PathBuf>,
        /// Compare with a file written by `--save` (same text, usually the unquantized model)
        #[arg(long)]
        against: Option<PathBuf>,
        #[command(flatten)]
        quant: QuantArgs,
    },
    /// Let a chat model answer a folder of prompts and save the tokens, to measure how well they could be guessed
    Guess {
        /// Chat model folder, such as Qwen3-1.7B
        model_dir: PathBuf,
        /// Folder of folders: each is a class of prompt, each .txt file in it one message to the model
        #[arg(long)]
        prompts: PathBuf,
        /// Most tokens in each reply
        #[arg(long, default_value_t = 400)]
        max_tokens: usize,
        /// Context window: the longest prompt plus a reply
        #[arg(long, default_value_t = 5120)]
        ctx: usize,
        /// Where to write the prompts and replies
        #[arg(long)]
        save: PathBuf,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[command(flatten)]
        quant: QuantArgs,
    },
    /// Replay replies saved by `guess` through the n-gram guesser: how often guesses are right, and the speed-up
    GuessReport {
        /// File written by `guess --save`
        transcripts: PathBuf,
        /// File written by `pass-cost --save` for the same weights: adds the projected speed-up
        #[arg(long)]
        costs: Option<PathBuf>,
        /// Matching tokens needed before a guess is made, for the detail tables
        #[arg(long, default_value_t = 2)]
        min_match: usize,
        /// Most tokens guessed at once, for the detail tables
        #[arg(long, default_value_t = 4)]
        draft: usize,
    },
    /// Time one pass over several tokens at several context lengths: what checking guesses costs
    PassCost {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        /// Text whose tokens fill the context (a few thousand tokens)
        #[arg(long)]
        text: PathBuf,
        /// Tokens already in the context when the passes run
        #[arg(long, value_delimiter = ',', default_value = "128,1024,4096")]
        contexts: Vec<usize>,
        /// Tokens fed in one pass
        #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,5,6,8,9,12,16")]
        sizes: Vec<usize>,
        /// Measured rounds; every round reads the context again for each size and times it
        #[arg(long, default_value_t = 10)]
        rounds: usize,
        /// Rounds before measuring, to warm the GPU
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Passes counted per size and round, timed back to back after the GPU has settled
        #[arg(long, default_value_t = 6)]
        reps: usize,
        /// Write the timings to a file for `guess-report --costs`
        #[arg(long)]
        save: Option<PathBuf>,
        /// Also profile each size stage by stage: attention time and key/value traffic
        #[arg(long)]
        stages: bool,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[command(flatten)]
        quant: QuantArgs,
    },
    /// Time the first token of a chat turn built on the cached conversation, against reading it all again
    Turns {
        /// Chat model folder, such as Qwen3-1.7B
        model_dir: PathBuf,
        /// Where the math runs
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        /// Conversation lengths in tokens before the new message (0 = a short first turn)
        #[arg(long, value_delimiter = ',', default_value = "0,2048,4096")]
        history: Vec<usize>,
        /// Most tokens in each reply (greedy)
        #[arg(long, default_value_t = 32)]
        reply_tokens: usize,
        /// Rounds timed (one more runs first and is thrown away)
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        /// Context window: the longest conversation plus two replies
        #[arg(long, default_value_t = 5120)]
        ctx: usize,
        /// Do not read the GPU's power
        #[arg(long)]
        no_energy: bool,
        /// Guess ahead by copying earlier text in both modes (GPU only), and show how many guesses were right
        #[arg(long)]
        speculate: bool,
        /// The message of the timed turn
        #[arg(long, default_value = "And what about at night?")]
        ask: String,
        #[command(flatten)]
        quant: QuantArgs,
    },
    /// Time the first token of a prompt after a restart: restore a saved start from disk, or read it all (GPU only)
    Restart {
        /// Folder with config.json, tokenizer.json and model.safetensors
        model_dir: PathBuf,
        /// save: read the prompt and write the snapshot; restore: read the same start plus a few tokens using it; cold: the
        /// same prompt with no snapshot
        #[arg(long, value_enum)]
        mode: RestartModeArg,
        /// Tokens of the saved start
        #[arg(long, default_value_t = 2048)]
        tokens: usize,
        /// Tokens added after the saved start when restoring or reading cold
        #[arg(long, default_value_t = 16)]
        extra: usize,
        /// Tokens generated after the prompt, to compare restored output with the live cache's
        #[arg(long, default_value_t = 1)]
        reply_tokens: usize,
        /// The snapshot file
        #[arg(long)]
        file: PathBuf,
        /// Do not read the GPU's power
        #[arg(long)]
        no_energy: bool,
        #[command(flatten)]
        quant: QuantArgs,
    },
    /// Experiment: decode real replies with and without guessing tokens ahead, interleaved, and compare (GPU only)
    SpecBench {
        /// Chat model folder, such as Qwen3-1.7B
        model_dir: PathBuf,
        /// Folder of folders: each is a scenario, each .txt file in it one message to the model
        #[arg(long)]
        prompts: PathBuf,
        /// Most tokens in each reply
        #[arg(long, default_value_t = 300)]
        max_tokens: usize,
        /// Context window: the longest prompt plus a reply
        #[arg(long, default_value_t = 5120)]
        ctx: usize,
        /// Rounds timed (one more runs first and is thrown away)
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        /// Ways of choosing tokens to time: greedy, chat (temperature 0.7, top-k 20, top-p 0.8, fixed seed)
        #[arg(long, value_delimiter = ',', default_value = "greedy,chat")]
        sampling: Vec<String>,
        /// Do not read the GPU's power
        #[arg(long)]
        no_energy: bool,
        /// Write every run as JSON
        #[arg(long)]
        save: Option<PathBuf>,
        #[command(flatten)]
        quant: QuantArgs,
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
        #[command(flatten)]
        kv: KvArgs,
    },
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
    pub quant: QuantArgs,
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
    /// Experimental: guess tokens ahead by copying text that repeats and check them in one pass (GPU only). The
    /// reply is the same; code edits come out faster
    #[arg(long)]
    pub speculate: bool,
}
