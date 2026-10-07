# Marin
<img width="532" height="331" alt="image" src="https://github.com/user-attachments/assets/ad91e0dd-1ff3-424b-8d2b-91652ac77165" />

Marin is a small language-model engine written from scratch in Rust, built to run well on ordinary laptops and desktops. No Python, no outside inference library. The goal is to understand every step of running a model, measure it, and make it faster on weak hardware. A chat assistant on top gives the engine real work to be judged by.

- **Models:** Qwen3 (1.7B is the reference), on the CPU or on an NVIDIA GPU with CUDA kernels written for this project.
- **Checked against Hugging Face**, token for token.
- **Faster where it matters:** tensor-core prefill, tuned attention, int8 / int4 weights, a chat cache that survives between turns, and opt-in guessing ahead.
- **Measured:** every optimization has before/after numbers, and tools to repeat them, in [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

You need Rust (the right version is picked from `rust-toolchain.toml`) and a model folder from Hugging Face (`config.json`, `tokenizer.json`, `model.safetensors`). The GPU path needs an NVIDIA GPU and CUDA.

```bash
cargo install --path crates/cli --locked
```

This installs two programs:

| Program | For |
|---|---|
| `marin` | the chat assistant |
| `marin-run` | the low-level tool: run base models, check correctness, measure |

```bash
marin --model ~/models/Qwen3-1.7B        # chat, type / for actions
marin run "explain a mutex in one line"  # one answer, then exit
marin-run MODEL -p "The capital of France is"
```

Marin uses the GPU when CUDA works and the model fits, and the CPU otherwise. `--backend cpu` or `--backend cuda` forces a choice. Add `--quantize int8` (or `int4`) for smaller weights: int8 costs almost no quality, int4 is clearly less accurate.

### `marin-run` subcommands

| Command | What it answers |
|---|---|
| `inspect MODEL` | How big is this model and how much memory will it need? |
| `verify MODEL --reference FILE` | Does the output match Hugging Face? |
| `bench MODEL [--profile] [--sample-gpu]` | How fast is it, and where does the time go (GPU load, power, clock)? |
| `trace MODEL` | How long does each stage of each layer take? |
| `compare before.json after.json` | Did the change help, or is it noise? |
| `probe` | What can this GPU do on its own (memory and math speed)? |
| `quality MODEL --text FILE` | How much accuracy does quantization cost? |
| `turns MODEL` | How much sooner is the first token of a chat turn when the cache is kept? |
| `spec-bench`, `guess`, `guess-report`, `pass-cost` | Guessing-ahead experiments |

## How it works

A model turns text into token numbers, runs them through a stack of identical layers, and picks the next token. Marin does each step itself.

1. **Tokenizer** splits text into tokens.
2. **Layers** normalize, run attention, then an MLP. Attention looks back at earlier tokens. Their keys and values live in the **KV cache**, so past work is never repeated.
3. **Sampler** picks the next token from the model's scores.

Reading the prompt is **prefill** (big passes, limited by arithmetic and, for long prompts, attention). Writing the answer is **decode**, one token per pass, limited by how fast weights can be read from memory, which is why smaller weights decode faster.

### The chat cache survives between turns

Each chat turn sends the whole conversation. Marin remembers which token ids its KV cache holds. On the next turn it finds the longest exactly shared start, cuts the cache back to it, and reads only what is new.

```text
cached : A B C D E F
turn 2 : A B C D E F G H I   -> keep 6, read G H I
edited : A B C X Y           -> keep 3, read X Y
```

The record is written only after a reply ends cleanly and is cleared on any error, so a half-written cache is never trusted. After 4,000 tokens of history the first token arrives in about 0.25 s instead of 9 s, with identical replies. Details in [BENCHMARKS.md](BENCHMARKS.md#10-keeping-the-cache-between-chat-turns).

### Guessing ahead (opt-in, GPU only)

When a reply repeats text it was just shown, Marin can guess the next few tokens by copying, then check them all in one pass. Wrong guesses are rolled back, so the output is exactly what normal decoding gives. It guesses only after the last 8 tokens match earlier text, and at most 7 at a time.

```bash
marin-run MODEL --speculate -p "..."
```

Code edits ran roughly 1.5 to 2.5 times faster. Normal chat and fresh code stay about the same speed.

## Speed

Qwen3-1.7B on a laptop GPU held to about 40 W (the power limit decides a lot, so every change is checked end to end):

| What | Result |
|---|---|
| int8 vs bf16 decode | 65% faster, half the weight memory |
| 3,500-token prompt, first token | about 7 s, 40% sooner after the prefill attention work |
| 4,000-token conversation, decode | 13 to 17% faster after the decode attention work |
| Chat turn after 4.1k tokens of history | first token 35x (bf16) / 37x (int8) sooner with the kept cache |

Methods, tables and commands to repeat each number are in [BENCHMARKS.md](BENCHMARKS.md).

## Repository layout

| Crate | What it does |
|---|---|
| `tensor` | math on f32 data: matmul, norms, rotary embeddings, quantization |
| `loader` | reads the config, tokenizer and safetensors weights |
| `model` | the Qwen3 network on CPU and GPU |
| `cuda` | GPU kernels behind a safe API (the only crate allowed `unsafe`) |
| `kv` | the CPU KV cache |
| `sampler` | greedy, temperature, top-k, top-p |
| `runtime` | the generate loop: prefill, decode, stop conditions, cache reuse, guessing ahead |
| `chat` | chat formatting and conversation history |
| `metrics` | timers, memory, CPU and GPU readings |
| `cli` | the `marin` and `marin-run` programs and the terminal interface |

`tests/fixtures` holds tiny models so the test suite runs without downloading weights; `tests/reference` holds Hugging Face reference output.

## Correctness

Output is compared with Hugging Face in float32: same tokens, logits within about 3e-5. Quantized models cannot match that exactly, so they are checked against their own 16-bit version. Every kernel change is tested against a plain reference. Kept-cache and guessing-ahead output must equal a fresh read token for token, and tests break the logic on purpose to prove they would notice.

## Development

```bash
cargo fmt --all
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
```

Project rules (the workspace lints and CI enforce most of them):

- Safe Rust only. `unsafe` lives in the `cuda` crate; every other crate has `#![forbid(unsafe_code)]`.
- No `unwrap`, `expect`, `panic!`, or indexing and arithmetic that can panic in production code (workspace lints deny them). Errors carry context.
- No lint suppressions: fix the cause.
- Rust files stay under 425 lines.
- An optimization needs a before/after measurement in [BENCHMARKS.md](BENCHMARKS.md), and correctness is checked before speed.
