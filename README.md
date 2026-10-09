# Marin

<img width="532" height="331" alt="image" src="https://github.com/user-attachments/assets/ad91e0dd-1ff3-424b-8d2b-91652ac77165" />

Marin is a small language-model engine written from scratch in Rust, built to run well on ordinary laptops and desktops. No Python, no outside inference library. The goal is to understand every step of running a model, measure it, and make it faster on weak hardware. A chat assistant on top gives the engine real work to be judged by.

- **Models:** Qwen3 (1.7B is the reference), on the CPU or on an NVIDIA GPU with CUDA kernels written for this project.
- **Checked against Hugging Face**, token for token.
- **Fast where it matters:** tensor-core prefill, tuned attention, int8 / int4 weights, a chat cache that survives between turns, a KV cache that grows with the conversation (stored as f16 on the GPU), prefix snapshots across restarts, opt-in guessing ahead, and SIMD matmul kernels on the CPU.
- **Measured:** every optimization has before/after numbers, and the commands to repeat them, in [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

You need Rust (the version is pinned in `rust-toolchain.toml`) and a model folder from Hugging Face (`config.json`, `tokenizer.json`, `model.safetensors`). The GPU path also needs an NVIDIA GPU and CUDA.

```bash
cargo install --path crates/cli --locked
```

| Program | For |
|---|---|
| `marin` | the chat assistant |
| `marin-run` | the low-level tool: run base models, check correctness, measure |

```bash
marin --model ~/models/Qwen3-1.7B        # chat, type / for actions
marin run "explain a mutex in one line"  # one answer, then exit
marin-run MODEL -p "The capital of France is"
```

Marin uses the GPU when CUDA works and the model fits, and the CPU otherwise; `--backend cpu` or `--backend cuda` forces a choice.

### Options worth knowing

| Option | What it does |
|---|---|
| `--quantize int8` | smaller weights (`marin-run` also takes `int4`). On the GPU: decode 33–63% faster, about 1.7 GB less memory, same first-token time. It rounds the weights, so wording often differs from bf16; bf16 stays the default. [Details](BENCHMARKS.md#16-int8-weights-in-chat) |
| `--kv-type f32\|f16` | KV cache precision. `marin` uses f16 on the GPU (half the memory, twice the context, about half the first-token time at 8k–16k) and f32 on the CPU. `marin-run` and `verify` stay on f32. [Details](BENCHMARKS.md#12-kv-cache-precision-f16) |
| `--snapshot FILE` | save the conversation's cache when the session ends and reuse the shared start next time. GPU only; nothing is saved unless you ask. [Details](BENCHMARKS.md#13-prefix-snapshot) |
| `--ctx N` | the longest conversation allowed (not a reservation, see below) |
| `--speculate` (`marin-run`) | guess ahead by copying repeated text. [Details](BENCHMARKS.md#5-guessing-ahead-opt-in-gpu-only) |

### `marin-run` subcommands

| Command | What it answers |
|---|---|
| `inspect MODEL` | How big is this model and how much memory will it need? |
| `verify MODEL --reference FILE` | Does the output match Hugging Face? |
| `bench MODEL [--profile] [--sample-gpu]` | How fast is it, and where does the time go? |
| `trace MODEL` | How long does each stage of each layer take? |
| `compare before.json after.json` | Did the change help, or is it noise? |
| `probe` | What can this GPU do on its own (memory and math speed)? |
| `quality MODEL --text FILE` | How much accuracy does quantization or a KV format cost? |
| `turns MODEL` | How much sooner is a chat turn's first token when the cache is kept? |
| `restart MODEL --mode save\|restore\|cold` | How much sooner is the first token after a restart with a snapshot? |
| `spec-bench`, `guess`, `guess-report`, `pass-cost` | Guessing-ahead experiments |

## How it works

A model turns text into token numbers, runs them through a stack of identical layers, and picks the next token. Marin does each step itself.

1. **Tokenizer** splits text into tokens.
2. **Layers** normalize, run attention, then an MLP. Attention looks back at earlier tokens, whose keys and values live in the **KV cache** so past work is never repeated.
3. **Sampler** picks the next token from the model's scores.

Reading the prompt is **prefill** (big passes, limited by arithmetic and, for long prompts, attention). Writing the answer is **decode**, one token per pass, limited by how fast weights can be read from memory, which is why smaller weights decode faster.

### Features

- **Chat cache kept between turns.** Marin remembers which token ids its KV cache holds, finds the longest exactly shared start on the next turn, cuts the cache back to it, and reads only what is new. The record is written only after a reply ends cleanly and cleared on any error. After 4,000 tokens of history the first token arrives in about 0.25 s instead of 9 s, with identical replies. [Details](BENCHMARKS.md#10-keeping-the-cache-between-chat-turns)

  ```text
  cached : A B C D E F
  turn 2 : A B C D E F G H I   -> keep 6, read G H I
  edited : A B C X Y           -> keep 3, read X Y
  ```

- **KV cache that grows (GPU).** The cache is kept in pages of 64 positions (14 MiB for Qwen3-1.7B), allocated eight at a time as the conversation needs them. A short chat with `--ctx 8192` holds about 117 MB of KV instead of 1.88 GB, at a cost of 0–4% speed at long contexts. A conversation that fills the GPU stops with a clear out-of-memory error. [Details](BENCHMARKS.md#11-paged-kv-cache)
- **Snapshots across restarts (GPU, opt-in).** After 4,000 tokens the first token arrives in 0.2–0.4 s instead of 6.5–7 s. The file is checked against the exact model, tokenizer, KV type and weight format; anything wrong with it means the prompt is simply read in full. [Details](BENCHMARKS.md#13-prefix-snapshot)
- **Guessing ahead (GPU, opt-in).** When a reply repeats text it was just shown, Marin guesses up to 7 tokens by copying, checks them in one pass, and rolls back wrong guesses, so the output equals normal decoding. Code edits run 1.5–2.5x faster; chat and fresh code are unchanged.

  ```bash
  marin-run MODEL --speculate -p "..."
  ```

- **CPU kernels.** On the CPU, weights are packed as bf16 pairs and multiplied with run-time-dispatched SIMD (AVX2 + FMA when present): a register-blocked GEMM for prompts and a streaming GEMV for decode. On one thread this is 2.7–4.4x faster at prefill and 1.5–1.8x at decode than the earlier loop, and decode reaches the memory ceiling. CPUs without AVX2 keep the earlier loop. Threads and CPU quantization are not built yet. [Details](BENCHMARKS.md#22-cpu-matmul-kernels-stage-1-one-thread)

## Speed

Qwen3-1.7B on a laptop GPU held to about 40 W (the power limit decides a lot, so every change is checked end to end). All numbers are before/after on the same machine.

| What | Result |
|---|---|
| int8 vs bf16 decode | 65% faster, half the weight memory |
| Prompt, first token (f16 cache, tensor-core prefill attention) | 4,000 tokens: 8.0 s → 2.8 s (energy 318 → 109 J); 8,000 tokens: 25 s → 6 s |
| Decode with 16-byte cache loads (f16 cache) | +13 / +34 / +65 / +95% (bf16) at 2k / 4k / 8k / 16k tokens of context |
| Chat turn after 4.1k tokens of history | first token 35x (bf16) / 37x (int8) sooner with the kept cache |
| Short chat, `--ctx 8192` | 117 MB of KV instead of 1.88 GB; GPU peak 1.8 GB lower |
| f16 KV cache | KV memory halved; 32k context fits (f32: 16k); first token about half at 8k–16k |
| Restart after 4,000 tokens, `--snapshot` | first token 0.2–0.4 s instead of 6.5–7 s |

On the CPU (Ryzen 7 260, **one thread**, Qwen3-0.6B at a 127-token prompt): prefill 26 → 95 tok/s and decode 11 → 17 tok/s; Qwen3-1.7B: prefill 7.7 → 33 and decode 3.6 → 6.6. llama.cpp on the same machine and thread count reaches 210 / 18.8 (0.6B, AVX-512) and 102 / 19.4 (AVX2-only build).

Methods, tables and commands to repeat each number are in [BENCHMARKS.md](BENCHMARKS.md).

## Repository layout

| Crate | What it does |
|---|---|
| `tensor` | math on CPU data: SIMD matmul, norms, rotary embeddings, quantization |
| `loader` | reads the config, tokenizer and safetensors weights |
| `model` | the Qwen3 network on CPU and GPU |
| `cuda` | GPU kernels and the paged KV cache behind a safe API |
| `kv` | the CPU KV cache |
| `sampler` | greedy, temperature, top-k, top-p |
| `runtime` | the generate loop: prefill, decode, stop conditions, cache reuse, guessing ahead |
| `chat` | chat formatting and conversation history |
| `metrics` | timers, memory, CPU and GPU readings |
| `cli` | the `marin` and `marin-run` programs and the terminal interface |

`tests/fixtures` holds tiny models so the test suite runs without downloading weights; `tests/reference` holds Hugging Face reference output.

## Correctness

Output is compared with Hugging Face in float32: same tokens, logits within about 3e-5, on the CPU and on the GPU. Quantized models and the f16 cache cannot match that exactly, so they are checked against their own higher-precision version with written pass/fail gates (see BENCHMARKS.md). Every kernel change is tested against a plain reference. Kept-cache, snapshot and guessing-ahead output must equal a fresh read token for token, and the tests break the logic on purpose to prove they would notice.

## Development

```bash
cargo fmt --all
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
```
