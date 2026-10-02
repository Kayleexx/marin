# Marin
<img width="532" height="331" alt="image" src="https://github.com/user-attachments/assets/ad91e0dd-1ff3-424b-8d2b-91652ac77165" />

Marin is a private, local-first intelligence layer built on a hardware-adaptive inference runtime for ordinary consumer hardware.

The core of the project is the runtime. It is a small inference engine written from scratch in Rust, with no Python and no outside inference library, built so every part of running a language model can be understood, measured and then optimized for weak machines. On top of it sits an assistant, which gives the engine a real workload to be judged against.

Where it is today: the engine runs Qwen3 models in bf16 on the CPU or an NVIDIA GPU, checks its output against Hugging Face, and has a chat assistant and tools to measure speed and memory.

## Where it is going

The goal is to squeeze as much useful intelligence as possible out of a fixed amount of compute, while collecting as little data as possible. Planned work, in the order the measurements justify:

- faster execution and quantization
- smarter KV cache handling, including moving it between GPU and RAM
- batching, scheduling and speculative decoding
- picking the model, precision and context size automatically for the machine it runs on
- an assistant that can act when asked, help with coding, remember ongoing work, and speak up only when it has something useful to say
- the same session following you between laptop and phone

## Try it

You need Rust (the right version is picked up automatically) and a model folder from Hugging Face.

```bash
cargo install --path crates/cli --locked
```

This gives you two programs:

- `marin` is the chat assistant. Point it at a chat model such as Qwen3-1.7B.
- `marin-run` is the low level tool for base models, checking and measuring.

```bash
marin --model ~/models/Qwen3-1.7B        # chat, type / for actions, Ctrl+O to inspect
marin run "explain a mutex in one line"  # one answer, then exit
```

Marin picks the GPU when CUDA works and the model fits, and falls back to the CPU otherwise. `--backend cpu` or `--backend cuda` forces a choice.

```bash
marin-run inspect ~/models/Qwen3-1.7B-Base                       # size and memory estimate
marin-run ~/models/Qwen3-1.7B-Base -p "The capital of France is" # continue some text
marin-run verify ~/models/Qwen3-1.7B-Base \
  --reference tests/reference/qwen3-1.7b-base.json              # compare with Hugging Face
marin-run bench ~/models/Qwen3-1.7B-Base --profile               # speed and where time goes
```

## How it works

A language model turns text into token numbers, runs them through a stack of identical layers, and picks the next token. Marin does each step itself:

1. The tokenizer splits text into tokens.
2. Each layer normalizes, does attention, then an MLP. Attention looks back at earlier tokens, and their stored keys and values (the KV cache) are reused so past work is never repeated.
3. The sampler chooses the next token from the model's scores.

Chat mode wraps your messages in the model's official chat format and keeps the whole conversation, so follow ups make sense.

## Layout

| Crate | What it does |
|---|---|
| `tensor` | math on f32 data: matmul, norms, rotary embeddings |
| `loader` | reads config, tokenizer and safetensors weights, and checks them |
| `model` | the Qwen3 network on CPU and GPU |
| `cuda` | the only crate with `unsafe`, for GPU kernels behind a safe API |
| `kv` | the attention memory |
| `sampler` | greedy, temperature, top-k, top-p |
| `runtime` | the generate loop: prefill, decode, stop conditions |
| `chat` | chat formatting and conversation history |
| `metrics` | timers and memory readings |
| `cli` | the `marin` and `marin-run` programs and the terminal interface |

## Correctness

Output is checked against Hugging Face in float32: same tokens, and logits within about 4e-5. `marin-run verify` runs that check on a real model. Small fixtures in `tests/` let the test suite run without downloading weights.

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
scripts/check-line-limit.sh   # source files stay under 425 lines
scripts/check-unsafe.sh       # unsafe only in the cuda crate
```

Rules the code follows: safe Rust outside `cuda`, no `unwrap` or panics in normal paths, small files, and every speed claim backed by a measurement.
