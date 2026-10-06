# Marin
<img width="532" height="331" alt="image" src="https://github.com/user-attachments/assets/ad91e0dd-1ff3-424b-8d2b-91652ac77165" />

Marin is a small language model engine written from scratch in Rust, built to run well on ordinary laptops and desktops. There is no Python and no outside inference library. The goal is to understand every step of running a model, measure it, and then make it faster on weak hardware. A chat assistant sits on top of the engine and gives it real work to be judged by.

## What works today

- Runs Qwen3 models on the CPU or on an NVIDIA GPU, using CUDA kernels written for this project.
- Checks its own output against Hugging Face, token for token.
- Optional 8-bit and 4-bit weights on the GPU: less memory, faster decoding.
- Fast prompt reading with tensor cores, and attention kernels tuned for both short and long prompts.
- An opt-in speed-up for text that repeats, like code edits.
- A chat program, plus tools to measure speed, memory, power and where the time goes.

## Try it

You need Rust (the right version is picked automatically) and a model folder from Hugging Face.

```bash
cargo install --path crates/cli --locked
```

This gives you two programs. `marin` is the chat assistant. `marin-run` is the low level tool for base models, checking and measuring.

```bash
marin --model ~/models/Qwen3-1.7B        # chat, type / for actions
marin run "explain a mutex in one line"  # one answer, then exit
```

Marin uses the GPU when CUDA works and the model fits, and the CPU otherwise. `--backend cpu` or `--backend cuda` forces a choice.

```bash
marin-run inspect MODEL                   # size and memory estimate
marin-run MODEL -p "The capital of France is"
marin-run verify MODEL --reference tests/reference/qwen3-1.7b-base.json
marin-run bench MODEL --profile           # speed and where the time goes
marin-run bench MODEL --sample-gpu        # also GPU load, power and clock
marin-run trace MODEL                     # time every stage of every layer
marin-run compare before.json after.json  # did it change, or is it noise?
marin-run probe                           # what the GPU itself can do
```

Add `--quantize int8` (or `int4`) to run with smaller weights. int8 costs almost no quality. int4 saves more memory but is clearly less accurate. `marin-run quality` measures the difference on a text file.

## How it works

A model turns text into token numbers, runs them through a stack of identical layers, and picks the next token. Marin does each step itself.

1. The tokenizer splits text into tokens.
2. Each layer normalizes, runs attention, then an MLP. Attention looks back at earlier tokens. Their keys and values are kept in the KV cache so past work is never repeated.
3. The sampler picks the next token from the model's scores.

Reading the prompt is called prefill and happens in big passes. Writing the answer is decode, one token per pass. Decode is limited by how fast the weights can be read from memory, which is why smaller weights make it faster. Prefill is limited by arithmetic, and for long prompts by attention.

### Guessing ahead (opt-in)

When a reply repeats text it was just shown, Marin can guess the next few tokens by copying, then check all the guesses in one pass. Wrong guesses are rolled back, so the output is exactly what normal decoding gives. It only guesses after the last 8 tokens exactly match earlier text, and at most 7 tokens at a time. It is off by default and GPU only:

```bash
marin-run MODEL --speculate -p "..."
```

Code edits ran roughly 1.5 to 2.5 times faster in our tests. Normal chat and fresh code stay about the same speed.

## Speed

Measured with Qwen3-1.7B on a laptop GPU held to about 40 W (the power limit decides a lot, so every change was checked end to end):

- int8 decodes 65% faster than bf16 and uses half the weight memory.
- A 3,500 token prompt gets its first token in about 7 seconds, 40% sooner than before the prefill attention work.
- With a long conversation (4,000 tokens) decoding is 13 to 17% faster since the decode attention work.
- Every optimization is listed with its numbers, method and the commands to repeat it in the benchmarks write-up at the top of the repository.

## Code layout

| Crate | What it does |
|---|---|
| `tensor` | math on f32 data: matmul, norms, rotary embeddings, quantization |
| `loader` | reads the config, tokenizer and safetensors weights |
| `model` | the Qwen3 network on CPU and GPU |
| `cuda` | GPU kernels behind a safe API |
| `kv` | the KV cache |
| `sampler` | greedy, temperature, top-k, top-p |
| `runtime` | the generate loop: prefill, decode, stop conditions, guessing ahead |
| `chat` | chat formatting and conversation history |
| `metrics` | timers, memory, CPU and GPU readings |
| `cli` | the `marin` and `marin-run` programs and the terminal interface |

## Correctness

Output is compared with Hugging Face in float32: same tokens, logits within about 3e-5. Quantized models cannot match that exactly, so they are checked against their own 16-bit version instead. Every kernel change is also tested against a plain reference, and small test models in `tests/` let the test suite run without downloading weights.

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
```
