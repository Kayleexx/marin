# Benchmarks

Numbers for the current engine, so later changes can be judged against them. Nothing here is tuned yet.

## Setup

- Model: Qwen3-1.7B-Base, bf16 weights, f32 math, greedy decoding, batch size 1
- Machine: AMD Ryzen 7 260, 16 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)
- Prompt of 32, 128 or 512 tokens, then exactly 64 generated tokens
- 1 warmup run, then the median of 5 runs

Reproduce one row:

```bash
marin-run bench ~/models/Qwen3-1.7B-Base --backend cuda --prompt-tokens 128 --tokens 64 --runs 5 --profile
```

`--json FILE` saves the full result. `--profile` adds one extra pass that times every stage. That pass is slower than a normal run, so it is used for shares of time, never for speed.

## GPU results

| prompt tokens | first token | prefill | decode | time between tokens (p50 / p95) | sampling per token |
|---|---|---|---|---|---|
| 32 | 93 ms | 358 tok/s | 58.6 tok/s | 16.8 / 18.6 ms | 3.8 ms |
| 128 | 336 ms | 386 tok/s | 56.0 tok/s | 17.5 / 20.5 ms | 3.8 ms |
| 512 | 1467 ms | 350 tok/s | 49.4 tok/s | 20.3 / 20.9 ms | 3.8 ms |

Loading takes about 2.5 s. The model uses about 3.5 GB of GPU memory. Each forward pass launches about 480 kernels and makes about 280 small GPU allocations.

## Where the time goes

Prefill (reading the prompt) is dominated by the MLP, then the attention projections:

| prompt tokens | MLP | q/k/v projections | attention |
|---|---|---|---|
| 128 | 70% | 16% | 4% |
| 512 | 64% | 14% | 14% |

Decode (one token at a time) is mostly MLP too, and attention grows with the prompt:

| prompt tokens | MLP | q/k/v projections | LM head | attention |
|---|---|---|---|---|
| 128 | 45% | 13% | 12% | 11% |
| 512 | 38% | 11% | 10% | 26% |

## What this suggests

- Decode reads all 3.4 GB of weights for every token, so it is limited by memory traffic. Smaller weights (quantization) are the most direct way to speed it up.
- Choosing the next token takes about 3.8 ms, roughly a fifth of each decode step, which is a lot for such a small job. It is the cheapest thing to fix.
- Prefill works out to about 1.3 TFLOP/s of useful math, and the MLP matmuls are most of it. The matmul kernel is simple, so a better one is the first thing to try there.
- Attention time grows with context length, so long prompts will need a faster attention kernel.
- The GPU is busy almost all of the time (under 5% of a decode step is outside the timed stages), so launch overhead is not the problem yet.

## Not measured yet

CPU numbers with this tool. The CPU run needs about 4 GB of free RAM and the machine did not have it when these were taken.
