# How Marin works, and what was measured

Marin is a small local LLM runtime written in Rust with CUDA kernels. This page is the short version: the design, what
worked, what was tried and dropped, the numbers, and how to repeat them. Six detailed logs are in
[docs/benchmarks/archive](docs/benchmarks/archive/INDEX.log); older logs stay on the author's machine.

## How it works

- **One way to run a model.** The terminal, `marin run`, code mode and the server all go through the same `Session`
  and `runtime::generate`. There is one sampler and one rule for reusing a cache.
- **A planner that says no.** Before loading, Marin estimates memory for the model and the context window. If it does
  not fit, it refuses and says why. It never quietly shrinks the window, swaps the model or changes precision.
- **Sessions are text.** A saved conversation is only the history, never the cache. An optional checkpoint keeps the
  model's processed state as a shortcut, and is used only when model, weights, precision and hashes all match.
- **Reuse instead of re-reading.** A follow-up turn reads only the new tokens. Qwen3.5 keeps a mark of its recurrent
  state so a failed or cancelled pass is undone without copying the whole cache (CPU and CUDA).
- **A server for several clients.** One session stays hot on the GPU and up to 16 wait in host memory (2 GiB). Up to 3
  requests share decode passes. Under GPU pressure nothing active is evicted: a pass that finds no memory is undone and
  retried or refused with 503.
- **Opt-in extras.** Guessing ahead (`--speculate`), model hibernation (`--idle-unload`), checkpoints and the background
  assistant are off by default. The assistant is rules only and never loads a model.
- **Strict code rules.** Safe Rust outside the `cuda` crate, no panicking code in production, files under 425 lines.
  All of this is checked in CI.

## Machine and method

Ryzen 7 260 (16 threads), 16 GB RAM, RTX 5050 laptop GPU with 8 GB, capped near 40 W. Rates are tokens per second.
Time to first token includes reading the prompt. RSS is the process peak, and VRAM is named as "process" or
"device-wide". At the 40 W cap, GPU decode moves by 10% or more between runs, so only matched, interleaved before/after
runs count as evidence.

## Models

| Model | Backends | Status |
|---|---|---|
| Qwen3 0.6B, 1.7B | CPU BF16 or INT8, CUDA | accepted |
| Qwen3.5-4B | CPU BF16 or INT8, CUDA INT8 (BF16 does not fit 8 GB) | accepted, coding experimental |
| Qwen3.5-2B base | CPU, CUDA | accepted, text continuation through `marin-run` |
| Granite 3.0 1B-A400M | CPU BF16 | accepted through `marin-run`; INT8 and CUDA are experimental (they miss the strict numerical gate) |
| Granite 4 H-Tiny, vision, MTP, CPU INT4 | none | not supported |

## Speed (single runs, warm file cache; see the logs for ranges)

| Model / backend | Prompt; threads | First token | Prefill | Decode | Peak memory |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B INT8 CPU | 256; 8 | 0.24 s | | 36.6 | 0.70 GB RSS |
| Qwen3-1.7B INT8 CPU | 256; 8 | 0.63 s | | 13.0 | 1.82 GB RSS |
| Qwen3.5-2B INT8 CPU | 512; 8 | 5.41 s | 94.7 | 9.45 | 2.14 GB RSS |
| Qwen3.5-4B INT8 CPU | 512; 8 | 11.56 s | 44.3 | 4.75 | 4.66 GB RSS |
| Qwen3.5-4B INT8 CUDA | 512 | 1.61 s | 319 | 32 to 36 | 0.9 GB RSS, about 4.7 GB process VRAM |
| Qwen3.5-2B BF16 CUDA | 512 | 1.59 s | 322 | 48.1 | 1.27 GB RSS |

## What worked

| Change | Result (same machine, before to after) |
|---|---|
| CUDA INT8 prefill (tensor-core kernel with a three-part split, [log 25](docs/benchmarks/archive/25-qwen35-int8-prefill.log)) | prefill 129 to 319 tok/s at 512 tokens (2.5x); first token 3.96 to 1.61 s; HF check unchanged |
| Reuse the CPU history (mark and rewind, [log 30](docs/benchmarks/archive/30-cpu-history-reuse-planner.log)) | 16 coding questions on the 4B CPU: 1,547 s to 468 s, 4.7 GB peak, same answers; planner estimate 8.32 to 6.52 GB |
| Cache-reuse sessions on the server | follow-up first token 1.6 to 0.42 s; with a checkpoint, a whole two-turn chat 46% faster than re-reading |
| Hot and cold sessions ([log 22](docs/benchmarks/archive/22-session-continuity.log)) | two alternating clients: 6 of 8 turns lost (404) to none; 8.3 s to 0.3 s to first token |
| Several requests at once ([log 06](docs/benchmarks/archive/06-server-scheduling.log)) | 3 decoders: 29 to 54.5 events/s (1.88x); no VRAM growth |
| Shared system prompt ([log 10](docs/benchmarks/archive/10-shared-prefix.log)) | short-request first token 0.333 to 0.112 s |
| In-place passes (no cache copy) | 14k-token decode 22.3 to 32.0 tok/s, VRAM peak 6.3 to 5.4 GB; the planner admits about 27k tokens instead of 15k |
| Memory-pressure recovery | turns with 340 to 600 MiB free: 29 to 42 s down to 6 to 7 s; two requests that do not fit now give 503, not 500 |
| Guessing ahead (`--speculate`, CUDA, opt-in) | code edits 39.9 to 81.2 tok/s (2.04x), chat 1.02x; greedy output identical in 64 of 64 runs |
| Hibernation (`--idle-unload`) | GPU memory 4.7 GB to 0 when idle; wake to first token 5 to 6 s; no tokens re-read |
| CPU prefill (state check once per pass) | +12 to 13% prefill, first token 11% lower |
| Code-mode call recovery ([log 34](docs/benchmarks/archive/34-v01-final-pass.log)) | malformed calls 11 to 5; 15 to 16 of 25 tasks (within noise) |

## What was tried and dropped

| Idea | Why not |
|---|---|
| Native CPU INT4 | decode already reads about 18 of 24 GB/s, so the ceiling is near 1.7x. Emulated 4-bit lost quality: 13 of 25 tasks against 16. Closed. |
| More CPU decode threads | one core already reaches the 25.5 GB/s memory limit; 2 to 4 threads were slower |
| Guessing ahead on the CPU | a 4-token pass costs 3.4x one token (997 vs 295 ms), so it cannot pay |
| Batching two prompts in one prefill | -8% to +13% against a 10% gate; prefill is already compute-bound |
| Switching models or thinking mode by task | thinking cost 2.2x the time and 4.2x the tokens for no gain; the 1.7B fails the tool format |
| Constrained tool output, an "answer cites files" guard | the guard scored 15 against 16 of 25 and was removed; constrained decoding needs a new sampler hook |
| DeltaNet CPU convolution rewrite | no prefill gain (46.0 to 45.1 tok/s); reverted |
| Tiered KV (offload old keys) | 37 ms per token at 8k context |

## Coding mode (experimental)

`marin code` is read-only (list, read, search). On 25 questions about this repository, Qwen3.5-4B INT8 finds the right
place 14 to 16 times (56 to 64%); the spread is one or two tasks. Misses are confident, so use it to find `path:line` and
then read the line. Two graders have drifted from the repo (`max_active` moved; `Store::evict` is gone), so compare runs
from the same commit only.

## Gates that stay in force

- **Hugging Face check:** Qwen3.5-4B INT8 on CUDA, 128 steps: token choices match, max logit diff 1.19e-5, max layer
  diff 1.15e-5, normalized max 4.4e-6, relative RMS 3.0e-6. Limits: normalized max at most 1e-4, relative RMS at most 1e-5.
- **Same answers on reuse:** reusing a cache must give the same tokens as reading the history again (greedy).
- **Tests:** CUDA-required suite 828 passed, forced-scalar suite 828 passed, 0 failed, 5 ignored (real-checkpoint tests).
- **Source guards:** `scripts/check-line-limit.sh`, `scripts/check-unsafe.sh`, Clippy `-D warnings`, `cargo fmt`.
- **Session-store race fixed:** a session that is being moved to host memory is now waited for instead of being
  reported missing (404). The unit test in `crates/cli/src/server/session_tests.rs` failed before the fix.

## Limits

- CPU decode is limited by memory bandwidth, so the 4B model runs at about 4.7 tok/s on this CPU.
- A session can still be dropped by design when host memory is short or the 2 GiB and 16-session limits are reached.
- The planner estimates memory and cannot reserve it against other programs. Below about 150 MiB of free GPU memory a
  single pass cannot run.
- Login autostart, multi-hour idle power draw and rare extreme low-VRAM recovery are not verified.
- The server has no authentication and binds loopback only.

## Reproduce

```bash
# speed: same model, context, prompt, threads and warmup on both sides
marin-run bench ~/models/Qwen3.5-4B-Q8 --backend cpu --threads 8 --quantize int8 \
  --prompt-tokens 512 --tokens 32 --ctx 2048 --runs 3 --warmup 1 --json cpu.json
marin-run bench ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 \
  --prompt-tokens 512 --tokens 32 --ctx 2048 --runs 3 --warmup 1 --json cuda.json
marin-run compare cpu.json cuda.json

# correctness: needs a matching Hugging Face reference file
marin-run verify ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 --reference hf-int8-128.json

# coding benchmark (from the repository root)
python3 -I scripts/code_eval.py run --model ~/models/Qwen3.5-4B-Q8 --quantize int8 --tag mine
python3 -I scripts/code_eval.py score --tag mine

# all checks
cargo fmt --all --check && cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
MARIN_REQUIRE_CUDA=1 MARIN_QWEN35_Q8_DIR=~/models/Qwen3.5-4B-Q8 cargo test --workspace --all-features --locked
scripts/check-line-limit.sh && scripts/check-unsafe.sh
```

The Hugging Face references are generated, not stored: `scripts/gen_reference.py`, `scripts/gen_chat_reference.py`,
`scripts/gen_chat_golden.py` and `tests/reference/{qwen35,granite}/*.py` (they need torch and transformers). The
128-step reference for the 4B is 226 MB and is not in the repository; make it with
`tests/reference/qwen35/full.py MODEL OUT --steps 128` on the BF16 checkpoint.

## Logs

| Log | What it shows |
|---|---|
| [06 server scheduling](docs/benchmarks/archive/06-server-scheduling.log) | several requests at once, gates and results |
| [10 shared prefix](docs/benchmarks/archive/10-shared-prefix.log) | why batching prefill was dropped and sharing the system prompt kept |
| [22 session continuity](docs/benchmarks/archive/22-session-continuity.log) | hot and cold sessions |
| [25 CUDA INT8 prefill](docs/benchmarks/archive/25-qwen35-int8-prefill.log) | the largest speed-up and its numerical proof |
| [30 CPU history reuse](docs/benchmarks/archive/30-cpu-history-reuse-planner.log) | 1,547 s to 468 s on the CPU |
| [34 final pass](docs/benchmarks/archive/34-v01-final-pass.log) | coding audit, the two server_cold root causes, release checks |

[INDEX.log](docs/benchmarks/archive/INDEX.log) lists every milestone, including the ones kept only on the author's machine.
