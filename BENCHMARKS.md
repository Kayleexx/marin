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
| Qwen3-0.6B INT8 CPU | 127; 8 | 0.24 s | | 36.6 | 0.70 GB RSS |
| Qwen3-1.7B INT8 CPU | 127; 8 | 0.63 s | | 13.0 | 1.82 GB RSS |
| Qwen3.5-2B INT8 CPU | 512; 8 | 5.41 s | 94.7 | 9.45 | 2.14 GB RSS |
| Qwen3.5-4B INT8 CPU | 512; 8 | 11.56 s | 44.3 | 4.75 | 4.66 GB RSS |
| Qwen3.5-4B INT8 CUDA | 512 | 1.61 s | 319 | 32 to 36 | 0.9 GB RSS, about 4.7 GB process VRAM |
| Qwen3.5-2B BF16 CUDA | 512 | 1.59 s | 322 | 48.1 | 1.27 GB RSS |

The two Qwen3 CPU rows were first labelled 256 prompt tokens; the log they come from (03-loader-qwen35-planner) used 127.
Do not compare them with runs at other prompt lengths (256 tokens gives 0.43 s and 1.23 s, the same speed per token).

## Final performance pass (2026-10-29)

Build 5f5b22b plus the DeltaNet kernel change below. Same machine and 40 W cap. Median of 3 rounds of 3 timed runs
after 1 warmup, old and new binaries run alternately, nothing else on the machine. Raw JSON stays local.

**CUDA, Qwen3.5-4B INT8** (`marin-run bench MODEL --backend cuda --quantize int8 --prompt-tokens N --tokens 64 --ctx 4096 --runs 3 --warmup 1`):

| Prompt | | TTFT | Prefill | Decode | Peak memory |
|---|---|---:|---:|---:|---|
| 512 | before | 1560 ms | 328 tok/s | 31.8 tok/s | 0.91 GB RSS, 5.9 GB VRAM (device-wide) |
| 512 | after | 1147 ms (-26%) | 447 tok/s (+36%) | 31.3 tok/s (-1.7%, within noise) | unchanged |
| 2048 | before | 6660 ms | 308 tok/s | 31.2 tok/s | |
| 2048 | after | 5082 ms (-24%) | 403 tok/s (+31%) | 30.5 tok/s (-2.2%, within noise) | |

**CUDA, Qwen3.5-2B BF16** (512 prompt): TTFT 1519 to 1347 ms (-11%), prefill 337 to 380 tok/s (+13%), decode 46.3 to 46.5 tok/s.

- **Change:** the DeltaNet recurrence kernel keeps each thread's column of state in registers for the whole call instead of
  reading and writing global memory per token. The recurrence was 46.6% of prefill (709 of 1522 ms) and is now 19% (219 ms);
  in decode it fell from 4.0 to 2.0 ms per token, but the weight-streaming stages measured about 10% slower in the same
  run (GPU at its 40 W cap, cause not proven), so decode is unchanged.
- **Same answers:** the arithmetic order is unchanged. Greedy output is byte-identical to the old binary on a 2k-token prompt,
  the Hugging Face check is unchanged (max logit diff 1.19e-5, relative RMS 3.0e-6), and `crates/cuda/tests/delta.rs`
  checks key dimensions 1 to 256 against a CPU loop and checks that splitting tokens across calls changes nothing.
- **Second step (same kernel):** the per-token set-up (normalising q and k) ran on one thread with 256 dependent global loads.
  Threads now load q and k in parallel and the sums keep the same order, so the output is still bit-identical (greedy output
  byte-identical, reference and split tests pass). Against the first version: 4B TTFT 1159 to 1102 ms (-5%) at 512 tokens and
  5321 to 4848 ms (-9%) at 2048; prefill +5% / +10%; decode unchanged. The recurrence itself fell from 219 to 62 ms, but the
  matmul stages ran about 20% slower in the same pass (see the power note), so the total moved much less.
- **Power cap, burst and sustained:** with `--sample-gpu` the 40 W cap is active in 100% of busy samples in both prefill and decode
  (clocks 43 to 61% of maximum). Decode of the 4B measures about 45 tok/s in short runs after a rest (six runs: 40.4 to 45.7) and
  about 31 tok/s after sustained load; one microbenchmark process went from 214 to 154 GB/s on the same kernel between its first and
  second pass. The tables here are the sustained state; a short chat turn runs near the burst rate. The mechanism is not proven.
- **Per-shape microbenchmark** (one-token INT8 matvec and 512-token matmul at every 4B projection shape, weights cycled through 300 MB):
  in the sustained state all big projections sit within 133 to 154 GB/s (about 50 to 55% of the 277 GB/s read ceiling), and in the burst
  state within 214 to 222 GB/s except attention q, k/v, o (139 to 174; those ran later in the pass, so the order is confounded). Prefill
  runs every projection at 5.0 to 5.6 TFLOP/s, set by the exact three-part activation split (three tensor-core multiplies per tile,
  chosen in log 25 to keep the Hugging Face gates). Only the 32-row `a`/`b` matrices are outliers (6 us and 118 us per call), worth
  0.6% of a decode token and 0.5% of TTFT. No shape class is far enough below the rest to justify a per-shape kernel change (the later integer prefill kernel changed the arithmetic for all of them; see the last two fixes section).
- **CPU baseline (127-token prompt, 8 threads, INT8):** Qwen3-0.6B 237 ms TTFT and 37.6 tok/s; Qwen3-1.7B 620 to 890 ms
  (noisy on this laptop) and 12.1 tok/s; Qwen3.5-4B (512 prompt) 11.2 s and 4.7 tok/s. These match the older rows. No regression.
- **CPU low memory:** Qwen3-0.6B runs under a 2 GB memory cap; the 4B is refused under 3 GB with the reason (checked earlier in this release cycle, not rerun in this pass).

## Last two fixes and the re-run comparison (2026-10-30)

Same machine (Ryzen 7 260, RTX 5050 laptop at the 40 W cap), Qwen3.5-4B INT8 unless noted. Before is the previous release binary.

**1. The Qwen3.5 CPU session re-read.** After a session answer the server read the whole history again unless the engine was Qwen3.5 on CUDA
(`crates/cli/src/server/completion.rs`, `session.rs`). The CPU engine already had marks, so the gate now includes it; a mark that cannot be
taken on the CPU falls back to the old re-read and reports `replayed`. Qwen3-0.6B (attention only) is cut back in place (`cut_to_history`, previous
section). Regression tests: `a_hybrid_cpu_session_turn_is_not_read_again_and_matches_a_full_replay` (`tests/server_hybrid_cpu.rs`: 102 of 117
tokens re-read before, 0 after; reuse equals full replay; a retry after an abandoned request equals replay).

| Qwen3.5-4B INT8 on the CPU server, session turn 2 (3 runs) | Before | After |
|---|---|---|
| Wait after the last token | 11.8 to 13.3 s | 0.00 s |
| Whole response | 18.3 to 20.6 s | 6.3 to 7.6 s |

CPU server matrix against llama-server (Qwen3-0.6B INT8, 2 passes, 3 reps, same drivers as the audit): later turns with the prefix cache,
Marin total 6.01 s to 2.32 s (llama-server 1.83 and 2.02 s; ratio 3.28x to 1.15x); coding workflow later turns 7.25 s to 2.89 s (ratio 3.16x to
1.23x); first visible token on later turns 0.30 s to 0.38 s (llama-server 0.29 and 0.31 s, unchanged within the spread).

**2. CUDA prefill: an integer exact matmul.** Profile (`marin-run bench --profile`, serialized stages): the projections are 87% of a 512-token
prefill (MLP 54%, qkv-stage 25%, attention output 8%), DeltaNet 6%, attention 4%. Per shape on the real sizes, the 3-part exact kernel reaches 5.0
to 5.7 TFLOP/s against 6.9 to 8.0 for the 2-part one: bound by tensor-core passes. A microbenchmark on this GPU shows int8 `mma.sync` `m16n8k32` at
about 110 TOP/s against 27.5 TFLOP/s for bf16 `m16n8k16` (both in short bursts). `matmul_q8_imma` (`crates/cuda/src/mma.cu`) therefore writes each
token's activations of one quantization group as 22-bit fixed point below the group's largest magnitude, splits them into three signed int8 limbs,
multiplies each limb by the int8 weight codes with int32 sums (exact) and recombines in f32 once per group. Groups of 32, 64 and 128 use it; any
other group keeps the three-part kernel. No extra memory is allocated.

- **Numerics:** it is not bit-identical. Against an f64 product of the dequantized weights its relative RMS error is 1.7e-7 on plain data and at most
  1.1e-6 with 30,000x outliers (the two-part kernel: about 2.2e-6). The limit is that elements much smaller than the group's largest value keep
  fewer bits: at 100x the error per output is about 3e-5 absolute. HF verify on the real 4B with unchanged tolerances: PASS, max logit diff
  2.01e-5 (was 1.19e-5), full tensors max normalized 9.3e-6 (gate 1e-4) and relative RMS 5.8e-6 (was 3.0e-6; gate 1e-5, so the margin is 1.7x).
  Greedy output is byte-identical to the previous binary on four long prompts (80 and 3 x 200 tokens). Test: `crates/cuda/tests/quant_exact.rs`.
- **A/B, same session, alternating, 3 rounds of 5 runs after a warmup, median of round medians:**

| Prompt tokens | First token before | After | Change | Decode before | After |
|---|---|---|---|---|---|
| 128 | 297 ms | 252 ms | -13% | 31.5 tok/s | 31.4 |
| 512 | 1,074 ms | 1,020 ms | -7% | 31.2 | 31.3 |
| 2048 | 4,835 ms | 4,597 ms | -5% | 30.6 | 30.7 |

  Every round of the new binary is faster than every round of the old one at all three sizes. RSS is unchanged (914 MB).
- **Against llama.cpp (engine level, 2 passes with the order swapped, 6 samples; prompt time Marin / llama.cpp):** 4.28x, 4.05x, 4.13x slower
  before; 3.49x, 3.48x, 3.98x after, at 128, 512 and 2048 tokens. The 2048-token cell is 4,666 ms against 4,643 ms in the audit, so that gain
  is not visible across sessions (the long prefill sits at the power cap), only in the same-session A/B above. Decode is unchanged: 45.5 tok/s
  (1.66x, burst), 31.4 (1.16x) and 30.5 (1.15x). End to end with 128 generated tokens: 3.05 s against 4.81 s (0.63x), 5.07 against 5.06 s, 8.83
  against 6.09 s (1.45x slower).
- **Server level (CUDA 4B, 2 passes, 3 reps):** Marin's own first visible token at about 128 / 512 / 2048 prompt tokens was 0.42 / 1.47 / 6.95 s and
  is 0.45 / 1.49 / 6.33 s, all within the ranges of the earlier run. The cold first request improved from 1.42 s to 1.12 s (n=2). Against
  llama-server the first visible token is still 2.4x, 3.5x and 4.3x slower; peak host RSS 1.25 GB against 6.9 GB.
- **Not claimed:** a gain on the server matrix (it is inside the noise), an energy gain (power is at the cap in both), or accuracy equal to Q8_0.
- **Commands:** `marin-run bench ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 --prompt-tokens {128,512,2048} --tokens 64 --ctx 4096 --runs 5
  --warmup 1` for each binary, alternating, three rounds; `marin-run bench ... --profile` for the stage table; the microbenchmarks and the
  per-shape test were throw-away files (not committed).
- **Validation of the final tree:** fmt, check, Clippy `-D warnings`, line-limit and unsafe guards clean; full CUDA-required suite 846 passed, 0
  failed, 5 ignored; forced-scalar suite 846/0/5; installed-binary E2E (chat CPU and CUDA, save and resume, Ctrl-C, streaming server, session
  turn, client disconnect) passed.

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
| DeltaNet recurrence kept in registers, then parallel set-up (Qwen3.5 CUDA, final pass) | 4B prefill 328 to 465 tok/s (+42%), TTFT 1560 to about 1100 ms (-29%) at 512 tokens; decode unchanged; greedy output identical |
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
| CPU thread policy (8 vs 16 threads, prefill vs decode) | paired runs within noise (0.6B 252 vs 239 ms, 1.7B 619 vs 595 ms); decode is one core at 92% of the memory limit |
| Persistent CPU thread pool | a scoped spawn costs about 85 us and a prefill pass has about 220 regions: at most 8% of the 0.6B, 3% of the 1.7B, 0.2% of the 4B |
| CUDA launch, sync and allocation overhead, CUDA Graphs | decode is GPU-bound (29.8 of 31.8 ms waiting on the GPU); all non-kernel time is 1 to 2 ms per token |
| Faster qkv projections (CUDA decode) | they read 980 MB per token at about 175 GB/s against 206 for the MLP; matching it saves 0.85 ms (2.7%) |
| Per-shape GEMV or matmul kernel work (CUDA) | the microbenchmark above shows no outlier shape; the only real prefill lever left is a two-part activation split (7 to 8 TFLOP/s instead of 5.5 to 6.4 in log 25), which gives up exactness and has not been tried against the strict Hugging Face gates |
| Staging 32 instead of 16 columns per block-wide sync in the tensor-core matmul (CUDA prefill, same accumulation order) | Qwen3.5-4B INT8, 3 alternating rounds of 5 runs: first token 512 tokens 1,070 to 1,054 ms (-1.4%), 2,048 tokens 4,823 to 4,658 ms (-3.4%), decode unchanged; the spread between rounds is about 3%. Below the gate, so reverted. Syncs are not the bottleneck. |
| Pipelined (`cp.async`) 3-part bf16 exact GEMM, pre-split activations | not built: per shape the 3-pass kernel is 1.3 to 1.5x slower than the 2-pass one (3 against 2 tensor-core passes), so it is bound by tensor-core passes (about 15 TFLOP/s of raw tensor work, 55% of the 28 sustained ceiling), not by loads or conversion. Replaced by the int8-limb kernel below. |
| Tiered KV (offload old keys) | 37 ms per token at 8k context |

## Cross-runtime comparison with llama.cpp (independent audit, 2026-10-29)

This part is separate from the history above: it compares Marin with other runtimes on the same machine, and it includes where Marin loses.
Nothing here is an optimization result.

**What "first token" means.** Marin's `first_token_ms` in `marin-run bench` is the time inside the generate call from just before the
prompt pass to the first sampled token (prefill plus one sample). It leaves out tokenizing, the chat template, request handling and
detokenizing, so it is an engine time, not "request to first visible token" (the older tables above use it). `llama-bench` also leaves out
tokenizing and sampling. The server tables below time the client side instead: request sent to the first streamed chunk with text.

**Setup.** AMD Ryzen 7 260 (8 cores, 16 threads, AVX2 and AVX-512 with BF16 and VNNI), 16 GB RAM, RTX 5050 Laptop 8 GB (40 W cap, driver
610.57.04), Arch Linux 7.2.4, on AC power. Marin 0.1.0 (5f5b22b plus the uncommitted CUDA kernel changes above), release build.
llama.cpp b11523 (e94acad), built out of tree in Release with `GGML_NATIVE=ON`; the CUDA build uses CUDA 13.3 for architecture 120.
Threads 8 for every CPU run, context 4096, temperature 0, 128 generated tokens. Each cell is 2 passes with the runtime order swapped
(Marin first, then llama.cpp first), 1 warmup and 3 timed runs per pass, so 6 timed samples per cell; 25 s of rest between runs. The
temperature before CPU runs was 48 to 81 C (median 61 C); The spread between samples is 2 to 32% on the CPU and 0.2 to 16% on CUDA (median CV over all cells 7.5%).
Medians are over all 6 samples. Raw JSON and the drivers stay on the maintainer's machine.

**Matched and unmatched pairs.**

| Pair | Same | Different |
|---|---|---|
| BF16 vs BF16 (Qwen3-0.6B, 1.7B; CPU and CUDA) | weights, architecture | Marin does the math in f32 with an exact split of the activations; llama.cpp converts the activations to its own formats and uses AVX-512 (CPU) or its own CUDA kernels |
| Marin INT8 (groups of 128, f32 activations) vs llama.cpp Q8_0 (groups of 32, fp16 scales, Q8_0 activations) | 8-bit weights from the same checkpoint | group size, scale format and activation precision; accuracy was not compared |
| Qwen3.5-4B | Marin reads the original checkpoint; llama.cpp reads a GGUF converted from the same files with `convert_hf_to_gguf.py --outtype q8_0` | llama.cpp's own gated-DeltaNet implementation |

**Engine level** (`marin-run bench MODEL --prompt-tokens P --tokens 128 --runs 3 --warmup 1` against
`llama-bench -m GGUF -t 8 -ngl 0|99 -p P -n 0`, `-p 0 -n 128 -d P` and `-p 0 -n 0 -pg P,128`, `-r 3`). Prefill and decode are separate; decode is
at a context of P tokens; the last pair is the time for the prompt plus 128 generated tokens. Ratios are Marin divided by llama.cpp.

| model / backend | prompt | prefill tok/s: Marin | llama.cpp | ratio | decode tok/s: Marin | llama.cpp | ratio | prompt+128 gen, ms: Marin | llama.cpp | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B BF16 CPU | 128 | 430 | 1076 | 0.40x | 18.6 | 18.7 | 0.99x | 7139 | 6855 | 1.04x |
| Qwen3-0.6B BF16 CPU | 512 | 531 | 1104 | 0.48x | 17.2 | 17.2 | 1.00x | 8337 | 7829 | 1.06x |
| Qwen3-0.6B BF16 CPU | 2048 | 348 | 831 | 0.42x | 13.2 | 15.1 | 0.87x | 15353 | 11168 | 1.37x |
| Qwen3-0.6B INT8/Q8_0 CPU | 128 | 395 | 889 | 0.44x | 31.5 | 33.3 | 0.95x | 4356 | 4116 | 1.06x |
| Qwen3-0.6B INT8/Q8_0 CPU | 512 | 410 | 799 | 0.51x | 27.4 | 29.9 | 0.92x | 6009 | 5073 | 1.18x |
| Qwen3-0.6B INT8/Q8_0 CPU | 2048 | 384 | 601 | 0.64x | 17.8 | 23.6 | 0.76x | 12975 | 8848 | 1.47x |
| Qwen3-1.7B INT8/Q8_0 CPU | 128 | 165 | 290 | 0.57x | 10.7 | 12.5 | 0.85x | 12862 | 11154 | 1.15x |
| Qwen3-1.7B INT8/Q8_0 CPU | 512 | 171 | 259 | 0.66x | 10.4 | 12.1 | 0.86x | 15184 | 12212 | 1.24x |
| Qwen3-1.7B INT8/Q8_0 CPU | 2048 | 166 | 255 | 0.65x | 9.2 | 10.3 | 0.89x | 27102 | 20880 | 1.30x |
| Qwen3-1.7B BF16 CPU | 512 | 157 | 417 | 0.38x | 6.3 | 6.7 | 0.94x | 23445 | 20453 | 1.15x |
| Qwen3.5-4B INT8/Q8_0 CPU | 512 | 39 | 89 | 0.44x | 4.1 | 4.8 | 0.85x | 45147 | 31622 | 1.43x |
| Qwen3-1.7B BF16 CUDA | 128 | 1329 | 3635 | 0.37x | 66.6 | 58.9 | 1.13x | 2004 | 2190 | 0.91x |
| Qwen3-1.7B BF16 CUDA | 512 | 1411 | 4424 | 0.32x | 50.4 | 39.4 | 1.28x | 2885 | 3456 | 0.83x |
| Qwen3-1.7B BF16 CUDA | 2048 | 776 | 3257 | 0.24x | 44.5 | 36.4 | 1.22x | 5543 | 4086 | 1.36x |
| Qwen3.5-4B INT8/Q8_0 CUDA | 128 | 399 | 1707 | 0.23x | 46.5 | 27.1 | 1.72x | 3028 | 4871 | 0.62x |
| Qwen3.5-4B INT8/Q8_0 CUDA | 512 | 483 | 1958 | 0.25x | 31.2 | 26.9 | 1.16x | 5146 | 5112 | 1.01x |
| Qwen3.5-4B INT8/Q8_0 CUDA | 2048 | 441 | 1824 | 0.24x | 30.6 | 26.4 | 1.16x | 8788 | 6041 | 1.45x |

- **Prefill: llama.cpp wins everywhere**, in both run orders: 1.5 to 2.9x on the CPU and 2.7 to 4.4x on CUDA. I did not test why. Candidates
  are AVX-512 and integer dot products on the CPU, and quantized activations on CUDA, against Marin's exact f32 path (the CUDA split is
  deliberate, see log 25).
- **Decode: parity on the CPU** (0.85 to 1.00x, 0.76x at 2048 tokens on the 0.6B) **and Marin ahead on CUDA** (1.13 to 1.28x for Qwen3-1.7B BF16;
  1.16x sustained for the 4B). The 1.72x at 128 tokens is a burst: in an alternating Marin, llama.cpp, llama.cpp, Marin check with 10
  samples each, Marin ran 46.5 tok/s for its first 5 runs and then 31.3 tok/s for the rest (both Marin blocks), while llama.cpp stayed at
  26.5 to 28.5 tok/s from the first sample at lower SM clocks (about 660 against 1020 MHz). Only the sustained 1.16x is a fair headline.
- **End to end** (prompt plus 128 tokens): llama.cpp is faster on the CPU (1.04 to 1.47x), mixed on CUDA: Marin faster for short prompts
  (4B at 128 tokens: 3.0 s against 4.9 s), equal at 512, and llama.cpp faster at 2048 (6.0 s against 8.8 s).
- **Memory.** Peak RSS on CPU runs is 0 to 18% higher for Marin. On CUDA, Marin's host RSS is 0.86 to 0.91 GB against 3.7 to 4.8 GB for
  llama-bench (and 8.6 GB for llama-server), which maps the weights and keeps host copies; device-wide VRAM is about the same
  (4.4 to 4.6 GB for the 4B against 4.7 to 5.1 GB; the figure includes other programs and is read every 0.2 s).

**Servers, measured from the client** (`marin serve` against `llama-server -np 1 -c 4096 -t 8`, the same OpenAI streaming request, temperature 0;
2 passes with the order swapped, 3 repetitions per pass; for the multi-turn runs a new conversation per repetition). Cold means a new process
with the operating system's file cache still warm (dropping it needs root), so it is not a cold disk. Times in seconds, median.

| Qwen3.5-4B INT8/Q8_0, CUDA | Marin | llama-server | Marin / llama-server |
|---|---:|---:|---:|
| process start to ready | 4.6 | 3.0 | 1.52x |
| first request after start, first visible token (512 tokens) | 1.42 | 0.25 | 5.7x |
| warm, 128-token prompt: first visible token / end to end | 0.42 / 3.95 | 0.15 / 4.95 | 2.8x / 0.80x |
| warm, 512-token prompt: first visible token / end to end | 1.47 / 5.62 | 0.36 / 5.26 | 4.1x / 1.07x |
| warm, 2048-token prompt: first visible token / end to end | 6.95 / 11.20 | 1.33 / 6.29 | 5.2x / 1.78x |
| 6-turn chat on a 1.5k-token document, cache on, later turns: first visible token | 0.45 | 0.15 | 3.1x |
| the same chat, cache off, later turns: first visible token | 5.66 | 1.23 | 4.6x |
| 6-turn coding conversation over 4 repository files, later turns: first visible token / total | 1.98 / 5.06 | 0.43 / 4.22 | 4.7x / 1.20x |
| peak RSS / peak VRAM (MB, device-wide) | 1210 / 5768 | 8583 / 5674 | |

| Qwen3-0.6B INT8/Q8_0, CPU | Marin | llama-server | Marin / llama-server |
|---|---:|---:|---:|
| process start to ready / first request, first visible token | 1.86 / 1.40 | 0.61 / 0.64 | 3.1x / 2.2x |
| warm 128 / 512 / 2048 tokens: first visible token | 0.44 / 1.04 / 4.80 | 0.15 / 0.77 / 3.26 | 2.9x / 1.35x / 1.48x |
| warm 128 / 512 / 2048 tokens: end to end | 4.61 / 5.55 / 11.18 | 3.83 / 4.82 / 8.48 | 1.20x / 1.15x / 1.32x |
| chat, cache on, later turns: first visible token | 0.30 | 0.29 | 1.05x |
| chat, cache off, later turns: first visible token | 3.69 | 2.67 | 1.38x |
| chat, cache on, later turns: total | 6.01 | 1.83 | 3.3x |

- **Prefix caching is real and works in both.** In the chat, later turns re-read only about 60 to 90 tokens of a 1.5 to 1.8k history in both
  runtimes. On CUDA Marin cut the first visible token on later turns from 5.66 s to 0.45 s (12x), llama-server from 1.23 s to 0.15 s (8x).
  On the CPU the first visible token on later turns is the same in both (0.30 against 0.29 s). Marin's absolute time is still longer on
  CUDA because its prompt pass is slower, so caching helps it more and does not make it faster than llama-server.
- **Anomaly (fixed afterwards): the Marin CPU server held the stream open after the last token in session mode.** About 2.6 s in an
  isolated test and 4 to 5 s in the matrix, against 0.04 s without a session; the text was visible on time.
  **Cause:** after the answer, a session turn on a CPU engine cleared the cache and read the whole conversation history again (97 of 112
  tokens in the test model) before it sent the final chunk and `[DONE]`. The CUDA path rewinds to a saved mark and never did this.
  **Fix** (`crates/cli/src/server/completion.rs`, `cut_to_history`): an attention-only cache (Qwen3 on CPU, dense CUDA) is cut back to the
  history length in place; a cache with recurrent state (Qwen3.5 on CPU) still takes the old re-read and reports it as `replayed`.
  Same machine, Qwen3-0.6B INT8 CPU server, 8 threads, 5.2k-character document, 48 then 35 tokens (`~/.cache/marin-ab/perf/tail.py`
  pattern: client-side timing of every streamed chunk), pre-fix HEAD against the fix:

  | Session turn 2 (3 runs) | Before | After |
  |---|---|---|
  | Wait after the last token | 3.38 to 3.43 s | 0.04 to 0.05 s |
  | Whole response | 5.0 to 5.5 s | 1.8 to 1.9 s |
  | Turn 1 (session `new`) wait after the last token | 2.75 s | 0.00 s |

  Without a session nothing changed (0.04 s). Regression test: `a_session_turn_does_not_read_the_history_again_after_the_answer`
  (`crates/cli/tests/serve.rs`) failed before (`marin_replayed_tokens` 97) and passes now. Not measured: Qwen3.5 on the CPU.
  The same measurement on the clean-installed release binary (`cargo install`): wait after the last token 0.04 s on all three session
  turns, whole response 1.75 to 1.87 s. Validation of this change and the final tree: full CUDA suite 843 passed, 1 failed, 5 ignored;
  the failure was a test setup step ("the hold left 154796032 bytes free, not about 83886080", `server_cold`), and the same test passed
  when rerun alone (651 s). Forced-scalar suite 844/0/5. HF verify PASS (max logit diff 1.19e-5, full tensors relative RMS 3.0e-6,
  thresholds unchanged). fmt, check, Clippy `-D warnings`, line-limit and unsafe guards clean. The llama.cpp CUDA cells were not
  re-run: no CUDA code changed in this pass, and llama-server has no session mode to compare the tail with.
- **Ollama 0.33.3** (imported from the same GGUFs, raw prompts, `/api/generate`, 5 runs after a warmup, no alternation, so indicative only):
  Qwen3.5-4B Q8_0 on CUDA: first token 0.15 / 0.36 / 1.40 s and end to end 4.66 / 5.09 / 6.24 s at 128 / 512 / 2048 tokens, decode 26 to 28 tok/s,
  matching llama-server within a few percent. Qwen3-0.6B Q8_0 on the CPU (8 threads): first token 0.18 / 0.69 / 3.27 s, decode 26 to 36 tok/s.

**Where Marin wins, loses, and what is unverified.**
- Wins: decode on CUDA (1.13 to 1.28x for dense BF16, 1.16x sustained and up to 1.7x in bursts for the 4B); end to end for short prompts
  on CUDA; host memory on CUDA (0.9 GB against 3.7 to 8.6 GB; VRAM is similar).
- Loses (before the 2026-10-30 fixes; CUDA prefill is 3.5 to 4x slower and the CPU session tail is gone, see that section): prefill by 1.5 to 4.4x on every backend and size; first visible token on a server by 1.35 to 5.7x; end to end for 2048-token prompts
  by 1.3 to 1.8x; process start; CPU decode by 0 to 24%.
- Practical on a low-end CPU: Qwen3-0.6B INT8 decodes about 31 tok/s and 1.7B about 10 tok/s, comparable to llama.cpp; a 4B model at
  4 tok/s and 45 s for a 512-token prompt and 128 tokens is slow in both runtimes (llama.cpp: 32 s).
- Unverified: why Marin's prefill is slower; the burst and sustained GPU behaviour (observed, not
  explained); whether Q8_0 and Marin INT8 give the same answers (not measured); a cold disk; llama.cpp tuned beyond its defaults.
- The earlier README speed table and the numbers above the line are Marin-only and do not claim anything about other runtimes.

## Coding mode (experimental)

`marin code` is read-only (list, read, search). On 25 questions about this repository, Qwen3.5-4B INT8 finds the right
place 14 to 16 times (56 to 64%); the spread is one or two tasks. Misses are confident, so use it to find `path:line` and
then read the line. Two graders have drifted from the repo (`max_active` moved; `Store::evict` is gone), so compare runs
from the same commit only.

What the harness supports (`scripts/code_eval.py`, audited by reading, not rerun in the 2026-10-29 audit): a question passes when
the final answer text matches a regular-expression rubric of file names and constants. It checks that the expected names appear,
not that the claim is right or that the line is correct, so "finds the right place" really means "names the expected file or
constant". Two rubrics are stale at this commit (`t1` expects `max_active` at line 26 of `sched.rs`; it is at line 30, and there is
a second one in `gate.rs`; `a5` expects `Store::evict`, which no longer exists). Seventeen of the 25 tasks are the development set the
rubrics were written against; only 8 (`b1` to `b8`) are held out. With n=25, 14 to 16 passes has a 95% interval of about 37 to 80%
(15/25: 41 to 77%), and the held-out part alone (n=8) is too small to say anything. So the whole 56 to 64% range is within noise, and
the claim is weakly supported: it holds for "usually names the right file on questions about this repository", not for correctness.
No other model or runtime was run on this harness, so there is no coding-quality comparison.

## Gates that stay in force

- **Hugging Face check:** Qwen3.5-4B INT8 on CUDA, 128 steps: token choices match, max logit diff 1.19e-5, max layer
  diff 1.15e-5, normalized max 4.4e-6, relative RMS 3.0e-6. Limits: normalized max at most 1e-4, relative RMS at most 1e-5.
- **Same answers on reuse:** reusing a cache must give the same tokens as reading the history again (greedy).
- **Tests:** CUDA-required suite 844 passed, forced-scalar suite 844 passed, 0 failed, 5 ignored (real-checkpoint tests). The earlier 828/828 was before the awareness and kernel tests.
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
- Prefill is still 3.5 to 4x slower than llama.cpp on the CUDA 4B and the first visible token on a server 1.8 to 4.3x slower (last two fixes section). A
  heavy outlier in an activation group costs the new CUDA prefill kernel precision (below); the real-model gates pass.
- A Qwen3.5 CPU session that cannot be marked (a failed state copy) is read again after the answer and says so as `replayed`.
- Laptop GPU speed depends on how long it has been loaded: the 4B decodes at 46 tok/s for roughly the first 15 to 20 s and at 31 tok/s after that, at the 40 W cap. Quote sustained numbers.

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

# cross-runtime comparison (llama.cpp built out of tree; the drivers that alternate the order and read the JSON stay local)
llama-bench -m MODEL.gguf -t 8 -ngl 0 -p 512 -n 0 -r 3 -o json          # prefill
llama-bench -m MODEL.gguf -t 8 -ngl 0 -p 0 -n 128 -d 512 -r 3 -o json   # decode at a context of 512 tokens
llama-bench -m MODEL.gguf -t 8 -ngl 0 -p 0 -n 0 -pg 512,128 -r 3 -o json  # prompt plus 128 tokens
marin-run bench MODEL --backend cpu --threads 8 --quantize int8 --prompt-tokens 512 --tokens 128 --ctx 4096 --runs 3 --warmup 1
llama-server -m MODEL.gguf -c 4096 -t 8 -ngl 0 -np 1 --port 8201   # and: marin serve --port 8200 --model DIR --threads 8 --ctx 4096
# the GGUF for Qwen3.5-4B: python convert_hf_to_gguf.py ~/models/Qwen3.5-4B --outfile Qwen3.5-4B-Q8_0.gguf --outtype q8_0

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
