# Benchmarks

What the Marin engine measures on a laptop, how each optimization changed it, and how to repeat every measurement. Every claim comes from a saved run. Numbers from different sessions are never mixed unless the text says so.

**Contents:** [Words](#words) · [Setup and method](#setup-and-method) · [Results at a glance](#results-at-a-glance) · [The power cap](#the-power-cap) · [Starting point](#starting-point) · [Optimizations 1 to 11](#optimizations) · [Appendix: design study](#appendix-the-design-study-before-paging) · [Not adopted](#tried-and-not-adopted) · [Repeating a measurement](#repeating-a-measurement)

## Words

| Term | Meaning |
|---|---|
| Prefill | Reading the prompt, in big passes. |
| TTFT | Time to first token. Mostly prefill. |
| Decode | Writing the reply, one token per pass. |
| KV cache | Keys and values of earlier tokens, kept so attention never recomputes them. Stored as f32. |
| Guessing ahead | Opt-in speculative decoding: copy likely next tokens from earlier text and check them all in one pass. |
| Kept cache | Reusing the KV cache of the previous chat turn instead of rebuilding it (section 10). |
| Page | 64 positions of keys and values for every layer: 14 MiB for Qwen3-1.7B. The unit the cache hands out and takes back. |
| Slab | 8 pages allocated together, so a layer's keys stay contiguous across them (V4). |

## Setup and method

- **Models:** Qwen3-1.7B-Base for the baseline and `verify`; Qwen3-1.7B (chat) for chat, guessing and sampling runs. bf16 weights unless stated, f32 math, batch size 1.
- **Machine:** AMD Ryzen 7 260 (16 threads), 16 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB, 20 multiprocessors, 32 MB L2, 384 GB/s memory speed on paper), driver 610.57.04, CUDA 13.3.
- **Power limit:** the GPU is held to about 40 W by software. The driver reports the cap active in every decode sample.
- **Desktop programs:** a browser and an editor were open during every run. They use 1.5 to 1.7 GB of GPU memory, so "peak GPU memory" is for the whole device.

How runs are compared:

- One warmup round, then medians. Old and new builds run interleaved (old, new, new, old) in one session.
- GPU clock and power state differ between sessions, so only ratios inside one session mean anything. Speeds in different tables are not comparable.
- `--profile` adds one extra pass that times every stage. It is slower than a normal pass: use it for shares of time, never for speed.
- `marin-run compare before.json after.json` calls a change "same" when the old and new min..max ranges overlap, so noise is not reported as a win. Take more `--runs` before believing a small difference.

## Results at a glance

| # | Change | What it did | Headline |
|---|---|---|---|
| 1 | Decode attention split across the context | one block per head used only 16 of 20 multiprocessors | decode at 4096 tokens +48% |
| 2 | Prefill matrix multiply on tensor cores | 5 to 8x faster matrix stages | prefill 2 to 3x faster up to 512 tokens |
| 3 | Fixed workspace, grouped launches | 339 allocations per token gone | speed unchanged |
| 4 | int8 / int4 weights | half / a third of the bytes to read | decode +65% / +91%; int8 loses almost no quality |
| 5 | Guessing ahead (opt-in) | copy repeated text, check in one pass | code edits 1.5 to 2.5x, chat unchanged |
| 6 | Shared-read attention for guessed tokens | each key/value row read once per pass | 4k edits 1.9 to 2.5x (was 1.4 to 2.3x) |
| 7 | Faster top-k / top-p sampler | 3.7 ms to 0.2 ms per token | plain decode 1 to 5% slower, guessing up to 17% faster |
| 8 | One-token attention | load more rows at once | 4k chat +13% (bf16), +17% (int8) |
| 9 | Prefill attention | fix the load pattern | 2048-token TTFT -35%, a real 3.6k prompt -40% |
| 10 | Kept cache between chat turns | read only what is new in each turn | first token after 2.3k / 4.1k tokens of history 21x / 35x sooner (bf16), 23x / 37x (int8) |
| 11 | Paged KV cache | allocate KV as the sequence grows, not for the whole `--ctx` | `--ctx 8192`, 64-token chat: 117 MB of KV instead of 1.88 GB, GPU peak -1.8 GB; speed within 0 to 4% |

Rows 7 and 9 did not simply get faster. The cause is the power cap, explained next.

## The power cap

With a 40 W limit the GPU trades clock speed for work. When it rests between tokens, the next burst runs at a higher clock. When it never rests, the clock settles lower. Measured with `bench --token-pause-ms N` (greedy, 128-token prompt, 3 interleaved rounds, warmed GPU):

| pause after each token | engine speed | GPU clock | GPU power | energy per token | tokens/s on the wall clock |
|---|---|---|---|---|---|
| 0 ms | 61.4 tok/s | 1383 MHz | 36.4 W | 0.78 J | 46.8 |
| 3 ms | 70.2 tok/s | 1641 MHz | 36.6 W | 0.82 J | 45.0 |
| 8 ms | 77.5 tok/s | 1890 MHz | 36.5 W | 0.95 J | 38.9 |

Power stays at the cap whatever share of time the GPU is busy, so energy per token follows time per token. Pauses raise the engine's own speed but lower the real speed, so idling gains nothing.

Where this showed up:

- The 3.7 ms CPU sampler cost only 1.1 ms of each step, because the GPU rested during it (61.5 vs 56.8 tok/s for greedy vs chat sampling).
- Making the sampler 19x faster made plain decoding 1 to 5% slower. The GPU's work per token grew by about the 3.5 ms the sampler gave back.
- The faster prefill attention made the matrix stages slower (clock 2074 to 1532 MHz), which took back about 19% of its saving.

So every kernel win here is checked end to end, with the cap on.

## Starting point

First measurements before any optimization (bf16, greedy, 64 generated tokens, median of 5 runs after a warmup, `--profile --sample-gpu`):

| prompt tokens | TTFT | prefill | decode | time between tokens p50 / p95 | sampling per token |
|---|---|---|---|---|---|
| 32 | 125 ms | 264 tok/s | 49.6 tok/s | 20.0 / 21.1 ms | 3.7 ms |
| 128 | 351 ms | 369 tok/s | 55.7 tok/s | 17.7 / 19.4 ms | 3.6 ms |
| 512 | 1477 ms | 348 tok/s | 49.1 tok/s | 20.4 / 20.9 ms | 3.7 ms |

(The 32-token row ran on a cold GPU and is the least steady.) Loading takes about 2.4 s, of which copying the 3.4 GB of weights to the GPU is 0.38 s.

Where the time went:

| | MLP | q/k/v projections | attention output proj | LM head | attention |
|---|---|---|---|---|---|
| prefill, 512 tokens | 64% | 14% | 7% | | 14% |
| decode, 128 tokens | 45% | 13% | 6% | 12% | 11% |
| decode, 512 tokens | 37% | 11% | 5% | 10% | 26% |

Decode reads all weights for every token, so it is limited by memory. Attention grows with the context and becomes the next cost. The CPU backend (one core) manages 3.7 tok/s decode and 8.3 tok/s prefill, with the MLP at 64 to 74% of the time. It is the correctness reference and the fallback, and is not optimized.

**What the GPU itself can do** (`marin-run probe`, no model, 4 runs of 4 s; "sustained" is the median after the opening burst):

| | opening burst | sustained |
|---|---|---|
| memory read (1 GB buffer) | 320 to 344 GB/s | 277 to 302 GB/s |
| fp32 math | 7.0 to 11.2 TFLOP/s | 7.0 to 11.2 TFLOP/s (follows the clock) |
| bf16 tensor cores | 15 to 28 TFLOP/s | 28.1 to 28.3 TFLOP/s |

Decode weight stages already sit near the memory limit (MLP 295 GB/s, LM head 351 GB/s), so smaller weights are the lever for decode speed.

## Optimizations

### 1. Decode attention split across the context

**Problem.** One block per (token, head) gave 16 blocks on a 20-multiprocessor GPU, each scanning the whole context alone.
**Change.** The context is split across about 80 blocks per layer. Each block reads a key/value row once for both query heads that share it, and a small second kernel merges the pieces (exact softmax, same result up to rounding).

| context | decode before -> after | attention per token before -> after |
|---|---|---|
| 128 | 60.0 -> 59.2 tok/s (same) | 1.98 -> 1.77 ms |
| 1024 | 44.9 -> 52.7 tok/s (+17.5%) | 9.17 -> 4.64 ms |
| 2048 | 33.4 -> 41.4 tok/s (+24.2%) | 16.47 -> 9.54 ms |
| 4096 | 23.1 -> 34.2 tok/s (+48.2%) | 30.64 -> 14.14 ms |

Weight stages do not change with context, so all the slowdown is attention. At 4096 tokens the cache is 0.94 GB (about 3 ms at 300 GB/s), so attention was still about 4x above that (see section 8). A 16-bit KV cache was tried and rejected: it fails `verify` at the existing tolerances (largest logit difference 2.8e-3 against 5e-4 allowed), and the tolerances were not loosened.

### 2. Prefill matrix multiply on tensor cores

**Change.** For passes of 16 or more tokens, a tensor-core kernel (bf16 `mma.sync`, f32 accumulation, 64 weight rows by 64 tokens per block) replaces the 32-bit multiply-add kernel. Activations are f32, so each is split into a bf16 high part and a bf16 low part and both are multiplied, because one rounding loses too much. Decode keeps the old kernel.

| prompt tokens | TTFT before -> after | prefill before -> after |
|---|---|---|
| 128 | 352 -> 127 ms | 364 -> 1009 tok/s |
| 512 | 1576 -> 519 ms | 325 -> 986 tok/s |
| 1024 | 3499 -> 1469 ms | 293 -> 697 tok/s |
| 2048 | 8531 -> 4351 ms | 240 -> 471 tok/s |

Matrix stages went from about 1.1 TFLOP/s to 4 to 8.7 TFLOP/s. Gains shrink for long prompts because attention was not a matrix kernel here (section 9). Crossover is at 16 tokens (equal speed); 48 tokens take 129 ms before and 51 ms after. Energy for a 2048-token prompt roughly halves (169 J to 86 J). `verify` prompts are under 16 tokens and never reach this kernel, so it was checked with greedy output on five long prompts (identical to the old build) and kernel tests against the CPU code at 1e-4.

### 3. Fixed workspace, fewer launches

Three changes, each measured against the one before: (1) every buffer a pass needs is allocated once and reused by all 28 layers; (2) q/k/v is one launch and gate/up is one launch (same arithmetic, same bits); (3) prompts longer than 512 tokens run in pieces of 512, which keeps the workspace near 55 MB (without it, peak memory at a 4096-token prompt rose from 5700 to 6100 MB).

| | before | after all three |
|---|---|---|
| device allocations per one-token pass | 339 | 0 |
| kernel launches per one-token pass | 507 | 423 |
| decode at 128 / 1024 / 4096 tokens | 61.3 / 54.2 / 39.5 tok/s | 62.1 / 54.3 / 39.6 tok/s |
| peak GPU memory at 4096 tokens | 5700 MB | 5691 MB |

Decode speed did not change beyond noise (grouped launches: +0.7% over 3 interleaved pairs). The remaining 423 launches cost about 0.55 ms of a 16 ms step. CUDA graphs gave +1.2% in a throwaway prototype, so they were not adopted; further fusion could save only a few tenths of a percent each.

### 4. Weight quantization

`--quantize int8|int4` stores the 7 projection matrices of each layer as integer codes with one f32 scale per group of 128 columns (`--quant-group`). The embedding and output head stay int8 whenever quantization is on, because a 4-bit head hurt quality badly. Codes are made at load time on the CPU (about 1.5 s extra). GPU only; the CPU backend refuses it. Quantized models cannot match Hugging Face logits, so `marin-run quality` measures perplexity and top-token agreement instead.

Accuracy (Qwen3-1.7B-Base on 2048 tokens of Vim release notes and of this repository's Rust code; loss change is the change in perplexity; "same top" is how often the first choice equals bf16's):

| | notes: loss change / same top | code: loss change / same top | weights |
|---|---|---|---|
| bf16 | | | 3441 MB |
| int8, groups of 128 | +0.03% / 98.2% | +0.11% / 98.9% | 1774 MB |
| int4, groups of 128 | +14.6% / 84.0% | +16.6% / 85.6% | 1070 MB |
| int4, groups of 32 | +9.1% / 86.9% | +8.2% / 90.1% | 1231 MB |

Speed (decode of 128 tokens after a 128-token prompt, interleaved, 40 W cap in every run):

| | decode | peak GPU memory | energy per token |
|---|---|---|---|
| bf16 | 56.8 tok/s | 5052 MB | 0.73 J |
| int8 | 93.6 tok/s (+65%) | 3404 MB | 0.45 J |
| int4 | 108.4 tok/s (+91%) | 2712 MB | 0.39 J |

**int8 is the setting to use.** int4 is clearly lossy even with small groups. At 4096 tokens the gain shrinks (36.6 to 49.8 tok/s for int8, 53.5 for int4) because attention still reads the same f32 cache. Peak memory at 4096 drops from 5990 MB to 4266 MB (int8), so about 1.7 GB more context fits. Prefill speed is unchanged. Not done: calibrated 4-bit weights (GPTQ style), per-layer widths, grouping q/k/v when quantized (about 1%).

### 5. Guessing ahead (opt-in, GPU only)

**Rule.** The prompt and reply so far are indexed by every run of 8 tokens. When the last 8 tokens exactly repeat an earlier run, up to 7 tokens that followed it are guessed. One pass checks all of them and returns logits for every position. Guesses are kept while they equal the token the sampler picks, and the KV cache is rolled back over the rest. The sampler is called once per delivered token, so seeded output is identical to plain decoding. Off unless asked for (`marin-run MODEL --speculate`); `marin` never turns it on; the CPU ignores it.

**First, a measurement only** (40 prompts replayed through the guesser, no engine changes). Copying only helps when the reply repeats text in view. Tokens settled per model pass, greedy, guessing after 3 matching tokens (1.00 means no gain):

| prompts | tokens settled per pass |
|---|---|
| chat, writing new code, explaining, summaries | 1.08 to 1.26 |
| whole-file code edits | 3.10 |

A longer pass costs more than a one-token pass. For bf16, 2 / 4 / 8 tokens cost 1.19 / 1.44 / 1.90x at 128 tokens of context and 1.57 / 1.91 / 2.91x at 4096 (int8 is dearer because its cheaper weights leave less time to hide the extra arithmetic). Passes of 9 tokens cost much more than 8, so guesses stop at 7.

**The match rule decides whether it is safe.** Guessing after only 3 matching tokens lost 25% on int8 fresh code and 13% on int8 chat at 4k context, because a wrong 8-token pass at 4k costs about 5 normal steps. Guessing after 8 gave up none of the edit gains. Pausing guesses after a wrong one did not help and was removed.

**In the engine** (`marin-run spec-bench`: the real `generate` loop, 12 prompts, up to 300 tokens, interleaved, 40 W cap held): every guessing reply was identical to the plain reply with the same seed, sampled replies included.

### 6. Shared-read attention for guessed tokens

**Problem.** Checking 2 to 8 tokens used the prompt kernel, one block per (token, head), so every guessed token re-read the whole cache: 15 GB asked for against 0.94 GB held at 4096 tokens. Attention was 71 to 77% of an 8-token pass, and a wrong guess wasted about 61 ms against 18 ms at 128 tokens.
**Change.** For 2 to 8 new tokens, one block reads each key and value row once and applies it to every (token, head) row, with a causal mask per row, split over 64-position blocks and merged. One-token decode, prefill and the cache layout are untouched.

| tokens in the pass (bf16, 4096 context) | attention before -> after | whole pass before -> after |
|---|---|---|
| 2 | 37.4 -> 14.5 ms | 49.9 -> 35.6 ms |
| 4 | 44.6 -> 13.7 ms | 60.3 -> 44.2 ms |
| 8 | 65.3 -> 21.6 ms | 92.1 -> 60.0 ms |

Gain over plain decoding with the same sampler, greedy, old kernel -> new kernel, and energy per token against plain decoding (new kernel):

| scenario | bf16 gain | int8 gain | energy bf16 / int8 |
|---|---|---|---|
| whole-file edit, short | 2.31x -> 2.50x | 1.77x -> 1.99x | -60% / -50% |
| whole-file edit, 4k | 1.85x -> 2.46x | 1.56x -> 2.18x | -59% / -54% |
| diff-style edit, short | 1.81x -> 1.91x | 1.51x -> 1.64x | -39% / -32% |
| diff-style edit, 4k | 1.69x -> 2.13x | 1.40x -> 1.87x | -53% / -46% |
| fresh code, short / 4k | 1.03x / 1.10x | 1.06x / 1.10x | -3% / -9% (bf16) |
| chat, short / 4k | 1.01x / 1.00x | 1.00x / 0.98x | -1% / 0% (bf16) |

- Edits gain the most; nothing is gained where nothing is copied. The only loss is int8 chat at 4k (0.98x, about 2% more energy), from one or two wrong 8-token guesses per reply.
- The chat sampler gains a little less than greedy (the CPU sampler is a fixed cost per token).
- Overhead per token: the guesser 0.4 to 1.9 microseconds, rollback under 0.02 microseconds, against 11 to 30 ms per step. Memory: under 0.4 MB plus about 25 MB of workspaces.
- Tokens settled by one pass arrive together, so the 95th percentile gap between tokens is longer where guessing fires (61 to 66 ms in 4k edits, plain decoding 26 to 36 ms). Time to first token is unchanged.
- Still limited by the kernel's own work, not memory: about 55 GB/s for the 8-token pass.
- Limits: 2 timed rounds, 12 prompts, whole-file and diff edits of files from this repository; per-run ranges are wide in some rows.

### 7. Faster top-k and top-p sampling

**Problem.** The chat sampler (temperature, top-k, top-p) fully sorted all 151,936 logits for every token: about 3.7 ms. Greedy is one pass over the logits (0.15 ms) and was never affected.
**Change** (same results, no approximation):

- Top-k up to 128: one scan keeps a short sorted list of the best `k`. Ties keep the old order (higher logit first, then the lower token id).
- No top-k, top-p below 1: only logits within 18 temperatures of the best are ranked. Anything further down weighs less than 2^-24, which cannot change an f32 sum that already holds the best token's 1.0, so every probability is bit for bit the same. If the top-p set is not complete inside that group, everything is ranked as before.
- Top-k above 128, or no top-k with top-p at 1: unchanged (full sort).

`crates/sampler/tests/exact.rs` compares the distribution with a plain full-sort reference bit for bit (ties, repeated maxima, +0.0 and -0.0, 1 to 20,000 logits, 7 temperatures, 9 top-k values around 128, 8 top-p values) and checks the seeded token sequence draw by draw. Shrinking the cutoff or breaking the tie rule makes it fail.

**Result.** The sampler took 3.7 ms before and 0.2 ms after in every row. Interleaved real replies (chat sampler, 4 runs per side per weight type):

| scenario | plain decode bf16 / int8 | guessing bf16 / int8 |
|---|---|---|
| whole-file edit, short | -1.7% / -0.7% | +15.7% / +17.4% |
| whole-file edit, 4k | -2.3% / -3.5% | +11.3% / +14.9% |
| diff-style edit, short / 4k | -4.5, -4.8% / -1.0, -4.0% | -1.8, +5.7% / +2.0, +8.6% |
| fresh code, short / 4k | -4.0, -2.6% / -3.5, -3.8% | -4.7, +13.2% / -3.2, -2.7% |
| chat, short / 4k | -4.1, -3.5% / -2.8, -4.1% | -4.4, -3.4% / -2.1, -3.3% |

Plain decoding got 0.7 to 4.8% slower, with 1 to 5% more energy per token (every row has the same sign). The cause is the power cap: the GPU's busy time per token grew by 3.6 to 5.1 ms, almost exactly the 3.5 ms the sampler gave back, and the mean clock fell (bf16 chat, short: 1116 to 822 MHz) while power stayed at the cap. A guessing pass samples every token it settles, so it gains about 3.5 ms per settled token and pays the slower clock once per pass (bf16 whole-file edit, short: 33.5 to 28.9 ms per pass). Rows where guessing rarely fires (about 1 token per pass) are 2 to 5% slower, like plain decoding.

### 8. One-token attention

**Problem.** At 4096 tokens attention took 14 to 16 ms of a 31 ms step.

**Where the time went** (a standalone program running the shipped kernels over 28 separate layer-sized buffers so the L2 cannot help; isolated runs drift about 10%):

- 128 blocks of 128 threads (6.4 per multiprocessor), 64 registers, 3 KB of shared memory. Up to 8 blocks fit, so no resource ran out.
- Reading keys and scoring took 5.7 to 6.9 ms, reading values 3.0 to 3.6 ms, for the same bytes. Merge (0.07 to 0.09 ms), softmax and shared-memory work (0.07 to 0.13 ms) and launches (0.10 ms) are about 3% together.
- The key loop was slow: a warp held one 512-byte row in flight at a time, then did 40 shuffle-reductions for it, because the loop ran over the maximum of 8 query-head groups when this model has 2. The value loop issued one load at a time.
- Realistic floor: a plain streaming read of the same bytes reached 199 to 245 GB/s (3.8 to 4.7 ms), so with merge and launches the kernel was about 2x above the floor, not 4 to 5x.

**Change** (split decode kernel only). A warp scores 4 positions at a time and loads all their rows before using any; only the real groups are reduced; 4 value rows are loaded before use. Every sum adds in the old order, so results are bit for bit the same. The attention kernels moved into their own file (a pure move).

**In the engine** (`pass-cost --stages`, 4 interleaved runs per side):

| | context | attention per token | key/value bandwidth | whole pass |
|---|---|---|---|---|
| bf16 | 1024 | 6.51 -> 3.82 ms (-41%) | 36 -> 62 GB/s | 22.6 -> 21.7 ms |
| bf16 | 4096 | 15.75 -> 10.29 ms (-35%) | 60 -> 91 GB/s | 31.4 -> 28.0 ms (-11%) |
| int8 | 1024 | 5.67 -> 3.54 ms (-38%) | 41 -> 66 GB/s | 14.8 -> 14.3 ms |
| int8 | 4096 | 13.80 -> 9.31 ms (-33%) | 68 -> 101 GB/s | 22.6 -> 20.1 ms (-11%) |

**Real replies** (chat sampler, 120 tokens, 2 interleaved runs per side; both builds have the new sampler):

| scenario | bf16 tok/s | int8 tok/s | energy per token (bf16 / int8) | gap p95 plain (bf16) |
|---|---|---|---|---|
| chat, short | 50.0 -> 50.9 (+1.7%) | 86.3 -> 84.7 (-1.9%) | 0.798 -> 0.784 / 0.462 -> 0.469 J | 24.8 -> 22.0 ms |
| chat, 4k | 33.4 -> 37.8 (+13%) | 46.7 -> 54.5 (+17%) | 1.198 -> 1.057 / 0.857 -> 0.735 J | 34.7 -> 30.3 ms |
| whole-file edit, 4k, guessing | 56.5 -> 62.2 (+10%) | 67.3 -> 74.0 (+10%) | 0.707 -> 0.643 / 0.594 -> 0.541 J | 59.4 -> 59.6 ms |

The gain survives the cap: power stayed at 39.7 to 40.0 W, and every 4k run of the new build beat every run of the old. Short context is flat within noise (the int8 short result may be a real slowdown under 2%). Greedy output of the real model is byte for byte identical, and `verify` still passes.

### 9. Prefill attention

**Problem.** Prefill uses the general attention kernel: one block of 128 threads per (token, head), 8,192 blocks per 512-token chunk, each walking every earlier position. Measured before changing anything (`bench --profile`, 3 runs after a warmup):

| bf16 prompt tokens | TTFT | attention | share of the pass | MLP | other stages |
|---|---|---|---|---|---|
| 128 | 91 ms | 14.7 ms | 16% | 48.9 ms | 27 ms |
| 512 | 537 ms | 224.7 ms | 50% | 158.4 ms | 64 ms |
| 1024 | 1337 ms | 878 ms | 65% | 334 ms | 130 ms |
| 2048 | 4404 ms | 3519 ms | 79% | 666 ms | 260 ms |

(int8 is the same to within a few percent.) Attention grows with the square of the prompt (x3.9 and x4.0 per doubling above 512), the other stages linearly.

**Why it was slow** (standalone program, one layer of a 2048-token prefill, 113.6 ms):

- At 2048 tokens there are 0.94 billion (token, head, position) triples, about 481 GFLOP. In 3.4 s that is 0.14 TFLOP/s, about 2% of the fp32 peak, so arithmetic was not the limit. A layer's keys and values (16.8 MB) fit in the L2 and occupancy was 100%, so neither memory nor occupancy was the limit.
- Each thread scores one position by reading its own key row, so every warp load touches 32 different cache lines. Removing the value loop left 83% of the time, removing the key read left 30%, and softmax plus barriers cost 0.7 ms. Computing `expf` once per position (my first guess) changed nothing alone (1.003x). No profiler is installed here, so "limited by the load pipeline" is inferred from these variants.

**Change** (general kernel only). Keys are read four values per load (when the head size is a multiple of 4), each position's weight is computed once by the thread that scored it, and four value rows are loaded before use. Sums keep their old order, so results are bit for bit the same. Registers 40 to 47, occupancy 100% to 83%.

| variant (one layer, isolated) | time vs old kernel |
|---|---|
| exp once + 4 value loads | 0.90x |
| float4 keys | 0.67x |
| float4 keys + 4 value loads | 0.49x |
| **float4 keys + exp once + 4 value loads (shipped)** | **0.44x** (114.5 -> 52.1 ms) |
| warp per position with shuffles (not bit-identical) | 0.56x |

**In the engine** (`bench --prompt-tokens N --tokens 2 --runs 3 --warmup 1 --profile --sample-gpu`, old, new, new, old per row):

| | prompt tokens | TTFT | attention (profiled) | rest of the pass | GPU clock | energy |
|---|---|---|---|---|---|---|
| bf16 | 128 | 84.6 -> 85.2 ms (+0.8%) | 14.3 -> 6.0 ms | 75 -> 81 ms | 1732 -> 1672 MHz | 0.6 -> 0.6 J |
| bf16 | 512 | 504 -> 318 ms (-37%) | 228 -> 114 ms | 228 -> 354 ms | 1724 -> 1766 MHz | 6.0 -> 4.2 J |
| bf16 | 1024 | 1359 -> 1044 ms (-23%) | 910 -> 409 ms | 475 -> 611 ms | 1954 -> 1566 MHz | 24.3 -> 17.3 J |
| bf16 | 2048 | 4378 -> 2838 ms (-35%) | 3473 -> 1581 ms | 902 -> 1262 ms | 2074 -> 1532 MHz | 85.5 -> 53.2 J |
| int8 | 128 | 92.5 -> 76.6 ms (-17%) | 15.3 -> 4.9 ms | 74 -> 74 ms | 2092 -> 1830 MHz | 0.7 -> 0.6 J |
| int8 | 512 | 584 -> 326 ms (-44%) | 219 -> 119 ms | 223 -> 376 ms | 1886 -> 1800 MHz | 7.7 -> 4.8 J |
| int8 | 1024 | 1428 -> 984 ms (-31%) | 893 -> 388 ms | 474 -> 593 ms | 1928 -> 1624 MHz | 25.2 -> 17.4 J |
| int8 | 2048 | 4205 -> 2748 ms (-35%) | 3360 -> 1448 ms | 879 -> 1172 ms | 2100 -> 1538 MHz | 80.8 -> 50.5 J |

A real 3,563-token prompt through the normal path (old, new, new, old): bf16 first token 11.79 and 11.53 s -> 7.04 and 6.90 s (-40%); int8 12.36 and 12.08 s -> 7.27 and 7.01 s (-42%).

- Attention is 45 to 57% shorter from 512 tokens up, matching the 2.25x seen in isolation. At 2048 tokens the prefill's energy falls 37 to 38%. Memory is unchanged (the kernel allocates nothing; process peak 854 to 855 MB for bf16).
- The cap takes part back: the rest of the pass slows (bf16, 2048: 902 -> 1262 ms) and the clock falls while power stays at the cap. Attention saved 1,892 ms and TTFT fell by 1,540 ms.
- Short prompts: no regression (bf16 128 is within noise, int8 128 is faster).
- **Correctness:** bit-identical to the old logic at 21 chunk starts; greedy replies of the real model are byte for byte identical for a short and a 3.6k-token prompt, bf16 and int8; `verify` shows the same logit differences as before (1.72e-5, 2.67e-5, 1.91e-5 against 5e-4 allowed). `compute-sanitizer --tool racecheck` finds the same 3,748 warnings and 0 errors as the old kernel (from the shared `block_max` helper); with the new barrier removed it reports 9,413 errors.
- **Tests** added to `attention.rs`: more than 8 tokens (the only way to reach this kernel), awkward history lengths, head sizes 10, 13, 18 and 30. A wrong weight, a dropped tail loop and the vector load for every head size each make them fail. Removing the barrier does not, which is why the race check was run.
- Attention is still the largest stage: 56% of the 2048-token pass (1,581 of 2,843 ms). Candidates, not measured: the two query heads that share keys read the same rows in separate blocks, and no key or value row is reused across tokens, which a tile-based kernel would fix.

### 10. Keeping the cache between chat turns

**Problem.** `generate` emptied the KV cache at the start of every reply, so each chat turn read the whole conversation again: system message, every earlier question and answer, then the new question. A turn after 4,000 tokens of history paid a 4,000-token prefill (about 9 s here) to read one new sentence.

**Change.** The chat session keeps a record of the exact token ids the cache holds. The next prompt keeps the cached positions it shares with that record from the first token on (the longest common prefix, compared id by id), cuts the cache back there, and reads only what follows. Nothing else changes: no paging, the same contiguous f32 cache, the same kernels.

```text
cached ids : A B C D E F
new prompt : A B C D E F G H I   -> keep 6, read G H I
new prompt : A B C X Y           -> keep 3 (A B C), read X Y
```

**Rules for trusting the cache**

| Situation | What happens |
|---|---|
| New session, new model, tokenizer or `--ctx` | record is empty, whole prompt is read |
| Reply ends normally (stop token, length limit, full context, cancelled) | record = prompt + every generated token that was fed back (all after a stop token, all but the last otherwise, never the stop token) |
| Reply fails | record is cleared first, so whatever the failed pass wrote is never trusted |
| Guesses rejected | cache and record are cut back together |
| Record length differs from the cache length | nothing is kept |
| Prompt is invalid or too long | refused before anything changes |
| `/profile`, next-token view, `spec-bench` touch the cache | they go through `Session::engine_and_kv`, which clears the record |
| `bench`, `spec-bench`, `guess generate` | still read every prompt from scratch |
| Prompt equals the cache | the last token is still read (the pass needs one to produce logits) |

Qwen3's template drops the empty reasoning block from earlier answers, so a turn keeps everything up to the last `<|im_start|>assistant` marker and reads the last answer, the new question and the framing again (52 tokens in the runs below).

**Correctness**

- **Unit and CPU tests:** a fake model that reads its history back out of the cache checks exact-prefix, partial, zero and shorter prefixes, mid-history divergence, stop token, cancellation, failure, guessing on and off, and the context limit. A CPU chat test compares 6-turn chats with a full replay. Thirteen deliberate breaks (prefix match, truncation, record after a rollback or failure, forgetting in the session tools) all make a test fail.
- **GPU tests** (`crates/model/tests/reuse.rs`): a kept cache against a fresh read on bf16 and int8, greedy and sampled (temperature 0.7 and 1.2), with and without guessing ahead, across 3-turn chats, an edit mid-history, a stop token and a cancelled reply. Breaking the truncation, the guess rollback or the prefix match fails them (the guess-rollback break fails only the GPU tests, so guessing really fires in them).
- **Rounding:** a kept position was computed by a pass of a different length than a full replay uses, so logits could differ in the last bits, as with guessing ahead. No token changed anywhere we looked.
- `verify` against Hugging Face is unchanged (1.72e-5, 2.67e-5, 1.91e-5 against 5e-4 allowed).

**Results, Qwen3-1.7B chat** (`marin-run turns`, greedy, 32-token replies, `--ctx 5120`, 3 timed rounds after a warmup, kept and replay interleaved; median with min..max). The conversation is a system message plus story exchanges built from the fixed benchmark sentence, then "What does a lighthouse keeper do?", the model's reply, and "And what about at night?". The last turn is timed. "Replay" clears the record first, which is what every turn did before. "Edited" changes the first message, so only the system message is shared.

| bf16 | prompt | kept | read | TTFT keep | TTFT replay | change | energy keep / replay |
|---|---|---|---|---|---|---|---|
| short, then a second turn | 96 | 44 | 52 | 80.8 ms (80.0..81.3) | 114.1 ms (108.5..115.5) | -29% | 27.4 / 28.6 J |
| same, first message edited | 99 | 31 | 68 | 103.1 ms (100.4..103.6) | 112.8 ms (107.9..116.6) | -9% | 28.0 / 28.9 J |
| after 2,280 tokens | 2,328 | 2,276 | 52 | 158.2 ms (158.0..159.6) | 3,324 ms (3,310..3,492) | -95% (21x) | 34.9 / 161.4 J |
| same, edited | 2,331 | 31 | 2,300 | 3,292 ms (3,281..3,293) | 3,323 ms (3,306..3,326) | -1% | 160.2 / 161.7 J |
| after 4,142 tokens | 4,190 | 4,138 | 52 | 264.2 ms (250.8..272.1) | 9,293 ms (9,137..9,366) | -97% (35x) | 44.2 / 399.7 J |
| same, edited | 4,193 | 31 | 4,162 | 9,469 ms (9,399..9,758) | 9,417 ms (9,254..9,513) | +1% (overlap) | 413.6 / 410.9 J |

| int8 | prompt | kept | read | TTFT keep | TTFT replay | change | energy keep / replay |
|---|---|---|---|---|---|---|---|
| short, then a second turn | 96 | 44 | 52 | 85.7 ms (82.9..88.3) | 117.0 ms (110.7..120.5) | -27% | 17.6 / 18.5 J |
| same, first message edited | 99 | 31 | 68 | 107.2 ms (102.6..109.3) | 114.2 ms (109.6..118.4) | -6% | 18.2 / 18.7 J |
| after 2,280 tokens | 2,328 | 2,276 | 52 | 154.3 ms (152.8..157.6) | 3,505 ms (3,352..3,587) | -96% (23x) | 24.8 / 159.5 J |
| same, edited | 2,331 | 31 | 2,300 | 3,502 ms (3,394..3,532) | 3,536 ms (3,457..3,661) | -1% | 155.0 / 161.2 J |
| after 4,142 tokens | 4,190 | 4,138 | 52 | 244.4 ms (244.1..247.8) | 8,955 ms (8,954..8,962) | -97% (37x) | 34.0 / 382.2 J |
| same, edited | 4,193 | 31 | 4,162 | 8,951 ms (8,937..8,951) | 8,991 ms (8,974..9,001) | -0.4% | 381.2 / 382.7 J |

- **Replies** were identical between kept and replay in every row and every round, bf16 and int8.
- **Tokens not read:** 2,276 of 2,328 (97.8%) after 2.3k tokens of history and 4,138 of 4,190 (98.8%) after 4.1k. The 52 that are read are the last answer, the new question and the template framing; reading them over 4.1k kept positions takes about 0.25 s.
- **An edit costs nothing extra.** With the first message edited the turn reads almost everything again and takes the same time as a replay (within 1% at 2.3k and 4.1k). The short edited row is 6 to 9% faster only because the 31 shared tokens are skipped.
- **Prefill time** equals TTFT to within 0.3 ms, so the whole saving is prefill.
- **Energy** is mean GPU power times the time of the whole turn (prompt pass plus 32 reply tokens), so the kept turn still pays for its decode: 34 to 44 J against 382 to 415 J at 4.1k. The GPU sat at the cap throughout (a spot reading: 39.8 W at 1,410 MHz), so energy follows time.
- **Peak VRAM** is the device's used memory sampled between tokens, with about 1 GB of desktop programs, so it differs between runs (int8 at 4.1k read 5,001 MB, then 4,040 MB when repeated). In every pair the kept turn read the same or lower, by 0 to 76 MB, except one short int8 pair that read 339 MB lower; that is mostly desktop noise (a replay's larger prefill workspaces may explain the small gap; not checked). The cache is reserved in full at load whatever it holds, so keeping it adds no memory.
- **The 4.1k int8 pair ran twice:** the first run was cut off before its edited pair, so those rows are a repeat of that case alone. The first run's unedited rows agree: 264.9 ms against 9,745 ms (37x).

**With guessing ahead on the real model** (`turns --speculate --ask "Repeat your last answer exactly, word for word." --history 0,2048,4096 --reply-tokens 128 --rounds 2 --ctx 5120 --no-energy`). The repeat copies earlier text, so the 8-token rule fires; on a fresh answer it hardly does (0 of 7 drafted tokens right in a first try). A kept cache and a full replay, both guessing, gave identical replies in all 24 rows, bf16 and int8, with guesses accepted and rejected so the rollback ran.

| | short | 2.3k history | 4.1k history | tokens accepted / drafted |
|---|---|---|---|---|
| bf16 kept / replay TTFT | 156 / 182 ms | 370 / 3,633 ms | 777 / 9,787 ms | 12/21, 24/42, 18/28 |
| int8 kept / replay TTFT | 143 / 156 ms | 361 / 3,651 ms | 777 / 9,787 ms | 28/42, 20/42, 8/14 |

These turns read 140 to 152 tokens instead of 52 because the previous 128-token answer and the longer question are read again. This checks that tokens are equal, not what guessing gains.

**Limits.** Greedy replies of 32 tokens in the main tables, one conversation shape, 2 to 3 rounds per row. A near-tie that a rounding difference flips could still appear in a longer or sampled real reply; none did in the real-model runs or in the GPU tests on the test model with sampling and guessing ahead.

### 11. Paged KV cache (V4)

**Problem.** The cache reserved the whole `--ctx` window for every layer at load, whatever the conversation held. At `--ctx 8192` a 64-token chat reserved 1.88 GB of KV (224 KiB per token) and the GPU peaked at 6.11 GB (tables in the appendix). Keeping the cache between turns (section 10) removed repeated *compute*; paging removes repeated *reservation*. They are separate: reuse decides which positions are valid, paging decides how much memory holds them.

**Change** (GPU only; the CPU cache stays contiguous):

| Part | What it is |
|---|---|
| Page | 64 positions, keys and values of every layer: 64 x 229,376 B = 14 MiB |
| Slab | 8 pages (112 MiB) allocated at once, the first time one of its pages is needed. Inside: `[layer][keys, values][page][row][kv_head][head_dim]` |
| `PagePool` | hands out page ids; freed ids are reused first, most recently freed first; the limit is `ceil(ctx / 64)` |
| `PageTable` | one sequence's logical page -> page id; a device copy holds each page's address and changes only when a page is added |
| Lookup | position `p` -> page `p / 64` -> table -> slab `id / 8`, slot `id % 8` -> row `p % 64` |

```text
position p -> page p/64 -> table -> page id -> slab (id/8), slot (id%8) -> row p%64
```

| Event | What happens |
|---|---|
| Append `n` tokens | `write` reserves pages for `[len, len + n)` (idempotent, so every layer asks for the same ones), maps new pages into the table, and copies each page's run with one keys copy and one values copy |
| `truncate(len)` | pages beyond `ceil(len / 64)` go back to the pool, nothing is freed on the GPU, and the next growth reuses them first |
| `clear` | `truncate(0)` |
| Past the window | an error before anything is allocated, as before |
| Attention | refused unless every page below the visible length is mapped (the safety check for the table the kernels follow) |
| Cross-turn reuse | unchanged: the record decides how far to `truncate`, the cache gives back whole pages and keeps the shared start |
| Guessing ahead | unchanged: a rejected guess rolls back with `truncate`, which frees at most one page, reused by the next pass |

The three attention kernels (prefill, one-token decode, few-token) only change how they find a row; every sum adds in the old order.

**Correctness**

- Tests: 9 bookkeeping tests that need no GPU (page counts at 0, 1, 63, 64, 65 tokens, the window edge, reuse after free, truncating inside, at and across pages, two sequences sharing a pool, a randomized check) and 12 GPU tests against the f64 reference (every 64-boundary, token-by-token growth, reversed and mixed page tables, rollback cycles like guessing, a window that is not a page multiple, refusing unmapped pages).
- Replies are byte-identical to the contiguous build: 20 of 20 (greedy and sampled, bf16 and int8, five prompts from a few tokens to about 4,000). `verify` shows the same logit differences as before (1.72e-5, 2.67e-5, 1.91e-5 against 5e-4 allowed).
- Kept cache against full replay on the paged cache: 24 of 24 rows identical, and with guessing ahead 24 of 24 again (accepted of drafted per reply: bf16 12/21, 24/42, 18/28; int8 28/42, 20/42, 8/14), so rollbacks across page boundaries ran.
- 17 deliberate breaks, each caught by a failing test: wrong page, row or slot; values read from or written over the keys (in each kernel and the host write); kernel and host disagreeing on page or slab size; the layer offset; truncate keeping or freeing the wrong pages; page counts at multiples of 64; the mapped-pages safety check removed.
- 498 tests passed at the end of V4 and 512 after the V4.1 pass below, with no GPU skips; clippy with `-D warnings`, the line-limit and the unsafe checks are clean, with no lint suppressions anywhere in the code.

**What the lookup cost, and the two designs that failed first**

| Design | Result | Verdict |
|---|---|---|
| Table lookup in the prefill kernel for every group of four value rows | 2048-token prefill TTFT +9.1% (2,591 -> 2,827 ms), attention +17.4%; at 512 tokens attention +33% | Rejected: each group of four loads waited on a pointer load |
| One lookup per page of value rows; one 14 MiB allocation per page, laid out `[page][layer][keys, values]` | 512 and 2048 tokens: TTFT +0.5% and +0.2%. At **4096 tokens: TTFT +12.9%** (8,009 -> 9,039 ms), energy +13% | Rejected: at the same ~38.9 W the mean clock fell 11.6% (1,456 -> 1,287 MHz) and every stage slowed, the untouched matmuls too (MLP +10.8%) |
| Same lookup, **slabs of 8 pages laid out `[layer][keys, values][page]`** | 4096 tokens: TTFT +2.3% in the profiled A/B, +4.0% in the plain one; clock -1.9%; energy +2.5% | Kept |

What the third design changed is where a layer's keys sit: in the second a layer's keys for 4,096 positions were 64 slices of 256 KiB spread 14 MiB apart; in the third they are contiguous in runs of 512 positions. That the layout, not the lookup, was the cause rests on the clock and every stage slowing under the same power cap and on the layout change removing it; no profiler is installed here, so the memory-system mechanism is not measured. Decode and few-token attention showed no lookup cost in either layout.

**Speed against the contiguous build** (final build, A B B A, medians, 40 W cap held):

| precision | context | tokens/s base -> paged | change | p95 token gap | TTFT change | energy per token |
|---|---|---|---|---|---|---|
| bf16 | 128 | 51.09 -> 51.17 | +0.2% | 20.81 -> 20.71 ms | -0.3% | 0.76 -> 0.76 J |
| bf16 | 2048 | 44.95 -> 44.30 | -1.5% | 23.20 -> 23.73 ms | +0.3% | 1.38 -> 1.39 J |
| bf16 | 4096 | 37.12 -> 36.08 | -2.8% | 27.69 -> 28.63 ms | +4.0% | 2.70 -> 2.79 J |
| int8 | 4096 | 51.64 -> 49.91 | -3.3% | 19.92 -> 20.62 ms | +3.7% | 2.41 -> 2.49 J |

- Prefill (profiled, bf16): 512 tokens TTFT +1.2% (attention -2.4%), 2048 tokens +0.1% (attention +0.5%), 4096 tokens +2.3% (attention +3.0%).
- Chat turns on the paged cache (section 10's benchmark again): first token after 4.1k tokens of history 251 ms kept against 8,842 ms replayed in bf16 (35x) and 240 against 8,801 ms in int8 (37x), replies identical.
- **Short-context sanity check** (not a full benchmark: context 128, 1 warmup, 2 timed rounds, one INT8 tie-break): few-token attention differs by 0.2 ms or less with mixed signs (bf16 1 / 2 / 4 / 8 tokens: 1.38 -> 1.31, 3.15 -> 3.04, 2.63 -> 2.73, 4.00 -> 4.13 ms) and whole bf16 passes are within +2.8% to -0.4%. INT8 first read -9.2% and +10.7% on the 1- and 2-token passes, the reversed tie-break +3.4% and -6.1%, and the two-run means -3.1%, +2.0%, +0.9%, -1.1%: noise. An earlier run on the second layout agreed (attention within -3.4% to +3.4%, one -11.6% row at 128 tokens). A longer four-run comparison was stopped for taking too long; its partial output is not used.
- Decode at 2 to 4k is 1.5 to 3.3% slower with a slightly lower clock. This build allocates a slab of 112 MiB every 512 tokens during decode; the plain benchmark reports p95 only, so the allocation spikes themselves were not measured.

**Memory** (`bench --ctx 8192`, GPU peak is device-wide with about 1 GB of desktop programs; live positions = prompt + 1):

| live positions | pages used | KV held | bf16 peak, paged / contiguous | int8 peak, paged / contiguous |
|---|---|---|---|---|
| 64 | 1 | 117 MB | 4.34 / 6.11 GB | 2.66 / 4.44 GB |
| 65 | 2 | 117 MB | 4.34 | 2.66 |
| 512 | 8 | 117 MB | 4.37 | 2.73 |
| 2,048 | 32 | 470 MB | 4.74 | 3.06 |
| 4,096 | 64 | 940 MB | 5.21 / 6.15 | 3.53 / 4.47 |
| 8,000 | 125 (of 128) | 1.88 GB | 6.15 / 6.11 | 4.47 / 4.47 |

KV memory now follows the live tokens, rounded up to a slab. Pages used x 14.7 MB matches the formula (64 pages = 940 MB, the same as 4,096 x 224 KiB). At 64 live tokens the cache holds 117 MB, not 1.88 GB, and the peak drops 1.8 GB. At 8,000 of 8,192 the two builds are equal, as they should be.

**Waste, honestly.** Two layers of rounding: the last page may be nearly empty (up to 63 positions, 14.4 MiB) and the last slab may hold up to 7 unused pages (103 MiB). A 65-token chat uses 2 pages (29 MB) inside a 117 MB slab, so 75% of what is held is unused; at 8,000 tokens 125 of 128 pages are used. Slabs of 8 were chosen because the one-allocation-per-page layout cost +13% at 4k; slab sizes between 1 and 8 were not tried.

**Slab-boundary token latency.** A slab is allocated when the sequence first needs one of its pages, so one decode step in 512 allocates. `bench --json` records every token's latency; a fresh process per row, three rows each. Run 1 allocates the next slab at the boundary; run 2 in the same process reuses it, so the same token position is compared with and without the allocation. Token gaps in ms (prompt 1000 for the 1024 rows, where the GPU is already warm after a long prefill, and 480 for the others):

| precision | boundary | token at the boundary, run 1 (allocates) | same token, run 2 | its 12 neighbours, run 1 (median) |
|---|---|---|---|---|
| bf16 | position 1024 | 15.6, 15.6, 16.6 | 17.1, 18.4, 20.0 | 15.7 |
| int8 | position 1024 | 10.8, 10.7, 11.0 | 10.2, 10.3, 10.4 | 10.2 |
| int8 | position 512 | 9.6, 12.2, 9.8 | 8.6, 9.6, 9.3 | 9.5 |
| int8 | page boundary 576, no allocation | 9.1, 9.0, 9.1 | 12.2, 10.4, 10.4 | 9.3 |

The allocating token costs the same as its neighbours, within about 1 ms, and the worst single sample was +2.7 ms (int8 at 512). BF16 and INT8 do not differ in a way that matters. The allocation itself was not timed (it is asynchronous, and a 112 MiB memset is about 0.4 ms at the 300 GB/s `probe` measured), so "harmless" here rests on the token gaps. One row is excluded: bf16 with a 480-token prompt, boundary 512, where every token near it is ~40 ms against 16 in run 2. That slow stretch starts at tokens 23 to 26, before the allocation at token 32, in all three processes, and run 2 never shows it, so it is the cold GPU settling under the 40 W cap, not the allocation. No fix is needed.

**What `--ctx` and the startup check mean now.** Before, the load-time check asked whether the weights plus the KV for the whole `--ctx` window plus 512 MiB fit. With a 32,768-token window that is 11.5 GB: `--backend cuda` refused (7,277 MB free), and `--backend auto` quietly moved to the CPU, which refused too (11.2 GB needed, 9.5 GB available). Now:

- `--ctx` is the logical maximum length, not a reservation.
- The GPU must have room for the weights, the first slab (112 MiB), the page table and the split-attention scratch (about 9 MB at `--ctx` 8192, 34 MB at 32768), plus the 512 MiB slack. If not, it is refused (`cuda`) or the CPU is used (`auto`), as before.
- If the whole window would not fit, the GPU is still used and a note says so: `note: the GPU has 7277 MB free but the whole --ctx window needs about 11494 MB: a conversation that fills it will stop with an out-of-memory error`. Nothing is silently overcommitted.
- The CPU path still reserves and checks the whole window. `inspect` shows both figures ("up to X for this context, Y before the first token on the GPU").

Checked on the real model: `--ctx 32768` with a short prompt answered on the GPU in both `cuda` and `auto`; the old binary refused both.

**When the GPU cannot allocate another slab.** The allocation happens in the first layer's write of a pass, before anything of that pass is written. A failure returns `CudaError::KvMemory { held, needed, positions }` and then:

- the length does not advance, the page table and the device table are unchanged (the pages taken for the attempt are returned), and no page is counted as held that has no storage; a second attempt fails the same way;
- attention refuses the page that was never mapped; what is cached still reads correctly, and the cache can be cleared and refilled inside the slabs it has;
- a failed run leaves the cross-turn record empty (it is taken at the start of every run), and so does a failure in the middle of a guessing run: the next turn reads from scratch and equals a fresh read;
- the CLI says `error: the GPU could not allocate 117440512 more bytes for the KV cache, which holds 3640655872 bytes for 15872 positions: ...`, with the hint to close other programs, start a new chat or use `--backend cpu`.

Tested with a slab larger than half the GPU, so the first fits and a second never can on any GPU, with nothing depending on how much is free (a first version that filled the GPU by hand was flaky, because the driver refuses very large allocations and small ones near the limit on its own, and was replaced). Two deliberate breaks, removing the rollback and counting handed-out ids as held pages, are each caught. End to end, a ~19,000-token prompt at `--ctx 32768` on the real model stopped after 31 slabs (15,872 positions, 3.64 GB held) with the error above and exit code 1, no panic, and the GPU memory was released when the process ended.

**Freed pages are reused.** A rollback returns whole pages to the pool, and regrowing takes them first: 600 tokens (10 pages, 2 slabs), truncate to 100, regrow to 600 made 0 GPU allocations (counted), kept the same bytes and the same 10 physical pages, and read correctly. Slabs are kept until the session ends, as under "Limits".

**Boundary coverage** (tests): page and slab boundaries (63/64/65, 511/512/513, 1023/1024/1025) across every kernel; truncating exactly on a page boundary and across a slab; rejected guesses across a page and across a slab; cross-turn reuse after a truncate; a window that is not a multiple of 64, smaller than a page, filled exactly and one token past it; a slab that cannot be allocated and a failure in the middle of a guessing run.

**Limits**

- Memory is not returned to the driver while a session lives: freed pages go back to the pool and the slabs stay allocated (a high-water mark).
- The smallest cache is one 117 MB slab, even for a window under 512 tokens.
- The split-attention scratch is still sized for `--ctx` (about 1 KB per position, 8 MB at 8192).
- Prefill at 4k costs +2 to +4% TTFT and decode at 2 to 4k 1.5 to 3.3%. That is the price of the memory saving above.
- Not tested: windows over 8,192, other models, more than one live sequence on the GPU (the pool and tables support it, nothing uses it yet).

**New bottleneck.** KV is no longer the memory floor: at 64 live tokens the GPU holds the weights (3.44 GB bf16, 1.77 GB int8) plus a slab. Prefill attention is still 72% of a 4096-token prefill and one-token attention 32% of a decode step.

**What it unlocks (V5, V6, V7)**

- **V5, tiering.** A page is the unit that moves between VRAM, pinned RAM and disk (512 KiB per layer is a sensible copy size). Decoding one token reads every position (0.94 GB per token at 4096 tokens; the only PCIe figure so far is a small 608 KB copy at 6.3 GB/s), so a cold page of the *active* sequence is needed every step. Tiering pays for parked sessions, shared prefixes and cold history, not the sequence being decoded now.
- **V6, prefix reuse.** Sequences share leading pages by reference count; only a diverging partial page is copied or recomputed (at most 63 tokens). Sharing is page-granular: a 3,500-token system prompt shares 54 of its 55 pages. A hash of each full page's tokens finds a reusable page.
- **V7, batching.** One pool serves many sequences: a table per sequence, admission control is the number of free pages, and a sequence no longer reserves its worst case.
- **Quantized KV** fits naturally: a scale per page and head can live in the page.

**In plain words**

While the model writes, it keeps a notebook of everything it has read. Before, Marin bought a notebook big enough for the longest conversation you allow before the first word (1.88 GB at `--ctx 8192`, even for "hello"). Now the notebook is cut into sheets of 64 words, kept in binders of 8 sheets, and a new binder is only taken out when the conversation needs one. Each conversation keeps an index card ("words 0 to 63 are on sheet 17, words 64 to 127 on sheet 3"), and attention reads the sheets through it. Taking words back (a rejected guess, or a changed message) hands whole sheets back to be used next. A short chat in a big window now holds one binder, 117 MB, instead of 1.88 GB; the price is 0 to 4% in speed at long contexts, and sheets are not given back to the GPU while the session lives. Keeping the notebook between turns (section 10) and paging the notebook are different things: one saves reading, the other saves space.

## Tried and not adopted

- **16-bit KV cache:** fails `verify` at the existing tolerances (section 1).
- **CUDA graphs:** +1.2% in a prototype, needs invasive state changes (section 3).
- **More kernel fusion:** a few tenths of a percent each.
- **4-bit output head:** hurts quality badly, so the head stays int8.
- **Guessing after 1 to 5 matching tokens, or pausing after a wrong guess:** slowdowns outside edits (section 5).
- **Chunk size for decode attention (32, 64, 128):** nothing on top of the shipped change.
- **Warp-per-position prefill attention:** not bit-identical and no faster than the shipped change.
- **Idle pauses between tokens:** raise engine speed but lower real speed (power cap section).
- **Table lookup for every group of four value rows in prefill:** +9.1% TTFT at 2048 tokens (V4).
- **One 14 MiB allocation per page, laid out `[page][layer][keys, values]`:** +12.9% TTFT at 4096 tokens, the clock down 11.6% at the same power (V4).

## Repeating a measurement

Save a result before a change and another after it with identical options, then `marin-run compare before.json after.json`. Build one binary before and one after, and alternate them.

```bash
# baseline: speed, stage shares, GPU load, power, memory
marin-run bench MODEL --backend cuda --prompt-tokens 128 --tokens 64 --runs 5 --profile --sample-gpu --json run.json
# what the GPU itself can do
marin-run probe
# decode attention and cost of multi-token passes, stage by stage
marin-run pass-cost MODEL --text LONG_TEXT_FILE --contexts 128,1024,2048,4096 --sizes 1,2,4,7,8 --stages
# prefill speed and attention share
marin-run bench MODEL --backend cuda --prompt-tokens 2048 --tokens 2 --runs 3 --warmup 1 --profile --sample-gpu --ctx 4096
# a real long prompt
marin-run MODEL -p LONG_PROMPT --temp 0 --ctx 5120 --max-tokens 8 --metrics
# chat turns on a kept cache vs read from scratch (add --quantize int8 for int8)
marin-run turns MODEL --history 0,2048,4096 --reply-tokens 32 --rounds 3 --ctx 5120
# the same with guessing ahead, asking for copied text so guessing fires
marin-run turns MODEL --speculate --ask "Repeat your last answer exactly, word for word." --history 0,2048,4096 --reply-tokens 128 --rounds 2 --ctx 5120
# KV memory held for a short live sequence in a large window (prompt 63 = 64 live positions; try 64, 511, 2047, 4095, 7999)
marin-run bench MODEL --backend cuda --prompt-tokens 63 --tokens 2 --runs 1 --warmup 0 --ctx 8192
# plain vs guessing on real replies (PROMPT_FOLDER holds one folder per scenario, one text file per prompt)
marin-run spec-bench MODEL --sampling chat --max-tokens 120 --prompts PROMPT_FOLDER --rounds 1
# accuracy of quantized weights
marin-run quality MODEL --quantize int8 --text TEXT_FILE
# output against Hugging Face
marin-run verify MODEL --reference tests/reference/qwen3-1.7b-base.json --backend cuda
```

Add `--quantize int8` to the first, third, fourth and sixth commands for int8 runs. The guessing replay tools are `marin-run guess`, `guess-report` and `pass-cost`. Every table lists its sample size next to it; most speed rows are 2 to 5 runs per side on one machine, so treat differences of a few percent as noise unless the ranges are clearly apart.

## Appendix: the design study before paging

The study done before building: how the cache worked, what it cost, and the plan. The measurements came from throwaway probes (the real `DeviceKv` and a standalone kernel harness) that are not in the repo.

### How the cache worked

- One key and one value buffer per layer, `[position][kv_head][head_dim]` in f32, all reserved at load (56 buffers plus 2 scratch): 229,376 B (224 KiB) per token for Qwen3-1.7B (2 x 28 x 8 x 128 x 4). The window (`--ctx`) was fixed for the cache's life; going past it was an error.
- A pass wrote its new keys and values at `[len, len + n)` (one copy per layer for keys and one for values) and committed them with `advance`. Rollback only set the length; nothing moved.
- Everything that assumed "physical row = logical position" was one expression, `position * kv_stride + kv_head * head_dim`, in the three attention kernels and in the cache write.
- `EngineCache` (the CPU `KvCache` or `Box<GpuCache>`) lives in the CLI `Session`; the `SequenceCache` trait (`used`, `capacity`, `clear`, `truncate`) is all `generate` sees. The CPU cache hands out one slice per layer, so it stays contiguous.

### What it cost

`DeviceKv` created directly, 4 repeated runs (identical each time). The driver reports free memory in 32 MiB steps, so "measured" rounds up to the next 32 MiB:

| context | formula (KV) | + scratch | measured | create + zero-fill | used by a 64-token prompt | reserved but unused |
|---|---|---|---|---|---|---|
| 128 | 28.0 MiB | 28.1 MiB | 32 MiB | 3.8 ms | 14 MiB | 50.0% |
| 1024 | 224 MiB | 225.0 MiB | 256 MiB | 1.0 ms | 14 MiB | 93.8% |
| 4096 | 896 MiB | 900.1 MiB | 928 MiB | 3.6 ms | 14 MiB | 98.4% |
| 8192 | 1792 MiB | 1800.1 MiB | 1824 MiB | 9.9 ms | 14 MiB | 99.2% |
| 16384 | 3584 MiB | 3600.2 MiB | 3616 MiB | 14.8 ms | 14 MiB | 99.6% |
| 32768 | 7168 MiB | | out of memory | | | |

- The formula holds. Engine-level growth matches it too: bf16 from 1024 to 4096 grew 0.71 GB (predicted 0.705), 4096 to 8192 grew 0.94 GB (0.94); int8 8192 to 16384 grew 1.88 GB (1.88).
- Decode and rollback make **0** allocations and **0** growth copies (2,000 token writes: 0 allocations; growing past capacity is refused with no allocation).
- Rollback (`truncate` plus `advance`) takes **1.7 ns**, with 0 launches and 0 bytes moved. In real guessing runs the whole rollback is under 0.02 microseconds per token.
- Writing one decode token's cache entry takes 123 microseconds (56 copies, 229,376 B): the 0.15 ms "kv cache write" stage of the decode profile.

Peak GPU memory (`bench`, a 64-token prompt, whole device including about 1.5 GB of desktop programs):

| context | KV reserved | bf16 grew / peak | int8 grew / peak |
|---|---|---|---|
| 128 | 29 MB | 3.49 / 5.13 GB | 1.81 / 3.49 GB |
| 1024 | 235 MB | 3.69 / 5.34 GB | 2.01 / 3.69 GB |
| 4096 | 940 MB | 4.40 / 6.04 GB | 2.75 / 4.40 GB |
| 8192 | 1.88 GB | 5.34 / 7.01 GB | 3.69 / 5.34 GB |
| 16384 | 3.76 GB | not tried (weights 3.44 GB + cache 3.76 GB = 7.2 GB before the desktop's 1.5 GB) | 5.57 / 7.22 GB |

Largest context that fits comfortably on this 8 GB card: **8192 for bf16** (peak 7.0 GB, 86% of the card) and **16384 for int8** (7.2 GB, 89%). The cache is the whole difference, and it is reserved in full at the start whatever the prompt length: a typical default chat (4096, 940 MB reserved) holding 1,000 tokens uses 229 MB, about a quarter of it.

### Does the layout slow decoding down?

A standalone copy of the shipped decode kernel where only the address of a key or value row changes (28 layers, 4096 tokens of context, 7 paired rounds against the real kernel; every layout gives bit-identical output):

| layout | time vs today |
|---|---|
| templated copy of today's layout (control) | 1.01x |
| head-major, `[head][position][dim]`, contiguous | **1.00x** |
| paged, shuffled physical pages, every row looks its page up | 1.04x to 1.06x (any page size, either layout inside the page) |
| paged, 256-token pages, page read once per block | 1.01x |
| paged, 64-token pages, block = page (chunk 64), page read once | 1.03x (its own contiguous chunk-64 control is also 1.02x) |
| paged, 64-token pages, 256-token block reading its 4 page ids into a small array | 1.09x (the array spills to local memory) |

- The order of data in memory is not a lever: head-major is no faster, so there is no reason to change the layout for speed.
- Scattered pages cost almost nothing by themselves. The 4 to 6% comes from looking the table up for every row. With one lookup per block the cost is about 1% of the attention kernel: about 0.1 ms of a 28 ms step (0.4%) at 4096 tokens.
- A block's positions must all lie in one page for the single lookup to be valid (a 256-token page with the 103-token blocks of a 1024-token context straddles pages and gave a wrong answer, which the harness flags). Block size is a separate knob: at 1024 tokens the paged chunk-64 variant was 5% faster than the shipped chunk-103 kernel, and at 4096 the contiguous chunk-64 control was 2% slower than chunk 256. Neither was tuned further.

### Four options

| | A. keep contiguous | B. paged, page table | C. hybrid | D. contiguous virtual range, physical memory mapped on demand |
|---|---|---|---|---|
| Idea | today | fixed-size pages in a pool, a table maps position to page | contiguous when small, paged when large | reserve a big virtual range, map physical chunks as the sequence grows (CUDA virtual memory) |
| Complexity | none | moderate | highest: two layouts, two kernel variants, a switch | moderate, CUDA-only (needs `unsafe` driver calls in the `cuda` crate) |
| Memory overhead | whole capacity up front | table (1 KiB per 16K tokens) plus at most 63 wasted tokens (14 MiB) per sequence | both | physical memory rounded to 2 MiB chunks per layer buffer (512 tokens of keys) |
| Runtime overhead | none | about 1% of attention with one lookup per block (4 to 6% if naive) | the paged cost beyond the switch | none: kernels unchanged |
| CUDA changes | none | 3 attention kernels read the table, the write is split per page | all of B, twice | none in kernels; new allocator in `cuda` |
| Unlocks | prefix reuse for one sequence (done, section 10) | sharing pages between sequences, page-sized tiering, batching without N x capacity | nothing B lacks | lazy growth and mapping a chunk into two sequences, but no page-sized tiering or eviction |

Supported here: the check program shows virtual memory management available, a 2 MiB minimum granularity, and two separate physical chunks mapped into one contiguous range.

### The plan: B, in small steps, with 64-token pages

Not C (more code than B for nothing B lacks). Not D alone (solves growth but not sharing or eviction, and ties the cache to one backend). A is right until V5 starts.

Where the build differed: pages are allocated in slabs of 8 and laid out per layer (not one allocation per page, which cost +13% at 4k); there is no identity-table stage on the GPU, because equality with the contiguous build was shown by byte-identical replies and a bookkeeping model instead; the pool grows on demand rather than starting as large as `--ctx`; and the prefill kernel does one lookup per page of rows, not a table staged in shared memory.

- **Page.** 64 tokens, all layers: 64 x 229,376 B = **14 MiB** (512 KiB per layer for keys plus values). It matches the shared-read kernel's 64-position block. Inside a page, positions stay in today's order, `[page][64][kv_head][head_dim]`, so writing a run of tokens is still a plain copy per page run.
- **Pool.** Per layer, one key slab and one value slab of `pool_pages x 64 x stride` floats (the same 56 allocations). A page id means the same slot in every layer, so one table serves all layers. At first the pool is as large as `--ctx` allows, so peak memory does not change.
- **Sequence.** A page table (`Vec<u32>` of page ids plus a length) owned by the cache. A device copy (4 B per page: 256 entries at 16K tokens) is refreshed when a page is added, once per 64 decoded tokens.
- **Allocate and free.** A free list of page ids. A page is taken when the length crosses a multiple of 64 and returned when `truncate` drops below it. Every page carries a reference count (always 1 for now) so sharing is possible later without redesign.
- **Rollback** stays O(1) in the common case (set the length, give back whole pages beyond it: at most one page for an 8-token pass). **Growth** appends a page id and never copies data. Running out of pages is the same `ContextFull` error.
- **Kernels.** One lookup per block. Decode: block = page (chunk 64). Shared-read: already 64. Prefill (thread per position): stage the page ids in shared memory once per block. The prefill kernel's lookup cost is **not measured yet** and must be before it ships.
- **Prefill write.** A 512-token chunk becomes 8 or 9 copies per layer for keys and for values instead of 1 (about 500 small copies per chunk, an estimated few ms against a 0.3 to 0.5 s chunk). A scatter kernel can replace them if it matters.
- **CPU** is unchanged. `SequenceCache` is the seam: the paged cache implements it for the GPU and the contiguous `KvCache` stays for the CPU fallback.
- **Order of work.** (1) Introduce `PagePool` and `PageTable` behind `SequenceCache` with an identity table, so nothing changes in the kernels and the exactness tests prove it. (2) Make the kernels read the table; test with shuffled pages that results are bit-identical to the contiguous cache. (3) Free pages on rollback and reuse them. (4) Optionally back the pool with mapped virtual memory so it grows with the sequence.
- **Risks to measure first:** the prefill kernel's lookup cost; chunk 64 against the current chunk at long contexts; the table upload size relative to a decode step.
