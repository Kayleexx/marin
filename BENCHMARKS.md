# Benchmarks

What the Marin engine measures on a real laptop, how each optimization changed it, and how to repeat every measurement. Each claim comes from a saved run. Numbers from different sessions are never mixed unless the text says so.

Words used below:

- **Prefill**: reading the prompt, in big passes. **TTFT**: time to the first token, mostly prefill.
- **Decode**: writing the reply, one token per pass.
- **KV cache**: the keys and values of earlier tokens, kept so attention does not recompute them. Here it is stored as f32.
- **Guessing ahead**: the opt-in speculative path. It copies likely next tokens from earlier text and checks them all in one pass.

## Setup

- Models: Qwen3-1.7B-Base for the baseline and `verify`, Qwen3-1.7B (chat) for chat, guessing and sampling runs. bf16 weights unless stated, f32 math, batch size 1.
- Machine: AMD Ryzen 7 260 (16 threads), 16 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB, 20 multiprocessors, 32 MB L2, 384 GB/s memory speed on paper), driver 610.57.04, CUDA 13.3.
- The GPU is held to a software power limit of about 40 W. During decode the driver reports the cap active in every sample.
- A browser and an editor were open during every run. They use 1.5 to 1.7 GB of GPU memory, so "peak GPU memory" below is for the whole device.

Method:

- One warmup round, then medians. Old and new builds are run interleaved (old, new, new, old) in one session.
- The GPU clock and power state differ from session to session, so only ratios inside one session mean anything. Speeds in different tables are not comparable.
- `--profile` adds one extra pass that times every stage. It is slower than a normal pass, so use it for shares of time, never for speed.
- `marin-run compare before.json after.json` calls a change "same" when the old and new min..max ranges overlap, so noise is not reported as a win. Take more `--runs` before believing a small difference.

## Results at a glance

| Change | What it did | Headline |
|---|---|---|
| Decode attention split across the context | one block per head used only 16 of 20 multiprocessors | decode at 4096 tokens +48% |
| Prefill matrix multiply on tensor cores | 5 to 8x faster matrix stages | prefill 2 to 3x faster up to 512 tokens |
| Fixed workspace, grouped launches | 339 allocations per token gone | speed unchanged |
| int8 / int4 weights | half / a third of the bytes to read | decode +65% / +91%, int8 loses almost no quality |
| Guessing ahead (opt-in) | copy repeated text, check in one pass | code edits 1.5 to 2.5x, chat unchanged |
| Shared-read attention for guessed tokens | each key/value row read once per pass | 4k edits 1.9 to 2.5x (was 1.4 to 2.3x) |
| Faster top-k / top-p sampler | 3.7 ms to 0.2 ms per token | plain decode 1 to 5% slower, guessing up to 17% faster |
| One-token attention | load more rows at once | 4k chat +13% (bf16), +17% (int8) |
| Prefill attention | fix the load pattern | 2048-token TTFT -35%, a real 3.6k prompt -40% |

The rows that did not simply get faster all have the same cause, the power cap, explained next.

## Why a faster kernel does not always mean faster decoding

The 40 W cap means the GPU trades clock speed for work. When the GPU rests between tokens, it runs the next burst at a higher clock. When it never rests, the clock settles lower. Measured with `bench --token-pause-ms N` (greedy, 128-token prompt, 3 interleaved rounds, warmed GPU):

| pause after each token | engine speed | GPU clock | GPU power | energy per token | tokens/s on the wall clock |
|---|---|---|---|---|---|
| 0 ms | 61.4 tok/s | 1383 MHz | 36.4 W | 0.78 J | 46.8 |
| 3 ms | 70.2 tok/s | 1641 MHz | 36.6 W | 0.82 J | 45.0 |
| 8 ms | 77.5 tok/s | 1890 MHz | 36.5 W | 0.95 J | 38.9 |

Power stays at the cap whatever share of time the GPU is busy, so energy per token follows time per token. Adding pauses raises the engine's own speed but lowers the real speed, so nothing is gained by idling.

Three results below are this effect:

- The 3.7 ms CPU sampler cost only 1.1 ms of each step, because the GPU rested during it (61.5 vs 56.8 tok/s for greedy vs chat sampling).
- Making the sampler 19x faster made plain decoding 1 to 5% slower. The GPU's work per token grew by about the 3.5 ms the sampler gave back.
- The faster prefill attention made the matrix stages slower (the clock fell from 2074 to 1532 MHz), which took back about 19% of its saving.

So every kernel win here is checked end to end, with the cap on.

## Starting point

First measurements of the engine before any optimization (bf16, greedy, 64 generated tokens, median of 5 runs after a warmup, `--profile --sample-gpu`):

| prompt tokens | TTFT | prefill | decode | time between tokens p50 / p95 | sampling per token |
|---|---|---|---|---|---|
| 32 | 125 ms | 264 tok/s | 49.6 tok/s | 20.0 / 21.1 ms | 3.7 ms |
| 128 | 351 ms | 369 tok/s | 55.7 tok/s | 17.7 / 19.4 ms | 3.6 ms |
| 512 | 1477 ms | 348 tok/s | 49.1 tok/s | 20.4 / 20.9 ms | 3.7 ms |

(The 32-token row ran on a cold GPU and is the least steady.) Loading takes about 2.4 s, of which copying the 3.4 GB of weights to the GPU is 0.38 s. Where the time went:

| | MLP | q/k/v projections | attention output proj | LM head | attention |
|---|---|---|---|---|---|
| prefill, 512 tokens | 64% | 14% | 7% | | 14% |
| decode, 128 tokens | 45% | 13% | 6% | 12% | 11% |
| decode, 512 tokens | 37% | 11% | 5% | 10% | 26% |

Decode reads all weights for every token, so it is limited by memory. Attention grows with the context and becomes the next cost. The CPU backend (one core only) managed 3.7 tok/s decode and 8.3 tok/s prefill, with the MLP at 64 to 74% of the time. It is the correctness reference and the fallback, and has not been optimized.

### What the GPU itself can do

`marin-run probe` measures the GPU with no model involved (4 runs of 4 s). "Sustained" is the median after the opening burst.

| | opening burst | sustained |
|---|---|---|
| memory read (1 GB buffer) | 320 to 344 GB/s | 277 to 302 GB/s |
| fp32 math | 7.0 to 11.2 TFLOP/s | 7.0 to 11.2 TFLOP/s (follows the clock) |
| bf16 tensor cores | 15 to 28 TFLOP/s | 28.1 to 28.3 TFLOP/s |

Decode weight stages already sit near the memory limit (the MLP reads 295 GB/s, the LM head 351 GB/s), so smaller weights are the lever for decode speed.

## Optimizations

### 1. Decode attention split across the context

**Problem.** Attention used one block per (token, head): 16 blocks on a 20-multiprocessor GPU, each scanning the whole context alone.

**Change.** The context is split across about 80 blocks per layer. Each block reads a key/value row once for both query heads that share it, and a small second kernel merges the pieces (exact softmax, same result up to rounding).

| context | decode before -> after | attention per token before -> after |
|---|---|---|
| 128 | 60.0 -> 59.2 tok/s (same) | 1.98 -> 1.77 ms |
| 1024 | 44.9 -> 52.7 tok/s (+17.5%) | 9.17 -> 4.64 ms |
| 2048 | 33.4 -> 41.4 tok/s (+24.2%) | 16.47 -> 9.54 ms |
| 4096 | 23.1 -> 34.2 tok/s (+48.2%) | 30.64 -> 14.14 ms |

Weight stages do not change with context, so all the slowdown is attention. At 4096 tokens the cache is 0.94 GB, about 3 ms at 300 GB/s, so attention was still about 4x above that (see section 8). Storing keys and values as 16-bit floats was tried and rejected: it fails `verify` at the existing tolerances (largest logit difference 2.8e-3 against 5e-4 allowed) and the tolerances were not loosened.

### 2. Prefill matrix multiply on tensor cores

**Change.** For passes of 16 or more tokens, a tensor-core kernel (bf16 `mma.sync`, f32 accumulation, 64 weight rows by 64 tokens per block) replaces the 32-bit multiply-add kernel. Activations are f32, so each is split into a bf16 high part and a bf16 low part and both are multiplied, because one rounding loses too much. Decode keeps the old kernel.

| prompt tokens | TTFT before -> after | prefill before -> after |
|---|---|---|
| 128 | 352 -> 127 ms | 364 -> 1009 tok/s |
| 512 | 1576 -> 519 ms | 325 -> 986 tok/s |
| 1024 | 3499 -> 1469 ms | 293 -> 697 tok/s |
| 2048 | 8531 -> 4351 ms | 240 -> 471 tok/s |

Matrix stages went from about 1.1 TFLOP/s to between 4 and 8.7 TFLOP/s. Gains shrink for long prompts because attention is not a matrix kernel here (section 9). Crossover is at 16 tokens (equal speed), and 48 tokens take 129 ms before and 51 ms after. Energy for a 2048-token prompt roughly halves (169 J to 86 J). `verify` prompts are shorter than 16 tokens and never reach this kernel, so it was checked with greedy output on five long prompts (identical to the old build) and kernel tests against the CPU code at 1e-4.

### 3. Fixed workspace, fewer launches

Three changes, each measured against the one before:

1. **Fixed workspace**: every buffer a pass needs is allocated once and reused by all 28 layers.
2. **Grouped launches**: q/k/v is one launch, gate/up is one launch (same arithmetic, same bits).
3. **Prompt pieces**: prompts longer than 512 tokens run in pieces of 512, which keeps the workspace near 55 MB. Without it peak memory at a 4096-token prompt rose from 5700 to 6100 MB.

| | before | after all three |
|---|---|---|
| device allocations per one-token pass | 339 | 0 |
| kernel launches per one-token pass | 507 | 423 |
| decode at 128 / 1024 / 4096 tokens | 61.3 / 54.2 / 39.5 tok/s | 62.1 / 54.3 / 39.6 tok/s |
| peak GPU memory at 4096 tokens | 5700 MB | 5691 MB |

Decode speed did not change beyond noise (grouped launches: +0.7% over 3 interleaved pairs). The remaining 423 launches cost about 0.55 ms of a 16 ms step. CUDA graphs were measured with a throwaway prototype and gave +1.2%, so they were not adopted; further fusion could save only a few tenths of a percent each.

### 4. Weight quantization

`--quantize int8|int4` stores the 7 projection matrices of each layer as integer codes with one f32 scale per group of 128 columns (`--quant-group`). The embedding and output head stay int8 whenever quantization is on, because a 4-bit head hurt quality badly. Codes are made at load time on the CPU (about 1.5 s extra). GPU only; the CPU backend refuses it. Quantized models cannot match the Hugging Face logits, so `marin-run quality` measures perplexity and top-token agreement instead.

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

int8 is the setting to use. int4 is clearly lossy even with small groups. At 4096 tokens of context the gain shrinks (36.6 to 49.8 tok/s for int8, 53.5 for int4) because attention still reads the same f32 cache. Peak memory at 4096 drops from 5990 MB to 4266 MB (int8), so about 1.7 GB more context fits. Prefill speed is unchanged.

Not done: calibrated 4-bit weights (GPTQ style) and per-layer widths, which is where the int4 loss could be attacked; grouping q/k/v when quantized (about 1%).

### 5. Guessing ahead (opt-in, GPU only)

**Rule.** The prompt and reply so far are indexed by every run of 8 tokens. When the last 8 tokens exactly repeat an earlier run, up to 7 tokens that followed it are guessed. One pass checks all of them and returns logits for every position. Guesses are kept while they equal the token the sampler picks, and the KV cache is rolled back over the rest. The sampler is called once per delivered token, so seeded output is identical to plain decoding. Off unless asked for (`marin-run MODEL --speculate`); `marin` never turns it on; the CPU ignores it.

**First, a measurement only** (40 prompts replayed through the guesser, no engine changes): copying only helps when the reply repeats text in view. Tokens settled per model pass, greedy, guessing after 3 matching tokens (1.00 means no gain):

| prompts | tokens settled per pass |
|---|---|
| chat, writing new code, explaining, summaries | 1.08 to 1.26 |
| whole-file code edits | 3.10 |

A longer pass costs more than a one-token pass: for bf16, 2 / 4 / 8 tokens cost 1.19 / 1.44 / 1.90x at 128 tokens of context and 1.57 / 1.91 / 2.91x at 4096 (int8 is dearer because its cheaper weights leave less time to hide the extra arithmetic). Passes of 9 tokens cost much more than 8, so guesses stop at 7.

**The match rule decides whether it is safe.** Guessing after only 3 matching tokens lost 25% on int8 fresh code and 13% on int8 chat at 4k context, because a wrong 8-token pass at 4k costs about 5 normal steps. Guessing after 8 gave up none of the edit gains. Pausing guesses after a wrong one did not help and was removed.

**In the engine** (`marin-run spec-bench`: the real `generate` loop, 12 prompts, up to 300 tokens, interleaved, 40 W cap held). Every guessing reply was identical to the plain reply with the same seed, sampled replies included.

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

- Edits gain the most; nothing is gained where nothing is copied, and the only loss is int8 chat at 4k (0.98x, about 2% more energy), from one or two wrong 8-token guesses per reply.
- The chat sampler gains a little less than greedy (the CPU sampler is a fixed cost per token).
- Overhead per token: the guesser 0.4 to 1.9 microseconds, rollback under 0.02 microseconds, against 11 to 30 ms per step. Memory under 0.4 MB plus about 25 MB of workspaces.
- Tokens settled by one pass arrive together, so the 95th percentile gap between tokens is longer where guessing fires (61 to 66 ms in 4k edits, plain decoding 26 to 36 ms). Time to first token is unchanged.
- Still limited by the kernel's own work, not memory: about 55 GB/s for the 8-token pass.

Limits: 2 timed rounds, 12 prompts, whole-file and diff edits of files from this repository; per-run ranges are wide in some rows.

### 7. Faster top-k and top-p sampling

**Problem.** The chat sampler (temperature, top-k, top-p) fully sorted all 151,936 logits for every token: about 3.7 ms. Greedy is one pass over the logits (0.15 ms) and was never affected.

**Change** (same results, no approximation):

- Top-k up to 128: one scan keeps a short sorted list of the best `k`, instead of sorting everything. Ties keep the old order (higher logit first, then the lower token id).
- No top-k, top-p below 1: only logits within 18 temperatures of the best are ranked. Anything further down weighs less than 2^-24, which cannot change an f32 sum that already holds the best token's 1.0, so every probability is bit for bit the same. If the top-p set is not complete inside that group, everything is ranked as before.
- Top-k above 128, or no top-k with top-p at 1: unchanged (full sort).

`crates/sampler/tests/exact.rs` compares the distribution with a plain full-sort reference bit for bit over many shapes (ties, repeated maxima, +0.0 and -0.0, 1 to 20,000 logits, 7 temperatures, 9 top-k values around 128, 8 top-p values) and checks the seeded token sequence draw by draw. Shrinking the cutoff or breaking the tie rule makes it fail.

**Result.** The sampler took 3.7 ms before and 0.2 ms after in every row. Interleaved real replies (chat sampler, 4 runs per side per weight type):

| scenario | plain decode bf16 / int8 | guessing bf16 / int8 |
|---|---|---|
| whole-file edit, short | -1.7% / -0.7% | +15.7% / +17.4% |
| whole-file edit, 4k | -2.3% / -3.5% | +11.3% / +14.9% |
| diff-style edit, short / 4k | -4.5, -4.8% / -1.0, -4.0% | -1.8, +5.7% / +2.0, +8.6% |
| fresh code, short / 4k | -4.0, -2.6% / -3.5, -3.8% | -4.7, +13.2% / -3.2, -2.7% |
| chat, short / 4k | -4.1, -3.5% / -2.8, -4.1% | -4.4, -3.4% / -2.1, -3.3% |

Plain decoding got 0.7 to 4.8% slower with 1 to 5% more energy per token (every row has the same sign). The GPU's busy time per token grew by 3.6 to 5.1 ms, almost exactly the 3.5 ms the sampler gave back, and the mean clock fell (BF16 chat, short: 1116 to 822 MHz) while power stayed at the cap. A guessing pass samples every token it settles, so it gains about 3.5 ms per settled token and pays the slower clock once per pass: BF16 whole-file edit, short, went from 33.5 to 28.9 ms per pass. Rows where guessing rarely fires (about 1 token per pass) are 2 to 5% slower, like plain decoding.

### 8. One-token attention

**Problem.** At 4096 tokens attention took 14 to 16 ms of a 31 ms step.

**Where the time went** (a standalone program running the shipped kernels over 28 separate layer-sized buffers so the L2 cannot help; isolated runs drift about 10%):

- 128 blocks of 128 threads (6.4 per multiprocessor), 64 registers, 3 KB of shared memory. Up to 8 blocks fit, so no resource was used up.
- Reading keys and scoring took 5.7 to 6.9 ms, reading values 3.0 to 3.6 ms, for the same bytes. Merge (0.07 to 0.09 ms), softmax and shared-memory work (0.07 to 0.13 ms) and launches (0.10 ms) are about 3% together.
- The key loop was slow: a warp held one 512-byte row in flight at a time, then did 40 shuffle-reductions for it, because the loop ran over the maximum of 8 query-head groups when this model has 2. The value loop issued one load at a time.
- Realistic floor: a plain streaming read of the same bytes reached 199 to 245 GB/s (3.8 to 4.7 ms), so with merge and launches the kernel was about 2x above the floor, not 4 to 5x.

**Change** (split decode kernel only; the 2 to 8 token kernel, prefill, cache layout, rollback and sampler are untouched): a warp scores 4 positions at a time and loads all their rows before using any, only the real groups are reduced, and 4 value rows are loaded before use. Every sum adds in the old order, so results are bit for bit the same. The attention kernels moved into their own file (a pure move).

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

**Problem.** Prefill uses the general attention kernel: one block of 128 threads per (token, head), 8,192 blocks per 512-token chunk, each block walking every earlier position. Measured again before changing anything (`bench --profile`, 3 runs after a warmup):

| bf16 prompt tokens | TTFT | attention | share of the pass | MLP | other stages |
|---|---|---|---|---|---|
| 128 | 91 ms | 14.7 ms | 16% | 48.9 ms | 27 ms |
| 512 | 537 ms | 224.7 ms | 50% | 158.4 ms | 64 ms |
| 1024 | 1337 ms | 878 ms | 65% | 334 ms | 130 ms |
| 2048 | 4404 ms | 3519 ms | 79% | 666 ms | 260 ms |

(int8 is the same to within a few percent.) Attention grows with the square of the prompt (x3.9 and x4.0 for each doubling above 512), the other stages linearly.

**Why it was slow** (standalone program, one layer of a 2048-token prefill, 113.6 ms):

- At 2048 tokens there are 0.94 billion (token, head, position) triples, about 481 GFLOP. In 3.4 s that is 0.14 TFLOP/s, about 2% of the fp32 peak, so arithmetic was not the limit. A layer's keys and values (16.8 MB) fit in the L2, and occupancy was 100%, so neither memory nor occupancy was the limit.
- Each thread scores one position by reading its own key row, so every warp load touches 32 different cache lines. Removing the value loop left 83% of the time, removing the key read left 30%, and the softmax and barriers cost 0.7 ms. Computing `expf` once per position, my first guess, changed nothing alone (1.003x). No profiler is installed here, so "limited by the load pipeline" is inferred from these variants.

**Change** (the general kernel only): keys are read four values per load (when the head size is a multiple of 4), each position's weight is computed once by the thread that scored it, and four value rows are loaded before use. Sums keep their old order, so results are bit for bit the same. Registers 40 to 47, occupancy 100% to 83%.

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

A real 3,563-token prompt through the normal path (old, new, new, old): bf16 first token 11.79 and 11.53 s -> 7.04 and 6.90 s (-40%), int8 12.36 and 12.08 s -> 7.27 and 7.01 s (-42%).

- Attention is 45 to 57% shorter from 512 tokens up, matching the 2.25x seen in isolation. At 2048 tokens the energy of the prefill falls 37 to 38%. Memory is unchanged (the kernel allocates nothing; process peak 854 to 855 MB for bf16).
- The cap takes part back: the rest of the pass slows (bf16, 2048: 902 -> 1262 ms) and the clock falls while power stays at the cap. Attention saved 1,892 ms and TTFT fell by 1,540 ms.
- Short prompts: no regression (bf16 128 is within noise, int8 128 is faster).
- Correctness: bit-identical to the old logic at 21 chunk starts; greedy replies of the real model are byte for byte identical for a short and a 3.6k-token prompt, bf16 and int8; `verify` shows the same logit differences as before (1.72e-5, 2.67e-5, 1.91e-5 against 5e-4 allowed). `compute-sanitizer --tool racecheck` finds the same 3,748 warnings and 0 errors as the old kernel (from the shared `block_max` helper); with the new barrier removed it reports 9,413 errors.
- Tests added to `attention.rs`: more than 8 tokens (the only way to reach this kernel), awkward history lengths, head sizes 10, 13, 18 and 30. A wrong weight, a dropped tail loop and the vector load for every head size each make them fail. Removing the barrier does not, which is why the race check was run.
- Still the largest stage: attention is 56% of the 2048-token pass (1,581 of 2,843 ms). Candidates, not measured: the two query heads that share keys read the same rows in separate blocks, and no key or value row is reused across tokens, which a tile-based kernel would fix.

## Tried and not adopted

- **16-bit KV cache**: fails `verify` at the existing tolerances (see section 1).
- **CUDA graphs**: +1.2% in a prototype, needs invasive state changes (section 3).
- **More kernel fusion**: a few tenths of a percent each.
- **4-bit output head**: hurts quality badly, so the head stays int8.
- **Guessing after 1 to 5 matching tokens, or pausing after a wrong guess**: slowdowns outside edits (section 5).
- **Chunk size for decode attention (32, 64, 128)**: nothing on top of the shipped change.
- **Warp-per-position prefill attention**: not bit-identical and no faster than the shipped change.
- **Idle pauses between tokens**: raise engine speed but lower real speed (power cap section).

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
# plain vs guessing on real replies (PROMPT_FOLDER holds one folder per scenario, one text file per prompt)
marin-run spec-bench MODEL --sampling chat --max-tokens 120 --prompts PROMPT_FOLDER --rounds 1
# accuracy of quantized weights
marin-run quality MODEL --quantize int8 --text TEXT_FILE
# output against Hugging Face
marin-run verify MODEL --reference tests/reference/qwen3-1.7b-base.json --backend cuda
```

Add `--quantize int8` to the first, third, fourth and sixth commands for int8 runs. The guessing replay tools are `marin-run guess`, `guess-report` and `pass-cost`. Every table above lists its sample size next to it; most speed rows are 2 to 5 runs per side on one machine, so treat differences of a few percent as noise unless the ranges are clearly apart.
