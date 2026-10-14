# Benchmarks

What Marin measures on a laptop, what each optimization changed, and how to repeat every number. Every claim comes from a saved run; numbers from different sessions are never mixed unless the text says so.

**Contents:** [Method](#method) · [Results at a glance](#results-at-a-glance) · [The power cap](#the-power-cap) · [Starting point](#starting-point) · [Optimizations 1–25](#optimizations) · [Not adopted](#tried-and-not-adopted) · [Repeating a measurement](#repeating-a-measurement)

## Method

**Terms**

| Term | Meaning |
|---|---|
| Prefill | Reading the prompt, in big passes. |
| TTFT | Time to first token. Mostly prefill. |
| Decode | Writing the reply, one token per pass. |
| KV cache | Keys and values of earlier tokens, kept so attention never recomputes them. |
| Guessing ahead | Opt-in speculative decoding: copy likely next tokens from earlier text and check them in one pass. |
| Kept cache | Reusing the previous chat turn's KV cache instead of rebuilding it (section 10). |
| Page / slab | 64 positions of KV for every layer (14 MiB for Qwen3-1.7B) / 8 pages allocated together (section 11). |

**Setup**

- **Models:** Qwen3-1.7B-Base for the baseline and `verify`; Qwen3-1.7B (chat) for chat, guessing and sampling; Qwen3-0.6B where stated. bf16 weights unless stated, f32 math, batch 1.
- **GPU machine:** AMD Ryzen 7 260 (16 threads), 16 GB RAM, RTX 5060 Laptop GPU (8 GB, 20 multiprocessors, 384 GB/s on paper), CUDA 13.3. The GPU is held to about **40 W** by software. A browser and an editor were open during every run (1.5–1.7 GB of GPU memory), so "peak GPU memory" is device-wide.
- **CPU runs (sections 21–22):** the same Ryzen 7 260 (Zen 4, 8 cores, AVX2 and AVX-512), one thread, 6.5 GB of RAM free.

**Comparing runs**

- One warm-up round, then medians. Old and new builds run interleaved (old, new, new, old) in one session.
- GPU clock and power differ between sessions, so only ratios inside one session mean anything.
- `--profile` adds one slower pass that times every stage: use it for shares of time, never for speed.
- `marin-run compare before.json after.json` calls a change "same" when the min..max ranges overlap. Differences of a few percent are noise unless the ranges are clearly apart.
- Each optimization section follows the same order: problem, change, result, verdict. Investigations that did not ship say so in the title.

## Results at a glance

| # | Change | Headline |
|---|---|---|
| 1 | Decode attention split across the context | decode at 4,096 tokens +48% |
| 2 | Prefill matmul on tensor cores | prefill 2–3x faster up to 512 tokens |
| 3 | Fixed workspace, grouped launches | 339 allocations per token gone; speed unchanged |
| 4 | int8 / int4 weights | decode +65% / +91%; int8 loses almost no quality |
| 5 | Guessing ahead (opt-in) | code edits 1.5–2.5x, chat unchanged |
| 6 | Shared-read attention for guessed tokens | 4k edits 1.9–2.5x (was 1.4–2.3x) |
| 7 | Faster top-k / top-p sampler | 3.7 ms → 0.2 ms per token; plain decode 1–5% slower (power cap) |
| 8 | One-token attention | 4k chat +13% (bf16), +17% (int8) |
| 9 | Prefill attention | 2,048-token TTFT −35%; a real 3.6k prompt −40% |
| 10 | Kept cache between chat turns | first token after 4.1k tokens of history 35x (bf16) / 37x (int8) sooner |
| 11 | Paged KV cache | `--ctx 8192`, 64-token chat: 117 MB of KV instead of 1.88 GB; speed within 0–4% |
| 12 | f16 KV cache (default of `marin` on the GPU) | KV memory halved, 32k context fits (f32: 15.9k); first token about half at 8k–16k |
| 13 | Prefix snapshot (`--snapshot FILE`) | restart to first token 3.4x sooner at 4k (f16), 12x at 8k (f32) |
| 14 | Parking idle sessions in RAM (not built) | 40–200x faster than re-reading, but Marin has one session: no-go |
| 15 | Draft-model speculation (not built) | +17% on bf16 code, −15–25% at long context, slower on int8: no-go |
| 16 | `marin --quantize int8` (option) | decode +33–63%, 1.7 GB less GPU memory; wording differs from bf16; bf16 stays default |
| 17 | Mixed precision, int8 KV values (not adopted) | ≤0.2 points of quality for 7–9% speed; no decode gain: no-go |
| 18 | Tensor-core prefill attention (f16 cache) | first token −34 / −49 / −65 / −77% at 511 / 2k / 4k / 8k tokens |
| 19 | Prefill matmul rewrite (reverted) | −12% first token, below the 15% gate |
| 20 | Decode attention with 16-byte loads | decode +13 / +34 / +65 / +95% (bf16) at 2k / 4k / 8k / 16k tokens |
| 21 | CPU inference (investigation) | Marin CPU: 1 core, prefill 26 and decode 11 tok/s vs llama.cpp 1,061 and 19.8 (8 threads) |
| 22 | CPU matmul kernels, stage 1 (one thread) | prefill 3.5x at 127 tokens (0.6B), 4.4x / 3.9x (1.7B at 127 / 511), 2.8x / 2.7x at 511 / 1,023 tokens (0.6B; scalar attention is now half the pass); decode 1.5–1.8x at the memory ceiling; `verify` unchanged; peak RAM unchanged; scalar fallback at parity |
| 23 | CPU attention and RoPE (one thread) | vector attention for prompts (8 query rows per vector, packed keys and values) and for decode (one streaming pass), RoPE angles computed once per token | 0.6B prefill 4.4x / 4.6x the section 21 rate at 511 / 1,023 tokens (first token 4.9 s / 10.8 s), 2.4x at 2,047; attention stage 4–7x faster; decode at 2,047 tokens 2.2x; `verify` unchanged; peak RAM unchanged |
| 24 | CPU threads for prompts (stage 2) | row-split matmul and kv-head-split attention on scoped std threads (no new dependency), vector SiLU and RMSNorm, buffers reused across layers, `--threads N` | 0.6B prefill 5.0x / 5.4x at 8 threads for 511 / 1,023 tokens (first token 1.06 s / 2.06 s), 1.7B 5.0x; decode unchanged (memory-bound); results bit-identical for any thread count; `verify` unchanged; peak RAM +10 to +20 MB |
| 25 | CPU INT8 weights (stage 3, opt-in) | decode 1.48–1.87x, weights 51.6% of BF16, RAM −0.56–0.58 / −1.63–1.67 GB; accepted after strict 0.6B HF and RTX 5050 CUDA/snapshot closure checks below; CPU INT4 remains no-go |

Rows 7 and 9 did not simply get faster; the cause is the power cap, explained next.

## The power cap

With a 40 W limit the GPU trades clock speed for work. When it rests between tokens the next burst runs at a higher clock; when it never rests the clock settles lower. Measured with `bench --token-pause-ms N`:

| pause after each token | engine speed | GPU clock | GPU power | energy per token | tokens/s on the wall clock |
|---|---|---|---|---|---|
| 0 ms | 61.4 tok/s | 1383 MHz | 36.4 W | 0.78 J | 46.8 |
| 3 ms | 70.2 | 1641 MHz | 36.6 W | 0.82 J | 45.0 |
| 8 ms | 77.5 | 1890 MHz | 36.5 W | 0.95 J | 38.9 |

Power stays at the cap whatever share of time the GPU is busy, so energy per token follows time per token, and idling gains nothing. It showed up three times: the 3.7 ms CPU sampler cost only 1.1 ms of each step because the GPU rested during it; making the sampler 19x faster made plain decoding 1–5% slower; and the faster prefill attention slowed the matrix stages (clock 2074 → 1532 MHz), giving back about 19% of its saving. **Every GPU win here is checked end to end with the cap on.**

## Starting point

Before any optimization (bf16, greedy, 64 generated tokens, median of 5 after a warm-up):

| prompt tokens | TTFT | prefill | decode | sampling per token |
|---|---|---|---|---|
| 32 | 125 ms | 264 tok/s | 49.6 tok/s | 3.7 ms |
| 128 | 351 ms | 369 | 55.7 | 3.6 ms |
| 512 | 1477 ms | 348 | 49.1 | 3.7 ms |

Loading takes about 2.4 s (copying 3.4 GB of weights: 0.38 s). Where the time went: prefill at 512 tokens was MLP 64%, q/k/v 14%, attention 14%; decode at 512 tokens was MLP 37%, attention 26%. Decode reads every weight once per token, so it is limited by memory; attention grows with context and became the next cost.

**What the GPU itself can do** (`marin-run probe`, sustained): memory read 277–302 GB/s, fp32 7.0–11.2 TFLOP/s, bf16 tensor cores 28.1–28.3 TFLOP/s. Decode weight stages already sit near the memory limit (MLP 295 GB/s), so smaller weights are the lever for decode speed.

## Optimizations

### 1. Decode attention split across the context

*Problem.* One block per head gave 16 blocks on a 20-multiprocessor GPU, each scanning the whole context alone. *Change.* The context is split across about 80 blocks per layer; a small second kernel merges the pieces (exact softmax).

| context | decode before → after | attention per token |
|---|---|---|
| 1,024 | 44.9 → 52.7 tok/s (+17.5%) | 9.17 → 4.64 ms |
| 2,048 | 33.4 → 41.4 (+24.2%) | 16.47 → 9.54 ms |
| 4,096 | 23.1 → 34.2 (+48.2%) | 30.64 → 14.14 ms |

A 16-bit KV cache fails `verify` at the existing tolerances (2.8e-3 against 5e-4); the tolerances were not loosened, and f16 is judged by separate gates (section 12).

### 2. Prefill matrix multiply on tensor cores

For passes of 16+ tokens, a bf16 `mma.sync` kernel (f32 accumulation) replaces the fp32 kernel. Activations are split into a bf16 high and low part and both are multiplied, because one rounding loses too much. Decode keeps the old kernel.

| prompt tokens | TTFT before → after | prefill |
|---|---|---|
| 128 | 352 → 127 ms | 364 → 1009 tok/s |
| 512 | 1576 → 519 ms | 325 → 986 |
| 2,048 | 8531 → 4351 ms | 240 → 471 |

Matrix stages went from about 1.1 to 4–8.7 TFLOP/s; gains shrink for long prompts because attention was not a matrix kernel yet (section 9). Energy for a 2,048-token prompt halves (169 → 86 J).

### 3. Fixed workspace, fewer launches

Every buffer a pass needs is allocated once; q/k/v and gate/up are one launch each; prompts over 512 tokens run in pieces of 512. Device allocations per one-token pass 339 → 0, launches 507 → 423, decode unchanged beyond noise (+0.7%). CUDA graphs gave +1.2% in a prototype and were not adopted.

### 4. Weight quantization

`--quantize int8|int4` stores each layer's 7 projection matrices as integer codes with one f32 scale per group of 128 columns. The embedding and head stay int8 whenever quantization is on (a 4-bit head hurt quality badly). Codes are made at load time. Quantized models cannot match Hugging Face logits, so `marin-run quality` measures perplexity and top-token agreement instead.

| (Qwen3-1.7B-Base, 2,048 tokens) | loss change, notes / code | same top choice | weights | decode | peak GPU | J/token |
|---|---|---|---|---|---|---|
| bf16 | | | 3441 MB | 56.8 tok/s | 5052 MB | 0.73 |
| int8, groups of 128 | +0.03% / +0.11% | 98.2% / 98.9% | 1774 MB | 93.6 (+65%) | 3404 MB | 0.45 |
| int4, groups of 128 | +14.6% / +16.6% | 84.0% / 85.6% | 1070 MB | 108.4 (+91%) | 2712 MB | 0.39 |
| int4, groups of 32 | +9.1% / +8.2% | 86.9% / 90.1% | 1231 MB | | | |

int8 is the setting to use; int4 is clearly lossy even with small groups. At 4,096 tokens the gain shrinks because attention still reads the same cache.

### 5. Guessing ahead (opt-in, GPU only)

The prompt and reply so far are indexed by every run of 8 tokens. When the last 8 tokens repeat an earlier run, up to 7 following tokens are guessed; one pass checks all of them, correct guesses are kept, and the KV cache is rolled back over the rest. The sampler runs once per delivered token, so seeded output is identical to plain decoding. Off unless asked for (`marin-run MODEL --speculate`).

Copying only helps when the reply repeats text in view: tokens settled per pass were 1.08–1.26 for chat and fresh code, 3.10 for whole-file edits. A longer pass costs more (bf16, 4 / 8 tokens: 1.44x / 1.90x a one-token pass at 128 tokens of context; int8 is dearer). Guessing after only 3 matching tokens lost 25% on int8 fresh code at 4k context; requiring 8 gave up none of the edit gains. Every guessing reply equalled the plain reply with the same seed.

### 6. Shared-read attention for guessed tokens

Checking 2–8 tokens used the prompt kernel, so each guessed token re-read the whole cache (attention was 71–77% of an 8-token pass). Now one block reads each key/value row once and applies it to every (token, head) row.

| bf16, 4,096 context | attention before → after | whole pass |
|---|---|---|
| 2 tokens | 37.4 → 14.5 ms | 49.9 → 35.6 ms |
| 8 tokens | 65.3 → 21.6 ms | 92.1 → 60.0 ms |

Gain over plain decoding: whole-file edit 2.31x → 2.50x short and 1.85x → 2.46x at 4k (bf16); fresh code and chat about 1.0x. The only loss is int8 chat at 4k (0.98x). Energy per token on edits −40% to −60%. Tokens settled by one pass arrive together, so the p95 gap between tokens is longer where guessing fires; TTFT is unchanged.

### 7. Faster top-k and top-p sampling

The chat sampler fully sorted all 151,936 logits per token (3.7 ms). Top-k up to 128 now keeps a short sorted list in one scan; top-p without top-k ranks only logits within 18 temperatures of the best (the rest weigh less than 2^-24 and cannot change an f32 sum). Results are bit-for-bit the same as a full sort (`crates/sampler/tests/exact.rs`). The sampler now takes 0.2 ms.

Plain decoding got 0.7–4.8% *slower* (1–5% more energy per token): the GPU's busy time per token grew by about the 3.5 ms the sampler gave back, and the clock fell at the same power. Guessing, which samples every settled token, gained up to 17%.

### 8. One-token attention

At 4,096 tokens attention took 14–16 ms of a 31 ms step, about 2x above a plain streaming read of the same bytes. The key loop held one 512-byte row in flight at a time and reduced over 8 query-head groups when this model has 2. Now a warp scores 4 positions at a time and loads their rows before using any; sums keep their old order (bit-identical).

| | context | attention per token | KV bandwidth | whole pass |
|---|---|---|---|---|
| bf16 | 4,096 | 15.75 → 10.29 ms (−35%) | 60 → 91 GB/s | 31.4 → 28.0 ms |
| int8 | 4,096 | 13.80 → 9.31 ms (−33%) | 68 → 101 GB/s | 22.6 → 20.1 ms |

Real 4k chat replies: bf16 33.4 → 37.8 tok/s (+13%), int8 46.7 → 54.5 (+17%); short context flat. Greedy output byte-identical; `verify` passes.

### 9. Prefill attention

Prefill attention used one 128-thread block per (token, head), each thread reading its own key row (every warp load touched 32 cache lines). It was 50% of a 512-token pass and 79% of a 2,048-token one, running at 2% of the fp32 peak. Keys are now read four values per load, each position's weight is computed once, and four value rows are loaded before use (bit-identical).

| | prompt | TTFT | attention | energy |
|---|---|---|---|---|
| bf16 | 512 | 504 → 318 ms (−37%) | 228 → 114 ms | 6.0 → 4.2 J |
| bf16 | 2,048 | 4378 → 2838 ms (−35%) | 3473 → 1581 ms | 85.5 → 53.2 J |
| int8 | 2,048 | 4205 → 2748 ms (−35%) | 3360 → 1448 ms | 80.8 → 50.5 J |

A real 3,563-token prompt: first token 11.8 s → 7.0 s (−40%). The power cap took part back (the rest of the pass slowed). Greedy replies byte-identical; `compute-sanitizer racecheck` finds the same warnings as the old kernel and no errors.

### 10. Keeping the cache between chat turns

*Problem.* `generate` emptied the KV cache at every reply, so a turn after 4,000 tokens of history paid a 4,000-token prefill (about 9 s) to read one new sentence. *Change.* The chat session records the exact token ids the cache holds; the next prompt keeps the longest shared start, cuts the cache back to it, and reads only what follows.

```text
cached : A B C D E F
turn 2 : A B C D E F G H I   -> keep 6, read G H I
edited : A B C X Y           -> keep 3, read X Y
```

The record is written only after a reply ends cleanly and is cleared on any error, so a half-written cache is never trusted. Qwen3's template drops earlier reasoning blocks, so a turn rereads the last answer, the new question and the framing (52 tokens).

| Qwen3-1.7B, last turn | prompt | read | TTFT kept | TTFT replay | change |
|---|---|---|---|---|---|
| bf16, after 2,280 tokens | 2,328 | 52 | 158 ms | 3,324 ms | −95% (21x) |
| bf16, after 4,142 tokens | 4,190 | 52 | 264 ms | 9,293 ms | −97% (35x) |
| int8, after 4,142 tokens | 4,190 | 52 | 244 ms | 8,955 ms | −97% (37x) |
| first message edited (4.1k) | 4,193 | 4,162 | 9,469 ms | 9,417 ms | +1% (an edit costs nothing extra) |

Replies were identical between kept and replay in every row, bf16 and int8; energy 44 J against 400 J at 4.1k. GPU tests (`crates/model/tests/reuse.rs`) compare a kept cache with a fresh read for greedy and sampled replies, with and without guessing, across edits, stop tokens and cancelled replies. Thirteen deliberate breaks were each caught. Limits: 32-token greedy replies, one conversation shape; a near-tie flipped by rounding could still appear in a long or sampled reply (none did).

### 11. Paged KV cache

*Problem.* The cache reserved the whole `--ctx` window for every layer at load (1.88 GB of KV at `--ctx 8192` even for a 64-token chat; GPU peak 6.11 GB). Section 10 removed repeated *compute*; paging removes repeated *reservation*.

*Change (GPU only).* A page is 64 positions of every layer's keys and values (14 MiB). Pages are allocated in slabs of 8 (112 MiB) the first time one is needed, laid out `[layer][keys, values][page][row]` so a layer's keys stay contiguous. A `PagePool` hands out ids (freed ids reused first), a `PageTable` maps logical page → id, and the three attention kernels follow the table (`p` → page `p/64` → slab `id/8`, slot `id%8` → row `p%64`). `truncate` returns whole pages to the pool; nothing is freed on the GPU until the session ends.

| live positions | KV held | bf16 peak, paged / contiguous | int8 peak, paged / contiguous |
|---|---|---|---|
| 64 | 117 MB | 4.34 / 6.11 GB | 2.66 / 4.44 GB |
| 2,048 | 470 MB | 4.74 | 3.06 |
| 4,096 | 940 MB | 5.21 / 6.15 | 3.53 / 4.47 |
| 8,000 | 1.88 GB | 6.15 / 6.11 | 4.47 / 4.47 |

Speed against the contiguous build (A B B A, 40 W): decode −1.5% at 2k and −2.8% (bf16) / −3.3% (int8) at 4k, TTFT +0.3% at 2k and +4.0% at 4k.

**Two designs failed first.** A table lookup per group of four value rows in prefill gave +9.1% TTFT at 2,048 tokens. One 14 MiB allocation per page (`[page][layer][keys, values]`) gave +12.9% TTFT at 4,096 tokens: the clock fell 11.6% at the same power and every stage slowed. The kept layout (slabs, keys contiguous) costs +2–4%.

**`--ctx` now means** the longest conversation allowed, not a reservation. Starting needs the weights, one slab and a small scratch; if the whole window would not fit, a note says so and a conversation that fills the GPU stops with a clear out-of-memory error (`the GPU could not allocate N more bytes for the KV cache…`). On a failed allocation the cache length and tables are unchanged, the cross-turn record is cleared, and the process exits cleanly (tested, including a real ~19,000-token prompt at `--ctx 32768`).

**Correctness.** 9 bookkeeping tests and 12 GPU tests against the f64 reference (every 64-boundary, reversed and mixed page tables, rollbacks, unmapped pages refused); 20 of 20 replies byte-identical to the contiguous build; `verify` unchanged; kept-cache and guessing runs identical; 17 deliberate breaks each caught.

**Limits.** Memory is not returned to the driver during a session; the smallest cache is one 117 MB slab; up to 63 positions and 7 pages of rounding waste; the split-attention scratch is still sized for `--ctx`; windows over 8,192 and several live sequences were not tested. Paging makes pages the unit for later work: moving them to RAM or disk (section 14), snapshots (13), sharing (not built).

### 12. KV cache precision (f16)

*Question.* Is it worth storing keys and values in fewer bits than f32 (224 KiB per token for Qwen3-1.7B)? bf16, f16 and q8 (one byte per value plus an f32 scale per token and head) were tried; attention math stays f32.

**Gates, fixed before any code** (against f32 KV, `marin-run quality` on 4,096 tokens of two texts, bf16 and int8 weights): loss at most +0.30%; same first choice at least 97.5%; 8 greedy replies, at least 6 identical; at most 55% of f32 bytes; decode at 2k/4k at most 3% slower and first token after 4k at most 5% slower; kernels match an f64 reference. To become a default, a format also had to beat f32 at 4,096 tokens and lose at most 0.15% loss and 1 point of agreement.

| | bytes per token | same first choice (Vim / code) | loss change | replies identical | result |
|---|---|---|---|---|---|
| f32 | 229,376 | reference | | | default elsewhere |
| bf16 | 114,688 | 99.6–99.9% | −0.02–0.01% | 6 of 8 | pass, but f16 is better: removed |
| **f16** | **114,688** | **99.9–100%** | **0.00%** | **8 of 8** | **kept** |
| q8 | 59,136 | 95.5% / 97.5% | +0.88% / +0.56% | 1 of 8 | fail: removed |

q8 fails because of the *keys* (keys only: +1.10%; values only: −0.01%; one scale per 32 values cut the loss 4.4x but missed the agreement gate by 0.1 point).

**Speed, f16 vs f32** (Qwen3-1.7B-Base, 40 W, four interleaved rounds): decode at 4k +6.5%, unchanged at 128 and 2k; first token 13% sooner at 2k and 19.5% sooner at 4k; energy per token at 4k 3.69 → 3.15 J. The likely reason is half the bytes read; the runs do not prove it.

**Long contexts and a second model** (f16 vs f32, `--ctx 32768`):

| Model | Prompt | Weights | First token | Decode | KV held | Peak GPU |
|---|---|---|---|---|---|---|
| 1.7B | 8,191 | bf16 | 39.9 → 22.1 s (−45%) | 33.7 → 35.7 (+5.9%) | 2.00 → 1.00 GB | 6.57 → 5.60 GB |
| 0.6B | 8,191 | bf16 | 39.1 → 19.6 s (−50%) | 51.1 → 52.1 | 2.00 → 1.00 GB | 4.32 → 3.31 GB |
| 1.7B | 16,383 | int8 | 217 → 108 s (−50%) | 26.7 → 28.3 | 3.88 → 1.94 GB | 6.80 → 4.86 GB |
| 1.7B | 16,383 | bf16 | out of memory at 14,848 positions | 24.2 (f16) | 1.94 GB | 6.54 GB |

A 32,000-token prompt reads fully in f16 (3.70 GB of KV); f32 runs out at 15,872 positions. Quality at 8k–16k tokens: the same first choice at 100.0% of about 100,000 scored positions, loss change below 0.00005 nats; 40 of 40 base-model replies identical; **7 of 8 chat replies identical** (one differed at a near-tie, "test cases" against "testing").

**Verdict.** `marin` defaults to **f16 on the GPU and f32 on the CPU**; `marin-run`, `verify`, `quality` and `bench` keep f32. `--kv-type f32|f16` overrides (an explicit f16 on the CPU is refused). Strict `verify` stays on f32 and passes (1.7e-5 to 2.7e-5); f16 fails it by design (2.83e-3 against 5e-4, the expected size of f16 rounding), and is judged by the gates above. Not lossless: one chat reply of eight changed. Open: two model sizes, one GPU, no long multi-turn chats, no f32 reference beyond 16k tokens. A pseudo-terminal test (`crates/cli/tests/marin_pty_kv.rs`) proves the flag reaches the loaded cache and survives a model switch; three deliberate breaks were caught.

### 13. Prefix snapshot

*What was built.* One snapshot file holds the KV cache of the last finished reply and its token ids. `marin --snapshot FILE` saves it when the session ends and, on the first prompt of the next session, loads the part the prompt shares; only the rest is read. GPU only, f32 and f16, any weights. Opt-in, because the file contains the conversation.

- **File:** an 80-byte header (magic, version, KV type, layout, positions, hashes of weights, config, token ids and the header), the token ids, one hashed block per 64-position page, and an end mark. A fingerprint of weights, config, tokenizer, quantization, KV type and layout means anything foreign or damaged is ignored and the prompt is read in full.
- **Restore rule:** the longest shared start, rounded down to whole pages (at most 63 tokens are recomputed). The token record is set only after the whole copy succeeded.
- **Save rule:** after a normal or cancelled reply, write a temp file, fsync, rename, fsync the folder. The weight hash (0.8 s) runs on its own thread while the model loads.

Restart to first token, Qwen3-1.7B-Base, separate processes, median of 3:

| saved start | KV | file | re-read: first token / process | restore: first token / process | save |
|---|---|---|---|---|---|
| 512 | f16 | 59 MB | 316 ms / 2.62 s | 65 ms / 2.48 s | 40 ms |
| 2,048 | f16 | 235 MB | 2149 ms / 4.42 s | 130 ms / 2.54 s | 138 ms |
| 4,096 | f16 | 470 MB | 6517 ms / 8.82 s | 224 ms / 2.62 s | 285 ms |
| 4,096 | f32 | 940 MB | 7177 ms / 9.49 s | 430 ms / 2.85 s | 870 ms |
| 8,192 | f32 | 1.88 GB | 39355 ms / 41.67 s | 974 ms / 3.41 s | 929 ms |

A restore is 15–50x shorter than reading the text and uses 1–2% of the GPU energy from 2k tokens up. Checksums cost about a third of a restore and stay (they make a damaged file safe). Below about 1k tokens it saves nothing a user would notice.

**Same output as the live cache:** a greedy 24-token reply after restoring was identical to the saving process's reply and to a cold read in all 16 combinations of start (512–4,096), KV type and weights. Tests cover the page rule around every 64/512 boundary, truncation and flipped bits at 15 places, snapshots refused for another KV type, quantization, config, tokenizer or weights, failed saves leaving the old file, and 12 deliberate breaks. Found while building: restoring only the first pages of a longer snapshot read the end mark from the wrong place (fixed).

*Why a file and not shared pages.* Marin runs one chat at a time, so there is no second live sequence to share pages with. For about 1,000 lines of code it turns 2–40 s of waiting into 0.2–1 s. Not built: shared pages, a prefix index, eviction, compression, background saving.

### 14. Parking idle sessions in RAM (investigated, not built)

*Question.* Is it worth copying an inactive conversation's KV to pinned RAM and back? A throwaway experiment (not in the repo) parked 2–3 sessions taking turns on Qwen3-1.7B-Base.

- Pinned copies run at 13.5–13.8 GB/s (4,096 f32 = 896 MiB in 69 ms); ordinary memory 7–12 GB/s; the page-by-page snapshot path only 2–3 GB/s; cold disk reads 2–3 GB/s.
- Switch time versus re-reading the history: 11 vs 443 ms (512 tokens, f16), 71 vs 6,808 ms (4,096 f16), 140 vs 22,765 ms (8,192 f16). **40–200x faster than re-reading, 5–6x faster than the snapshot disk path**, at about the energy of keeping every session resident.
- It matters for memory only when live sessions together hold more than about 16k tokens in f32 (33k in f16): three 8,192-token f32 sessions ran out of memory when resident but worked when parked.
- Parked sessions resumed bit-identically.

**Verdict: no-go for now.** The gain exists only for several simultaneous conversations, and `Session` owns exactly one cache. For one conversation, snapshots (section 13) already restore a session in 0.2–1 s. Revisit together with multi-session support; the numbers and two design points (the page-pointer table must be rewritten after slabs are reallocated; memory is released a slab at a time) carry over.

### 15. Draft-model speculative decoding (investigated, not built)

*Question.* Would Qwen3-0.6B proposing tokens for Qwen3-1.7B beat plain decoding or the n-gram guesser on this 40 W GPU? A throwaway experiment loaded both models (identical tokenizers, 28 layers, 8 KV heads) and used the real check pass; output followed the target's samples exactly in 229 of 234 runs (the other 5 were one floating-point near-tie).

Decode speed, tokens per second (Qwen3-1.7B target, f16 KV):

| Weights | Workload | Plain | n-gram | Draft k=2 |
|---|---|---|---|---|
| bf16 | chat | 62.1 | 62.3 | 61.4 |
| bf16 | code | 62.4 | 63.1 | 73.2 (+17%) |
| bf16 | edit | 59.1 | 129.9 (+120%) | 77.2 (+31%) |
| bf16 | long, 6k | 37.0 | 36.4 | 31.3 (−15%) |
| int8 | code | 105.8 | 105.7 | 89.9 (−15%) |
| int8 | long, 6k | 46.3 | 44.9 | 34.8 (−25%) |

*Why it loses.* A 0.6B draft step still costs 43–53% of a target step at short context (28 layers of small kernels) and 79–88% at 6k (same KV size); and a 3-token check costs 1.74x one token with int8 (1.22x with bf16), so rounds rarely pay (int8 code needs 2.83 tokens per round, gets 2.47). Even a free check would give +20% at best. It also costs 0.6–1.2 GB of weights, a second KV cache that halves the longest context, and seconds of extra prompt processing.

**Verdict: no-go.** The n-gram guesser already gives +70–120% on edits for no memory. Better next steps, in measured order: int8 weights in `marin` chat (section 16); a cheaper several-token pass with int8; only then a draft that shares the target's KV cache.

### 16. INT8 weights in chat

`marin --quantize none|int8` (with `--quant-group N`, default 128) uses the same quantizer and GPU kernels as `marin-run`. bf16 stays the default; the Model tab names the weight format; at this stage an explicit `--quantize int8` on the CPU was refused (CPU support arrives in section 25).

**Bug or quantization noise?** Replies often differ from bf16's in wording. Three runs of the same text were compared: original bf16, real int8, and the same weights rounded to int8 by an independent numpy script and run on the plain bf16 path. Real int8 agrees with the ideal rounded weights at 99.6–99.8% of positions, but with the original at only 98.1–98.7%: the difference is the weight rounding (0.7% relative error per matrix), not the kernels. At about 1.5% of positions choosing another token, a 200-token greedy reply stays identical with a probability near 5%.

**Correctness, not wording** (264 items scored by program: 16 Python functions with assertions, 32 facts, 12 instruction checks, 6 long-document lookups; both models, greedy and chat sampler): bf16 passes 217 (82.2%), int8 214 (81.1%). The paired difference is −1.1 points with a 95% interval of −3.3 to +1.1: not shown worse, a loss up to 3 points not excluded. One real, reproduced semantic regression on code ("rewrite this slow Python": int8 produced a set intersection that loses order).

| Model | Prompt | First token bf16 → int8 | Decode tok/s | Peak GPU | J/token |
|---|---|---|---|---|---|
| 1.7B | 127 | 109 → 105 ms | 51.4 → 83.8 (+63%) | 4.56 → 2.87 GB | 0.75 → 0.44 |
| 1.7B | 2,047 | 2391 → 2338 ms | 44.5 → 67.2 (+51%) | 4.74 → 3.06 GB | 1.58 → 1.30 |
| 1.7B | 8,191 | 23.4 → 22.6 s | 29.6 → 39.4 (+33%) | 5.38 → 3.69 GB | 8.57 → 8.06 |
| 0.6B | 127 | 71 → 34 ms | 142.6 → 221.3 (+55%) | 2.11 → 1.54 GB | 0.24 → 0.13 |

Start-up grows 1.2–1.6 s (quantizing at load). Compatible with snapshots (a mismatch is refused), the f16 cache and `marin-run --speculate` (guessed output equals plain decoding).

**Verdict: option, not the default.** The gains are large and the kernels are correct, but int8 changes the text of most replies, its same-first-choice rate (98.1–98.7%) misses the 99% the f16 default had to meet, and the small correctness set cannot exclude a loss of up to 3 points. What would decide it: a larger executable code set (about 150 tasks) with a pass rule written first.

### 17. Mixed precision and int8 KV values (not adopted)

Three ideas tried alone with the other settings fixed (experiment code not committed).

1. **bf16 output head with int8 layers.** The head was already int8 (the old help text was wrong; fixed). Rounding error is spread over all 197 tensors roughly in proportion to their size; the head is not special. A real bf16 table gained 0.0–0.2 points of agreement and 7–10% of drift for 7–9% of decode speed and 0.3 GB more memory. Groups of 64 (`--quant-group 64`) recover as much for a fifth of the cost. **No-go.**
2. **f16 keys + int8 values** (scale per 32 values or per head). Quality is good (99.6–99.9% same choice, correctness within one item), and memory drops 22–24% (28–32% more positions). But decode was unchanged (−1%), first token 8–13% slower and energy 7–11% higher. The attention stage took the same time with 44% fewer value bytes because the kernel was not limited by memory speed. **No-go** (worth re-measuring after section 20's kernel).
3. **What moves speed or error:** finer groups are the cheapest lever and exist; for the cache, kernel structure, not bytes, set the speed at that point.

*Correction (found in section 20):* the `pass-cost` "KV held" column prints the f32 size whatever the cache type, so the "96 GB/s" attention figure here was really 48 GB/s.

### 18. Tensor-core prefill attention

*Why.* A 4,095-token prompt took 6.7 s, of which attention was 71.5% at 0.40 TFLOP/s: 1.4% of the 28.3 TFLOP/s tensor-core ceiling.

*Design.* A FlashAttention-2-style kernel ([paper](https://arxiv.org/pdf/2307.08691)) for the f16 cache, head size 128, passes of more than 8 tokens: one block per (64 query positions, query head), four warps, keys and values staged in shared memory 64 positions at a time, `mma.sync.m16n8k16` with f32 accumulation, online softmax, causal mask only on the diagonal tile. Everything else keeps the old kernel; `MARIN_ATTENTION_SCALAR=1` forces it.

**Gates, fixed before code:** kernel error ≤2e-3 (max) / 2e-4 (mean) against an f64 reference at 12 lengths; strict `verify` unchanged; quality (same choice ≥99.5%, loss ≤+0.001, 264-item set within 2 passes, 6 of 8 chat replies identical); first token ≥25% / 40% / 50% lower at 2k / 4k / 8k and not slower at 511; prompt energy ≥30% lower at 4k; decode within 2%, memory within +50 MB.

| Prompt | First token old → new | GPU joules | Attention stage |
|---|---|---|---|
| 511 | 432 → 285 ms (−34%) | 13 → 9 | 126 → 9.8 ms |
| 2,047 | 2702 → 1391 ms (−49%) | 104 → 52 | 1330 → 66 ms |
| 4,095 | 8024 → 2815 ms (−65%) | 318 → 109 | 5107 → 250 ms |
| 8,191 | 25350 → 5904 ms (−77%) | 1012 → 233 | 20510 → 934 ms |

At 4,095 tokens the kernel runs at 7.7 TFLOP/s (27% of ceiling). The power cap slowed the matmul stages about 10% (MLP +12%), but the gain was not cancelled. Quality: kernel error worst 2.3e-4 (gate 2e-3); same first choice 100% (one run 99.9%); 217 passes on the 264-item set with identical text on 264 of 264; 8 of 8 chat and 3 of 3 long-document replies identical; strict `verify` unchanged (1.91e-5).

**Verdict: kept.** Every gate holds for one 150-line kernel and one fallback switch. Next prefill bottleneck: the matrix multiplies (86% of a 4k pass).

### 19. Prefill matrix multiplication (investigated; best prototype reverted)

After section 18 the three matmul stages are 80–90% of a prompt pass and run at 5.0–6.1 TFLOP/s of the 28.3 ceiling. Ablations on the real shapes: memory movement and launch overhead are not the limit (loads and conversion alone take a fifth of the time); the multiply phase alone reaches 76% of the ceiling; the production kernel is slower because loading, converting and multiplying run one after another behind two barriers per 16 columns (38% of the time is not tensor work), and the power cap sets the clock it runs at.

**Gates, fixed before any change:** output bit-identical for bf16/int8/int4; strict `verify` unchanged; matmul stages ≥1.25x faster at 2k/4k/8k; first token ≥10% lower at 2k and ≥15% at 4k and 8k; decode within 2%; energy at 4k ≥10% lower.

| Variant (gate/up at 4,096 tokens, bit-identical) | Time vs production (15.4–16.2 ms) |
|---|---|
| next tile fetched early, two shared tiles | −6% |
| 32 columns per step | +3% (slower) |
| `ldmatrix` alone | 0% |
| pipelined + `ldmatrix` | −11% |
| + 2x2 warps | −14% |
| cuBLAS bf16 (single multiply), for reference | 5.25 ms (×2 for the high/low split = 10.5) |

End to end (best candidate, bit-identical): first token −20.6% / −11.5% / −13.2% / −11.5% at 511 / 2k / 4k / 8k tokens, energy −13% at 4k, matmul stages 1.15x. **Missed:** stages 1.15x (gate 1.25x) and first token −13% at 4k and 8k (gate −15%). **No-go as specified; the prototype was reverted.** A stronger version needs a different design (a cuBLAS-style GEMM with 3–4 stage `cp.async` pipelines and activations split once in a pre-pass, estimated −20–25% first token, ~300 lines) or dropping the low part on the f16 path (1.5–2x on matmuls, changes every matmul's numerics).

### 20. Decode attention with 16-byte loads

*Profile.* The weight stages read at 250–333 GB/s (the practical ceiling here: 277 sustained), but decode attention read the cache at only 37–64 GB/s and flattened near 60 GB/s, so at 16k tokens attention was 70% (bf16) and 78% (int8) of a decode step. The old kernel did eight 2-byte loads per lane per row.

*Diagnosis.* Stand-alone ablations on all 28 layers: the keys phase held the bandwidth down (10.7 of 14.0 ms at 8k tokens); 16-byte loads in both phases made the step 2.2x faster at 8k and 3.5x at 2k with outputs equal to 1e-8; vectorizing only the values phase made it slower (register pressure); changing the split size moved the step by less than 10%. The old kernel was nowhere near the power cap at 2k tokens.

*Change* (f16 cache, head size 128; the f32 cache and `MARIN_ATTENTION_SCALAR=1` take the old code): a half-warp of 16 lanes reads a 256-byte row with one 16-byte load per lane, eight positions per batch (`scores_f16_vec`); the values phase gives each of 8 half-warps every 8th position and adds the 8 partial sums in a fixed order (`values_f16_vec`). About 110 lines of CUDA and 25 of Rust.

| Weights | Context | Decode tok/s old → new | Decode J/token | Peak GPU |
|---|---|---|---|---|
| bf16 | 2,047 | 50.9 → 57.4 (+13%) | | 4.89 → 4.89 GB |
| bf16 | 4,095 | 39.0 → 52.4 (+34%) | 1.00 → 0.74 | 5.09 → 5.11 |
| bf16 | 8,191 | 27.3 → 44.9 (+65%) | 1.43 → 0.87 | 5.53 → 5.56 |
| bf16 | 16,383 | 18.0 → 35.0 (+95%) | 2.17 → 1.11 | 6.51 → 6.49 |
| int8 | 4,095 | 51.8 → 80.5 (+55%) | 1.01 → 0.65 | 3.39 → 3.40 |
| int8 | 16,383 | 20.1 → 46.2 (+130%) | 1.94 → 0.84 | 4.78 → 4.83 |

The attention stage now reads the cache at 119–155 GB/s (40–50% of the ceiling). Gates (all met): +10 / 20 / 35 / 50% decode at 2k / 4k / 8k / 16k; not slower at 127 tokens (+3%); kernel error 1.2e-7 (gate 1e-5); same first choice 100% on all six quality runs; 264 of 264 identical replies; 11 of 11 chat and long-document replies identical; strict `verify` unchanged. **Kept.** Next: attention is still 46–56% of a 16k step; a double-buffered `cp.async` pipeline could reach about 250 GB/s (+30% at 16k); the few-token kernel that checks guesses still uses 2-byte loads; int8 KV values should be re-measured on this kernel.

### 21. CPU inference (investigation, no code changed)

The CPU backend is what a machine without a usable NVIDIA GPU gets (`--backend auto` falls back to it). Measured on the Zen 4 laptop with Qwen3-0.6B and Qwen3-1.7B; every result is a ratio against a reference on the same hardware. The "older CPU" cases are builds without AVX-512, a proxy, not a real old CPU.

**Audit.** No threads (1 of 16 cores busy). The baseline x86-64 build (SSE2, no `target-cpu`, crates are `forbid(unsafe_code)`). The matmul converted each bf16 weight row to f32 and called one `dot` per token with no tiling. Attention is a scalar per-head loop; the KV cache is f32 with the whole window reserved; there is no CPU quantization; RMSNorm, SiLU and RoPE are scalar loops.

**Marin CPU at that point** (bf16 weights, f32 math, 1 thread):

| Model | Prompt | First token | Prefill | Decode | Weight reads | Peak RAM |
|---|---|---|---|---|---|---|
| 0.6B | 127 / 511 / 1,023 | 4.8 / 21.4 / 49.2 s | 26.4 / 23.8 / 20.8 tok/s | 11.0 / 9.5 / 8.5 | 13.2 / 11.3 / 10.1 GB/s | 1.30 / 1.41 / 1.57 GB |
| 1.7B | 127 / 511 | 16.5 / 67.4 s | 7.7 / 7.6 | 3.6 / 3.5 | 12.4 / 12.2 GB/s | 3.55 / 3.69 GB |

Rebuilding unchanged with `-C target-cpu=x86-64-v3` gave only +24–27% prefill and +14–20% decode; `native` was no better.

**Reference: llama.cpp** (BF16 GGUF of the same checkpoints; *native* = AVX-512 with bf16 dot products, *AVX2* = no AVX-512). Prefill / decode, tokens per second:

| Model | 1 thread, native | 8 threads, native | 1 thread, AVX2 build |
|---|---|---|---|
| 0.6B | 210 / 18.8 | 1061 / 19.8 | 102 / 19.4 |
| 1.7B | 76 / 6.8 | 373 / 7.0 | 33.9 / 6.8 |

Findings: one thread of llama.cpp prefills 8–10x faster than Marin (4x with only AVX2). Decode is memory-bound at about 23 GB/s whatever the thread count, so Marin sat at 52–58% of the ceiling. Prefill scales about 5x to 8 threads; 16 threads is slower than 8 (SMT). Quantized llama.cpp decodes the 0.6B at 36.7 (Q8_0) and 56.9 (Q4_K_M) tok/s at 8 threads and the 1.7B at 13.0 and 21.4, with peak RSS 2.10 and 2.05 GB against 3.68 GB for BF16. Marin's 1.7B in bf16 is killed at a 3 GB memory limit and runs at 4 GB. Energy was not measurable (RAPL is root-only).

**Top three bottlenecks:** (1) the matmul kernel; (2) no threads; (3) bf16-only weights on the CPU. Next in line: f32 whole-window KV (1.9 GB at 8k), scalar attention, scalar elementwise ops.

**SIMD without `unsafe` outside `cuda`:** `std::simd` is nightly-only, and calling a `#[target_feature]` function without the feature needs `unsafe`, so run-time dispatch needs a maintained library, per-feature-level builds with a launcher, or `unsafe` in a second crate. Section 22 chose the first.

**Staged plan** (each stage measured against the table above): 1 kernels (one thread); 2 threads on physical cores (≥70% scaling to 4 threads); 3 CPU quantization (decode ≥1.7x int8, ≥2.5x int4; quality gates of sections 4 and 16); 4 memory (growing f16/int8 KV, no second weight copy; 1.7B int4 with an 8k window under 3 GB); 5 the hardware planner chooses format, threads and window.

### 22. CPU matmul kernels, stage 1 (one thread)

*Scope.* Only `tensor::Matrix::matmul`, on one thread: no threads, no quantization, no change to attention or the elementwise ops. Gates were written before the code. The first version missed three (scalar fallback, peak RAM, long-prompt prefill); a second pass closed two and explained the third.

**SIMD choice.** `std::simd` is nightly-only and baseline builds get SSE2 only, so run-time dispatch needs a library. Considered: `pulp` (the layer under `faer` and `gemm`, about 13 small crates), `gemm` (no bf16 input), `archmage`, `fearless_simd`, `wide`, `multiversion`. **Chosen: `pulp`**: it picks AVX2+FMA at run time and our crates stay `#![forbid(unsafe_code)]` (the `unsafe` is inside the dependency). It has no SSE level, so other CPUs use the fallback below.

**Design** (files in `crates/tensor`: `kernel.rs`, `gemm.rs`, `gemv.rs`, `dispatch.rs`, `accuracy_tests.rs`):
- **No bf16 conversion:** two bf16 are read as one `u32` (`v << 16` is the low one as f32, `v & 0xFFFF0000` the high one). Activations are split into even and odd elements, so a dot is `low·even + high·odd`.
- **Storage:** the loader's `Vec<u16>` is viewed as `u32` pairs in place (`bytemuck`, via `pulp`; needs an even width, little-endian, 4-byte alignment). Otherwise it is repacked block by block, shrinking the source as it goes.
- **Prefill (2+ tokens):** register-blocked GEMM, 4 weight rows × 2 tokens per tile (4×3 and 6×2 spill registers and run at half speed), weights and activations packed into panels (256 pairs, 64 tokens per block).
- **Decode (1 token):** GEMV, one weight row at a time with two accumulators. Four rows at a time was slower: with 1,024 columns the rows sit 2 KiB apart and the streams interfere (16–19 GB/s against 24–28).
- **Fallback:** CPUs without AVX2+FMA run the original loop (convert a row to f32 once, one dot per token). `MARIN_MATMUL_SCALAR=1` forces it on any machine; that switch is how it was tested, since no older CPU was available.

**Results** (one thread; old = section 21 and a same-day run of the old build; a browser was running):

| Model | Prompt | First token | Prefill | Decode | Weight reads | Peak RAM |
|---|---|---|---|---|---|---|
| 0.6B | 127 | 4.8 → 1.35 s | 26.4 → 94.4 tok/s (**3.5x**) | 11.0 → 17.5 (**1.54x**) | 13.2 → 20.9 GB/s | 1.30 → 1.30 GB |
| 0.6B | 511 | 21.4 → 7.4 s | 23.8 → 69.4 (2.8x) | 9.5 → 15.5 (**1.6x**) | 11.3 → 18.5 | 1.41 → 1.42 |
| 0.6B | 1,023 | 49.2 → 18.4 s | 20.8 → 55.7 (2.7x) | 8.5 → 13.3 (**1.6x**) | 10.1 → 15.9 | 1.57 → 1.58 |
| 1.7B | 127 | 16.5 → 3.8 s | 7.7 → 33.6 (**4.4x**) | 3.6 → 6.6 (**1.8x**) | 12.4 → 22.8 | 3.55 → 3.55 |
| 1.7B | 511 | 67.4 → 17.3 s | 7.6 → 29.6 (**3.9x**) | 3.5 → 6.2 (**1.8x**) | 12.2 → 21.5 | 3.69 → 3.69 |

Against llama.cpp on one thread: decode is at 90–93% of its BF16 figure (17.5 vs 18.8; 6.6 vs 6.8); prefill is at 93–99% of its AVX2-only build (94.4 vs 102; 33.6 vs 33.9) and about 45% of its AVX-512 build.

**Correctness.** `marin-run verify` on Qwen3-1.7B-Base (CPU, against Hugging Face) passes at the unchanged tolerance on both paths: max logit diff 3.24e-5 / 3.43e-5 / 3.05e-5 (SIMD) and 3.05e-5 / 4.20e-5 / 3.05e-5 (scalar, identical to before). For the 0.6B-Base, which has no saved reference, greedy output (3 prompts, 48 tokens) is byte-identical between the old build, the SIMD path and the scalar path. Kernel error is below 1e-5 relative to an f64 reference over 528 shapes (odd widths, remainders, 1–129 tokens) on both paths. Full suite: 65 programs, 561 tests.

**Gates**

| Gate | Result |
|---|---|
| Correctness, tolerance unchanged | met |
| Decode ≥1.4x | met (1.5–1.8x) |
| Prefill ≥3x | met for the 1.7B and the 0.6B at 127 tokens; **missed for the 0.6B at 511 (2.8x) and 1,023 tokens (2.7x)** |
| Scalar fallback at most 5% slower | met: 0.6B prefill 26.9 vs 26.7 tok/s, decode 11.35 vs 11.36; 1.7B decode 3.95–3.98 vs 3.91–4.01 (interleaved) |
| Peak RAM at most +50 MB | met: +0 to +10 MB everywhere |

**Peak RAM: live against retained.** Live memory is the weights plus KV cache and pass buffers. The first version's excess (+70 to +130 MB) was retained, not live: it packed into a new buffer and freed the 16-bit source, and glibc kept the freed heap chunks (+80 MB at load for the 0.6B), then a whole-prompt activation copy added up to 50 MB more. Raising the allocator's trim or mmap thresholds brought the peak back to 1.28 GB, which located the cause. The fix removes the allocation rather than tuning the allocator: weights are used in place and activations are split per 64-token block (about 0.5 MB, reused). The allocator settings now change nothing.

**Long-prompt prefill: attention.** Profile of the 0.6B at 1,023 tokens:

| Stage | Old | New |
|---|---|---|
| matrix stages (MLP, q/k/v, output) | 36.1 s | 8.8 s (4.1x) |
| attention | 8.85 s (19%) | 8.78 s (**48%**) |
| rest | 0.6 s | 0.6 s |
| pass total | 45.6 s | 18.2 s |

Attention did not change and is now half the pass. It runs at about 14 GFLOP/s (120 GFLOP over 28 layers) against 90–100 for the matrix stages. The 3x gate needs about 16.4 s, so attention must get about 22% faster; at matrix-stage speed (an estimate, not built) the first token would be near 12 s. At 127 tokens attention is 7.5% of the pass, so the gate holds there. The scalar attention loop (`crates/model/src/qwen3/attention.rs`) was left alone on purpose.

**Failures on the way.** Four-row GEMV (1.16x decode); 4×3 and 6×2 tiles (50 vs 91–101 GFLOP/s); packing that held both copies of a tensor; `pulp`'s array-backed `Scalar128b`/`Scalar256b` as a fallback (13–23x slower than the old code); a fallback unpacking loop that did not vectorize (4x slower decode).

**Trade-offs and recommendation.**
- The fallback was tested only through the forced switch on an AVX2 machine; those CPUs keep the old speed, not the new one, and there is no SSE2 kernel.
- `pulp` (and `bytemuck` with it) is a new dependency; the AVX-512 path was not tried.
- The in-place view relies on a 4-byte-aligned loader buffer (checked at run time, with the repack fallback).
- Single-thread decode is at the memory ceiling (21–23 GB/s); more needs threads (stage 2) or smaller weights (stage 3).
- **Stage 1 is closed with one gate open:** the one miss, 0.6B prefill at 511 and 1,023 tokens, comes from unchanged scalar attention, not from the matmul work. Vectorizing attention is the natural first item of the next CPU step, since it also limits what threads can give at long prompts.

### 23. CPU attention and RoPE (one thread)

*Problem.* After section 22 the 0.6B at 1,023 tokens spent 8.8 s of an 18.2 s prefill in attention, running at about 14 GFLOP/s against 90–100 for the matrix stages. Scope: CPU attention, RoPE and the elementwise stages, one thread, existing safe Rust and `pulp` dispatch; the scalar fallback and strict `verify` stay.

**Profile before the change** (0.6B, one profiled run per size):

| Prompt | prefill | attention | RoPE | norms | decode step | decode attention |
|---|---|---|---|---|---|---|
| 127 | 1.30 s | 103 ms (8%) | 50 ms (3.8%) | 13 ms | 55 ms | 4.1 ms (7%) |
| 511 | 6.9 s | 2.17 s (31%) | 201 ms (2.9%) | 52 ms | 65 ms | 13.9 ms (22%) |
| 1,023 | 17.9 s | 8.59 s (48%) | 406 ms (2.3%) | 103 ms | 77 ms | 27.3 ms (36%) |
| 2,047 | 72.5 s | 53.96 s (74%) | 838 ms (1.2%) | 207 ms | 155 ms | 102.6 ms (66%) |

Attention grew faster than the square of the context while everything else is linear. **Diagnosis** (a stand-alone copy of the loop, one layer): dot products 56% of the time (18.6 ns per query-key pair), weighted sum of values 34%, softmax 10% (scalar `exp`). Causes: (1) the baseline build gives 4-wide SSE2 without FMA and every 128-wide dot ended in a serial horizontal sum; (2) each (token, head) swept its kv head's whole slice again, more than the L2 at 2,047 positions, and the two query heads sharing a kv head swept it separately; (3) scalar `exp`. Allocations were not a factor. RoPE recomputed `powf` and `sin_cos` for every head of every layer (24 times more than needed). Norms and other elementwise stages were 0.2–1% and were left alone.

**Gates (written before the code).**

| Area | Gate |
|---|---|
| Correctness | within 1e-5 of an f64 reference on 1–66 tokens, 0–130 earlier positions, six head layouts; `verify` unchanged; greedy output byte-identical to the previous build and on the scalar path |
| RoPE | bit-identical |
| Peak RAM | at most +20 MB |
| Speed (0.6B) | prefill at least 3x the section 21 rate at 511 and 1,023 tokens (first token at most 7.2 s and 16.4 s); attention at 1,023 tokens at least 2.5x faster |
| No regression | prefill at 127 tokens, decode and the 1.7B within 2% or better; scalar fallback within 5% |

**Change** (`crates/tensor`: `attention.rs`, `attention_simd.rs`, `attention_one.rs`; the model calls `tensor::attention`):
- **Prompts (2+ tokens):** one (token, head) row per vector lane: the 8 rows are the heads of one kv head group over 4 tokens, so a score is a multiply-add of a broadcast key element with a transposed query vector and needs no horizontal sum. Keys and values are packed once per kv head (`[position][head_dim]`, and `[tile of 64][column chunk of 8][position][8]`) so the loops stream from L1. Softmax runs across lanes with a vector `exp` (Cephes polynomial); the causal mask touches only the last few positions.
- **Decode (1 token):** keys then values streamed once, row by row, all 16 query heads scored as each row passes (four independent accumulators), vector softmax.
- **RoPE:** `rope_tokens` computes the frequencies once and one `sin_cos` per pair per token, shared by all heads.
- **Fallback:** the original per-head loop, used when AVX2+FMA is absent, the head size is not a multiple of 8, or `MARIN_MATMUL_SCALAR=1`.

**Results** (one thread, current build = the commit before this change, interleaved runs; a browser was running):

| Model | Prompt | First token | Prefill | Decode | Peak RAM |
|---|---|---|---|---|---|
| 0.6B | 127 | 1.30 → 1.17 s | 97 → 109 tok/s | 17.6 → 19.4 | 1.30 GB both |
| 0.6B | 511 | 6.69 → 4.87 s | 76 → 105 (**4.4x** the section 21 rate) | 16.0 → 18.3 | 1.42 both |
| 0.6B | 1,023 | 17.5 → 10.8 s | 59 → 95 (**4.6x**) | 13.4 → 17.0 | 1.58 both |
| 0.6B | 2,047 | 62.3 → 25.4 s | 33 → 80 (2.4x) | 6.6 → 14.5 (**2.2x**) | 1.88 both |
| 1.7B | 511 | 16.4 → 15.0 s | 31 → 34 | 6.4 → 6.6 | 3.69 both |
| 1.7B | 1,023 | 39.0 → 31.2 s | 26 → 33 | 5.9 → 6.3 | 3.87 both |

Stage times (0.6B, before → after): prefill attention 103 → 39 ms (127), 2.17 → 0.53 s (511), 8.59 → 2.03 s (1,023, 4.2x), 53.96 → 7.97 s (2,047, 6.8x, now about 60 GFLOP/s); RoPE 406 → 21 ms at 1,023; attention per decoded token 4.1 → 1.4 ms (127), 27.3 → 10.5 ms (1,023), 102.6 → 19.0 ms (2,047, 5.4x, about 24 GB/s of cache: the memory ceiling).

**Gates:** all met. `verify` (Qwen3-1.7B-Base, CPU) passes with max logit diff 3.05e-5 for all three cases; the scalar path is identical to before; greedy 0.6B-Base output is byte-identical across the old build, the new build and the scalar path; the new attention is within 1e-5 of the reference on both paths; RoPE equals the per-head formula bit for bit; scalar fallback 28.2 vs 27.5 tok/s prefill; peak RAM identical; the 1.7B is not slower.

**What failed on the way.**
- The first prefill kernel (no packing, a slice taken per position and column chunk) was only 2.7–3.5x faster than the old loop; per-position slicing dominated. Packing values once per kv head reached 4.1–5.0x.
- Packing the last value tile at its real length misaligned the chunks of blocks using fewer positions (errors up to 2.0); tests caught it, tiles are now padded to 64 positions.
- The first decode kernel never ran: the guard still rejected one-token calls, and unit tests could not tell, since the scalar loop gives the same answers. An ablation that changed nothing in the timings exposed it; fixed, decode attention was 5x faster.

**Left over.** At 2,047 tokens the 0.6B prefill (25 s) is MLP 41%, attention 31%, projections 17%; prefill attention runs at about 60 GFLOP/s, below the roughly 110 the instruction mix allows (the weighted-sum loop is the larger part). Decode attention is at the cache-read ceiling, so only a smaller cache (stage 4) or threads help. Both kernels need AVX2+FMA and a head size that is a multiple of 8. The vector `exp` differs from `f32::exp` by a few ulps, inside every tolerance above. Next CPU step: threads (section 24).

### 24. CPU threads for prompts (stage 2)

*Why.* Prefill is compute-bound and the machine has 8 cores (16 threads); llama.cpp reaches 5x at 8 threads (section 21). Scope: threads for the matmul, attention and elementwise stages of a prompt pass. No quantization.

**Measured before designing** (stand-alone programs, Ryzen 7 260: 8 cores / 16 threads, 1 MiB L2 per core, 16 MiB L3):
- *Memory bandwidth does not scale with threads.* Streaming a 1 GiB buffer: 25.8 GB/s with 1 thread, 26.2 with 8, 26.3 with 16. One thread already reads at 85–90% of what the chip can (Marin decode reads weights at 21–23 GB/s), so **decode cannot gain more than about 15% from threads here**; llama.cpp's flat decode curve (18.8 tok/s at 1 thread, 19.8 at 8) agrees.
- *Compute scales, until SMT.* A pure AVX2 multiply-add loop gives 155 GFLOP/s per thread for 1 to 8 threads (8.0x) and 99 per thread at 12 threads: the 16 logical threads are 8 cores, and more than one thread per core only hurts.
- *Thread start-up is not free:* a scoped spawn and join costs 28 us for 2 threads, 47 us for 4 and 85 us for 8. A prompt pass has about 8 parallel regions per layer.
- *Under 8 threads of GEMM* the clock falls from 5.0 to 4.7 GHz, and each of 8 concurrent jobs runs about 25% slower than one alone (3.8 to 4.9 ms for the same share of a matrix).

**Design.**
- Threads for prompt passes only; decode stays on one thread (nothing to gain, and a spawn per region would cost 15–40% of a decode step).
- `std::thread::scope` per parallel region: no dependency, no `unsafe`. A region runs on the caller plus helpers pulling jobs from a shared queue, and only when it has at least about 16 million multiply-adds per thread, so short prompts run as before. A persistent pool (`rayon` 1.12, about four small crates) was measured for the same 8 jobs and was no faster (4.9 ms against 5.0 ms), so it was not added.
- Matmul: weight rows split into ranges aligned to the 4-row tile; each thread packs only its own rows and writes its own columns of every token's output (`split_at_mut`, no copies). Attention: split by kv head (8 here). Every output element is computed by the same instructions as on one thread, so **results are bit-identical for any thread count** (tested for 1, 2, 3, 4, 8 and 16 threads on shapes with remainders).
- Thread count: `--threads N` on `marin` and `marin-run`; default one per physical core, read from `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`. `--threads 1` is the old single-thread path with no spawn.
- Serial work that mattered once the matrices were fast: SiLU (a scalar `exp` loop, 8 ms per layer at 511 tokens) and RMSNorm now run as vector kernels; the even/odd split of the activations (done by every thread) became a loop the compiler vectorizes (fixed cost per job 0.52 ms to 0.13 ms); and the CPU forward pass now allocates its intermediate buffers once per pass instead of about 35 MB per layer.

**Gates (written before the code) and results.**

| Area | Gate | Result |
|---|---|---|
| Correctness | outputs bit-identical for 1–16 threads; scalar path correct with threads; existing tests unchanged; `verify` at the unchanged tolerance with 1 and 8 threads; greedy output identical | **met:** `verify` max logit diff 2.10e-5 / 2.67e-5 / 1.72e-5 at both 1 and 8 threads (was 3.05e-5 / 3.43e-5 / 3.05e-5); greedy 0.6B-Base output (3 prompts, 48 tokens) byte-identical for 1 thread, 8 threads, 8 threads on the scalar path, and the previous build |
| 0.6B scaling | at 511 and 1,023 tokens at least 2.8x at 4 threads and 4.5x at 8; 3x at 8 threads for 127 tokens; first token at most 1.2 s at 511 and 3 s at 1,023 tokens | **met:** 4 threads 3.3x / 3.4x; 8 threads **5.0x at 511 and 5.4x at 1,023 tokens**, 4.1x at 127; first token 1.06 s and 2.06 s (best runs 0.85 s and 1.75 s) |
| 1.7B | at 511 tokens at least 4x faster than one thread | **met:** 5.0x at 8 threads (16.7 s to 3.3 s) |
| One thread | no slowdown, no thread started | met (`--threads 1` is the old path; the vector SiLU and norm make it a little faster) |
| Decode | unchanged, never starts a thread | met (within noise: 0.6B 15.8 against 15.6 tok/s at 1,023 tokens of context) |
| Short prompts | 16 tokens or fewer not slower | met: 8 tokens 164–173 ms against 160–169 (no thread starts), 16 tokens 234 to 169 ms, 32 tokens 370 to 155 ms |
| Peak RAM | at most +24 MB at 8 threads and 2,047 tokens | met: 1.87 to 1.89 GB (+10 to +20 MB) |
| CPU cost | extra CPU over one thread below 30% at 8 threads | met: a 1,023-token run takes 11.6 s of CPU on 1 thread and 13.9 s on 8 (+20%) |

**Results, 0.6B prefill** (tokens per second, speedup against 1 thread; first token in brackets; this session's machine ran about 10% slower than the best runs of section 23):

| Prompt | 1 thread | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 127 | 98.3 (1.29 s) | 160.8 (1.6x) | 262.9 (2.7x) | **402.8 (4.1x; 0.32 s)** | 343.9 |
| 511 | 96.8 (5.28 s) | 173.6 (1.8x) | 320.6 (3.3x) | **481.8 (5.0x; 1.06 s)** | 388.1 |
| 1,023 | 91.5 (11.2 s) | 175.5 (1.9x) | 314.6 (3.4x) | **496.7 (5.4x; 2.06 s)** | 430.0 |

The best 8-thread runs reached 473 / 603 / 583 tok/s (first token 0.27 s, 0.85 s, 1.75 s). 16 threads is slower than 8 at every size. **1.7B** at 8 threads: 138.8 / 153.1 / 155.0 tok/s at 127 / 511 / 1,023 tokens (4.4x / 5.0x / 5.0x). A 2,047-token prompt on the 0.6B: 24.8 s on 1 thread, 4.6 s on 8 (5.3x). The scalar fallback also threads: 24.1 tok/s on 1 thread, 95.6 on 8 (4.0x) at 511 tokens; scalar attention stays serial.

**What the stage profile showed.** At 8 threads and 511 tokens the first version reached only 3.8x. The matrix stages scaled 4.0x and about 14% of the pass was serial: RMSNorm 57 ms, scalar SiLU inside the MLP stage (about 8 ms per layer), 70 ms of untimed copies and allocations, RoPE, residual adds, the cache write and the LM head (14 ms, one token). Each fix was measured:
- vector SiLU: 1.9 to 0.31 ns per element (the first version was 10x slower, see below); vector RMSNorm: 57 to 12 ms per pass; vectorized activation split: 8-thread prefill +8%.
- buffers reused across layers: 8-thread wall time −12% on a 1,023-token run and system time 2.8 s to 0.5 s. The process had been allocating and page-faulting about 1 GB per prompt, and page faults from 8 threads contend; CPU cost over one thread went from +45% to +20–23%.

**What failed or did not help.**
- A vector SiLU written with a closure inside the SIMD function ran 10x slower than the scalar loop (18.5 against 1.9 ns per element): the closure was not compiled with the AVX2 target features. As a plain `#[inline(always)]` function it is 6x faster than scalar.
- Splitting the matmul into 2 jobs per thread (so early finishers take more): 444 and 436 tok/s against 467 and 449, no gain.
- A shared pool of scratch buffers for the matmul and attention kernels: first token within noise with and without it (511 tokens: 1055, 1064, 1078 ms with; 1048, 1017, 1061 without), so it was removed.
- `rayon`'s persistent pool against scoped threads: 4.9 against 5.0 ms for 8 jobs; allocator settings (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_`) cut system time but only as a diagnosis, since safe code cannot set them.

**Trade-offs and what is left.**
- Decode is single-threaded by design: memory bandwidth is the limit here. A machine where one core cannot saturate memory (many DDR4 desktops) might gain from 2–4 decode threads; that needs a persistent pool and a measurement on such a machine, neither of which exists yet.
- About 10% of an 8-thread prompt pass is still serial (untimed copies of the hidden state, RoPE, residual adds, cache write, the one-token LM head). 16 threads is never better than 8; the default (one per physical core) is the best setting measured. No thread pinning (safe Rust cannot set it), so a busy machine can place two threads on one core.
- Scalar attention (CPUs without AVX2) is serial; the scalar matmul threads.
- **Recommendation: close stage 2.** Next CPU step: stage 3 (CPU quantization), since decode now sits at the memory ceiling and the 1.7B at 4 GB needs smaller weights anyway.

### 25. CPU INT8 weights (stage 3)

**Acceptance gates, fixed before implementation (2026-10-11).** BF16 remains
the default and the strict Hugging Face `verify` tolerances stay unchanged.
All mandatory gates apply to both Qwen3-0.6B and Qwen3-1.7B; an unmeasured
gate is open, not passed. INT4 work starts only after every INT8 gate passes.

| Area | Gate |
|---|---|
| Kernel correctness | INT8 GEMV/GEMM within `1e-5 * max(1, sum(abs(w*x)))` of an independently decoded f64 dot; zero groups, signed extremes, row/token remainders, groups 32/64/128/256, malformed shapes; scalar and SIMD checked |
| BF16 correctness | Existing full tests and strict HF verification pass, without tolerance changes |
| Quality | On at least 1,024 teacher-forced tokens each of prose and code: same top-1 ≥97.5%, mean loss increase ≤0.02 nats/token and perplexity increase ≤1%; eight fixed greedy prompts per model, inspect semantic regressions and report exact-match count (wording equality is not a gate) |
| Decode | Median ≥1.25x BF16 at 127 tokens and ≥1.10x at 2,047, on both 1 and 8 configured threads; at least three measured rounds after one warm-up |
| Prefill / TTFT | At 127 and 2,047 tokens, TTFT no more than 10% worse than BF16, on 1 and 8 threads |
| Weight / peak RAM | Stored weights ≤55% of BF16; process peak RSS at the same context window falls by ≥35% of BF16 weight bytes; no whole-model dequantization |
| Cold start | Fresh-process load penalty ≤3 seconds (0.6B) / ≤6 seconds (1.7B); separately report filesystem-cold versus page-cache-warm loads and process-to-first-token |
| Compatibility | Both CLIs select INT8 and report it; BF16 default, invalid groups, CPU INT4 rejection, auto fallback, scalar fallback, multithread prefill, KV reuse and existing GPU snapshots remain covered |
| Reference / energy | Local llama.cpp BF16 and Q8_0 at matching threads, contexts, CPU-only and f32 KV; distinguish pp/tg microbenchmarks from end-to-end TTFT. Report CPU energy only if a readable package counter exists; unavailable energy is not a failure |

**Implementation plan.** Reuse the tensor quantizer and model `Store`, add
INT8 storage to the CPU matrix, use safe `pulp` dispatch and existing scoped
prefill threads. Keep f32 activations and accumulation, f32 norms and KV.
Measure fused decode and bounded prefill dequantization before considering
activation quantization or a different format.

**Audit and reference research.** The loader validates tensor names, BF16
dtype and shapes before reading one tensor at a time. CPU matrices use
row-major BF16 pairs; embedding and tied output head share the same matrix.
CUDA already consumes symmetric signed INT8 codes with f32 scales, with
the embedding and head kept at INT8 for the INT4 experiment. CPU INT8
reuses that quantizer, loader and byte accounting. Norms, activations and
CPU KV remain f32. CPU cache reuse is unchanged; snapshots remain GPU-only.
AVX2+FMA dispatch and the scalar override are shared with BF16. Decode uses
one thread, prefill partitions output rows through the existing thread runner.
Quantization at load still uses the quantizer's logical-core workers,
independently of the prompt `--threads` setting.

| Candidate | Storage / CPU computation | Decision |
|---|---|---|
| Marin grouped INT8 | 128 signed bytes plus f32 scale = 8.25 bits/weight; f32 activations, fused dequantization for decode | Reuse the already validated quantizer; isolate weight error from activation error |
| llama.cpp Q8_0 | 32 signed bytes plus f16 scale = 8.5 bits/weight; Q8 activation blocks and integer SIMD dots | Useful bandwidth and throughput reference; numerically different from weight-only INT8 |
| llama.cpp Q4_K_M | Q4_K superblocks with local scales/minima; selected tensors use higher precision | Candidate for later feasibility; not equivalent to Marin's naive symmetric INT4 |
| AWQ / GPTQ | Calibration-dependent channel scaling / second-order reconstruction | Quality-preserving candidates, but calibration and compatible kernels need separate measurements |

Sources: llama.cpp's [block layouts](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-common.h),
[x86 quantized dots](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-cpu/arch/x86/quants.c)
and [quantization guidance](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/tools/quantize/README.md),
the [AWQ paper](https://arxiv.org/abs/2306.00978) and
[GPTQ paper](https://arxiv.org/abs/2210.17323). The papers describe methods and
results on their own models/hardware; those results are not Marin measurements
and do not establish quality on Qwen3. The local llama.cpp comparison uses
the pinned revision above; current upstream source was also checked.

**CPU implementation.** Row-major signed bytes and separate f32 group scales;
the decode kernel reads four codes per u32, extracts four f32 vectors and
accumulates before applying each group scale. Activations are packed once
per call. Prefill dequantizes only four rows × 512 columns into f32 panels,
reusing them across 64 tokens with the existing 4×2 tile output code. Scratch
is bounded per worker; neither a full matrix nor the model is expanded to
BF16/f32. An unaligned byte buffer or a CPU without AVX2 keeps the scalar
row-at-a-time path. Both CLIs accept `--backend cpu --quantize int8` with
the existing group option; CPU INT4 remains refused. INT8 is reported by
the engine, benchmark header, REPL and chat UI; strict `verify` stays BF16.

**Failures found before the final matrix.** On a 127-token 0.6B spot check,
the first INT8 kernel reached only 7.0 decode and 42.7 prefill tok/s versus
BF16's 18.2 / 100.5. It reloaded the same SIMD codes for each activation
plane and performed excessive indexing in the tile loop. Packing the four
activation planes once and loading codes once recovered 34.9 decode tok/s;
packing prefill inputs in the microkernel's read order and using a 512-column
panel recovered 102.1 prefill tok/s. These are single-run diagnostic results,
not the acceptance matrix.

The first three-run long-context RAM comparison saved 415,956,992 bytes,
below the 417,234,944-byte gate. The quantizer copied per-worker code buffers
into a second full code matrix, retaining allocator memory. Workers now write
disjoint slices of the final codes/scales directly. Existing serial/parallel
quantization tests check identical codes and scales for both INT8 and INT4.
Measurements before this fix are kept separately from the final build.

The full suite also exposed an existing auto-backend bug when libcuda was
installed but there was no visible device. `CUDA_ERROR_NO_DEVICE` during
context creation now becomes `Unavailable`, so auto selection falls back to
CPU; explicit CUDA still fails clearly. A subprocess test checks auto INT8
with all GPUs hidden. Other driver failures are still propagated.

**Final local speed and memory matrix.** Ryzen 7 260, release build,
AVX2+FMA, CPU-only, f32 KV, INT8 group 128. Qwen3 chat models, 32 output
tokens, one warm-up and three measured rounds per case, medians below.
Each pair used the same binary and context window (`prompt + 33`); formats
ran serially with alternating order. TTFT starts after model loading.
Peak RSS is process-wide across load, warm-up and measured rounds.
Threads configure prefill; Marin decode remains single-threaded in every row.
Saved JSON schema 2 retains the historical `precision` reference string;
`quantization` records the actual weight format. Thread counts are in the
command manifests, not an added schema field.
All values are **BF16 / INT8**; GB and MB are decimal.

| Model | Threads | Prompt | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Decode gain | Peak RSS (GB) |
|---|---|---|---|---|---|---|---|
| 0.6B | 1 | 127 | 1.208 / 1.228 | 105.2 / 103.5 | 18.49 / 34.55 | 1.87x | 1.300 / 0.726 |
| 0.6B | 1 | 2047 | 26.912 / 26.420 | 76.1 / 77.5 | 13.35 / 19.94 | 1.49x | 1.873 / 1.313 |
| 0.6B | 8 | 127 | 0.303 / 0.284 | 419.3 / 447.3 | 18.79 / 33.74 | 1.80x | 1.307 / 0.727 |
| 0.6B | 8 | 2047 | 4.848 / 4.617 | 422.3 / 443.4 | 13.41 / 19.84 | 1.48x | 1.892 / 1.331 |
| 1.7B | 1 | 127 | 3.979 / 4.122 | 31.9 / 30.8 | 6.12 / 11.39 | 1.86x | 3.557 / 1.898 |
| 1.7B | 1 | 2047 | 67.219 / 71.421 | 30.5 / 28.7 | 5.69 / 9.43 | 1.66x | 4.210 / 2.575 |
| 1.7B | 8 | 127 | 0.887 / 0.722 | 143.2 / 175.8 | 6.35 / 11.72 | 1.85x | 3.571 / 1.899 |
| 1.7B | 8 | 2047 | 13.386 / 11.441 | 152.9 / 178.9 | 5.63 / 9.46 | 1.68x | 4.226 / 2.593 |

All eight decode and TTFT gates pass. The largest TTFT regression is 6.25%
(1.7B, one thread, 2,047 tokens). Stored weight bytes are 1.192 / 0.615 GB
on 0.6B and 3.441 / 1.774 GB on 1.7B, about 51.6% of BF16. Peak RAM falls
560–580 MB / 1.63–1.67 GB, exceeding the required 417 MB / 1.204 GB.
These are local measurements on this machine, not a forecast for DDR4 PCs.

The initial llama.cpp comparison failed before inference: this revision's
`llama-bench` parser accepts f16/bf16/quantized KV but omits f32, although the
core and regular CLI support it. A workspace copy of its benchmark source
adds only the three-line f32 parser branch, linked against the existing
AVX2/native libraries. The original checkout and binaries are unchanged.
A smoke run checks the reported KV types are both f32 and GPU layers zero.
The parser failure and copied source/build command are saved with the results.

**Teacher-forced quality.** The first 64 tokens prime the cache; the next
1,024 positions are scored with the same input token ids for BF16 and INT8.
Prose is Vim's `usr_01.txt` + `usr_02.txt`; code is this revision's
`runtime/src/generate.rs` + `tensor/src/matrix.rs`. These are two small fixed
corpora, not a broad evaluation. All four comparisons pass, but 0.6B prose
has only one token of margin over the top-1 gate. Loss reductions on two
corpora are local rounding effects, not evidence that INT8 improves models.

| Model | Corpus | Scored | Same top-1 | Loss, BF16 / INT8 | Δ loss (nats) | Δ perplexity | Gate |
|---|---|---|---|---|---|---|---|
| 0.6B | prose | 1,024 | 97.559% | 2.8196 / 2.8226 | +0.00294 | +0.294% | pass |
| 0.6B | code | 1,024 | 97.949% | 2.1013 / 2.0983 | -0.00304 | -0.303% | pass |
| 1.7B | prose | 1,024 | 98.047% | 2.7687 / 2.7651 | -0.00356 | -0.356% | pass |
| 1.7B | code | 1,024 | 98.730% | 1.9435 / 1.9466 | +0.00304 | +0.304% | pass |

**Fresh-process startup.** Eight configured prefill threads; three fresh
processes per format/cache condition; seven prompt tokens, two outputs,
no warm-up. Load includes conversion at every startup. The last column is
`load + request TTFT`, excluding tokenizer/backend setup before the load timer;
it is not the separately measured real CLI process-to-visible-output time.

| Model | File cache | Load, BF16 / INT8 (s) | INT8 load penalty (s) | Load + first token, BF16 / INT8 (s) |
|---|---|---|---|---|
| 0.6B | cold | 1.224 / 1.572 | +0.348 | 1.369 / 1.738 |
| 0.6B | warm | 0.657 / 1.045 | +0.389 | 0.806 / 1.206 |
| 1.7B | cold | 3.197 / 3.682 | +0.485 | 3.455 / 3.962 |
| 1.7B | warm | 1.968 / 2.500 | +0.532 | 2.233 / 2.775 |

The measured load penalties pass the 3 s / 6 s caps. Cold means per-file
`POSIX_FADV_DONTNEED` before launch; no global cache drop. Load-time physical
reads were 0.978–0.983 GB / 3.069–3.072 GB on the two sizes, versus zero in
all warm cases. This establishes disk I/O, but does not prove every source
page was evicted or reproduce a cold boot. Source files still contain BF16
and an unused tied head; INT8 reduces runtime storage, not files on disk.

**Final bottleneck profile.** Separate single instrumented runs at eight
threads, 2,047 prompt tokens, 16 outputs; these do not supply speed gates.
Weight-reading decode stages roughly halve: 0.6B MLP 343 → 186 ms and head
196 → 102 ms over 15 decode passes; 1.7B MLP 1,526 → 818 ms and head
401 → 208 ms. Attention stays about 311–325 ms because KV remains f32.
It grows from 29% to 43% of 0.6B decode after weight compression; 1.7B decode
is still MLP-dominated (52%). INT8 prefill is also MLP-dominated (39% / 63%).
Assuming one read of each weight and the populated f32 KV per step gives
roughly 20–22 GB/s combined traffic at long context; this is a byte-count
estimate, not a memory-controller counter. The remaining long-context
limit is KV traffic plus matrix work, not a reason to expand all weights.

**Local llama.cpp comparison.** Pinned revision `e94acad853c0`, CPU-only,
f32 K/V, flash attention off, batch/ubatch 2,048, three measured rounds and
the tool's default warm-up. AVX2/FMA build for ISA parity (no AVX-512/BLAS),
native build at eight threads for this machine's ceiling. Medians recomputed
from saved `samples_ts`; these are pp/tg kernel workloads, **not generated
reply TTFT**. Generation uses random tokens and excludes token sampling.
`-d 127` / `-d 2047` explicitly prime the same starting depths as Marin;
ordinary zero-depth `tg32` is saved but not used for those comparisons.

| Build | Model | Weights | Threads | pp127 | pp2047 | tg32 @127 | tg32 @2047 |
|---|---|---|---|---|---|---|---|
| avx2 | 0.6B | BF16 | 1 | 92.10 | 51.28 | 17.47 | 12.03 |
| avx2 | 0.6B | BF16 | 8 | 544.65 | 305.26 | 18.63 | 13.11 |
| avx2 | 0.6B | Q8_0 | 1 | 99.32 | 51.85 | 30.83 | 17.15 |
| avx2 | 0.6B | Q8_0 | 8 | 643.31 | 296.58 | 31.93 | 19.16 |
| avx2 | 1.7B | BF16 | 1 | 31.77 | 25.56 | 6.33 | 5.53 |
| avx2 | 1.7B | BF16 | 8 | 210.41 | 159.25 | 6.90 | 5.92 |
| avx2 | 1.7B | Q8_0 | 1 | 36.07 | 25.80 | 11.26 | 8.90 |
| avx2 | 1.7B | Q8_0 | 8 | 232.26 | 157.69 | 12.48 | 9.86 |
| native | 0.6B | BF16 | 8 | 947.18 | 396.07 | 18.63 | 13.34 |
| native | 0.6B | Q8_0 | 8 | 763.16 | 335.60 | 32.98 | 19.25 |
| native | 1.7B | BF16 | 8 | 361.49 | 240.26 | 6.85 | 5.89 |
| native | 1.7B | Q8_0 | 8 | 299.36 | 186.99 | 12.28 | 9.75 |

Marin INT8 decode is competitive with AVX2 Q8_0: 0.6B is 34.55 / 19.94
versus 30.83 / 17.15 tok/s on one thread, and 33.74 / 19.84 versus
31.93 / 19.16 at eight configured threads. On 1.7B: 11.39 / 9.43 versus
11.26 / 8.90 on one thread, and 11.72 / 9.46 versus 12.48 / 9.86 at eight.
A few-percent difference is not a strong win. Short-prompt prefill still
favors llama.cpp at eight threads (Q8_0: 643 / 232 versus Marin's 447 / 176);
long prefill on 0.6B favors Marin with these f32-KV settings, while 1.7B is
closer. Native BF16 prefill benefits much more than native Q8_0 on this CPU.
Q8_0 also quantizes activations; it is not a numerically identical reference.

GGUF BF16 samples match the local HF checkpoints: four 32-value samples in
each of 310 loaded tensors per model, including norms promoted to f32.
This is a provenance check, not full-file equality or a quality evaluation
of the pre-existing quantized GGUF files. Those were not reconverted.

**CLI startup and reference RAM.** Actual `marin-run -p "The capital of
France is" --max-tokens 1 --ctx 64 --backend cpu --threads 8` launches,
three per cache/format condition; first stdout byte includes process and
HF tokenizer setup. This measures first visible output, not a universal
first-internal-token timestamp.

| Model | Cache | Process → visible output, BF16 / INT8 (s) |
|---|---|---|
| 0.6B | cold | 1.640 / 1.991 |
| 0.6B | warm | 1.041 / 1.435 |
| 1.7B | cold | 3.893 / 4.271 |
| 1.7B | warm | 2.576 / 3.062 |

Only `llama-bench` is available locally. Its startup rows below measure
**process load/setup plus one token evaluation**, with no warm-up, not
real generated-output TTFT or pure loader time. Three fresh processes per
cache/format condition, eight threads, explicit mmap. Preconverted Q8 GGUF
avoids Marin's BF16 read and quantization cost at startup.

| Model | GGUF | Cold / warm process + evaluation (s) | Peak RSS @127 / @2047 (GB) |
|---|---|---|---|
| 0.6B | BF16 | 0.806 / 0.346 | 1.328 / 2.158 |
| 0.6B | Q8_0 | 0.411 / 0.282 | 0.767 / 1.593 |
| 1.7B | BF16 | 1.545 / 0.520 | 3.581 / 4.428 |
| 1.7B | Q8_0 | 1.029 / 0.421 | 1.971 / 2.804 |

Reference peak RSS uses one combined `pp+tg` run (`-pg 127,32` or
`2047,32`), no warm-up, same f32 KV and batch settings; `getrusage` is
collected by a separate wrapper process for each child. Different cache
and attention workspaces still affect RSS. Cold is the same per-file
fadvise method; a fully cold boot is unmeasured. CPU package energy
counters exist but return permission denied; no joules/token are claimed.

**Greedy output and cache compatibility.** Eight fixed no-thinking ChatML
prompts per model, temperature zero, at most 128 outputs. Replies are byte
identical on 7/8 (0.6B) and 8/8 (1.7B). The changed 0.6B arithmetic reply
only changes whitespace/punctuation and still gives 323. No new semantic
regression was observed; both precisions remove only adjacent duplicates
in `unique_in_order`, both ignore number-only arithmetic formatting, both
add Markdown trailing spaces to the word-per-line response, and 1.7B uses
nonstandard “Bonjour matin” in both. Baseline errors are not INT8 successes.

Real-model INT8 `turns` checks at eight threads, history targets 127/2,047,
eight reply tokens, one warm-up and one measured round: retained and fresh
reads match in all four continuation/edit cases on both sizes. Actual
history lengths are 420/2,280 tokens on 0.6B and 420/2,280 on 1.7B (the tool
builds whole chat turns). Fixture tests cover BF16 and INT8, greedy and
seeded sampling, rollback and context sliding. Snapshots remain GPU-only;
CPU INT8 does not add a snapshot format.

**Final checks:** `cargo fmt --all -- --check`, workspace/all-targets `check`,
all-targets/all-features `clippy -- -D warnings`, full workspace/all-features
tests (**579 passed**), line-limit and unsafe scripts all pass. Forced-scalar
release kernel tests and isolated SIMD/scalar prefill tests at 1/2/8/16
threads pass. CUDA runtime tests skip without a visible GPU; the count does
not imply GPU execution. No dependencies or unsafe Rust were added.

**Verification and decision.** BF16 strict HF verification passes for both
1.7B chat and Base, at 1 and 8 threads; chat maximum logit differences are
1.72e-5 / 1.91e-5 / 1.91e-5, Base 2.10e-5 / 2.67e-5 / 1.72e-5. Forced
scalar Base verification also passes, with the existing tolerances. At the
initial run there was no saved 0.6B HF reference, and the default Python
lacked PyTorch/Transformers. Its kernel references and BF16-relative quality tests passed,
but strict 0.6B HF parity was **open** at that point. The closure audit below
now supplies and passes its independent HF reference.

| Gate | Status before the RTX 5050 recheck |
|---|---|
| Decode, prefill/TTFT, stored weights, peak RAM | **pass**, both sizes, 1/8 threads, short/long contexts |
| Relative output quality | **pass** on the four 1,024-position corpora; fixed greedy prompts show no new semantic regression |
| Cold-load penalty | **pass** for measured fresh-process warm/fadvise-cold conditions; full cold boot unmeasured |
| Kernel correctness and scalar/thread paths | **pass**, independent f64 tests, group 96 across 512-column panels, isolated SIMD/scalar checks at 1/2/8/16 threads |
| Strict BF16 HF parity | **pass** on 1.7B chat/Base and, in the closure audit below, 0.6B chat |
| CLI selection, default BF16, validation, auto fallback, KV reuse | CPU tests and real-model INT8 reuse pass |
| CPU INT8 snapshot round trip | **out of scope**: pre-existing unsupported CPU capability; the original gate explicitly covers existing GPU snapshots |
| Existing CUDA inference and GPU snapshots | compile coverage and existing tests retained; runtime tests skip without a visible GPU, so runtime regression verification is **open** |
| llama.cpp comparison / energy | local comparable pp/tg and RAM/startup measured; generated-output llama.cpp TTFT unavailable; package energy permission-denied |

**Initial go/no-go (before the RTX 5050 was visible):** performance, RAM and measured relative quality support keeping
CPU INT8 as an **experimental opt-in**. Full Stage 3 acceptance was not
closed: the closure audit supplies strict 0.6B HF verification, but
CUDA/snapshot runtime verification remained unavailable in that environment.
CPU snapshot support is not an original Stage 3 gate.
**No INT4 feasibility experiment was run.**
BF16 remains the default/reference. Close the remaining verification gaps
before Stage 3 acceptance. The closure audit extends conversation quality
coverage; no lower-bit work is authorized or performed here. Results are
from one Zen 4 laptop, not an older DDR4 PC; non-x86 hardware, larger
models, other group sizes and conversation quality beyond 1,088 tokens
are unmeasured.

**Closure audit (2026-10-11).** Started from clean commit `9d41f88` with
all three original stashes intact. No runtime behavior, defaults, dependencies
or weight formats changed in this audit: changes are reference data, tests
and this evidence. BF16 stays default, INT8 opt-in. The measured executable
is exactly the original Stage 3 binary (SHA256
`eff4b9b9346468ae93446b14313581fd85e9fb031d3df1b6b9372e64f85b7477`);
its historical runtime tag is `961096041874-dirty`, not the current git HEAD.
Follow-up artifacts and complete command manifests are in
`bench-results/cpu-stage3/closure/`; they do not replace the original matrix.

**0.6B HF gap closed.** An existing local Python environment was found at
`/home/mitali/miracles/ralph/.venv/bin/python`: PyTorch `2.13.0+cu130`,
Transformers `5.17.0`, CPU execution, offline HF loading. The unchanged
`scripts/gen_reference.py` keeps source weights BF16 and upcasts linear and
embedding math to f32. It generated `tests/reference/qwen3-0.6b.json` from
the real local chat checkpoint: three prompts, 16 greedy decode steps each,
five reference logits per step and 28 layer norms per prompt. Generator,
checkpoint, config and reference SHA256 hashes and exact environment are in
`tests/reference/qwen3-0.6b.provenance.json`. This is an HF-generated
reference, not Marin output saved as its own reference.

| Path | Threads | Max logit differences, three cases | Max relative layer differences, three cases | Greedy choices |
|---|---|---|---|---|
| SIMD | 1 | 1.91e-5 / 1.43e-5 / 1.81e-5 | 6.28e-7 / 2.70e-7 / 6.06e-7 | 48/48 |
| SIMD | 8 | 1.91e-5 / 1.43e-5 / 1.81e-5 | 6.28e-7 / 2.70e-7 / 6.06e-7 | 48/48 |
| Scalar fallback | 8 | 3.24e-5 / 2.67e-5 / 3.81e-5 | 9.90e-7 / 9.03e-7 / 1.41e-6 | 48/48 |

All pass the unchanged strict limits: absolute logit tolerance `5e-4` plus
`1e-4 * abs(reference)`, relative norm tolerance `1e-4`, exact greedy ids.
HF verification intentionally tests BF16; quantized quality is evaluated
relative to that reference path, not by loosening HF tolerances.

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/mitali/miracles/ralph/.venv/bin/python scripts/gen_reference.py reference /home/mitali/models/Qwen3-0.6B tests/reference/qwen3-0.6b.json --steps 16
# Repeat SIMD at --threads 1 and 8; add MARIN_MATMUL_SCALAR=1 for the fallback.
marin-run verify /home/mitali/models/Qwen3-0.6B --reference tests/reference/qwen3-0.6b.json --backend cpu --threads 8
```

**Snapshots and switches: supported behavior versus missing capability.**
`Engine::kv_layout()` returns `None` on CPU and CPU page export/import
returns `Unsupported`. `Session::snapshot_identity()` therefore never
restores or saves a CPU cache. This predates INT8. The requested CPU INT8
save/restart/restore gate **cannot pass** on this implementation.

On both real chat models and both BF16/INT8, the normal `marin` CLI was run
fresh and then in two fresh processes with a missing snapshot, and two with
an existing invalid file: **20 launches total**. Same-format replies match
the fresh run byte-for-byte; missing files stay missing, existing bytes
stay unchanged. This verifies safe replay and preservation, **not** a
successful CPU snapshot save/restore or payload corruption detection.
The extended fixture test also asserts zero reused positions and no restore
result across both precisions and both file states. A new header test checks
all nine ordered BF16/INT8-group-32/INT8-group-128 identity pairs: only
identical formats are accepted. Header validation alone does not verify
real GPU page import or checksum-failure recovery.

Actual terminal tests exercise INT8 CPU reload, CPU → CUDA failure → CPU
recovery, CPU → auto fallback, a switch to a second model configuration,
continued generation after every switch, retained INT8/f32 labels and a
restored terminal. The model-change test observes an empty reply state;
the inspected switch path also constructs a new chat for a different model.
These use the real CLI and loaded tiny HF fixture, not a mocked backend. Real
0.6B INT8 explicit CUDA selection fails; hidden-GPU auto selection succeeds
on CPU. Successful CPU ↔ CUDA execution remains **unverified**:
`nvidia-smi` cannot contact the driver and `/dev/nvidia0`, `/dev/nvidiactl`
and `/dev/nvidia-uvm` are absent. Running
`MARIN_REQUIRE_CUDA=1 cargo test -p cli --test snapshot --test snapshot_safety -- --nocapture`
returns 101: all four tests in the first snapshot binary fail with
`no CUDA-capable device is visible`; Cargo then stops, so the second binary
is not reached. This deliberate required-device failure is retained in
`cuda-required-snapshots.log`; ordinary full-suite skips are not GPU evidence.

**Storage and memory audit.** `CpuStore::matrix` consumes one `Vec<u16>`
from the file loader, `Matrix::quantized` retains only the exclusive
`Storage::Int8(Q8)` variant, and the source BF16 vector drops on return.
The file loader retains paths/tensor metadata, not an mmap or a full-file
BF16 buffer; its read scratch is at most 4 MiB. Tied embedding/head share
one matrix. Norms intentionally retain small f32 vectors. The existing
`weight_bytes` estimate counts these norms at their BF16 source width;
actual CPU weight payload is 131,072 bytes higher on 0.6B and 247,808
bytes higher on 1.7B, for both weight formats, before vector/struct overhead.
This small accounting difference does not change the RAM gates. Decode expands
bytes into vector registers; SIMD prefill expands at most a four-row,
512-column panel into f32 (8 KiB), plus 2 KiB scratch and 128 KiB packed
activations, per worker. Scalar fallback expands one row at a time. There is no retained model-sized BF16
or f32 reconstruction. A new internal test checks code/scale byte lengths,
contents and allocation pointers remain unchanged through prefill and
decode; runtime RSS provides a separate check at real model scale.

There is a known startup inefficiency when `--snapshot` is requested on CPU:
`load_for_snapshot` still starts a full source-file hash, and the first
prompt joins it before discovering that CPU identity is unsupported.
That adds disk/CPU work without a usable snapshot; it does not retain a
full weight copy. Fixing the missing CPU snapshot capability requires a
separate coherent page-export/import change and real round-trip tests.
No unsupported behavior was relabeled as snapshot compatibility.

**Follow-up quality, speed and startup.** Two additional fixed conversation
corpora concatenate the eight previously saved prompts and their BF16 replies
in ChatML order (repeated four times), never using INT8 output as a target.
The first 64 tokens prime the cache and the following 1,024 positions are
scored. These are narrow tests of the same tasks, not a held-out benchmark
or proof of general model quality.

| Model | Same top-1 | Mean loss, BF16 / INT8 | Δ loss (nats) | Δ perplexity | Gate |
|---|---|---|---|---|---|
| 0.6B | 99.414% | 1.2607 / 1.2627 | +0.001962 | +0.196% | pass |
| 1.7B | 99.316% | 1.7411 / 1.7299 | -0.011156 | -1.109% | pass |

The lower measured 1.7B loss is a local rounding result; it does not establish
that INT8 improves the model. The original prose/code gates still apply.
Six real `marin` INT8 launches also compare a fixed roughly 200-token user
prompt at 1 and 8 threads and forced scalar: replies are byte-identical
within each model. The terminal switch tests and isolated kernel tests cover
actual prefill dispatch; the scalar override tests the Rust fallback, not
an artificially SIMD-disabled compiler.

Diagnostic repeats use the same original executable, eight prefill threads,
32 reply tokens, one warm-up and three measured rounds per case. Formats
run serially in blocks. This host now shows substantial timing drift and
RAM pressure; these measurements are **not** a replacement for the earlier
controlled 1/8-thread acceptance matrix or evidence of a new optimization.
For example, 0.6B BF16 short TTFT is now 766 ms rather than 303 ms, despite
identical executable bytes. Available RAM and cache residency also changed;
no single cause of the timing change was established. Values below are
BF16 / INT8; process RSS includes loading and all requests.

| Model | Prompt / context | TTFT (s) | Decode (tok/s) | Peak RSS (GB) |
|---|---|---|---|---|
| 0.6B | 127 / 160 | 0.766 / 0.361 | 12.80 / 28.59 | 1.308 / 0.727 |
| 0.6B | 2047 / 2080 | 7.835 / 7.799 | 11.34 / 14.91 | 1.892 / 1.331 |
| 1.7B | 127 / 160 | 1.306 / 0.923 | 5.26 / 10.28 | 3.567 / 1.900 |
| 1.7B | 2047 / 2080 | 18.027 / 13.716 | 4.99 / 8.39 | 4.224 / 2.606 |

The repeated RAM savings are 561–582 MB on 0.6B and 1.62–1.67 GB on 1.7B,
meeting the original RAM gate. Long 1.7B INT8 RSS is 2.606 GB versus the
original 2.593 GB: a 13 MB change, not an additional BF16 model allocation.
Both formats are slower in absolute time than the original run; INT8 still
has lower TTFT in these repeats and higher decode throughput. These
non-interleaved ratios should not be treated as stable performance estimates.

Fresh-process startup repeats use seven prompt tokens, two outputs, 16
configured positions, eight threads, no warm-up, three alternating-order
pairs per cache attempt. Fadvise-cold means per-file `DONTNEED`, not a cold
boot. The first 0.6B startup set overlapped full correctness checks; it is
preserved under `startup-overlap/` and excluded below. A separate
`startup-recheck-manifest.json` records the serial repeats after all checks
and other model runs completed.

| Model / condition | Median load, BF16 / INT8 (s) | INT8 penalty (s) | Physical load reads |
|---|---|---|---|
| 0.6B / fadvise-cold | 1.413 / 1.913 | +0.500 | 0.978–0.980 GB BF16; 0.978 GB INT8 |
| 0.6B / pre-read attempts | 0.904 / 1.440 | +0.536 | 2/3 zero-I/O runs per format; first pair read 1.075 / 0.088 GB |
| 1.7B / pre-read attempts | refused / 4.634 | unavailable | INT8 2.678–3.007 GB |
| 1.7B / fadvise-cold | refused / 4.561 | unavailable | INT8 3.078–3.183 GB |

The 0.6B penalty remains below its 3 s cap. Pre-reading a checkpoint did
**not** guarantee cache warmth under these conditions: the first 0.6B pair
and all 1.7B INT8 pre-read trials still performed physical I/O. Do not label
the mixed pre-read medians as guaranteed page-cache-warm latency. All six
1.7B BF16 startup attempts were refused by the memory guard before model
loading (e.g. 3.71 GB needed, 3.64 GB available); INT8 completed all six,
with peak RSS about 1.899 GB. Thus a fresh 1.7B load-penalty comparison
cannot be computed here. The original complete cold/warm evidence is kept,
but repeating its warm comparison on an uncontended host remains open.
Manifest entries retain exact errors, available memory and load averages;
failed commands are not silently omitted or assigned zero load time.

**Closure checks and decision.** Workspace/all-targets check,
all-targets/all-features clippy with `-D warnings`, fmt, line-limit and unsafe
checks pass. Full workspace/all-features tests report **583 passed**;
GPU-dependent test functions may return early, so this count is not CUDA
runtime coverage. Release forced-scalar INT8 integration kernels pass, as
do isolated SIMD/scalar 1/2/8/16-thread prefill and immutable-storage tests.
The explicit required-CUDA failure described above is a separate failed
hardware gate, not a suppressed full-suite failure. No tolerances or skips
were changed. Command arrays and return codes are in `checks.json`.

**Earlier closure decision (before the hardware recheck): BLOCKED.** Strict 0.6B HF parity and CPU storage/compatibility
checks are now established; extended measured relative quality passes.
Successful CUDA switches and GPU snapshot recovery were unavailable, and
the fresh 1.7B startup pair was blocked by RAM admission. CPU snapshots are
unsupported in both formats and outside the original gate, not an INT8
regression or an acceptance blocker. Keep BF16 default and INT8 opt-in.
Do not start Stage 4 implementation or INT4 from this audit. The plan below
is conditional, not permission to bypass these remaining gates.

**CPU Stage 4 proposal — measured baseline, no implementation.** Both
models have 28 layers, eight KV heads, head dimension 128. The present CPU
cache reserves the complete contiguous f32 window with `vec![0.0; ...]`;
Linux demand paging means reservation is not the same as resident RAM.
`Footprint::ensure_fits` admits the full window plus a fixed 256 MiB slack,
while CPU attention requires contiguous `&[f32]` keys/values. GPU slab/page
ownership and `SequenceCache` commit/truncate rules are useful references;
porting GPU pointers or adopting paging immediately is unnecessary.

| Positions | f32 KV bytes | f16 KV bytes (projection) |
|---|---|---|
| 128 | 29,360,128 | 14,680,064 |
| 2,048 | 469,762,048 | 234,881,024 |
| 8,192 | 1,879,048,192 | 939,524,096 |

At 127 populated tokens, increasing configured context from 160 to 8,192
raises INT8 peak RSS only 727 → 756 MB (0.6B) and 1,900 → 1,936 MB
(1.7B); unused zero pages are mostly not resident. In contrast, 1.7B BF16
with an 8K reservation is refused before loading: estimated need 5.59 GB,
4.68 GB available at that launch. Growable KV should improve committed
capacity/admission and growth control; do not claim a 1.88 GB startup RSS
saving on this Linux host.

F16 is the first lower-precision candidate: section 12's earlier CUDA
measurements passed quality, while Q8 keys failed. Those are GPU results,
not CPU predictions. Fully populated 8K **INT8 weights + f16 KV** would be
2,714,234,880 bytes of payload, including f32 norms, before workspace/allocator
overhead; adding current 256 MiB slack gives 2,982,670,336 bytes. A 3 GB target has almost no margin.
CPU `hidden`/`Buffers`/`MlpBuffers` contain
`4 * (4*hidden + 2*q_dim + 2*kv_dim + 2*intermediate)` bytes per prompt token:
64 KiB on 0.6B and 104 KiB on 1.7B. At 8,192 tokens that is 537 / 872 MB
of named buffers alone, excluding vector capacity, attention scratch and
quantized-kernel panels. A lower KV format alone cannot satisfy that budget.
A 512-token chunk would bound these named buffers to 34 / 55 MB; this is
an arithmetic projection, not an implemented or timed optimization.

Proposed small sequence, conditional on closing Stage 3 first:

1. **Grow contiguous f32 KV first.** Keep the logical context limit separate
   from allocated positions; start with 256 positions and grow in bounded
   increments, with checked sizes and fallible reservation. Preserve prefix,
   clear/truncate and all-layer commit rules on allocation failure. Expose
   actual vector capacities/bytes, not RSS alone. Gates: exact f32 reference
   logits and replies, strict HF unchanged, 1/8-thread behavior, no committed
   length change on failure; at 127 used positions with an 8K limit, KV
   allocation at most one 256-position increment. Growth-event p95 latency,
   TTFT and decode must be measured; no more than 5% TTFT or 3% decode loss
   against the existing f32 path at 127/2K/8K. Measure resize-copy peaks before
   deciding whether CPU pages earn an attention-layout change.
2. **Opt-in f16 KV.** Store 16-bit data and decode bounded attention tiles,
   retaining f32 accumulation and the f32/scalar reference. Do not maintain
   a full f32 mirror. Reuse section 12's independently rounded reference
   and test finite-range/subnormal/saturation boundaries. Gates on both
   models, BF16 and INT8 weights, 1/8 threads: KV bytes at most 55% of f32;
   at least 4,096 scored positions each of prose/code/conversation, loss
   increase at most 0.30%, same top-1 at least 97.5%, at least 6/8 identical
   greedy replies; decode no more than 3% slower at 2K/4K/8K and TTFT no
   more than 5% worse. Keep f32 default until separate default gates pass.
   Do not start Q8 keys or any INT4 experiment.
3. **Bound whole-process RAM.** Account for source conversion temporaries,
   actual KV allocation/growth, all prompt buffers, panels, tokenizer and
   allocator headroom. Compare current full-prefill with 256/512-token
   chunks using existing cache append semantics before choosing a chunk.
   Validate with real 2/3/4 GiB process limits on a host that exposes the
   limits, plus the 8 GB/16 GB machine profiles; stop admission/growth
   cleanly and preserve the last committed prefix. Target 1.7B INT8/f16,
   fully populated 8K, peak RSS below **3,000,000,000 bytes**, with no hidden
   weight/KV mirror, swapping or OOM. Report virtual reservation, RSS/PSS,
   cold/warm startup, TTFT, decode, resize peaks, page faults and correctness
   at equal populated contexts. This target remains unmeasured and may
   require bounded prefill; it is not inherited from the old INT4 roadmap.

No Stage 4 source, format, dependency or CLI option was added. No commits,
pushes or stash changes were performed.

**Final Stage 3 acceptance recheck (2026-10-11; RTX 5050 available).**
**GO: every mandatory original gate passes for Qwen3-0.6B and Qwen3-1.7B,
CPU INT8 group 128.** This supersedes the earlier BLOCKED decisions above.
BF16 remains the default/reference and INT8 remains opt-in. No Stage 4 or
INT4 experiment, runtime change, dependency, unsafe code, commit, push or
stash change was made. The added test extends the existing terminal helper
to cover successful CPU → CUDA → CPU → CUDA switches and a model switch,
alongside the retained hidden-GPU failure/recovery case.

The recheck uses the unchanged release binary whose SHA256 is recorded
above, on the same Ryzen 7 260 / 16 GB host with the powersave governor and
AC power. The RTX 5050 Laptop GPU now reports 8,151 MiB, driver 610.57.04.
Fresh command arrays, return codes, hardware/source hashes, admission
conditions and measurements are in `bench-results/cpu-stage3/acceptance/`.
Tests finish before timing; model/format loads and timing runs are serial.
The earlier Stage 4 proposal is retained from the previous request; this
recheck does not investigate or implement that proposal.

| Original gate / scope | Classification | Exact evidence |
|---|---|---|
| Kernel correctness | **passed** | Independent decoded f64 GEMV/GEMM tests, zero/signed-extreme groups, shape rejection, remainder cases and groups 32/64/96/128/256/512; full SIMD tests plus release forced-scalar `scalar-kernels.log` |
| BF16 / FP32 reference correctness | **passed** | Fresh CPU and CUDA strict HF checks on both chat models: three prompts × 16 exact greedy steps per model/backend; `verify-{0.6,1.7}-{cpu,cuda}.log`. Earlier 1/8-thread and scalar HF checks remain valid; tolerances unchanged |
| Relative quality | **passed** | Four fresh 1,024-position prose/code pairs below; all eight scoring arrays exactly reproduce original top-1 and target log-probabilities. Original eight-prompt semantic review remains 7/8 byte-exact on 0.6B, 8/8 on 1.7B, no new observed semantic regression |
| Decode | **passed** | All eight fresh pairs below exceed ≥1.25x short / ≥1.10x long; one warm-up and three measured rounds per format, 32 output tokens |
| Prefill / TTFT | **passed** | All eight fresh pairs stay within +10%; worst observed regression +5.0204%, 1.7B / one thread / 127 tokens |
| Stored weights / peak RAM | **passed** | Storage 51.57% of BF16; every matched-window peak-RSS saving exceeds 35% of BF16 weight bytes. Exclusive INT8 storage, per-tensor loading and bounded panels remain unchanged; immutable-code/scale tests pass |
| Cold start | **passed** | Three filesystem-cold and three verified zero-read warm fresh loads per format/model; penalties +0.304/+0.384 s (0.6B cold/warm), +0.495/+0.776 s (1.7B). Separate process-to-first-output measurements below |
| CLI / dispatch / threads / KV reuse | **passed** | Both CLIs, default BF16, invalid options, CPU INT4 refusal, hidden-GPU auto fallback, successful and failed backend/model switching; scalar kernels and isolated SIMD/scalar 1/2/8/16-thread prefill/storage tests pass. Existing real-model retained-cache evidence remains valid |
| Existing CUDA / GPU snapshots | **passed** | Full suite with `MARIN_REQUIRE_CUDA=1`: quantized-model suite 4/4 (3 GPU, 1 CPU rejection), snapshots 4/4, snapshot safety 4/4 (3 GPU, 1 CPU unsupported), CLI precision 5/5, inspect/reload 3/3, INT8 switching 2/2 (combined CPU/GPU cases). GPU restore/continuation, corruption/incomplete files, incompatible weights/KV/model identity and fresh-process CLI snapshots pass; group incompatibility is additionally covered by header identity tests |
| Local llama.cpp reference | **passed** | Original pinned local BF16/Q8_0 results above and `final/` artifacts, matching 1/8 threads, short/long depths, CPU-only and f32 KV. Unchanged source/binary; pp/tg remain distinct from generated-output TTFT |
| CPU energy availability reporting | **passed** | Package-counter reads still return permission denied, recorded in `environment.json`; original gate explicitly permits unavailable energy |
| Numerical CPU energy measurement | **blocked, optional** | No readable package counter; no joules/token claim and no acceptance blocker |
| CPU snapshot implementation | **out of scope** | Unsupported before INT8 and still unsupported for BF16/INT8; the original gate explicitly says existing GPU snapshots. Safe ignored-request tests are not presented as successful CPU restoration |
| Full cold boot, older/non-x86 PCs, larger models, other group-size quality, llama.cpp generated-output TTFT, Stage 4 / INT4 | **out of scope** | Not required by the fixed Stage 3 gates or not authorized here; no results inferred |

Fresh medians below use prompts 127/2,047 and identical f32-KV windows
160/2,080 respectively; GB means decimal bytes. Each row is BF16 / INT8.

| Model | Threads | Prompt | TTFT (s) | Decode (tok/s) | Peak RSS (GB) |
|---|---:|---:|---:|---:|---:|
| 0.6B | 1 | 127 | 1.129 / 1.063 | 19.64 / 36.37 | 1.301 / 0.727 |
| 0.6B | 1 | 2,047 | 25.674 / 24.518 | 13.80 / 22.02 | 1.874 / 1.314 |
| 0.6B | 8 | 127 | 0.304 / 0.263 | 19.11 / 34.99 | 1.308 / 0.728 |
| 0.6B | 8 | 2,047 | 5.262 / 4.958 | 13.43 / 20.15 | 1.890 / 1.329 |
| 1.7B | 1 | 127 | 3.600 / 3.781 | 6.91 / 12.78 | 3.557 / 1.899 |
| 1.7B | 1 | 2,047 | 66.521 / 66.952 | 6.29 / 10.55 | 4.212 / 2.576 |
| 1.7B | 8 | 127 | 0.781 / 0.611 | 7.17 / 13.11 | 3.570 / 1.900 |
| 1.7B | 8 | 2,047 | 11.829 / 10.064 | 6.25 / 10.37 | 4.227 / 2.591 |

Decode speedups are 1.500–1.852x; peak-RSS savings are 560–580 MB on
0.6B and 1.636–1.670 GB on 1.7B. Stored weight accounting remains
1,192,099,840 / 614,739,968 bytes and 3,441,149,952 / 1,774,462,976 bytes.
The small common f32-norm accounting difference documented earlier does
not change either storage or RAM verdict. Source/storage inspection and
tests confirm no retained full-weight BF16 copy; temporary per-tensor BF16,
f32 rows/panels and activations remain bounded as described above.

| Model / corpus | Same top-1 (out of 1,024) | Loss change (nats/token) | Perplexity change |
|---|---:|---:|---:|
| 0.6B / prose | 999 (97.5586%) | +0.002940 | +0.294410% |
| 0.6B / code | 1,003 (97.9492%) | −0.003038 | −0.303324% |
| 1.7B / prose | 1,004 (98.0469%) | −0.003562 | −0.355518% |
| 1.7B / code | 1,011 (98.7305%) | +0.003040 | +0.304416% |

These establish relative quality preservation on the fixed corpora, not
a broad improvement in quality. The 0.6B prose result has only one matched
position above the minimum passing count. The earlier semantic review's
baseline errors still apply; neither lower loss nor wording equality proves
general task success. Fresh strict HF checks retain `5e-4 + 1e-4*abs(logit)`
and `1e-4` relative layer-norm tolerances. CUDA maximum logit errors are
1.91e-5 (0.6B) and 1.72e-5 (1.7B); all greedy choices match.

**RAM guard and startup.** Admission still reserves weights + the complete
f32 KV window + 256 MiB. BF16 1.7B requires 3,713,255,424 bytes at context
16 or 5,588,633,600 at 8,192; previous available RAM was below those limits,
so refusal was correct. Fresh admission logs show sufficient host RAM;
all process/ancestor cgroups have unlimited `memory.max`/`memory.high` and
zero OOM events. No limit was changed or bypassed. Both formats also pass
the 8,192-window diagnostic naturally, with peak RSS 3.585 / 2.021 GB;
unused f32 KV pages are demand-paged, so this is not a populated-8K result.

| Model / cache condition | Load BF16 / INT8 (s) | Load penalty (s) | Process to first output BF16 / INT8 (s) |
|---|---:|---:|---:|
| 0.6B / zero-read warm | 0.612 / 0.996 | +0.384 | 0.921 / 1.334 |
| 0.6B / fadvise-cold | 1.142 / 1.446 | +0.304 | 1.466 / 1.827 |
| 1.7B / zero-read warm | 1.497 / 2.273 | +0.776 | 2.006 / 2.829 |
| 1.7B / fadvise-cold | 3.054 / 3.549 | +0.495 | 3.063 / 3.879 |

Load uses seven prompt/two output tokens, context 16; first-output uses
the normal `marin-run` generation path, fixed five-token prompt, one output,
context 64. Both configure eight threads and group 128; three fresh
processes per format/condition. Warm loads and first-output trials report
zero physical reads. Cold loads read 977.69–977.79 MB (0.6B) and
3,069.28–3,077.33 MB (1.7B); cold first-output trials read 977,690,624 and
3,069,280,256 bytes respectively. Cold means per-file `POSIX_FADV_DONTNEED`,
not a cold boot or system-wide cache drop. Initial single-pre-read trials
with residual I/O are retained as diagnostics and excluded from the
zero-read warm medians. `fincore`, process I/O, admission conditions and
every attempt are recorded in `warm-manifest.json` / `first-output-io.json`.
INT8 adds measurable load-time conversion cost; the load and TTFT limits
pass, and observed RAM/quality results reproduce the earlier evidence.

**Final verification:** fmt, locked workspace/all-targets check,
all-targets/all-features clippy `-D warnings`, locked full workspace tests
with CUDA required (**584 passed**), line-limit and unsafe checks all pass.
Release forced-scalar kernels/storage and isolated SIMD/scalar prefill
thread tests pass. `checks.json` and logs record commands and outcomes;
`summary.json`, `verified-warm-summary.json` and
`first-verified-summary.json` contain exact unrounded values. Prior
uncommitted changes/reference fixtures and all three stash identities are
preserved; HEAD is still `9d41f88`. No genuine Stage 3 runtime regression
was found. **GO for Stage 3 under the fixed gates and measured scope;
no Stage 4 or INT4 work is started or authorized by this result.**

Raw original artifacts are in `bench-results/cpu-stage3/final/`: command manifests,
JSON speed/quality/load/profile records, reference logs, first-visible-output
and semantic outputs/review, GGUF provenance samples, hardware/build metadata,
and the isolated llama f32 parser build script. Parent-directory results
record earlier failed kernels and the pre-memory-fix measurements; they are
not mixed into the final matrix. These artifacts are ignored by git under
the existing repository convention.

Reproduce the measured settings (repeat for both models, threads 1/8 and
prompts 127/2,047; omit `--quantize` for BF16):

```bash
cargo build --release -p cli --locked
marin-run bench MODEL --backend cpu --threads 8 --quantize int8 --quant-group 128 --prompt-tokens 2047 --tokens 32 --ctx 2080 --runs 3 --warmup 1 --json int8.json
marin-run quality MODEL --backend cpu --threads 8 --text TEXT --tokens 1088 --save bf16-quality.json
marin-run quality MODEL --backend cpu --threads 8 --quantize int8 --text TEXT --tokens 1088 --against bf16-quality.json --save int8-quality.json
# This pinned llama-bench needs the documented f32 parser branch.
llama-bench -m MODEL-Q8_0.gguf -ngl 0 -t 8 -p 127,2047 -n 0 -r 3 -ctk f32 -ctv f32 -fa off -b 2048 -ub 2048 -o json
llama-bench -m MODEL-Q8_0.gguf -ngl 0 -t 8 -p 0 -n 32 -d 2047 -r 3 -ctk f32 -ctv f32 -fa off -b 2048 -ub 2048 -o json
```

## Tried and not adopted

- **bf16 head with int8 layers; int8 values in the KV cache:** ≤0.2 points of quality for 7–9% of speed; no decode gain and a slower first token (section 17).
- **bf16 and q8 KV cache:** bf16 had no advantage over f16; q8 failed the quality gates (section 12).
- **CUDA graphs:** +1.2% in a prototype, needs invasive state changes (section 3). More kernel fusion: a few tenths of a percent each.
- **4-bit output head:** hurts quality badly.
- **Guessing after 1–5 matching tokens, or pausing after a wrong guess:** slowdowns outside edits (section 5).
- **Idle pauses between tokens:** raise engine speed but lower real speed (power cap).
- **Table lookup per group of four value rows in prefill; one 14 MiB allocation per page:** +9.1% and +12.9% TTFT (section 11).
- **Draft-model speculation; RAM parking of sessions; prefill matmul prototype:** see sections 15, 14 and 19.

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
# chat turns on a kept cache vs read from scratch (add --quantize int8 for int8)
marin-run turns MODEL --history 0,2048,4096 --reply-tokens 32 --rounds 3 --ctx 5120
# KV memory held for a short sequence in a large window
marin-run bench MODEL --backend cuda --prompt-tokens 63 --tokens 2 --runs 1 --warmup 0 --ctx 8192
# plain vs guessing on real replies
marin-run spec-bench MODEL --sampling chat --max-tokens 120 --prompts PROMPT_FOLDER --rounds 1
# KV cache format: quality against f32 KV
marin-run quality MODEL --text TEXT_FILE --tokens 4096 --kv-type f32 --save f32.json
marin-run quality MODEL --text TEXT_FILE --tokens 4096 --kv-type f16 --against f32.json
# restart: save a prefix, then restore it and read the cold baseline in separate processes
marin-run restart MODEL --mode save --tokens 4096 --file snap.bin --kv-type f16
marin-run restart MODEL --mode restore --tokens 4096 --file snap.bin --kv-type f16
marin-run restart MODEL --mode cold --tokens 4096 --file snap.bin --kv-type f16
# accuracy of quantized weights
marin-run quality MODEL --quantize int8 --text TEXT_FILE
# output against Hugging Face (GPU, or CPU with --backend cpu)
marin-run verify MODEL --reference tests/reference/qwen3-1.7b-base.json --backend cuda
# CPU speed, one thread
marin-run bench MODEL --backend cpu --prompt-tokens 127 --tokens 32 --runs 2 --warmup 1 --profile
```

Add `--quantize int8` to the first, third and fourth commands for int8 runs. The guessing replay tools are `marin-run guess`, `guess-report` and `pass-cost`. Every table lists its sample size next to it; most speed rows are 2–5 runs per side on one machine, so treat differences of a few percent as noise unless the ranges are clearly apart.

### 26. Adaptive CPU KV (stage 4): gates fixed before implementation

Stages are sequential: A growable FP32 storage, B RAM budgeting and bounded
prefill, C opt-in FP16 feasibility. No later implementation starts while an
earlier mandatory gate is failed or unmeasured. BF16 weights and FP32 CPU KV
remain defaults. CPU snapshots are unsupported and outside this milestone.

| Stage / area | Mandatory acceptance gate |
|---|---|
| A correctness | Unchanged strict HF verification for both Qwen3 models; identical greedy outputs against the saved Stage 3 binary with BF16 and INT8 weights; scalar/SIMD, 1/8 threads, growth boundaries, clear/reuse, truncation, failed writes and rollback preserve every committed layer's prefix |
| A allocation | Logical context stays independent of allocation; at 127 held tokens and context 8192, actual KV vector capacities cover at most 256 positions; report actual capacity bytes, not estimated RSS savings; arithmetic overflow and allocator refusal return errors without committing tokens |
| A speed | Both models, INT8, 127/2047/4095/8159 prompt tokens, 32 outputs, context 8192, 1/8 threads: one warm-up and three measured rounds; median TTFT regression ≤5%, decode regression ≤3% versus saved Stage 3 binary. BF16 correctness and short/2k speed also checked. Growth latency/copy peaks reported separately |
| A peak RAM | Same workloads and process lifetime: peak RSS no more than 64 MiB above baseline; no full-model dequantization or second full KV mirror |
| B correctness | Chunked versus unbounded prefill passes strict HF tolerances and identical greedy outputs with both weight formats; an admission/growth failure preserves the previously committed prefix; explicit unsupported context still fails clearly |
| B RAM / OOM | Effective cgroup-v2 4 GiB and 8 GiB limits, swap disabled, both models at short/2k/4k/8k; report memory.max/current/events, peak RSS and KV allocation. Zero OOM kills; clean rejection where weights + KV + bounded workspace cannot fit. Account ancestor limits and retain safety headroom; no override of genuine limits |
| B performance | Bounded prefill at 512 tokens maximum; same A timing matrix, TTFT regression ≤10% and decode ≤3% versus A; report startup warm/cold and process-to-first-output, allocator/growth peaks and output checks |
| C FP16 | Only after A/B pass: safe scalar conversion and available SIMD paths measured; storage ≤55% of FP32; no full FP32 KV mirror; both models and weight formats, ≥4096 teacher-forced positions each prose/code/conversation: top-1 ≥97.5%, relative mean loss increase ≤0.30%, ≥6/8 identical greedy prompts; TTFT ≤5% and decode ≤3% regression at 2k/4k/8k. FP32 remains default |
| All compatibility / checks | INT8 opt-in, scalar fallback, threading, KV reuse and RTX 5050 CUDA/snapshot regressions preserved; fmt, workspace all-target check, all-feature clippy -D warnings, full all-feature tests, 425-line and unsafe checks pass |

Baseline binaries and pre-change working-tree hashes are retained under ignored
`bench-results/cpu-stage4/baseline/`. The old Stage 3 proposal above remains
historical planning, not measurements or Stage 4 acceptance. Linux demand-zero
full-context vectors need not consume their entire virtual size in RSS.
Fallible reservation preserves data on allocator refusal ([Rust Vec docs](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve_exact));
it does not prevent a later physical-memory OOM under overcommit. Stage B must
therefore use actual memory constraints ([kernel cgroup-v2 documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)),
not treat an address-space limit as a RAM limit.


CPU allocation audit: KV is `[position][kv_head][head_dim]` per layer,
with a separate committed length. Truncation/clear retain storage; runtime
rollback uses the same length. Each Qwen3 model needs 229,376 FP32 KV
bytes per position (1,879,048,192 bytes at 8192). CPU attention reads only
the visible prefix; SIMD prefill temporarily packs one KV head per worker,
not a second full-cache mirror. Decode reads the cache directly. Growable
storage reuses the GPU cache's logical-limit/high-water distinction without
introducing CPU page tables or changing CUDA allocation.

The existing CLI admission estimate still charges the full logical FP32
window and reads host `MemAvailable`; it does not yet account cgroup limits.
The CPU forward pass still sizes reusable intermediates to the entire
prompt: named activation buffers are 64/104 KiB per token for 0.6B/1.7B,
excluding attention packing, quantization panels and logits. These are
source-derived estimates, not measured peak RSS. Stage A alone therefore
does not establish graceful low-RAM inference; that remains Stage B's gate.

FP16 API audit only: Rust's [native `f16`](https://doc.rust-lang.org/std/primitive.f16.html)
is experimental. The [half crate](https://docs.rs/half/latest/half/#hardware-support)
offers safe slice conversion, independent F16C detection and software fallback.
AVX2 dispatch alone does not establish F16C support. These are published API
facts, not conversion throughput or model-quality measurements. No dependency
or CPU FP16 inference path has been added; feasibility remains gated on A/B.

**Stage A: GO (2026-10-12).** Growable contiguous FP32 KV passes its fixed
gates; stages B and C have no implementation yet. Storage starts
empty, reserves all layers before a forward pass, grows in 256-position
increments and retains its high-water allocation on clear/truncation. Only
visible storage is initialized; attention kernels and weights are unchanged.
Both CLIs expose actual allocated bytes separately from the window limit.

All six CLAUDE checks pass, including 587 workspace tests with the RTX 5050
visible (`MARIN_REQUIRE_CUDA=1`). Both models pass unchanged strict HF
verification at 1/8 threads with SIMD and forced scalar dispatch: eight
commands, 48 greedy steps each, with layer observations. A separate 128 MiB
**address-space** limit forces partial all-layer allocator refusal: vector
capacity grows from 58,720,256 to 123,731,968 bytes before refusal; every
committed prefix survives and clear/reuse succeeds. This is not a physical
4 GiB RAM test. Raw evidence is under `bench-results/cpu-stage4/`.

Completed pairs: context limit 8192, 32 outputs, one warm-up plus three
measured rounds per process. The two repeated cases below use all fifteen
original and repeat rounds per build; every other row uses three. The table
compares the saved Stage 3 binary with growable KV. These
are local Ryzen 7 260 measurements, not published performance results.

| Model | Threads | Prompt | Weights | TTFT ms (old / grown) | Decode tok/s (old / grown) | Peak RSS delta MiB | KV capacity MiB |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.6B | 8 | 127 | int8 | 264.2 / 254.9 | 35.244 / 34.792 | -83.44 | 56 |
| 0.6B | 8 | 127 | bf16 | 303.4 / 294.2 | 18.760 / 19.025 | -0.27 | 56 |
| 1.7B | 8 | 127 | int8 | 750.2 / 715.7 | 11.906 / 12.020 | -25.65 | 56 |
| 1.7B | 8 | 127 | bf16 | 955.3 / 853.8 | 6.525 / 6.592 | -10.57 | 56 |
| 0.6B | 8 | 2047 | int8 | 4620.3 / 4389.1 | 20.810 / 20.839 | -36.94 | 504 |
| 0.6B | 8 | 2047 | bf16 | 4714.0 / 4820.7 | 14.017 / 13.968 | +19.54 | 504 |
| 1.7B | 8 | 2047 | int8 | 11030.9 / 11010.0 | 9.557 / 9.588 | +50.36 | 504 |
| 1.7B | 8 | 2047 | bf16 | 13192.7 / 13141.0 | 5.792 / 5.873 | +37.82 | 504 |
| 0.6B | 8 | 4095 | int8 | 16174.4 / 16185.7 | 14.544 / 14.646 | -21.60 | 952 |
| 1.7B | 8 | 4095 | int8 | 32521.2 / 27228.7 | 8.150 / 8.143 | +52.82 | 952 |
| 0.6B | 8 | 8159 | int8 | 80275.5 / 79816.8 | 9.302 / 9.270 | -10.78 | 1792 |
| 1.7B | 8 | 8159 | int8 | 104095.1 / 100743.4 | 6.259 / 6.268 | +0.19 | 1792 |
| 0.6B | 1 | 127 | int8 | 1227.0 / 1134.1 | 36.173 / 36.744 | -83.51 | 56 |
| 0.6B | 1 | 127 | bf16 | 1156.9 / 1142.5 | 19.347 / 19.445 | -0.51 | 56 |
| 1.7B | 1 | 127 | int8 | 3699.4 / 3815.1 | 12.770 / 12.853 | -88.67 | 56 |
| 1.7B | 1 | 127 | bf16 | 3669.9 / 3542.4 | 6.909 / 6.862 | +4.59 | 56 |
| 0.6B | 1 | 2047 | int8 | 24044.5 / 24303.4 | 22.103 / 21.911 | -41.22 | 504 |
| 0.6B | 1 | 2047 | bf16 | 24486.8 / 24168.3 | 14.447 / 14.367 | +0.13 | 504 |
| 1.7B | 1 | 2047 | int8 | 64818.5 / 66604.3 | 10.296 / 10.259 | +43.77 | 504 |
| 1.7B | 1 | 2047 | bf16 | 69012.9 / 63271.2 | 5.985 / 6.082 | +37.45 | 504 |
| 0.6B | 1 | 4095 | int8 | 65892.4 / 61595.3 | 15.314 / 15.320 | -47.72 | 952 |
| 1.7B | 1 | 4095 | int8 | 142854.8 / 141528.6 | 8.573 / 8.588 | -38.28 | 952 |
| 0.6B | 1 | 8159 | int8 | 182569.6 / 180900.2 | 9.676 / 9.636 | +0.14 | 1792 |
| 1.7B | 1 | 8159 | int8 | 350754.5 / 335899.9 | 6.507 / 6.536 | +0.04 | 1792 |

The initial 0.6B/8-thread/2k INT8 pair missed decode by 3.11%. A
follow-up used four alternating-order pairs, all twelve measured
rounds per build: TTFT 4684.293/4371.755 ms, decode 20.7996/20.8988 tok/s
(baseline/grown). The pooled ratios 0.9333/1.0048 pass. All initial and
repeat results are retained; no sample was dropped. The measurement
controller was paused while an existing BF16 child completed, to serialize
these retries. Its manifest wall duration includes that controller pause;
CLI-internal TTFT/decode values do not.

The initial 1.7B/1-thread/127-token INT8 pair missed TTFT by 9.44%.
Four alternating-order pairs (twelve rounds per build, no discarded samples)
give pooled TTFT 3694.699/3758.526 ms and decode 12.7680/12.8490 tok/s:
ratios 1.0173/1.0063 pass. Original and repeat data are retained in
`retry-short-*.json`. The controller pause again affects only manifest
wall duration, not the already running child's internal timing.

Pooling the original three rounds with every repeat gives **15 samples
per build**. The 0.6B/8-thread/2k ratios are 0.94995 TTFT and 1.00143
decode; the 1.7B/1-thread/short ratios are 1.03127 and 1.00655. Both
pass with the original misses included (`retry-*-all-samples.json`).

All 32 semantic comparisons (eight prompts × two models × two weight formats)
and 16 long-prompt comparisons are byte-identical to the saved Stage 3 binary.
The latter feed actual 124/1987/3975/7920-token text after decode/re-encode;
the speed matrix feeds exact 127/2047/4095/8159-token IDs. Both use context
8192 and eight threads. Original CLI stdout/stderr are retained.

The separate synthetic growth experiment writes zero K/V to all 28 layers,
256 positions at a time through 8192, ten rounds per build (320 steps each).
It measures reserve/initialize/write together, excludes cache construction,
and uses fresh caches. These are allocator measurements, not inference or
4/8 GiB physical-RAM tests.

| Growth microbenchmark | Stage 3 full-window vectors | Growable vectors |
|---|---:|---:|
| Median step / p95 / maximum (ms) | 8.168 / 17.739 / 21.259 | 20.985 / 28.410 / 30.001 |
| Median total through 8192 (ms) | 226.725 | 715.847 |
| Maximum process RSS (bytes) | 1,882,804,224 | 1,882,959,872 |
| Final KV capacity (bytes) | 1,879,048,192 | 1,879,048,192 |

Incremental initialization/reallocation adds 489.123 ms in this synthetic
sequence and 155,648 bytes to its peak RSS. The measured inference matrix
still clears the speed gate. Partial allocator-refusal evidence above
covers preserved committed prefixes; it cannot prove physical OOM safety.

| Stage A gate | Decision | Exact evidence |
|---|---|---|
| Numerical / outputs / rollback | PASS | Eight strict HF commands, 384 greedy steps; 32 semantic + 16 long outputs exact; growth/truncate/clear/replay and partial allocator-failure prefix checks |
| Actual allocation / limits | PASS | Empty initial vectors; 127 positions use 256-position capacity (58,720,256 bytes); logical 8192 limit retained; checked overflow/refusal tests |
| Speed | PASS | All 24 pairs; worst accepted TTFT regression 3.13%, decode regression 1.28%; two cases pool all 15 original/repeat samples |
| Peak RAM / representation | PASS | Worst matrix RSS delta +52.82 MiB <64 MiB; repeated-case peak includes all five processes per build; CPU weights remain packed INT8 and attention packs one head per worker |
| Compatibility / required checks | PASS | Six checks exit 0, 587 tests passed with RTX 5050 required; scalar/SIMD and 1/8-thread HF verification; GPU switching/quantization/snapshot regressions exercised |
| Stage B low-RAM / bounded prefill | UNMEASURED | Not implemented yet; Stage A does not establish graceful physical-memory OOM handling |
| Stage C FP16 | GATED | No conversion throughput, quality or model-speed claim; FP32 remains default |

Raw evidence: `stage-a-acceptance.json`, `stage-a-manifest.json`, all original
and retry benchmark JSON, `retry-peak-rss-all-processes.json`, `correctness.json`,
`long-correctness.json`, `growth-summary.json`, and six check logs under
`bench-results/cpu-stage4/`. Stage A is accepted; overall Stage 4 is not yet
accepted. Real 4/8 GiB RAM runs, bounded prefill and FP16 remain outstanding.


**Stage B implemented, validation pending (2026-10-12).** CPU prefill is split
into at most 512 tokens per pass, reusing the existing layer buffers. A failure
in any chunk rolls committed length back to the original prefix; observers
still report only the final token once per layer. Twelve saved Stage A
numerical cases (512/513/1025 positions, both models and weight formats,
eight greedy steps each) pass unchanged logit and norm tolerances and every
greedy choice. These comparisons preceded the memory-budget implementation;
they will be repeated on the final build.

CPU admission now budgets weights, the full requested FP32 window, bounded
named workspace and 256 MiB headroom against host MemAvailable and visible
cgroup-v2 ancestor memory.max/high limits. Default contexts may shrink;
explicit contexts never shrink silently. CPU tensor reads and each forward
chunk recheck available memory before allocation, with 64 MiB growth
headroom, actual initialized KV high-water bytes, possible single-vector
copy space and attention packing capacity. Retained vectors that already
fit are not charged a reallocation copy. No CUDA allocation code changed.

The reclaimable-file estimate excludes shmem, dirty, writeback and
unevictable pages from memory.stat file bytes, then constrains it by
memory.current and host MemAvailable. This uses the documented counters;
it is a conservative planning estimate, not a kernel reservation or a
guarantee against concurrent external pressure. Unreadable identified
Linux limits fail closed. Physical 4/8 GiB runs and final speed/startup
comparisons remain unmeasured; Stage B and overall Stage 4 are not accepted.


First Stage B build: all six checks passed (593 tests), including unchanged
trace assertions after moving memory preparation into the embedding timer.
The earlier trace-order failure and interrupted copy-budget check attempt
are retained separately. Final-build numerical/HF checks passed. The real
384 MiB, swap-off ballast test rejected growth in both weight formats at
57,573,376 / 56,643,584 available bytes, preserved all committed layers,
and resumed with exact replay after release; memory.peak 346,013,696 bytes,
all memory.events counters zero.

The first 4 GiB/1.7B/INT8/8k profile **rejected during growth**, despite
nominal window admission: peak RSS 4,046,729,216 bytes; a subsequent chunk
needed 327 MB while only 236 MB remained. memory.max was 4,294,967,296,
memory.swap.max 0; max 12397, oom 0, oom_kill 0. Its wrapper exit 0 records a
verified clean rejection, not successful inference. No output-parity pass
is claimed for that case. BF16/8k rejected at admission under 4 GiB;
BF16/short also rejected under the 3 GiB process budget that reserves 1 GiB
for a hypothetical 4 GiB system's OS/apps. BF16/8k completed under 8 GiB,
peak RSS 5,680,910,336 bytes, exact Stage 3 output, all event counters zero.

Investigation found that reused named buffers were charged again on each
chunk, and attention packing performed repeated vector growth. The next
build credits only already initialized buffer values within the new chunk's
shape and reserves each head's final packing size upfront. Neither the
64/256 MiB headroom nor actual memory limits are reduced. These changes
require new correctness, speed and RAM evidence; no new GO is claimed.
Initial raw data are retained under `stage-b-before-buffer-fix/` and
`ram-before-buffer-fix/`. The measurement controller was stopped after its
active child finished; partial timing results remain historical, not final
acceptance. FP16 remains gated on Stage B.


Final Stage B candidate (`marin-run` SHA-256
`99905c1cc4e0a425cc3dff2a55045cec60def510fae6241c220fc177618a5286`):
all six required checks pass, 594 tests with CUDA required. The twelve
512/513/1025-position numerical comparisons and eight strict Hugging Face
verification commands pass again. The 384 MiB swap-disabled rollback/replay
check also passes on this build. No numerical tolerances changed.

The revised 4 GiB/1.7B/INT8/8k run still **rejects cleanly during growth**:
peak RSS 4,038,946,816 bytes, requesting another 280 MB with 244 MB available.
Its cgroup reached memory.peak 4,294,967,296 bytes; max 12428, oom 0,
oom_kill 0, swap current 0. This establishes graceful refusal, not support
for this workload under that budget. Buffer credit and packing reservation
reduce the estimated next allocation but do not make it fit; legitimate
limits and headroom remain intact. Final timing, the full RAM matrix and
startup comparisons are running serially. Stage B remains unaccepted.


**Final Stage B timing and RAM matrix (2026-10-12).** All 24 timing cases
and 64 swap-disabled cgroup profiles completed on the frozen candidate above.
The original speed gate is **FAILED**: 1.7B INT8, one thread, 8159 prompt
tokens has median TTFT 595819.3 ms versus 335899.9 ms for Stage A (+77.38%).
All three measured candidate samples are retained (595727.5–596200.8 ms).
The other 23 cases pass TTFT ≤10%; every case passes decode ≤3%.
Stage B is **NO-GO** on the current measurements; FP16 remains blocked.
Diagnosis and the remaining startup/CLI evidence are pending.

| Model | Threads | Prompt IDs | Weights | A / B TTFT (ms) | A / B decode (tok/s) | B−A peak RSS (MiB) | Actual KV held (MiB) |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.6B | 8 | 127 | int8 | 254.9 / 240.0 | 34.792 / 36.821 | +0.02 | 56 |
| 0.6B | 8 | 127 | bf16 | 294.2 / 267.9 | 19.025 / 19.914 | -0.67 | 56 |
| 1.7B | 8 | 127 | int8 | 715.7 / 618.4 | 12.020 / 12.945 | +0.60 | 56 |
| 1.7B | 8 | 127 | bf16 | 853.8 / 764.3 | 6.592 / 7.016 | -0.09 | 56 |
| 0.6B | 8 | 2047 | int8 | 4389.1 / 4379.4 | 20.839 / 21.586 | -84.20 | 504 |
| 0.6B | 8 | 2047 | bf16 | 4820.7 / 4579.1 | 13.968 / 14.583 | -91.77 | 504 |
| 1.7B | 8 | 2047 | int8 | 11010.0 / 10309.7 | 9.588 / 10.347 | -142.68 | 504 |
| 1.7B | 8 | 2047 | bf16 | 13141.0 / 12355.4 | 5.873 / 6.207 | -145.66 | 504 |
| 0.6B | 8 | 4095 | int8 | 16185.7 / 15252.0 | 14.646 / 15.033 | -141.93 | 952 |
| 1.7B | 8 | 4095 | int8 | 27228.7 / 28386.1 | 8.143 / 8.658 | -345.70 | 952 |
| 0.6B | 8 | 8159 | int8 | 79816.8 / 79584.3 | 9.270 / 9.535 | -282.01 | 1792 |
| 1.7B | 8 | 8159 | int8 | 100743.4 / 104091.6 | 6.268 / 6.419 | -554.04 | 1792 |
| 0.6B | 1 | 127 | int8 | 1134.1 / 1092.9 | 36.744 / 37.064 | +0.09 | 56 |
| 0.6B | 1 | 127 | bf16 | 1142.5 / 1124.8 | 19.445 / 19.852 | +0.53 | 56 |
| 1.7B | 1 | 127 | int8 | 3815.1 / 3649.5 | 12.853 / 13.141 | -0.09 | 56 |
| 1.7B | 1 | 127 | bf16 | 3542.4 / 3613.2 | 6.862 / 7.049 | -0.07 | 56 |
| 0.6B | 1 | 2047 | int8 | 24303.4 / 23837.5 | 21.911 / 22.066 | -88.09 | 504 |
| 0.6B | 1 | 2047 | bf16 | 24168.3 / 25641.8 | 14.367 / 14.653 | -89.32 | 504 |
| 1.7B | 1 | 2047 | int8 | 66604.3 / 63737.6 | 10.259 / 10.776 | -152.24 | 504 |
| 1.7B | 1 | 2047 | bf16 | 63271.2 / 61224.7 | 6.082 / 6.278 | -148.50 | 504 |
| 0.6B | 1 | 4095 | int8 | 61595.3 / 63515.2 | 15.320 / 15.532 | -163.76 | 952 |
| 1.7B | 1 | 4095 | int8 | 141528.6 / 141893.6 | 8.588 / 8.751 | -348.40 | 952 |
| 0.6B | 1 | 8159 | int8 | 180900.2 / 185106.5 | 9.636 / 9.794 | -316.73 | 1792 |
| 1.7B | 1 | 8159 | int8 | 335899.9 / 595819.3 | 6.536 / 6.608 | -571.80 | 1792 |

Stage A comparison uses its original three rounds, except the two previously
repeated cases use all 15 original/repeat samples. RSS deltas compare the
original A process and the B process. No B speed miss has been discarded.

RAM profiles below use eight threads, one cold-requested fresh benchmark
process with 32 output tokens, then a separate exact-output companion.
Prompts have 127/2015/4063/8159 exact IDs for context 256/2048/4096/8192;
companion text has 124/1987/3975/7920 actual IDs. Every successful companion
is byte-identical to Stage 3. A per-file DONTNEED hint is requested; actual
load I/O is in the raw JSON. Limits 3/6 GiB simulate leaving 1/2 GiB for
OS/apps on 4/8 GiB machines; these are process constraints on this Zen 4
laptop, not measurements on physically smaller or older PCs.

| Process cap / profile | Model | Weights | Context | Outcome | TTFT (ms) | Decode (tok/s) | Peak RSS (MiB) | Actual KV (MiB) |
|---|---|---|---:|---|---:|---:|---:|---:|
| 4 GiB (4GiB) | 0.6B | bf16 | 2048 | exact output | 4416.6 | 14.816 | 1714.97 | 448 |
| 4 GiB (4GiB) | 0.6B | bf16 | 256 | exact output | 274.2 | 20.140 | 1249.62 | 56 |
| 4 GiB (4GiB) | 0.6B | bf16 | 4096 | exact output | 15119.9 | 11.268 | 2241.22 | 896 |
| 4 GiB (4GiB) | 0.6B | bf16 | 8192 | exact output | 80161.6 | 7.836 | 3275.43 | 1792 |
| 4 GiB (4GiB) | 0.6B | int8 | 2048 | exact output | 4453.7 | 22.458 | 1180.41 | 448 |
| 4 GiB (4GiB) | 0.6B | int8 | 256 | exact output | 243.8 | 36.991 | 704.56 | 56 |
| 4 GiB (4GiB) | 0.6B | int8 | 4096 | exact output | 14643.7 | 15.237 | 1686.62 | 896 |
| 4 GiB (4GiB) | 0.6B | int8 | 8192 | exact output | 79444.3 | 9.670 | 2776.24 | 1792 |
| 4 GiB (4GiB) | 1.7B | bf16 | 2048 | clean rejection | — | — | 130.04 | — |
| 4 GiB (4GiB) | 1.7B | bf16 | 256 | exact output | 911.9 | 7.032 | 3407.93 | 56 |
| 4 GiB (4GiB) | 1.7B | bf16 | 4096 | clean rejection | — | — | 130.07 | — |
| 4 GiB (4GiB) | 1.7B | bf16 | 8192 | clean rejection | — | — | 130.45 | — |
| 4 GiB (4GiB) | 1.7B | int8 | 2048 | exact output | 10058.9 | 10.320 | 2327.85 | 448 |
| 4 GiB (4GiB) | 1.7B | int8 | 256 | exact output | 589.9 | 12.776 | 1823.70 | 56 |
| 4 GiB (4GiB) | 1.7B | int8 | 4096 | exact output | 27523.4 | 8.571 | 2853.12 | 896 |
| 4 GiB (4GiB) | 1.7B | int8 | 8192 | clean rejection | — | — | 3851.84 | — |
| 3 GiB (4GiB-OS-reserve) | 0.6B | bf16 | 2048 | exact output | 4456.4 | 14.396 | 1716.34 | 448 |
| 3 GiB (4GiB-OS-reserve) | 0.6B | bf16 | 256 | exact output | 266.3 | 19.701 | 1248.19 | 56 |
| 3 GiB (4GiB-OS-reserve) | 0.6B | bf16 | 4096 | exact output | 15199.4 | 11.476 | 2237.60 | 896 |
| 3 GiB (4GiB-OS-reserve) | 0.6B | bf16 | 8192 | clean rejection | — | — | 132.11 | — |
| 3 GiB (4GiB-OS-reserve) | 0.6B | int8 | 2048 | exact output | 4453.4 | 22.280 | 1180.42 | 448 |
| 3 GiB (4GiB-OS-reserve) | 0.6B | int8 | 256 | exact output | 240.9 | 36.997 | 704.90 | 56 |
| 3 GiB (4GiB-OS-reserve) | 0.6B | int8 | 4096 | exact output | 14498.5 | 15.449 | 1687.93 | 896 |
| 3 GiB (4GiB-OS-reserve) | 0.6B | int8 | 8192 | exact output | 79873.5 | 9.625 | 2748.26 | 1792 |
| 3 GiB (4GiB-OS-reserve) | 1.7B | bf16 | 2048 | clean rejection | — | — | 132.24 | — |
| 3 GiB (4GiB-OS-reserve) | 1.7B | bf16 | 256 | clean rejection | — | — | 131.33 | — |
| 3 GiB (4GiB-OS-reserve) | 1.7B | bf16 | 4096 | clean rejection | — | — | 130.78 | — |
| 3 GiB (4GiB-OS-reserve) | 1.7B | bf16 | 8192 | clean rejection | — | — | 132.02 | — |
| 3 GiB (4GiB-OS-reserve) | 1.7B | int8 | 2048 | exact output | 10190.9 | 10.252 | 2332.62 | 448 |
| 3 GiB (4GiB-OS-reserve) | 1.7B | int8 | 256 | exact output | 604.9 | 12.322 | 1823.05 | 56 |
| 3 GiB (4GiB-OS-reserve) | 1.7B | int8 | 4096 | exact output | 27021.6 | 8.507 | 2855.77 | 896 |
| 3 GiB (4GiB-OS-reserve) | 1.7B | int8 | 8192 | clean rejection | — | — | 131.71 | — |
| 8 GiB (8GiB) | 0.6B | bf16 | 2048 | exact output | 4448.0 | 14.750 | 1714.86 | 448 |
| 8 GiB (8GiB) | 0.6B | bf16 | 256 | exact output | 271.3 | 20.042 | 1249.29 | 56 |
| 8 GiB (8GiB) | 0.6B | bf16 | 4096 | exact output | 15133.9 | 11.265 | 2237.21 | 896 |
| 8 GiB (8GiB) | 0.6B | bf16 | 8192 | exact output | 80070.0 | 7.802 | 3275.54 | 1792 |
| 8 GiB (8GiB) | 0.6B | int8 | 2048 | exact output | 4554.3 | 22.094 | 1180.11 | 448 |
| 8 GiB (8GiB) | 0.6B | int8 | 256 | exact output | 242.9 | 36.574 | 704.63 | 56 |
| 8 GiB (8GiB) | 0.6B | int8 | 4096 | exact output | 14825.0 | 15.470 | 1685.26 | 896 |
| 8 GiB (8GiB) | 0.6B | int8 | 8192 | exact output | 79527.6 | 9.552 | 2757.93 | 1792 |
| 8 GiB (8GiB) | 1.7B | bf16 | 2048 | exact output | 12079.7 | 6.396 | 3875.75 | 448 |
| 8 GiB (8GiB) | 1.7B | bf16 | 256 | exact output | 751.3 | 7.222 | 3408.29 | 56 |
| 8 GiB (8GiB) | 1.7B | bf16 | 4096 | exact output | 29893.7 | 5.656 | 4392.79 | 896 |
| 8 GiB (8GiB) | 1.7B | bf16 | 8192 | exact output | 112734.2 | 4.495 | 5412.17 | 1792 |
| 8 GiB (8GiB) | 1.7B | int8 | 2048 | exact output | 9893.5 | 10.570 | 2328.16 | 448 |
| 8 GiB (8GiB) | 1.7B | int8 | 256 | exact output | 633.0 | 13.047 | 1823.86 | 56 |
| 8 GiB (8GiB) | 1.7B | int8 | 4096 | exact output | 27242.2 | 8.756 | 2855.02 | 896 |
| 8 GiB (8GiB) | 1.7B | int8 | 8192 | exact output | 102678.3 | 6.512 | 3948.66 | 1792 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | bf16 | 2048 | exact output | 4499.5 | 14.695 | 1714.60 | 448 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | bf16 | 256 | exact output | 270.3 | 20.145 | 1248.37 | 56 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | bf16 | 4096 | exact output | 14853.9 | 11.318 | 2237.66 | 896 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | bf16 | 8192 | exact output | 79971.1 | 7.860 | 3277.97 | 1792 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | int8 | 2048 | exact output | 4523.5 | 22.131 | 1180.38 | 448 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | int8 | 256 | exact output | 252.8 | 36.733 | 704.49 | 56 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | int8 | 4096 | exact output | 14638.7 | 15.176 | 1689.62 | 896 |
| 6 GiB (8GiB-OS-reserve) | 0.6B | int8 | 8192 | exact output | 79322.7 | 9.627 | 2762.19 | 1792 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | bf16 | 2048 | exact output | 11997.6 | 6.344 | 3875.84 | 448 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | bf16 | 256 | exact output | 838.7 | 7.364 | 3408.27 | 56 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | bf16 | 4096 | exact output | 29999.8 | 5.639 | 4392.84 | 896 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | bf16 | 8192 | exact output | 110827.3 | 4.604 | 5415.75 | 1792 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | int8 | 2048 | exact output | 9921.7 | 10.511 | 2327.91 | 448 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | int8 | 256 | exact output | 600.1 | 13.008 | 1824.05 | 56 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | int8 | 4096 | exact output | 26168.9 | 8.674 | 2855.93 | 896 |
| 6 GiB (8GiB-OS-reserve) | 1.7B | int8 | 8192 | exact output | 104134.2 | 6.502 | 3916.65 | 1792 |

All 64 profiles have memory.swap.max 0 and zero memory.events oom, oom_kill
and oom_group_kill. There are 54 completed inferences and 10 clean rejections.
Raw before/after memory.max/high/current/peak/events and swap counts, actual
RSS, load I/O, companion process-to-first-output and exact-output results
are in `bench-results/cpu-stage4/ram/*.json`. Rejected runs emit no benchmark
JSON or KV-capacity metric; no allocation or quality pass is invented for them.
The outer unrestricted measurement-controller service reported a small swap
peak; each limited inference service independently enforces swap-off.

The first startup harness was invalid: bench requires at least two output
tokens, but it requested one. It failed argument validation before loading
the model; the log is retained in `startup-invalid-one-token/`. The corrected
harness uses two benchmark tokens and a separate one-token streaming CLI
process. This is a harness error, not an inference regression.



**Final startup and CLI adaptation evidence.** Forty-eight alternating-order
fresh-process samples (three per build/model/format/cache condition) passed
exact one-token output comparison. Bench uses 127 prompt IDs, two outputs,
8192 context, eight threads, no warm-up; the separate one-token CLI streams
plain text and supplies process-to-first-stdout. Medians below compare
Stage A / Stage B. Peak RSS is the maximum streaming-process RSS among
three samples; separate benchmark RSS is retained in `startup/records.json`.

| Model | Weights | Requested cache state | A / B load (s) | A / B process-first-output (s) | A / B benchmark TTFT (ms) | A / B peak streaming RSS (MiB) | A / B load disk reads (bytes) |
|---|---|---|---:|---:|---:|---:|---:|
| 0.6B | bf16 | warm | 0.6216 / 0.6717 | 0.9472 / 1.0062 | 292.0 / 287.8 | 1211.79 / 1211.57 | 0 / 0 |
| 0.6B | bf16 | cold | 1.1744 / 1.2391 | 1.5262 / 1.5942 | 343.5 / 283.6 | 1211.11 / 1210.98 | 977690624 / 977690624 |
| 0.6B | int8 | warm | 1.0155 / 1.0683 | 1.3449 / 1.4058 | 253.6 / 250.2 | 672.07 / 672.50 | 0 / 0 |
| 0.6B | int8 | cold | 1.5397 / 1.6160 | 1.8994 / 1.9342 | 250.1 / 276.6 | 672.21 / 671.52 | 977690624 / 977690624 |
| 1.7B | bf16 | warm | 1.7982 / 1.8691 | 2.2622 / 2.3450 | 868.0 / 886.7 | 3359.25 / 3359.16 | 0 / 0 |
| 1.7B | bf16 | cold | 2.6693 / 2.7402 | 3.1618 / 3.2029 | 789.4 / 869.6 | 3359.27 / 3359.07 | 3069280256 / 3069280256 |
| 1.7B | int8 | warm | 2.3909 / 2.3991 | 2.8340 / 2.9352 | 692.7 / 671.0 | 1810.44 / 1810.41 | 0 / 0 |
| 1.7B | int8 | cold | 3.6194 / 3.5521 | 4.1179 / 4.0512 | 720.0 / 746.3 | 1810.39 / 1810.37 | 3069280256 / 3069280256 |

Cold-requested load reads were 977,690,624 / 3,069,280,256 bytes for
0.6B / 1.7B in these medians; these are actual I/O counters, not a claim
that every source byte was cold. Warm load reads were zero. Source files
also contain unused LM-head data. CPU memory checks before each tensor
add safety work: observed median load differences span −67 to +76 ms;
0.6B BF16 warm load is +50 ms (+8.06%), cold +65 ms (+5.51%). This startup
cost is reported rather than hidden; the fixed Stage B startup gate requires
measurement, with no invented speed threshold. Host load and cold-cache
hints remain measurement limitations.

All 32 semantic prompts are byte-identical to Stage 3 on the final build.
Together with 54 successful constrained long-output companions, twelve
chunk-boundary numerical cases and eight strict HF commands, this passes
the stated FP32 correctness/output checks. Raw evidence is
`stage-b/semantic.json`, `numeric-b-check.log`, `stage-b/verify-*.log`
and `ram/*.json`.

A real 1 GiB cgroup-v2, swap-off run tested both normal CLI entry points
with isolated XDG configuration/state directories. Qwen3-0.6B INT8 defaults
shrank to 277 (`marin-run`, default 2048) and 275 (`marin`, default 4096)
positions, and both generated successfully. Both explicit --ctx 8192
requests rejected instead of shrinking; both BF16-default requests rejected
instead of silently quantizing. Aggregate memory.peak 736,202,752 bytes,
all memory.events counters zero. Raw logs and selected-context evidence are
in `default-context/summary.json`. No saved user model selection changed.



**Long-prefill diagnosis and final decision.** A separate safe Rust harness
links the unchanged release model/tensor libraries and feeds the exact
benchmark's 8159 repeated lighthouse IDs to Qwen3-1.7B INT8 with one thread,
8192 context and a fresh cache. It times one forward pass, not the benchmark's
warm-up plus three reused-cache runs. Wall time is 352.494463 s.

| Fresh-cache diagnostic stage | Calls | Seconds |
|---|---:|---:|
| Embedding, memory checks and KV preparation | 16 | 1.901225 |
| Norm | 1360 | 0.277029 |
| QKV projections | 448 | 35.598477 |
| RoPE | 448 | 0.177864 |
| KV writes | 448 | 0.130733 |
| Attention | 448 | 121.209289 |
| Attention output projection | 448 | 17.836592 |
| MLP | 448 | 174.585637 |
| Residual addition | 896 | 0.286496 |
| LM head | 16 | 0.251848 |

MLP and attention dominate this fresh-cache diagnostic; allocation/admission
work is not its dominant cost. This pass does **not** reproduce the warmed
595.819 s acceptance result and cannot replace those three samples. The
warm-state/performance discrepancy remains unresolved. CPU policy sampled
in the diagnostic was powersave / balance_performance, active core about
5.016 GHz; no equivalent per-core frequency series was collected during the
original failed case. Allocator/cache placement and changing runtime/power
conditions are hypotheses, not established causes. No speculative kernel
fix, unsafe floating-point mode, power-policy change, discarded sample or
gate relaxation was used. Raw stage trace and diagnostic source are
`profile-b.csv`, `profile-b.rs`, `profile-b.logits` and `profile-b-power.jsonl`.

| Stage 4 acceptance gate | Classification | Evidence / limitation |
|---|---|---|
| A growable storage / rollback / reuse | PASSED — GO | Stage A table above, allocation/refusal tests and 24 before/after pairs; 127 positions hold 256-position payload instead of the full 8192 window |
| B numerical correctness / outputs | PASSED | Twelve 512/513/1025 boundary comparisons with unchanged HF tolerances; eight strict HF commands; 32 semantic + 54 successful constrained long-output comparisons exact |
| B context policy / bounded work | PASSED | At most 512 positions per chunk; whole-request length rollback after later-chunk failure; real 1 GiB CLI adaptation to 277/275 positions; explicit windows and BF16 defaults reject rather than silently changing |
| B RAM / OOM | PASSED for tested profiles | 64 effective 3/4/6/8 GiB cgroup-v2 cases, swap disabled, zero OOM events/kills; 54 completions / 10 clean rejections; 384 MiB ballast preserves prefix and resumes exact replay; rejected CLI runs emit no KV-capacity metric |
| B decode performance | PASSED | All 24 comparisons satisfy ≤3% regression; measured ratios 1.00706–1.07914 versus Stage A |
| B prefill performance | FAILED — NO-GO | 23/24 satisfy ≤10%; 1.7B INT8, one thread, 8k TTFT 335.900 → 595.819 s (+77.38%); all original samples retained, warm-state discrepancy unresolved |
| B startup / process-first-output / peak reporting | PASSED as a reporting gate | 48 alternating-order fresh-process samples; exact output; warm/cold-requested load, actual I/O, TTFT, streaming latency and RSS table above; small startup regressions disclosed |
| BF16/FP32 reference, INT8 opt-in, scalar/SIMD, threads, CUDA | PASSED | Defaults unchanged; source stays packed INT8; strict scalar/SIMD 1/8-thread HF checks; full CUDA-required workspace suite on RTX 5050 |
| Required checks | PASSED | fmt, check, clippy -D warnings, workspace/all-features tests (594), line limit and unsafe checks all exit 0 on the frozen candidate |
| CPU snapshots | OUT OF SCOPE | Existing unsupported CPU save/restore remains unsupported; CUDA snapshot regression suite passes; no CPU snapshot implementation |
| C CPU FP16 conversion / storage / quality / speed | BLOCKED | Stage B prefill gate fails; only the previously recorded official API audit exists. No FP16 path, conversion benchmark or model-quality pass is claimed |
| Physical older CPU / physical 4/8 GiB PC | NOT MEASURED | Same Zen 4 laptop with enforced process limits; forced scalar checks are compatibility evidence, not older-hardware timing |

**Decision: Stage A GO; Stage B NO-GO; overall Stage 4 NO-GO.** The growable
FP32 and bounded-prefill/memory-budget candidate remains uncommitted for
review. FP32 KV and BF16 weights remain defaults; CPU INT8 stays opt-in.
Stop here: no CPU FP16, INT4, CPU snapshots or later features. Before a new
acceptance attempt, the measured next step is an alternating-order A/B
1.7B/one-thread/8k comparison with identical cache reuse and per-stage timing,
plus allocator/RSS and CPU-frequency observations. Preserve the original
three samples and retain every follow-up; diagnose the warm-state gap before
changing kernel or allocation policy.

The memory checks estimate available capacity; they do not reserve it against
concurrent external allocations. Graceful refusal is demonstrated for the
listed workloads, not a universal OOM guarantee. HEAD, all three stash IDs,
all pre-existing non-document files and the original BENCHMARKS prefix
remain unchanged. No dependencies, commits, pushes or stash operations were
introduced.


### 27. Stage 4 bounded-prefill regression investigation

The accepted Stage 3 gates in section 25 and the original Stage 4 gates in
section 26 remain unchanged. The +77.38% 1.7B INT8 / one-thread / 8k TTFT
failure is retained. No new feature is authorized by this investigation.

Before changing production source, archive the exact failed candidate
(SHA-256 `99905c1cc4e0a425cc3dff2a55045cec60def510fae6241c220fc177618a5286`)
and rerun its original 8159-ID prompt / 32-output / 8192-window / one-warm-up /
three-measured-round command. Compare the saved growable-only Stage A binary
on the same workload, serially. Record CPU frequency/policy, faults, RSS,
swap and commands; do not change machine power policy or memory limits.

The controlled diagnostic factors are prefill chunk size, actual KV growth
versus already allocated/initialized reusable storage, and profiling state.
Use the same model weights, input IDs, dispatch and thread count. Sweep
smaller legal bounds as well as larger/unbounded diagnostic batches to
separate batching efficiency from repeated KV packing. Larger diagnostic
batches do not relax the production bound of at most 512 positions.
Break down each chunk's projections/MLP GEMM, attention packing/computation,
KV reads/writes, reservation/copy work and memory-guard time. Where allocation
and copying cannot be separated without an unsafe allocator hook, report
the combined measured duration and explicitly distinguish pointer/capacity
observations and source-derived copy bounds from hardware traffic.

Accept only a measured root-cause fix that retains growable KV, legitimate
memory limits/headroom, rollback/reuse, BF16/FP32 defaults and INT8 opt-in.
Rerun the original 24-case Stage 4 timing matrix (both models, 1/8 threads,
INT8 all four depths; BF16 short/2k), strict numerical/output checks, scalar
fallback, full CUDA-required suite and all constrained RAM profiles. No
sample dropping, threshold change or format default change. If bounded
prefill cannot pass, disable/revert that component under the user's explicit
fallback instruction, keeping independently validated growth/safety work;
do not relabel a disabled component as passing the 512-token gate.

Evidence and preservation hashes are retained under ignored
`bench-results/cpu-stage4-fix/`. The unchanged-binary reproduction is running;
no new performance or root-cause claim is made yet. No FP16, INT4, CPU
snapshots, Stage 5, dependency, commit, push or stash change is introduced.


**Unmodified same-session reproduction.** Candidate B and archived growable-only
A ran serially with the original 8159/32/8192, one-thread, one-warm-up,
three-measured-round settings. B: TTFT 370689.686 ms (367024.885–372912.933),
decode 5.88469 tok/s. A: TTFT 374361.124 ms (374255.981–374458.282),
decode 6.21664 tok/s. Same-session B/A ratios are 0.99019 TTFT and 0.94660
decode: the archived +77.38% TTFT failure is **not reproduced in this first
process**, and decode still misses the original ≤3% gate. Both the original
failure and every new sample are retained; no pass or code fix is claimed.

Frequency/policy, faults, RSS and swap are recorded per process every five
seconds. System-wide CPU and other-process activity were additionally observed
from partway through B. CPU policy was powersave/balance_performance and
observed active-core frequencies were near 5 GHz. Timing services had no
artificial RAM limit; B/A observed small nonzero process swap (436/2056 KiB
in late samples), disclosed rather than treated as swap-disabled RAM tests.
These observations do not reconstruct the missing telemetry from the earlier
595.819 s run. Raw commands, results and telemetry are in
`bench-results/cpu-stage4-fix/`. Production Rust and binary hashes are unchanged.

**Public-kernel batching/allocation sweep.** Synthetic finite inputs use the
real 1.7B gate/down dimensions, group-128 INT8 and the unchanged public Matrix
API. Each case is a fresh process with one unmeasured and three measured
rounds, processing 8159 rows; all outputs are bit-identical to the same
kernel's unbatched reference. This is a GEMM calibration, not full-model
quality or acceptance. KV allocation primers use the actual KvCache API:
none, full 8192 reservation, or sixteen 512-position growth steps. Both
nonempty primers finish with 1,879,048,192 bytes of KV capacity. Allocation
and input-copy durations, pointer phases and commands are retained.

| GEMM shape | Batch | No KV primer (s) | Direct full reservation (s) | Incremental KV growth (s) |
|---|---:|---:|---:|---:|
| 6144 × 2048 | 128 | 2.169363 | 2.187002 | 2.167221 |
| 6144 × 2048 | 256 | 2.180028 | 2.177217 | 2.185392 |
| 6144 × 2048 | 512 | 2.178438 | 2.165138 | 2.459139 |
| 6144 × 2048 | 1024 | 2.169980 | 2.166572 | 2.128451 |
| 6144 × 2048 | 2048 | 2.120636 | 2.119729 | 2.159751 |
| 6144 × 2048 | 8192 | 2.188202 | 2.174362 | 2.158127 |
| 2048 × 6144 | 128 | 2.034616 | 2.044572 | 1.963643 |
| 2048 × 6144 | 256 | 2.005822 | 1.989797 | 1.961229 |
| 2048 × 6144 | 512 | 2.006230 | 2.001394 | 2.147177 |
| 2048 × 6144 | 1024 | 2.003367 | 2.008048 | 1.933839 |
| 2048 × 6144 | 2048 | 1.932880 | 2.086566 | 2.008591 |
| 2048 × 6144 | 8192 | 1.998999 | 2.018857 | 2.013122 |

The 512-row incremental-primer cases are slower: gate 2.4591 versus
2.1651 s (+13.58%), down 2.1472 versus 2.0014 s (+7.29%). Smaller batches
do not show that interaction in this calibration. Other batches and primers
are close enough that arithmetic batching alone does not establish a
77% model slowdown. Pointer phases differ between allocation histories;
cache/alignment sensitivity is a hypothesis requiring a focused control,
not an established hardware cause.

**Warmed chunk profile on unchanged libraries.** A genuine runtime warm-up
processes the full prompt and 31 decode passes (32 sampled outputs), then
clears committed KV length and times all 16 bounded chunks. Growable-history
prefill is 368.218526 s: MLP 185.774342 s, attention 125.234731 s, QKV
37.827698 s, output projection 17.991617 s. Embedding/guards/reservation
sum only 0.009019 s and KV writes 0.129182 s. Every returned logit is
bit-identical to the saved cold profile. Per-layer traces and per-chunk
totals are in `warm-profile-grown.csv` and `warm-chunks-grown.json`.
The direct-reservation control and exact-kernel alignment calibration are
still running. No production optimization has been made.


**Completed allocation controls; fallback decision.** Direct preallocation followed
by the same real warm-up measures 358.202962 s versus incremental-history
368.218526 s. Direct-control MLP is 177.593947 s and attention 124.763385 s;
guards/reservation total 0.009161 s. Both profiles return bit-identical logits.
The model's tensor dependency fingerprint matches the explicitly linked tensor
library. Synthetic kernels are still calibrations, not full-model measurements.

An exact unchanged INT8 GEMM kernel with controlled scratch pointer phases
measures 1.943952 s with both buffers cache-line aligned versus 2.198063 s
with both offset by 16 bytes (three measured samples per phase). This ~13%
sensitivity does not establish the archived 77% cause. No speculative alignment
optimization is accepted. The unchanged CLI under a process-local static
malloc mmap threshold of 8192 bytes measures 367734.074 ms at 8k (one measured
round after one warm-up), also without the archived failure. Short-prompt
normal/static-threshold TTFT is 4108.423/4130.837 ms. This allocator control
changes neither memory limits nor machine-wide policy; it is not a production
setting or a gate pass.

Attention batch sizes 128/256/512/1024/2048/4096/8192 were measured with the
unchanged public kernel and an exact-source private packing calibration.
All outputs are bit-identical to that kernel's unbatched reference. At 512,
repeated source KV packing processes 570155008 logical bytes per layer;
unbounded processes 66838528. Packing takes about 0.10 versus 0.014 seconds
per synthetic layer. These are source-derived logical copy sizes, not hardware
traffic counters. Synthetic attention timings do not represent real-model
attention latency. Raw fresh/primed packing and full-attention samples are
retained in `attention-sweep.csv`; no unsafe allocator hook was introduced.

The unchanged same-session run does not reproduce the archived +77.38%
TTFT failure, and still fails decode by 5.34%. A causal fix is therefore
**unproven**. Under the user's explicit fallback authorization, bounded prefill
is removed rather than accepted on these controls. The single whole-prompt
pass retains growable KV, checked full-input/context arithmetic and commit
only after successful inference. CPU admission now budgets full-window
activation memory, including when selecting an omitted default context;
the before-growth guard still budgets resident KV, copy/packing headroom and
the actual input workspace against host/cgroup availability. No memory-limit
bypass is introduced. Full-input workspace can legitimately cause additional
low-RAM rejections compared with bounded prefill.

The original ≤512 production bound is **DISABLED / NOT PASSED**, not relaxed.
Original timing, correctness and RAM gates will be rerun on the fallback;
Stage 4 as originally scoped remains NO-GO while bounded prefill is absent.
No FP16, INT4, snapshots, Stage 5 or default-format change is introduced.


**Final validation scope, explicitly narrowed by the user.** After more than
two hours of diagnostics, the user requested convergence: keep the fallback,
stop expanding investigation, and use focused checks rather than another
exhaustive 8k single-thread matrix. This supersedes the investigation's
planned full rerun above; it does not change any original gate or make
unmeasured cells pass. Only the required verification suite and one serial
focused-validation controller remain active. No competing CPU timing jobs
are scheduled. The final controller checks both models/formats, scalar
reference verification, 1/8-thread short A/B timings, eight-thread INT8 2k
A/B timings, boundary numerics, real pressure rollback, focused 4/8 GiB
swap-disabled profiles and both CLIs' adaptive defaults under 1 GiB.
The complete final timing/RAM matrices and cold-start reruns are explicitly
unverified; the existing archived results are historical evidence.


**Final fallback targeted timing — all samples retained.** Final `marin-run`
SHA-256 is `05d194ea2ae87fe5793af4b3bc3905f828d18e080ae0b0d6b42e4b1148ca1931`.
Ten serial, alternating-order A/fallback pairs use context 8192, 32 outputs,
one warm-up and three measured rounds per binary. A is the archived accepted
growable-only Stage A binary; fallback is whole-prompt prefill with the
retained guards and full-workspace admission. This is a targeted comparison,
not a replacement for A's original 24-case comparison against Stage 3.
The 10% TTFT / 3% decode thresholds below are the unchanged comparison
against A; they do not replace A's stricter 5% versus Stage 3 gate.

| Model | Weights | Threads | Prompt | A / fallback TTFT (ms) | A / fallback decode (tok/s) | TTFT change | Decode change | RSS delta (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B | bf16 | 1 | 127 | 1136.734 / 1116.034 | 19.28232 / 18.91943 | -1.8210% | -1.8820% | +0.0703 |
| 0.6B | bf16 | 8 | 127 | 326.712 / 289.462 | 19.43884 / 19.24654 | -11.4014% | -0.9892% | -0.6836 |
| 0.6B | int8 | 1 | 127 | 1105.109 / 1205.811 | 36.67452 / 35.86439 | +9.1125% | -2.2090% | +0.1797 |
| 0.6B | int8 | 8 | 127 | 254.488 / 267.228 | 36.08887 / 35.58839 | +5.0063% | -1.3868% | +0.2617 |
| 1.7B | bf16 | 1 | 127 | 3598.949 / 3625.199 | 6.51802 / 6.56685 | +0.7294% | +0.7492% | -0.0234 |
| 1.7B | bf16 | 8 | 127 | 826.086 / 812.912 | 6.53733 / 6.55247 | -1.5948% | +0.2315% | +0.4062 |
| 1.7B | int8 | 1 | 127 | 3879.441 / 3825.295 | 12.00675 / 12.37575 | -1.3957% | +3.0732% | -0.3203 |
| 1.7B | int8 | 8 | 127 | 685.498 / 757.319 | 11.73164 / 12.26128 | +10.4772% | +4.5146% | -0.2969 |
| 0.6B | int8 | 8 | 2047 | 4435.728 / 4675.380 | 21.58436 / 20.87638 | +5.4028% | -3.2801% | +2.0898 |
| 1.7B | int8 | 8 | 2047 | 10936.079 / 10980.033 | 9.56583 / 9.58155 | +0.4019% | +0.1643% | -2.9844 |

**Two focused failures remain:** 1.7B INT8 / eight threads / 127 positions
has +10.4772% TTFT; 0.6B INT8 / eight threads / 2047 positions has -3.2801%
decode. All other targeted cells satisfy those thresholds; every RSS delta
is below +64 MiB. No failing sample is removed, no threshold is changed, and
no further sweep/retake is scheduled after the user's convergence instruction.
These misses do not establish a causal memory-guard or allocation regression;
no causal fix or complete performance acceptance is claimed. The original
+77.38% failure is likewise retained as historical evidence, not explained
away by the new unchanged-binary run.

Initial KV is 58,720,256 bytes at the short prompt (256-position payload)
and 528,482,304 bytes after the 2047-position warm-up (2304-position payload,
including decode crossing 2048). The final forward still reserves only actual
used positions and keeps the warmed allocation for reuse. It does not allocate
an 8192-position KV window for a short prompt. Load durations are retained in
`final-timing.json`; these cache conditions are uncontrolled and are not a
replacement for cold/zero-read-warm startup measurements.


**Final focused correctness and memory evidence.** All twelve boundary checks
(512/513/1025 positions, both models and BF16/INT8) match the saved Stage A
logits, hidden-state values, layer norms and eight greedy choices bit for bit:
maximum absolute/relative difference zero. Eight fresh strict HF runs pass
both models with SIMD/scalar and 1/8 threads, using BF16 weights and FP32 math;
HF tolerances remain unchanged. The full suite covers grow/truncate/reuse,
whole-pass failure prefix preservation, INT8 storage/dispatch/threading, real
RTX 5050 backend/model switching, CUDA quantization and existing GPU snapshots.
CPU snapshot support remains unchanged and unsupported.

Focused RAM runs use effective cgroup-v2 limits below, swap disabled, eight
threads, one fresh-process measured round with no warm-up and 32 outputs.
Per-file DONTNEED is requested; actual load disk reads and process-to-first
stdout are retained, not treated as a verified cold boot. Every successful
case has a separately generated text-prompt companion that exactly matches
the archived Stage 3 output. Actual text-token counts are recorded separately
from benchmark IDs. Rejections emit no KV measurement; dash means unavailable,
not zero allocation or an output-quality pass.

| Limit | Model | Weights | Context | Result | TTFT (ms) | Decode (tok/s) | Process peak RSS (MiB) | Cgroup peak (MiB) | Held KV (MiB) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 4GiB | 0.6B | bf16 | 2048 | exact output | 4834.944 | 13.89028 | 1785.020 | 2945.762 | 448.000 |
| 4GiB | 0.6B | int8 | 2048 | exact output | 4504.940 | 20.76173 | 1251.023 | 2396.480 | 448.000 |
| 4GiB | 1.7B | bf16 | 2048 | clean refusal | — | — | 130.922 | 131.906 | — |
| 4GiB | 1.7B | bf16 | 8192 | clean refusal | — | — | 134.055 | 141.031 | — |
| 4GiB | 1.7B | int8 | 2048 | exact output | 11385.742 | 9.51117 | 2445.809 | 4096.000 | 448.000 |
| 4GiB | 1.7B | int8 | 8192 | clean refusal | — | — | 131.984 | 134.023 | — |
| 8GiB | 1.7B | bf16 | 4096 | exact output | 33155.489 | 5.09845 | 4683.441 | 4991.652 | 896.000 |
| 8GiB | 1.7B | int8 | 4096 | exact output | 28914.500 | 7.95161 | 3124.117 | 6264.055 | 896.000 |

All 8 profiles have **zero `oom` and `oom_kill` events**, zero
swap, 5 exact-output completions and 3 clean refusals.
Admission and growth respect the actual limits; no override was used.

The real 384 MiB pressure test passes with BF16 and INT8: 320 MiB touched
ballast forces the before-growth guard to refuse, every layer's committed
five-token prefix is unchanged, and release/replay returns exactly the fresh
result. Cgroup peak is 347082752 bytes; OOM/kill/max events
are zero and swap is disabled. Under 1 GiB, both normal CLIs adapt an omitted
INT8 context and produce output; explicit 8192 and BF16 defaults fail clearly.
Selected INT8 contexts are marin-run-int8-default: 330, marin-int8-default: 328.
All six CLI cases preserve zero OOM kills. Full-window budgeting can refuse
work that formerly fit with bounded workspace; this is an intentional honest
resource limit, not evidence that the old low-RAM capability is retained.

**Final checks:** fmt, locked workspace/all-target check, all-target/all-feature
clippy `-D warnings`, locked full all-feature suite with `MARIN_REQUIRE_CUDA=1`
(**594 passed**), 425-line and unsafe checks all pass. Final source hashes,
commands, return codes and raw samples are in `final-build.json`,
`final-manifest.json`, `checks/`, `final-timing.json`, `ram/`,
`default-context/` and `final-acceptance.json` under
`bench-results/cpu-stage4-fix/`.

| Component / gate | Final decision | Evidence / remaining limit |
|---|---|---|
| Growable FP32 KV independently | **GO** | Accepted Stage A's original 24-case speed/RAM matrix remains intact; KV implementation is unchanged in this investigation. Final growth/reuse/truncation/refusal tests, exact boundary numerics and strict HF pass |
| Full-prefill workspace budgeting / existing RAM guards | **Focused correctness and safety PASS; performance NO-GO** | Full input/context memory is budgeted; pressure rollback and eight constrained profiles pass. Two targeted A/B performance misses are retained above; no blanket performance acceptance |
| Bounded prefill | **NO-GO / REMOVED** | Original +77.38% TTFT failure remains; unchanged new run did not reproduce it, but lost 5.34% decode. Allocation, packing and alignment controls do not establish its cause. No bounded-prefill gate pass is claimed |
| Final fallback short / 2k speed | **Two focused failures** | +10.4772% TTFT at 1.7B INT8 / 8 threads / 127; -3.2801% decode at 0.6B INT8 / 8 threads / 2047 versus A. Other eight cells pass the corresponding limits; no retries or sample exclusion |
| Final full 24-case timing matrix / populated 8k one-thread speed | **UNVERIFIED** | Not rerun after fallback under the user's convergence instruction. Archived A/B and unchanged-binary reproduction remain separate historical measurements |
| Final complete 4/8 GiB and OS-reserve RAM matrix / controlled cold and warm startup | **UNVERIFIED** | Focused profiles do not replace the original 64 profiles or 48 controlled startup samples; the full-prefill workspace changed, so those old B results cannot be transferred |
| BF16 default / INT8 opt-in / scalar / threads / CUDA | **PASS** | Defaults unchanged, eight HF runs and 594-test CUDA-required suite pass |
| FP16 / INT4 / CPU snapshots / new stages | **NOT IMPLEMENTED** | Outside this authorized fallback; CPU snapshots remain unsupported |

**Decision: growable FP32 KV GO; bounded prefill NO-GO and removed;
complete Stage 4 NO-GO.** The fallback is finished with full-workspace
admission and existing RAM safety intact. Remaining narrow timing failures
and unverified cells are explicit; no claimed root-cause fix for +77% and
no claim that every original Stage 4 gate passes. Investigation stops here
at the user's request. BF16/FP32 defaults, INT8 opt-in, prior uncommitted work,
HEAD and all three stash identities are preserved. No commit or push.


**Timing environment limit:** focused A/B timing processes were not swap-disabled.
The final controller's systemd completion log reports 10.1M peak swap over
its entire lifetime; individual focused timing processes were not sampled
for swap, so that number cannot be attributed to either narrow miss. Normal
desktop background work was not stopped. These are additional reasons not to
infer a proven causal regression or broad speed acceptance from close results.
The separate constrained RAM units explicitly disable swap, and both their
benchmark and exact-output companions have zero OOM/kill events and zero swap.
The 4 GiB 1.7B INT8 2k case reaches the cgroup limit including filesystem cache;
limit/reclaim events are preserved in its raw record and are not hidden as
unused headroom. `final-service.log` retains the controller completion accounting.
All Marin Stage 4 benchmark/validation services are now inactive; no further
benchmark or background investigation is scheduled.


### 28. Stage 4 closure and Stage 5 CPU INT4 feasibility (2026-10-12)

**Scope and gates fixed before pilot code.** Growable FP32 KV is independently
GO; bounded prefill is NO-GO and removed. The final +10.4772% TTFT (1.7B INT8,
eight threads, short prompt) and -3.2801% decode (0.6B INT8, eight threads,
2k) remain unresolved failures. The 10% / 3% gates in sections 26–27 do not
change. Stage 4 is closed as a partial result, not accepted in full. The user
explicitly authorizes this separate INT4 feasibility study despite that
partial result; it does not authorize reconstructing bounded prefill or V9.

Study cap: 45 minutes overall, 20 minutes of serial benchmark execution;
stop after a representative pilot. No production/native INT4 implementation,
new dependency, unsafe Rust, FP16 KV, planner, commit, push or stash mutation.
Use the current frozen Marin libraries and pinned existing AVX2 llama.cpp
libraries. Both models, canonical identical HF token IDs, FP32 KV, CPU-only,
127/2047 prompt positions and 32 subsequent evaluations; short 1/8 threads,
long eight threads only. One warm-up plus two measured rounds is a pilot,
not the three-round full acceptance matrix. Separate conversion/load, model
prefill-to-logits and one-token forward evaluation; no fake generated-output
TTFT from random-token llama-bench results. Explicitly report library-level
versus real CLI sampling timing. Quality uses two fixed 128-position prose/
code pilot segments after a 64-token prefix, BF16 cross-engine logits,
same top-1 and target log-probabilities; calibration/evaluation separation
and a broad task suite are required for eventual acceptance.

| Native CPU INT4 acceptance area | Gate before future implementation |
|---|---|
| Kernel / safety | Independent decoded-f64 GEMV/GEMM within 1e-5*max(1,sum(abs(w*x))); zero/extreme/odd/remainder/malformed cases, scalar and AVX2, checked metadata/size limits; no unsafe Marin code or whole-model BF16/f32 expansion |
| Reference / defaults / compatibility | Unchanged strict BF16 HF verification on both models; INT8 gates/defaults untouched, BF16 default, INT4 opt-in; scalar, 1/8-thread prefill, KV reuse/rollback and existing CUDA/snapshots pass |
| Quantized relative quality | At least 1024 held-out scored positions each prose/code/conversation per model: top-1 agreement >=95%, mean loss increase <=0.03 nats/token AND perplexity increase <=3%; eight fixed greedy tasks with no new critical semantic failures. These are new INT4 gates, not changes to INT8 or HF gates |
| Decode / prefill | Both models, 1/8 configured threads, 127/2047 depths: >=1.25x INT8 short decode and >=1.10x long; TTFT <=10% worse than INT8; warm-up plus >=3 measured rounds. Published/external Q4 performance does not pass a native kernel gate |
| Memory | Native stored weights <=70% INT8 and <=40% BF16; peak RSS saving >=20% INT8 weight bytes at matched populated context; panels bounded per worker, no second model/KV mirror; effective 4/8 GiB limits and zero OOM kills or graceful refusal |
| Startup | Fresh cold/warm conversion penalty over INT8 <=2s (0.6B)/4s (1.7B), disk reads and process-to-first-output reported separately; preconverted GGUF startup must not be called a native HF-to-INT4 conversion measurement |
| Decision | A pilot can justify a scoped prototype or reject a design; it cannot close full native INT4 acceptance. Unmeasured mandatory gates remain open |

Implementation complexity and candidate storage/kernel design will be estimated
before any native code is authorized. Only ignored measurement harnesses are
used here. Raw commands, fingerprints, IDs and results go under
`bench-results/cpu-stage5/`; all pre-existing work is preserved.


**Stage 4 memory audit — no production change.** Current CPU CLI admission is
`reported_weights + KV(context) + activation(context) + logits + 256 MiB`.
For both models FP32 KV is 229376 bytes/position. Named activation buffers
are 65536/106496 bytes per input position (0.6B/1.7B), plus 607744 bytes of
last-token logits. These are checked capacity estimates, not measured RSS.
At eight threads Q8 prefill panels remain bounded: four rows x 512 columns,
64-token packed inputs and 512-float scratch per worker, rather than a full
weight reconstruction. Attention packing is separately guarded. Weights and
activation precision are independent; growable KV changes allocation timing,
not bytes per populated position.

| Model / weights | Reported weights (MiB) | KV at 8k (MiB) | Full 8k activations + logits (MiB) | Margin (MiB) | 2k / 4k / 8k admission (GiB) |
|---|---:|---:|---:|---:|---:|
| 0.6B / bf16 | 1136.875 | 1792.000 | 512.580 | 256 | 1.923 / 2.486 / 3.611 |
| 0.6B / int8 | 586.262 | 1792.000 | 512.580 | 256 | 1.386 / 1.948 / 3.073 |
| 1.7B / bf16 | 3281.736 | 1792.000 | 832.580 | 256 | 4.096 / 4.737 / 6.018 |
| 1.7B / int8 | 1692.260 | 1792.000 | 832.580 | 256 | 2.544 / 3.184 / 4.466 |

CPU norms actually retain FP32: reported weights undercount payload by
131072/247808 bytes, independently of BF16/INT8. Actual payloads are
1192230912/614871040 and 3441397760/1774710784 bytes. Vector/struct metadata
and allocator retention are also outside weight payload. The existing margin
covers these small differences; this audit does not silently fix counters
or relabel estimated weights as exact RSS. Inference admission cannot prove
future safety against other applications suddenly consuming RAM.

The before-growth guard rechecks effective host/cgroup-v2 ancestor availability:
actual input workspace, net newly initialized KV through the pass end,
largest temporary old-vector growth copy, conservative rounded attention
packing and 64 MiB headroom. At 8k the packing allowance is 64 MiB. Loading
also checks every source tensor with `2*raw_tensor_bytes + 64 MiB`; this is a
conservative conversion allowance, not another model kept resident. Clean
reclaimable filesystem cache is credited; shared/dirty/locked pages are not.
Failures preserve committed KV prefixes. Non-Linux is host-policy unavailable;
cgroup v1 has only host accounting. These compatibility limits remain open.

**Simplest safe policy: keep current explicit format selection, full-window
admission and per-request growth checks; shrink only an omitted default
context, never a user's explicit context or BF16 default.** Report weights,
actual held KV, full-request activations and margin separately. Refuse a
non-fitting request before allocation with a smaller-context/INT8 hint.
Do not tune allocators, rebuild chunks, silently switch formats, or add a V9
planner. 1.7B INT8's full 8k estimate is 4794969600 bytes before metadata, so
4 GiB rejection is legitimate; BF16 needs 6461656576 bytes and cannot promise
an 8k whole-prefill under a 6 GiB OS-reserve budget. The original focused
timing failures remain NO-GO; independent growable storage remains GO.

**Smallest native candidates, estimated before implementation.** Keep row-major
packed unsigned nibbles, two per byte; affine groups decode as `scale*q+min`.
Store only packed codes and per-group metadata in an exclusive Matrix storage
variant. Reuse existing Pulp AVX2/FMA dispatch, scoped prefill threads,
4x2/512-column/64-token panel loop and one-row scalar fallback. Decode should
unpack into registers, accumulate FP32 and apply scale/minimum; prefill expands
only bounded panels. Embedding/tied head initially stay INT8. No whole-model
BF16/f32 expansion. Existing symmetric max/7 CUDA INT4 is not the chosen
quantizer. Weighted affine fitting requires a separately recorded activation
calibration corpus and a held-out quality check; it is not implied by storage.

| Model | Affine group | Storage estimate (MB) | Fraction of INT8 payload | Fraction of BF16 payload |
|---|---:|---:|---:|---:|
| 0.6B | 32 | 491.008 | 0.7986 | 0.4118 |
| 0.6B | 64 | 435.958 | 0.7090 | 0.3657 |
| 0.6B | 128 | 408.433 | 0.6643 | 0.3426 |
| 1.7B | 32 | 1378.349 | 0.7767 | 0.4005 |
| 1.7B | 64 | 1202.188 | 0.6774 | 0.3493 |
| 1.7B | 128 | 1114.108 | 0.6278 | 0.3237 |

These include two FP32 metadata values per group and INT8 tables. Group 32
already fails the fixed <=40% BF16 and <=70% INT8 storage gates. Group 64
fails the INT8 storage gate on 0.6B (70.90%); group 128 passes storage on both
but its quality is unmeasured and its coarse groups must not be accepted from
Q4_K_M results. The alternative is a Q4_K-style 256-value superblock with
32-value local scales/minima and compressed metadata; keeping FP32 rather
than FP16 super-scales takes 148 instead of 144 bytes (4.625 versus 4.5 bpw).
It needs weighted fitting and a measured sensitive-tensor policy. IQ4_XS
adds nonlinear codebook/scales and a different fitting problem; defer it as
a first native format.

Estimated engineering effort, not measured implementation: 4–6 engineer-days
for a scalar affine quantizer/independent f64 oracle and sample quality study;
another 4–6 for fused AVX2 decode, reused-panel GEMM, both CLI/storage/load
callers and compatibility tests. A Q4_K-style compressed scale fitter and
calibration/mixed-tensor policy would likely take 2–4 weeks total. Touches:
tensor Matrix/storage/quantizer/dispatch/panels; model CpuStore and byte
accounting; CLI format validation/reporting; kernel, HF, quality, memory,
scalar/thread and CUDA compatibility tests. No dependency or native code is
introduced during this study. The first future implementation checkpoint is
quality of decoded packed weights, before performance kernel work.


**Format/kernel research.** Pinned measured llama.cpp is
`e94acad853c0d11a3b47974e3167ec12f263d86a`; current official upstream layouts,
quantizers and x86 sources were also checked. Q4_K encodes 256 values in
144 bytes: eight 32-value groups, packed nibbles, six-bit local scales/minima,
and two FP16 super-scales (4.5 bpw). Q4_K_M is a tensor-selection policy,
not a uniform four-bit block: selected value/down projections and tables
can use higher precision. IQ4_XS stores 256 values in 136 bytes (4.25 bpw),
using a fixed nonlinear 16-value codebook, signed six-bit local scales and
an FP16 super-scale. It is not Marin's linear symmetric max/7 format.
[Official block layouts](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-common.h).

Q4_K AVX2 dots mask/shift nibbles, use byte/integer multiply-adds against
Q8_K activation blocks, and apply minimum corrections using activation sums.
IQ4_XS uses byte shuffles to look up nonlinear codes before integer dots.
The CPU backend also repacks Q4_K into eight-row interleaved layouts for
GEMV/GEMM. This reduces decode traffic but adds activation quantization and
can retain a second quantized layout; it does not reconstruct BF16 weights.
[Quantized x86 kernels](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-cpu/arch/x86/quants.c),
[repacking/GEMM](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-cpu/repack.cpp).

Importance collection accumulates squared input activations per channel;
weighted fitting then minimizes reconstruction error using those statistics.
Q4_K fitting searches group scale/minimum combinations rather than just
rounding by a maximum. IQ4_XS similarly accepts optional fitting weights;
its name does not mean calibration is mandatory for every quantization.
[Quantizer](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-quants.c),
[importance-matrix guidance](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/tools/imatrix/README.md).
AWQ uses activation-aware channel scaling; GPTQ uses second-order error
compensation. Their published results are on their own models/hardware,
not Marin quality evidence. A 2026 unified llama.cpp study evaluates
Llama-3.1-8B, not these Qwen3 checkpoints; no published speed or accuracy
number is transplanted into the local pilot.
[AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323),
[2026 quantization study](https://arxiv.org/abs/2601.14277).

**Local provenance and measurement scope.** The local Q4_K_M files report
Unsloth importance-matrix metadata (688/685 chunks, 196 entries), but the
calibration datasets themselves are unavailable. Both have 168 Q4_K,
29 Q6_K and 113 FP32 tensors. Payload is 390753280/1101457408 bytes;
file size is 396705472/1107409472 bytes. Every norm vector matches the local
HF BF16 source after FP32 promotion; four 256-value blocks from every one
of the 197 quantized matrices give sampled relative weight RMSE
6.9855%/7.0096%. All tensor shapes and the 758 used token IDs match. This
is sample provenance, not complete checkpoint identity or a calibration audit.
No whole tensor/model was expanded during this check: only 256-value blocks.
IQ4_XS has no local artifact and was researched, not benchmarked.

Ryzen 7 260, 8 cores/16 logical threads, Linux 7.2.4, Rust 1.96.1,
powersave/balance_performance; machine policy and desktop applications were
not changed. llama.cpp is the existing AVX2/FMA, no-AVX512/no-BLAS/no-CUDA
build; no GPU layers, flash attention off, FP32 K/V, batch/ubatch 2048.
Marin decode stays single-threaded; llama.cpp decode follows configured threads.
Both public-API harnesses read canonical HF IDs without adding BOS/chat tokens.
They evaluate the same prompt and 31 identical subsequent IDs, giving 32
logit evaluations. They measure inference-to-logits, not sampling or real
CLI generated-output TTFT. One warm-up and two measured rounds; medians
below are pilot estimates, not complete native acceptance.

An initial request of context 160/2080 was rounded by llama.cpp to 256/2304.
Those initial Marin comparisons are retained only as diagnostics. Only the
12 affected Marin cases were rerun at the actual matching 256/2304 windows;
llama.cpp samples are unchanged. Total initial plus corrected benchmark span
was 936.9 seconds, within the original 20-minute cap. No 8k or one-thread
2k matrix, new quantization implementation, or repeated acceptance sweep.
All inference processes were serial under 8 GiB cgroup-v2 limits with swap
disabled. Zero OOM/kill events; the first controller hit its limit through
reclaimable file cache (`max` events 35702), explicitly retained in records.

| Model | Engine / weights | Threads | Prompt | Actual window | Prefill-to-logits (ms) | Prefill (tok/s) | Decode evaluations/s | Peak RSS (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B | marin / bf16 | 1 | 127 | 256 | 1171.517 | 108.41 | 19.432 | 1.2992 |
| 0.6B | marin / int8 | 1 | 127 | 256 | 1183.368 | 107.32 | 35.309 | 0.7345 |
| 0.6B | llama / BF16 | 1 | 127 | 256 | 1355.334 | 93.70 | 18.603 | 1.3232 |
| 0.6B | llama / Q4_K_M | 1 | 127 | 256 | 991.429 | 128.10 | 49.191 | 0.7356 |
| 0.6B | marin / bf16 | 8 | 127 | 256 | 313.178 | 405.52 | 19.463 | 1.3059 |
| 0.6B | marin / int8 | 8 | 127 | 256 | 270.048 | 470.29 | 35.095 | 0.7353 |
| 0.6B | llama / BF16 | 8 | 127 | 256 | 217.395 | 584.19 | 19.560 | 1.3233 |
| 0.6B | llama / Q4_K_M | 8 | 127 | 256 | 163.396 | 777.25 | 50.819 | 0.7391 |
| 0.6B | marin / bf16 | 8 | 2047 | 2304 | 4453.790 | 459.61 | 14.174 | 1.9311 |
| 0.6B | marin / int8 | 8 | 2047 | 2304 | 4395.672 | 465.69 | 21.394 | 1.3704 |
| 0.6B | llama / BF16 | 8 | 2047 | 2304 | 6548.668 | 312.58 | 14.064 | 2.1572 |
| 0.6B | llama / Q4_K_M | 8 | 2047 | 2304 | 6024.518 | 339.78 | 25.501 | 1.5648 |
| 1.7B | marin / bf16 | 1 | 127 | 256 | 3711.381 | 34.22 | 6.784 | 3.5557 |
| 1.7B | marin / int8 | 1 | 127 | 256 | 3719.091 | 34.15 | 12.327 | 1.9059 |
| 1.7B | llama / BF16 | 1 | 127 | 256 | 3860.007 | 32.90 | 6.786 | 3.5772 |
| 1.7B | llama / Q4_K_M | 1 | 127 | 256 | 2765.901 | 45.92 | 20.361 | 1.9069 |
| 1.7B | marin / bf16 | 8 | 127 | 256 | 800.143 | 158.72 | 6.803 | 3.5678 |
| 1.7B | marin / int8 | 8 | 127 | 256 | 690.632 | 183.89 | 11.998 | 1.9074 |
| 1.7B | llama / BF16 | 8 | 127 | 256 | 626.339 | 202.77 | 7.032 | 3.5820 |
| 1.7B | llama / Q4_K_M | 8 | 127 | 256 | 477.206 | 266.13 | 20.200 | 1.9098 |
| 1.7B | marin / bf16 | 8 | 2047 | 2304 | 12474.412 | 164.10 | 5.988 | 4.2626 |
| 1.7B | marin / int8 | 8 | 2047 | 2304 | 10432.615 | 196.21 | 9.759 | 2.6342 |
| 1.7B | llama / BF16 | 8 | 2047 | 2304 | 12448.715 | 164.43 | 6.223 | 4.4275 |
| 1.7B | llama / Q4_K_M | 8 | 2047 | 2304 | 10168.055 | 201.32 | 14.791 | 2.7495 |

Compared with Marin INT8, external Q4_K_M short decode gains are 1.39–1.68x;
2k gains are 1.19x/1.52x. At 2k, 0.6B Q4 prefill is **37.06% slower**,
and peak RSS is **194.38 MB higher**; 1.7B prefill is 2.54% faster but
RSS is **115.34 MB higher**. Short Q4 RSS is also approximately equal to
INT8 (1–4 MB higher), despite much smaller weight files. Thus the external
pilot does not establish the native RAM or TTFT gates. Local load logs show
0.6B Q4 retains 370.96 MiB mapped weights plus 204.75 MiB CPU_REPACK;
FP32 KV at 2304 is 504 MiB and reserved compute buffer is 1269.06 MiB.
Reserved compute bytes are not all resident RSS. Native panel reuse could
avoid some of this overhead, but that is an unmeasured design opportunity.

**Cold/warm pilot startup.** Cold requests use per-file DONTNEED; warm requests
pre-read files. Physical I/O below is whole-process `ru_inblock*512`, not a
pure loader counter. mmap can defer weight faults into the first prefill;
loader time alone therefore overstates Q4 startup gains. First-logits time
starts at model loading, includes context creation and the cold/warm-up
prefill, and excludes Marin ModelDir/tokenizer setup before its model timer.
It is not process-to-visible-output latency. One fresh process per condition;
no fully cold boot or three-sample startup gate pass is claimed.

| Model | Engine / weights | Requested cache | Model load (s) | Load-start to first logits (s) | Process physical block reads (MB) |
|---|---|---|---:|---:|---:|
| 0.6B | marin / bf16 | cold | 3.2440 | 4.5027 | 1928.724 |
| 0.6B | marin / int8 | cold | 1.5470 | 2.7482 | 981.430 |
| 0.6B | llama / BF16 | cold:BF16 | 0.5959 | 2.0014 | 1051.050 |
| 0.6B | llama / Q4_K_M | cold:Q4_K_M | 0.3041 | 1.3057 | 392.897 |
| 0.6B | marin / bf16 | warm | 0.7457 | 1.1092 | 0.000 |
| 0.6B | marin / int8 | warm | 1.0495 | 1.3084 | 0.000 |
| 0.6B | llama / BF16 | warm:BF16 | 0.2029 | 0.4433 | 0.000 |
| 0.6B | llama / Q4_K_M | warm:Q4_K_M | 0.2296 | 0.4209 | 0.000 |
| 1.7B | marin / bf16 | cold | 3.4764 | 7.2487 | 3184.767 |
| 1.7B | marin / int8 | cold | 3.5315 | 7.2839 | 3071.926 |
| 1.7B | llama / BF16 | cold:BF16 | 1.5403 | 5.5792 | 3279.614 |
| 1.7B | llama / Q4_K_M | cold:Q4_K_M | 1.0764 | 3.8830 | 1106.051 |
| 1.7B | marin / bf16 | warm | 2.5748 | 3.3954 | 1078.510 |
| 1.7B | marin / int8 | warm | 2.5810 | 3.2137 | 0.020 |
| 1.7B | llama / BF16 | warm:BF16 | 0.2887 | 0.9154 | 0.033 |
| 1.7B | llama / Q4_K_M | warm:Q4_K_M | 0.4792 | 0.9494 | 0.000 |

**Quality pilot.** Each prose/code segment scores 128 positions after 64
prefix tokens. Comparison uses identical teacher-forced IDs and the same
151669 usable vocabulary entries, excluding padding. Results are relative
to Marin BF16 weights/FP32 math. This is a narrow pilot, not 1024-position
acceptance, a held-out calibration guarantee or an eight-task semantic suite.

| Model | Corpus | Precision / engine | Same top-1 /128 | Mean loss | Loss change (nats) | Perplexity change | Eight-step continuation exact? |
|---|---|---|---:|---:|---:|---:|---|
| 0.6B | prose | marin / int8 | 123 | 2.722597 | +0.001689 | +0.1690% | yes |
| 0.6B | prose | llama / BF16 | 128 | 2.721215 | +0.000306 | +0.0306% | yes |
| 0.6B | prose | llama / Q4_K_M | 108 | 2.720822 | -0.000087 | -0.0087% | no |
| 0.6B | code | marin / int8 | 125 | 2.439954 | +0.007463 | +0.7491% | no |
| 0.6B | code | llama / BF16 | 128 | 2.433623 | +0.001132 | +0.1133% | yes |
| 0.6B | code | llama / Q4_K_M | 112 | 2.464047 | +0.031556 | +3.2059% | no |
| 1.7B | prose | marin / int8 | 124 | 2.650019 | +0.000738 | +0.0738% | yes |
| 1.7B | prose | llama / BF16 | 128 | 2.650321 | +0.001040 | +0.1041% | yes |
| 1.7B | prose | llama / Q4_K_M | 115 | 2.611521 | -0.037760 | -3.7056% | no |
| 1.7B | code | marin / int8 | 127 | 2.116909 | -0.003081 | -0.3076% | yes |
| 1.7B | code | llama / BF16 | 128 | 2.117654 | -0.002336 | -0.2333% | no |
| 1.7B | code | llama / Q4_K_M | 116 | 2.077427 | -0.042563 | -4.1670% | no |

Q4 top-1 is 84.375%/87.5% on 0.6B and 89.84375%/90.625% on 1.7B:
**all four miss the fixed 95% gate**. 0.6B code also misses loss (+0.031556)
and perplexity (+3.20594%). The same-engine llama.cpp BF16 comparison gives
identical Q4 top-1 counts and +0.030424 nats on 0.6B code, so that miss is
not explained away by changing the comparator. Lower loss on 1.7B is a
rounding/evaluation outcome, not a general quality improvement. Greedy
continuations differ in all four Q4 cases; wording equality alone is not
semantic correctness. No claim that Q4 universally fails on larger models.
The short INT8 pilot's 0.6B prose top-1 (123/128) does not replace its
passing original 1024-position Stage 3 quality gate.

Fresh strict Marin HF verification passes both models, eight threads,
without changed tolerances. llama.cpp BF16 also rounds matrix activations
into BF16, whereas Marin keeps them FP32; Q4 dots use Q8_K activations.
The first 64-token BF16 cross-engine maximum logit gaps are 0.047–0.068,
so llama.cpp BF16 is **not** relabelled as strict FP32 HF numerical parity,
even though pilot top-1 agrees 128/128 in all four cases. Q4 comparisons
include runtime/activation precision effects, not weight error alone.
[CPU type traits](https://github.com/ggml-org/llama.cpp/blob/e94acad853c0d11a3b47974e3167ec12f263d86a/ggml/src/ggml-cpu/ggml-cpu.c).

**Bandwidth.** A safe-Rust AVX2-build streaming checksum reads a 1 GiB buffer,
one unmeasured and five measured rounds. Median read throughput is
21.130 GB/s (one thread, range 20.479–23.571) and 24.221 GB/s (eight,
24.170–24.767). It is a fresh stream calibration, not an inference hardware
counter. Inference estimates in `timing-summary.json` multiply decode rate
by one logical weight payload plus FP32 KV at mean decode depth; they omit
extra packing/copies and may include cache hits. `perf` is unavailable;
no memory-controller bandwidth counter is claimed. Package energy reads
return permission denied, so no joules/token result.

| Decision / gate | Status | Evidence / limit |
|---|---|---|
| Stage 4 growable FP32 KV | **GO independently** | Storage unchanged; original accepted A matrix and latest correctness evidence retained |
| Stage 4 bounded prefill / full acceptance | **NO-GO; bounded component removed** | +10.48% TTFT / -3.28% decode failures unchanged; this different comparison does not close them |
| Native CPU INT4 implementation now | **NO-GO** | External pilot fails all four top-1 checks and 0.6B loss/2k TTFT; no native quality proof or measured kernel exists |
| External Q4 decode potential | **Promising pilot only** | 1.39–1.68x short, 1.19–1.52x 2k versus current INT8; two samples, one AVX2 laptop |
| External Q4 low-RAM inference | **Not demonstrated versus INT8** | Smaller files/payload, but short RSS equal and 2k RSS higher due to layouts/workspaces |
| IQ4_XS / older PCs / native scalar and AVX2 kernels / full native gates | **UNMEASURED** | Source/design audit only; no local IQ4 artifact, real old CPU or implemented native format |

**Next steps toward 3B–7B and adaptation.** First a separate, small quantizer
quality experiment: weighted affine/Q4_K-style groups with explicit INT8
head protection, independently calibrated and held-out corpora. Evaluate
packed scalar rows/panels directly; never expand a model into BF16. Require
the quality/storage gates before fused SIMD work. The 0.6B storage budget
also limits how many sensitive tensors can remain INT8. Do not adopt the
old symmetric CUDA INT4 as a shortcut or start with IQ4_XS/GPTQ machinery.

Then validate one supported dense model around 3B–4B, using its actual
config, tied/untied head and byte counts, before a 7B–8B model. Weight-only
estimates: 3B needs 6 GB BF16 / 3.094 GB grouped INT8; 7B needs 14 GB /
7.219 GB. A realistic 5–5.5 bpw mixed Q4 budget is 1.875–2.063 GB and
4.375–4.813 GB respectively, **before KV, activations and margin**. These
are estimates, not loaded-model measurements. KV must come from each
model's actual layers/heads/head size; do not transplant the two small
models' 224 KiB/position. Whole-prefill workspace remains material.

Keep the simple admission policy for measured Linux/cgroup-v2 environments;
report unknown budgets on non-Linux/v1 rather than advertising portable
OOM guarantees. Next adaptive step is an inspectable estimate of weights,
held/planned KV, input workspace, margin, usable RAM and SIMD availability.
Use measured physical-core prefill settings and existing single-thread
decode; keep formats explicit until quality is proven. Profile larger-model
TTFT and decode before changing threads, reintroducing chunks, lowering KV
precision or building a planner. No V9 implementation is warranted now.

**Verification and stop.** No production Rust, dependency, configuration,
unsafe code, default or CUDA path changed. Fresh fmt, workspace check,
all-feature clippy `-D warnings`, line-limit and unsafe checks pass; both
fresh strict HF runs pass. The unchanged source hashes match the earlier
594-test CUDA-required validation; that full suite was not redundantly
rerun for this documentation/measurement-only study. Harnesses assert
stable prefill outputs, valid IDs, finite scored logits and correct stream
checksums; every subprocess succeeded and zero OOM/kill events were recorded.
All three stashes and HEAD remain unchanged. No commit or push. Full native
quality/speed/RAM/startup gates, older hardware and IQ4_XS remain unverified.
Raw evidence is in ignored `bench-results/cpu-stage5/` (initial fingerprints,
canonical IDs, provenance samples, commands, contexts, every attempt,
quality/logit arrays, source/library audit, bandwidth, HF and check logs).
**Stop after this pilot: retain CPU INT8; native CPU INT4 is NO-GO until a
quality-preserving quantizer is established.**

## 29. Offline scalar 4-bit accuracy and 4B memory feasibility (2026-10-12)

This study has a 45-minute wall-clock cap, beginning 12:26:55 UTC. It does
not authorize a production INT4 path or SIMD kernels. Section 28's gates
remain unchanged: at least 95% top-1 agreement with Marin BF16, no more
than 0.03 nats loss increase and 3% perplexity increase, and the original
storage, resident-memory, startup and speed gates. A 128-position pilot
is a screening result, never full acceptance. Executable generated code
must be checked separately from token agreement; unrun checks are blocked.

The experiment uses a private copy of the existing model/tensor crates,
packed unsigned nibbles and per-group FP32 affine parameters. It decodes
one weight row at a time, retaining packed storage throughout inference.
No production source or CLI changes are planned. Controls reuse identical
HF checkpoints, canonical IDs, vocabulary limits and teacher-forced
scoring from section 28; llama.cpp's different activation arithmetic and
incompletely audited Q4 calibration provenance remain limitations.

### Offline method, controls and causal limits

The representative pilot uses an ignored, private workspace in
`bench-results/cpu-q4-quality/`, copied from the unchanged production sources.
The existing CPU loader, attention, FP32 KV, threading and INT8 table kernels
are reused. Only private matrix storage/dispatch and named loading are extended;
there is no production INT4 option or new SIMD kernel. Packed layer matrices
remain packed during inference. The scalar decoder expands one FP32 row,
then uses the existing dot implementation. It never retains a whole decoded
matrix/model. This is a quality tool, not a candidate performance engine.

The affine quantizer stores two unsigned nibbles per byte, FP32 scale/offset
per group and a checked 16-byte header. Group size is 64; signed INT8 tables
retain the existing group-128 format and norms retain FP32. It tries min/max
and 95%-range initialization with four weighted least-squares/reassignment
steps, accepting only lower local weighted error. Calibration weights are
per-input-channel mean squared activations from two disjoint 256-token
texts (README and hand-written prose/code), collected with the installed
HF BF16 model. Evaluation remains the frozen Vim prose and Marin code IDs,
64-token prefix, vocabulary 151669, FP32 KV and identical teacher forcing.
The 0.6B main comparison scores 128 positions per corpus; 1.7B and the final
selective probe score 32. Small samples can reject candidates but cannot
establish the original >=1024-position/corpus quality gate.

The selective probe uses group-32 Q4 for remaining projections, retaining
**all value and down projections in INT8**, plus the INT8 tied table.
Its protected tensors are a hypothesis, not a measured optimal selection.
The pilot uses diagonal activation importance; it does not implement AWQ's
channel rescaling or GPTQ's Hessian error compensation. Research distinction:
[AWQ](https://arxiv.org/abs/2306.00978) motivates activation-selected channel
scaling; [GPTQ](https://arxiv.org/abs/2210.17323) uses approximate second-order
information. Their published accuracy does not establish Marin's 95% gate.

The original llama Q4 artifacts already contained imatrix metadata and
Q6 protection, but their exact calibration data/input provenance was unknown.
This run additionally checks **every used BF16 GGUF tensor/value** against the
HF checkpoint, then freshly quantizes that exact BF16 GGUF with the pinned
llama.cpp library's Q4_K_M recipe, without an imatrix. Both 310-tensor inputs
match exactly (596,049,920 / 1,720,574,976 values). Fresh artifacts remove the
checkpoint-identity ambiguity; they do not substitute for a well-calibrated
Q4 recipe. Existing original-artifact results are retained separately.
llama.cpp still uses Q8_K activation dots for Q4 versus Marin FP32 activations,
so its comparison is an end-to-end backend comparison, not a weight-only
causal experiment. The private BF16/INT8 controls are checked against the
frozen production probe's entire first-logit bytes, 32 teacher-forced scores
and greedy IDs on both models.

### Measured quality (BF16 is the reference)

| Model | Format | Corpus | Scored | Top-1 | PPL | Loss delta, nats | PPL change |
|---|---|---|---:|---:|---:|---:|---:|
| 0.6B | bf16 | prose | 128 | 128/128 (100.00%) | 15.1941 | +0.000000 | +0.000% |
| 0.6B | int8 | prose | 128 | 123/128 (96.09%) | 15.2198 | +0.001689 | +0.169% |
| 0.6B | llama original Q4_K_M | prose | 128 | 108/128 (84.38%) | 15.1928 | -0.000087 | -0.009% |
| 0.6B | uniform64 | prose | 128 | 99/128 (77.34%) | 18.7377 | +0.209631 | +23.322% |
| 0.6B | weighted64 | prose | 128 | 103/128 (80.47%) | 15.9155 | +0.046384 | +4.748% |
| 0.6B | bf16 | code | 128 | 128/128 (100.00%) | 11.3872 | +0.000000 | +0.000% |
| 0.6B | int8 | code | 128 | 125/128 (97.66%) | 11.4725 | +0.007463 | +0.749% |
| 0.6B | llama original Q4_K_M | code | 128 | 112/128 (87.50%) | 11.7523 | +0.031556 | +3.206% |
| 0.6B | uniform64 | code | 128 | 107/128 (83.59%) | 15.9090 | +0.334395 | +39.710% |
| 0.6B | weighted64 | code | 128 | 106/128 (82.81%) | 12.6315 | +0.103699 | +10.927% |
| 1.7B | bf16 | prose | 32 | 32/32 (100.00%) | 15.3620 | +0.000000 | +0.000% |
| 1.7B | int8 | prose | 32 | 30/32 (93.75%) | 15.4550 | +0.006032 | +0.605% |
| 1.7B | llama original Q4_K_M | prose | 32 | 27/32 (84.38%) | 14.4787 | -0.059222 | -5.750% |
| 1.7B | weighted64 | prose | 32 | 27/32 (84.38%) | 16.4101 | +0.065995 | +6.822% |
| 1.7B | bf16 | code | 32 | 32/32 (100.00%) | 14.3365 | +0.000000 | +0.000% |
| 1.7B | int8 | code | 32 | 32/32 (100.00%) | 13.8796 | -0.032394 | -3.187% |
| 1.7B | llama original Q4_K_M | code | 32 | 27/32 (84.38%) | 13.4895 | -0.060900 | -5.908% |
| 1.7B | weighted64 | code | 32 | 29/32 (90.62%) | 14.5511 | +0.014858 | +1.497% |
| 0.6B | INT8 matched 32 | prose | 32 | 31/32 (96.88%) | 18.6019 | +0.011888 | +1.196% |
| 0.6B | llama fresh Q4_K_M | prose | 32 | 25/32 (78.12%) | 16.5796 | -0.103199 | -9.805% |
| 0.6B | weighted64 matched 32 | prose | 32 | 23/32 (71.88%) | 17.4319 | -0.053070 | -5.169% |
| 0.6B | selective32 | prose | 32 | 26/32 (81.25%) | 18.8830 | +0.026888 | +2.725% |
| 0.6B | INT8 matched 32 | code | 32 | 31/32 (96.88%) | 16.6836 | +0.002065 | +0.207% |
| 0.6B | llama fresh Q4_K_M | code | 32 | 26/32 (81.25%) | 21.3525 | +0.248809 | +28.250% |
| 0.6B | weighted64 matched 32 | code | 32 | 25/32 (78.12%) | 19.8870 | +0.177710 | +19.448% |
| 0.6B | selective32 | code | 32 | 28/32 (87.50%) | 19.0542 | +0.134931 | +14.446% |
| 1.7B | INT8 matched 32 | prose | 32 | 30/32 (93.75%) | 15.4550 | +0.006032 | +0.605% |
| 1.7B | llama fresh Q4_K_M | prose | 32 | 25/32 (78.12%) | 15.4842 | +0.007921 | +0.795% |
| 1.7B | weighted64 matched 32 | prose | 32 | 27/32 (84.38%) | 16.4101 | +0.065995 | +6.822% |
| 1.7B | INT8 matched 32 | code | 32 | 32/32 (100.00%) | 13.8796 | -0.032394 | -3.187% |
| 1.7B | llama fresh Q4_K_M | code | 32 | 27/32 (84.38%) | 18.5411 | +0.257180 | +29.328% |
| 1.7B | weighted64 matched 32 | code | 32 | 29/32 (90.62%) | 14.5511 | +0.014858 | +1.497% |

Calibration improves the unweighted 0.6B candidate's loss, but the main
candidate still fails top-1 on both models. Lower perplexity on a short
segment is insufficient when token agreement fails. The 95% threshold is
unchanged. Q4_K_M's original misses cannot be attributed solely to missing
calibration: it was already importance-aware and mixed Q4/Q6. Error
accumulation, activation arithmetic, calibration coverage and sensitive
matrices remain distinct hypotheses. No universal claim that four-bit
accuracy is impossible follows from this small study.

### Executable code, rather than token similarity

One identical no-thinking ChatML prompt asks for a pure Python `clamp`.
Generation is greedy, limited to 128 tokens, and uses the same canonical
prompt IDs. AST validation excludes imports, attribute access and arbitrary
calls; min/max and Boolean comparisons are allowed. The generated function
runs in an isolated process with CPU/address-space limits. Six cases cover
below/inside/above range, equal bounds, negative values and floats. The
initial harness rejected a valid Boolean expression from llama Q4; that
harness error was corrected, existing output reevaluated, and the original
record retained. This is one task, not acceptance of the original eight-task
gate.

| Model | Format | Executed assertions | Task |
|---|---|---:|---|
| 0.6B | bf16 | 6/6 | PASS |
| 0.6B | int8 | 6/6 | PASS |
| 0.6B | weighted64 | 1/6 | FAIL |
| 0.6B | llama-Q4_K_M | 6/6 | PASS |
| 0.6B | llama-fresh-Q4_K_M | 6/6 | PASS |
| 1.7B | bf16 | 6/6 | PASS |
| 1.7B | int8 | 6/6 | PASS |
| 1.7B | weighted64 | 6/6 | PASS |
| 1.7B | llama-Q4_K_M | 6/6 | PASS |
| 1.7B | llama-fresh-Q4_K_M | 6/6 | PASS |

The calibrated 0.6B candidate generated
`return min(x, low) if x < low else max(x, high)` and passed only 1/6:
a genuine semantic failure, not a formatting or tokenizer mismatch.

### Format overhead and startup

Q4_K blocks use 144 bytes/256 weights (4.5 bits/weight); IQ4_XS uses
136/256 (4.25 bits), with nonlinear codebook values and compact group scales.
The local affine group-64 design is **5 bits/weight**, including 25% metadata
over the nibble bytes. Group-32 is 6 bits/weight, 50% metadata. Compact
metadata is therefore part of the design problem, not a minor header cost.
Pinned definitions and tensor-selection rules are in llama.cpp
`ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c` and `src/llama-quant.cpp`
at e94acad853c0d11a3b47974e3167ec12f263d86a. Q4_K_M protects selected
value/down matrices and the output/tied table; IQ4_XS is not simply uniform
linear INT4. No IQ4_XS quality claim is made.

| Model | Candidate | Whole payload incl. headers | / INT8 | / BF16 | Original storage gate |
|---|---|---:|---:|---:|---|
| 0.6B | weighted64 | 435,960,896 B | 70.903% | 36.567% | FAIL |
| 0.6B | selective32-estimate | 524,040,384 B | 85.228% | 43.955% | FAIL (estimate) |
| 1.7B | weighted64 | 1,202,191,424 B | 67.740% | 34.933% | PASS |
| 1.7B | selective32-estimate | 1,493,956,800 B | 84.180% | 43.411% | FAIL (estimate) |

The selective candidate's better precision cannot qualify by hiding its
extra INT8 tensors: it exceeds both original storage ratios on both models.
The original external Q4 RAM miss also persists as an implementation issue:
0.6B had 370.96 MiB mapped weights **plus 204.75 MiB repack**, with KV and
compute buffers additional. At matched 2k context its RSS was 194.38 MB
above Marin INT8; 1.7B was 115.34 MB above. Compact GGUF file size alone
was never evidence of an inference RSS win (section 28).

Cold-requested load-only measurements below use DONTNEED on every source
and packed tensor, then a fresh process; block reads document actual I/O.
These are one-shot observations, not a cold boot or a statistical startup
gate. The offline path deliberately still reads one BF16 source tensor
before loading its packed counterpart, keeping existing validation and
memory guards. Its additional file I/O is avoidable in a future direct
packed loader; it is not the cost of a production implementation.

| Model | Format | Weight-load s | Whole startup s | Peak RSS, MiB | Actual block reads, MB |
|---|---|---:|---:|---:|---:|
| 0.6B | bf16 | 1.376 | 1.610 | 1202.46 | 982.94 |
| 0.6B | int8 | 1.639 | 1.848 | 667.02 | 977.69 |
| 0.6B | weighted64 | 1.529 | 1.755 | 529.53 | 1255.06 |
| 1.7B | bf16 | 3.388 | 3.606 | 3349.24 | 3132.59 |
| 1.7B | int8 | 3.558 | 3.788 | 1804.73 | 3079.91 |
| 1.7B | weighted64 | 3.296 | 3.507 | 1289.27 | 3962.54 |

Scalar teacher-forced runtimes are approximately 150–154 seconds for
0.6B/128 positions and 149–152 seconds for 1.7B/32, including repeated
prefill/greedy checks. They are not native INT4 decode speed estimates.
No speed acceptance is claimed from this deliberately scalar experiment.

### Actual 4B configurations and conservative memory budgets

[Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json)
and [Qwen3-4B-Instruct-2507 config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/config.json)
both imply **4,022,468,096 parameters**, hidden 2560, FFN 9728,
36 layers, 32 query heads, 8 KV heads, head dimension 128 and one tied
151936×2560 table. Their RoPE/context limits differ (40960/1e6 versus
262144/5e6). Marin's current config parser accepts both and reports the
same exact parameter/workspace counts without loading their weights.
Importantly, query width is 4096, not hidden width 2560; the existing
non-square projection/attention dimensions already account for that.
There is no fabricated official Qwen3-3B result here: 3B-class conclusions
are interpolation estimates, not a measured checkpoint.

4B weight payloads: BF16 **8,045,328,384 B**; current group-128 INT8
**4,148,752,384 B**; proposed affine Q4/G64 + INT8 table
**2,672,717,824 B**; source-recipe Q4_K_M estimate **2,491,323,904 B**.
The Q4_K_M value is computed from current selection rules and compact block
sizes, not a downloaded or measured 4B GGUF. Native headers/allocator costs
would be additional. Q4/G64 has not passed quality.

FP32 KV is **288 KiB/token**: 576/1152/2304 MiB at 2k/4k/8k.
Named full-prefill activation workspace is **156 KiB/token**, plus
607744-byte last-token logits: 312/624/1248 MiB at those contexts.
Attention packing adds 16/32/64 MiB. Budget totals also include a separate
**128 MiB unmeasured runtime allowance**, **256 MiB safety margin**, and
**64 MiB growth headroom**. They conservatively charge full attention
packing and a full cache during prefill. Existing populated-cache growth
may additionally need the largest old vector transiently (up to 32/64/128
MiB at 2k/4k/8k); near-boundary fits must include it before admitting work.
Fresh full-prefill starts without old-vector copies.
OS allowances are 1 GiB on a 4 GiB machine and 2 GiB on an 8 GiB machine,
leaving 3/6 GiB for Marin. These are stated planning assumptions, not
measurements. Decimal 4GB/8GB hardware provides less than 4/8 GiB.

| Format | Context | Total estimated GiB | Fits 4 GiB / 1 GiB OS | Fits 8 GiB / 2 GiB OS |
|---|---:|---:|---|---|
| BF16 | 256 | 8.041 | no | no |
| BF16 | 2048 | 8.814 | no | no |
| BF16 | 4096 | 9.696 | no | no |
| BF16 | 8192 | 11.462 | no | no |
| INT8 | 256 | 4.412 | no | yes, estimate |
| INT8 | 2048 | 5.185 | no | yes, estimate |
| INT8 | 4096 | 6.068 | no | no |
| INT8 | 8192 | 7.833 | no | no |
| affine-Q4-G64/INT8-table | 256 | 3.038 | no | yes, estimate |
| affine-Q4-G64/INT8-table | 2048 | 3.810 | no | yes, estimate |
| affine-Q4-G64/INT8-table | 4096 | 4.693 | no | yes, estimate |
| affine-Q4-G64/INT8-table | 8192 | 6.458 | no | no |
| Q4_K_M-recipe-estimate | 256 | 2.869 | yes, estimate | yes, estimate |
| Q4_K_M-recipe-estimate | 2048 | 3.641 | no | yes, estimate |
| Q4_K_M-recipe-estimate | 4096 | 4.524 | no | yes, estimate |
| Q4_K_M-recipe-estimate | 8192 | 6.290 | no | no |

**Practical conclusion:** current 4B INT8 is an 8 GiB / roughly 2k-context
candidate, not a 4 GiB solution. BF16 4B does not fit either budget. Compact
Q4 could allow 8 GiB / 4k, but 8k with FP32 KV/full prefill exceeds the
6 GiB process budget. On 4 GiB, only the compact Q4 recipe's very short
256-token estimate fits, narrowly; it is not quality accepted and repacking,
OS variation or a decimal-GB machine can erase that fit. No safe general
4GB/4B support promise follows. For a hypothetical 3B with these same
36-layer KV/activation dimensions, subtracting about 1.022B matrix weights
would lower the 2k INT8 total to about 4.20 GiB and compact Q4 total by
roughly 0.54 GiB; even that 3B Q4/2k estimate remains near/above a 3 GiB
process allowance. Real 3B architectures require their own configs.

The previously measured 1/8-thread streaming medians are 21.13/24.22 GB/s
on this Ryzen, not on a low-end DDR4 PC. At 4B/2k, streaming-only ideal
ceilings are approximately **4.45 tok/s for INT8** and **6.83 tok/s for
compact Q4_K_M** using the single-thread bandwidth. At 8k they fall to
3.22/4.31. These divide measured bandwidth by estimated weights+live FP32
KV traffic; compute, nonlinear ops, table access and dequantization can
only reduce them. Eight-thread STREAM bandwidth does not make current
single-thread decode eight-threaded. At 10 GB/s (illustrative older-PC
bandwidth, unmeasured), 2k ceilings are about 2.10/3.23 tok/s.
4B linear/attention work is roughly 2.57× the measured 1.7B setup;
projecting its 10.433-second INT8 2k/8-thread prefill gives approximately
27 seconds, **an estimate**, not an actual 4B TTFT. Large models on small
PCs can be memory feasible and still feel slow.

### Acceptance and next milestone

| Gate / question | Decision | Evidence |
|---|---|---|
| Packed decoded computation correctness | PASS for pilot | Independent f64 products, constant/nibble round trips, malformed headers, dimensions, lengths and nonfinite metadata |
| Original >=95% top-1 | FAIL | Calibrated primary candidate misses on both models; tables above |
| Original loss/PPL gates | FAIL | 0.6B both corpora and 1.7B prose fail; larger acceptance corpora unrun |
| Executable code correctness | FAIL for primary candidate | 0.6B clamp passes 1/6 versus BF16/INT8 6/6 |
| Original storage ratios | FAIL for family | G64 0.6B 70.903% of INT8; selective design 84–85% |
| Full native speed / startup / RSS acceptance | BLOCKED | No production/native compute path; scalar/cold-requested observations do not qualify it |
| General 4GB / 3B–4B support | NO-GO promise | Short-context compact-Q4 estimate is narrow and quality unaccepted |
| 8 GiB / 4B INT8 around 2k | Feasible estimate, unverified | Conservative total 5.185 GiB; actual load/quality/TTFT still required |
| Native CPU INT4 implementation now | **NO-GO** | Mandatory quality/storage failures; thresholds unchanged |

Next architecture milestone: **direct, versioned loading of already packed
INT8 weights with explicit peak-memory admission**, followed by one 4B
INT8/2k pilot on an 8 GiB budget. Preserve strict checkpoint identity and
fallback semantics. Avoid BF16 source rereads/requantization every start;
count tables, per-tensor load transients, initialized KV, full workspace,
attention packing, runtime and copy headroom separately. Use one conservative
memory policy with current guards; do not revive chunked prefill or the V9
planner. SIMD dispatch, arbitrary valid Qwen dimensions, INT8 arithmetic,
threaded full prefill and growable FP32 KV already generalize. Direct packed
checkpoint loading, a quality-preserving compact quantizer and later KV
memory reduction need new, separately gated work. No such feature is
implemented here. Before native INT4, test a narrowly scoped AWQ/channel
scaling or GPTQ calibration export with compact metadata and measured tensor
selection on held-out prose/code/conversation; stop immediately if 95% or
storage fails. Do not invest in a packed kernel before that.

### Verification and preservation

The isolated workspace passed `cargo fmt`, all-target workspace check,
all-feature/all-target clippy with `-D warnings`, and **596 tests, zero
failures/ignores**. A later wider independent f64 test was added and the
full tensor suite rerun: 40 unit + 2 accuracy + 11 quantization tests pass.
All copied Rust files were checked against 425 lines; introduced Rust is
safe. Root fmt/line-limit/unsafe checks pass. Production Rust source hashes
remain identical to section 28. All previous BENCHMARKS content is retained,
README is untouched this task, all three stashes and HEAD are unchanged.
Raw sources, commands, full logits/scores, generated code/assertion outcomes,
calibration metadata, manifests, provenance, memory estimates and check logs
remain in ignored `bench-results/cpu-q4-quality/`. No commit or push.



## 30. Bounded L2/INT8 packing audit (2026-10-12)

Added to the section 29 timebox at the user's request; no production kernel
changes. Inspection covers `matrix.rs`, `kernel.rs`, `gemm.rs`, `gemv.rs`,
`q8.rs`, `q8_gemm.rs`, `q8_gemv.rs`, dispatch and row threading.

BF16 storage is row-major packed pairs, usually a checked zero-copy view of
loader u16s; only incompatible alignment/odd columns require load-time
repacking. BF16 GEMV splits the current activation vector into even/odd
planes, then streams rows without repacking weights. BF16 GEMM uses
MR=4, NR=2, KC=256 pairs (512 values), TB=64 tokens, repacking small weight
and activation panels for each token/K block. Its per-worker even/odd
activation copies cover the current 64-token block.

**INT8 weights are already quantized/packed once at load:** immutable
row-major signed byte codes and separate FP32 group scales. No per-call
requantization or whole-weight repack occurs. Existing tests confirm code
and scale contents/pointers remain unchanged after GEMM/GEMV calls. INT8
GEMV forms four planes of the *activation vector*, then extracts signed
bytes directly from checked u32 views; it does not make a decoded weight
copy. INT8 GEMM has MR=4, NR=2, KC=512 values, TB=64. It decodes/packs one
4×512 FP32 weight panel repeatedly, once per token block/K/row group,
reusing it across 64 tokens. Activations are packed once per token block/K
**per row worker**; eight workers duplicate that packing. These are bounded
scratch conversions, not repeated packing of the persistent INT8 format.

On this host sysfs reports 1 MiB private L2 and 16 MiB L3; `perf` is absent,
`perf_event_paranoid=2`. L2 misses, cycles, actual DRAM bytes, and exclusive
packing-time counters are **unmeasured**. Analytical traffic is not a
hardware counter. Baseline INT8 activation panels are 128 KiB/worker;
TB=128 makes them 256 KiB. Weight panels remain 8 KiB plus 2 KiB decode
scratch. Logical weight decoding scales as ceil(tokens/TB), while activation
packing volume is roughly tokens×cols×4 per worker. Doubling TB reduces
repeated weight conversion but enlarges cache working sets; it need not
improve overall locality. Keeping all decoded FP32 panels at load would
cost about four bytes per layer weight instead of one, defeating low-RAM
INT8. A one-time interleaved *byte* layout might avoid panel rearrangement,
but still needs dequantization and a verified GEMV/scalar access strategy;
no evidence here justifies its kernel/layout complexity.

The isolated pilot changes **only INT8 TB from 64 to 128**, preserving KC,
MR/NR, arithmetic and all production sources. It loads the real layer-0
gate projection from each exact HF checkpoint: 3072×1024 and 6144×2048.
Each subprocess quantizes once, warms once and records five matmul calls
at 128/1024 tokens and 1/8 threads. Execution order is baseline A1,
candidate B, baseline A2, serially after the quantization/code/reference
jobs. No CPU benchmarks overlap. Every entire output buffer is byte-identical
between A1/B/A2; this is a projection microbenchmark, not model TTFT.
The 0.6B short/eight-thread baseline drifts 25%; its tiny timings are noisy.
The longer 1.7B result has under 1% baseline drift.

| Model | Tokens | Threads | A1 median ms | A2 median ms | TB128 median ms | Change vs mean baseline | Baseline drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B | 128 | 1 | 8.128 | 8.252 | 8.020 | -2.08% | +1.53% |
| 0.6B | 1024 | 1 | 58.859 | 59.733 | 59.555 | +0.44% | +1.48% |
| 0.6B | 128 | 8 | 1.407 | 1.759 | 1.986 | +25.45% | +24.97% |
| 0.6B | 1024 | 8 | 9.846 | 10.371 | 12.903 | +27.64% | +5.33% |
| 1.7B | 128 | 1 | 34.749 | 34.990 | 33.553 | -3.77% | +0.70% |
| 1.7B | 1024 | 1 | 278.978 | 279.517 | 258.423 | -7.46% | +0.19% |
| 1.7B | 128 | 8 | 6.949 | 7.163 | 7.633 | +8.18% | +3.08% |
| 1.7B | 1024 | 8 | 37.156 | 37.457 | 38.930 | +4.35% | +0.81% |

TB128 improves the long 1.7B single-thread projection by **7.46%**, but
regresses that model's eight-thread long case **4.35%** and short case
**8.18%**. 0.6B long/eight-thread regresses **27.64%**, with 5.33% baseline
drift; its single-thread long result is essentially unchanged (+0.44%).
This is no robust cross-model/prefill win. No end-to-end TTFT speedup is
claimed; section 28's measured full-model INT8 2k/eight-thread baselines
remain the comparison reference (1.7B prefill 10.433 s). Candidate load
quantization functions are unchanged and no persistent weight copy is
added. Scratch grows 128 KiB per active worker. Process RSS/startup values
below include the micro-harness's source buffers, reference outputs and
serialization; they are not whole-model peak-RAM/startup measurements.

| Model/run | Peak RSS MiB | Read + quantize projection, s |
|---|---:|---:|
| locality-0.6-A1 | 55.40 | 0.012430 |
| locality-0.6-B | 56.65 | 0.007200 |
| locality-0.6-A2 | 55.29 | 0.007792 |
| locality-1.7-A1 | 123.30 | 0.032929 |
| locality-1.7-B | 123.91 | 0.022076 |
| locality-1.7-A2 | 123.48 | 0.024349 |

**NO-GO for implementing TB128 or a new one-time weight layout now.**
The existing path already packs persistent weights once, and the measured
tradeoff hurts the important threaded-prefill workload. A later targeted
profile may investigate duplicated activation packing/shared panels or a
thread-count-specific block choice, but only after hardware-counter or
packing-time evidence and a full-model win. No such change is implemented.
The isolated TB128 tensor crate passes fmt, strict all-target clippy and
all 40 unit tests, including quantized correctness/thread/scalar coverage.
Raw byte-exact outputs, five-round samples, A/B/A commands, RSS and logs
are in `bench-results/cpu-q4-quality/locality-*`. SIMD, INT8 quantization
quality, defaults and all three stashes are preserved.


## 31. Consolidation audit and direct INT8 loading assessment (2026-10-12)

What this repo does: Marin loads a local Qwen3 checkpoint and runs generation
on CPU or CUDA, with shared tokenization, sampling and cache-reuse semantics.
The assumed workload is one interactive user/model/session, not a concurrent
server. This audit is capped at 45 minutes from 13:20:14 UTC. No new loader,
model support, quantization, kernels or planner is implemented.

### Real execution paths and boundaries

`marin` / `marin-run` arguments → `cli::load::open` → `ModelDir::open`
(config, tokenizer, safetensors headers) → `Footprint` / backend selection →
`Engine::load` → common `Qwen3Weights::load` / `Store` → CPU `Matrix` or
CUDA `Weight` → `Session` → `generate_reusing` → `CausalLm::forward` →
attention/KV commit → sampler → token stream/rendering.

The CPU model consumes BF16 tensors one at a time. BF16 matrices own the
loader vector or a necessary alignment/odd-width repack. INT8 matrices own
only codes/scales; norms are FP32. CPU prefill uses bounded SIMD panels and
row/head threading, **not bounded prompt chunks**. CPU KV grows, commits
length after a successful pass, and retains its high-water allocation after
rollback/clear. CUDA uploads one tensor at a time, owns device buffers,
uses the separate proven 512-token prefill chunks, paged f32/f16 KV and
persistent decode/verification workspaces. CPU/CUDA cache mismatch returns
an error. Model/backend switching drops the old session before loading and
rebuilds/replays state rather than sharing stale KV.

The existing `Store` trait genuinely has two implementations and shares
model shape/weight validation. `Engine`/`EngineCache` enums and the small
`CausalLm`/`SequenceCache` contracts are justified. CPU and GPU forward
orchestration should remain separate: ownership, synchronization and
workspace lifetimes differ. Moving them into a generic device-tensor layer
would add coupling without a demonstrated need. CUDA is dynamically loaded;
CPU execution does not require a working driver, although `model` still has
a compile-time CUDA dependency. Removing that dependency or public backend
APIs would be a separate compatibility change.

| Category | Code/evidence | Decision |
|---|---|---|
| Production | CPU/CUDA BF16, opt-in CPU/CUDA INT8, SIMD/threading, growable CPU KV, GPU paging/f16, reuse, GPU snapshots | Keep |
| Necessary fallbacks/reference | CPU scalar/non-AVX2, alignment/odd-width handling, GPU f32/scalar attention, strict HF verification, independent numerical tests | Keep |
| Opt-in measured execution | GPU n-gram speculation and its rollback path | Keep |
| Active diagnostics/experiments | `quality`, `trace`, `probe`, guess/replay/pass-cost commands; explicit GPU INT4 | Keep; callers/tests exist |
| Historical artifacts | Rejected CPU bounded prefill, CPU Q4 scalar/TB128 private workspaces, failed CUDA prototypes and their measurements | Preserve; not reachable through root Cargo/CLI |

The runtime n-gram index uses one fixed match length; the CLI replay index
searches multiple lengths for experimental reports. They share a tiny hash
routine but have different costs/semantics. Merging them would either broaden
the runtime interface or add indexing work; no such refactor is justified.
Likewise CPU pass buffers, per-worker packing, scoped thread startup and
GPU per-prefill workspaces still allocate. They are bounded and used, not
abandoned prototypes. Profile their cost before changing their lifetimes.
The failed TB128 experiment remains rejected (section 30).

### Safely removed or simplified

- Removed the unused **loader → tensor dependency edge**, plus only that
  dependency-list entry in Cargo.lock. No loader source, exports, tests,
  macros or build scripts use it. Standalone loader now depends on serde,
  serde_json, thiserror and tokenizers; workspace tensor/SIMD dependencies
  remain because inference uses them. No package version changed.
- CPU snapshot requests no longer spawn/join a full source-weight hash whose
  result cannot produce a CPU snapshot identity. CUDA snapshot hashing is
  unchanged. The existing CPU safety test now checks the hash handle is
  absent, BF16/INT8 output unchanged, missing files remain missing and
  existing snapshot bytes remain untouched. The added assertion failed on
  the old path and passes after the conditional change. CPU snapshots remain
  unsupported; this does not implement them. No speed gain is claimed.
- Deleted one stale CPU attention comment referring to the removed bounded
  prefill path. Exact reservation itself remains necessary bounded scratch
  management and is retained. No mathematical code changed.
- Corrected the at-a-glance Stage 3 row, which still called completed HF/CUDA
  closure checks open. Its original failure/closure evidence remains intact.
  The section 25 historical note about unnecessary CPU snapshot hashing is
  superseded by this cleanup, not by new snapshot support.

No function, CLI option, compatibility path, test, fallback or historical
artifact was deleted. Public APIs that lack an internal caller are not
proven dead for external/debug use. Net production Rust line count is not
a useful acceptance criterion; this removes real work and a dependency edge.

### Ranked remaining findings (left unchanged)

1. **Memory diagnostics can say “fits” when CPU admission will refuse.**
   `cli/memory.rs::footprint/available_bytes` uses host MemAvailable and
   weights+full KV+slack; `fit_cpu` uses cgroup-aware memory and also includes
   full-prefill workspace. A verified **512 MiB memory.max** child ran
   `marin-run inspect Qwen3-1.7B --ctx 8192 --quantize int8` and printed
   “needs about 3.92 GB, 10.49 GB free, fits”. No model was loaded in that
   child. Actual CPU load/growth guards are still active; this is misleading
   telemetry, not evidence they were bypassed. The same output labels the
   1.77 GB INT8 payload “stored as bf16”. `describe` also uses a filename
   heuristic and says chat templates are unsupported when reused by chat UI.
   Small future fix: have inspection use the real precision and CPU budget
   calculation, label backend-specific estimates, and avoid claiming general
   chat capability from a filename. No planner is needed.

2. **Default threads ignore CPU affinity when sysfs topology is present.**
   `tensor/threads.rs::physical_cores` counts all host sibling lists. Fresh
   process: available_parallelism=16, physical/default=8. A `taskset` child
   with one allowed CPU: available_parallelism=1, physical/default still=8.
   Explicit `--threads 1` works. A future change should cap usable cores by
   affinity/quota without altering unconstrained defaults; do not add a pool
   or scheduler before measuring that fix. [Rust's documented resource
   limits](https://doc.rust-lang.org/std/thread/fn.available_parallelism.html)
   explain why host topology alone is insufficient.

3. **Weight-byte reports count norms at source BF16 width.**
   `model/qwen3/weights.rs::loaded_bytes` counts norms at two bytes although
   both stores keep four-byte norms. The underestimate is 131072 / 247808 B
   on 0.6B/1.7B, and 392192 B for the 4B config. It did not invalidate the
   measured RAM gates, but a new packed loader must account for actual typed
   payloads. Its associated tests currently encode the source-width estimate;
   changing it is a deliberate accounting correction, not a deletion.

4. **Older NVIDIA GPUs do not have an explicit capability fallback.**
   `cuda/gpu.rs::Gpu::new` compiles compute_80 tensor-core PTX before checking
   fit/capabilities; `Backend::Auto` falls back for Unavailable/fit failures,
   not arbitrary module/JIT errors. Pre-Ampere hardware therefore risks a
   startup error instead of graceful CPU selection. [NVIDIA's MMA
   requirements](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
   make the minimum architecture relevant. This is a code-derived risk;
   no older NVIDIA card was available. Check capability before compiling,
   classify unsupported hardware explicitly, and preserve genuine driver
   errors. Do not broaden fallback to swallow every CUDA error.

5. **New artifacts need stronger trust-boundary rules than the current lite
   safetensors reader.** The reader checks size/shape/bounds but does not
   enforce disjoint spans or reject duplicate JSON keys. Shard paths are
   joined without forbidding absolute/parent paths. Public quantization
   structs can also be constructed with invalid groups/buffer lengths;
   the current production quantizer validates its inputs, but an artifact
   constructor must validate independently. Reject malformed new artifacts
   before allocating/using them. Restrict new cache manifests to contained
   paths; do not silently break legacy external-shard users in this cleanup.

6. **Current persistence guarantees differ.** Snapshots use temp-file,
   sync and rename; `Home::save_config/save_history` use truncating fs::write.
   A crash or failed write can lose remembered settings/prompt history.
   Atomic writes are a small separate reliability task. Snapshot temp names
   use PID + File::create rather than exclusive creation; do not copy that
   collision/symlink behavior into a new artifact publisher. Existing 64-bit
   content hashes are documented compiler-dependent, so a long-lived
   portable artifact needs an explicitly versioned stable digest scheme.
   [DefaultHasher](https://doc.rust-lang.org/std/collections/hash_map/struct.DefaultHasher.html)
   does not promise a stable algorithm. Neither format migration nor these
   unrelated persistence changes is implemented here.

On non-Linux systems `cpu_available_bytes` returns None. RAM guards are
therefore not an established low-RAM guarantee on those systems. The current
Linux/cgroup evidence must not be marketed as Windows/older-PC validation.

### Actual load ownership and fresh measurements

`Weights` retains paths and tensor metadata, not the BF16 checkpoint.
`read_bf16` allocates one Vec<u16> and at most 4 MiB read scratch; the scratch
drops on return. During quantization that one BF16 tensor coexists with its
new codes/scales. `Matrix::quantized` consumes the source vector and retains
only Q8. CUDA similarly releases host source/quantized vectors after upload.
A tied head uses the embedding matrix once. A fresh CPU cache holds zero KV
bytes before any token, independent of the 2048 context limit in this probe.

Four fresh, serial production-model load-only probes record all 310 progress
callbacks, /proc RSS/HWM, wall time and actual I/O. The callback runs after
each BF16 read and before conversion/storage, so it measures an in-flight
source tensor; HWM also captures conversions between callbacks. No weight
snapshot hash is requested. Cache conditions were **uncontrolled**, and
callback sampling adds overhead: times below are observations, not an A/B
startup gate or a cold/warm speed comparison.

| Model/format | Actual retained weight bytes incl. FP32 norms | Load s | Process peak RSS MB | Physical reads MB |
|---|---:|---:|---:|---:|
| 0.6B / bf16 | 1,192,230,912 | 1.352 | 1260.57 | 972.80 |
| 0.6B / int8 | 614,871,040 | 1.087 | 699.55 | 0.00 |
| 1.7B / bf16 | 3,441,397,760 | 3.457 | 3511.87 | 2591.00 |
| 1.7B / int8 | 1,774,710,784 | 3.522 | 1891.98 | 2526.51 |

For a live matrix with N values, conversion needs roughly 2N BF16 bytes plus
1.03125N packed bytes, beside already loaded matrices. The existing guard
allows 2×source bytes + 64 MiB, conservatively. Large tied embeddings are
loaded first, so their conversion peak is not added to the full model.
Later projections can add one BF16 matrix transient; they do not double the
whole model. Direct loading should remove that transient and conversion,
**not promise another 50% reduction in steady-state inference RAM**.

### Smallest direct packed-INT8 milestone (proposal only)

1. Export one tensor at a time with existing `tensor::quantize`; write and
   discard each result rather than loading a whole model before serializing.
   Use the existing Qwen expected tensor definitions. Produce a standalone,
   explicitly selected CPU artifact bundle with config/tokenizer and optional
   generation/tokenizer-config files, one weight payload and a manifest.
   Ordinary HF loading and BF16 default remain unchanged. No automatic stale
   source-directory cache or hidden format fallback in version 1.
2. Reuse [safetensors' row-major little-endian
   container](https://github.com/safetensors/safetensors#format): original
   matrix names contain I8 `[rows, cols]`, paired FP32 scales
   `[rows, cols/group]`; norm names contain exact promoted FP32 values.
   Group, signed-byte encoding, table precision, tied-head policy, quantizer
   revision, file/layout version, exact source/config/tokenizer identities
   and payload digests belong in the manifest. One tied table, no unused
   duplicate head. Choose/version a stable digest before coding; no new
   dependency is added by this audit.
3. Add checked typed I8/F32 reads to loader and a consuming packed-Q8 matrix
   constructor in tensor. Validate shapes, group divisibility/limits,
   offsets/overlap, duplicates, exact code/scale lengths and finite norm/scales
   before use. Move codes/scales into existing Q8; do not clone a complete
   encoded tensor, introduce mmap/unsafe, or decode whole weights. Keep file
   parsing independent of math; model validates the Qwen semantic mapping.
4. Extend the existing model reader for this explicit source format and
   actual typed-byte accounting. CLI export/selection/inspect are the only
   product plumbing. Force CPU for the first artifact path; original CPU/CUDA
   HF selection stays intact. No general weight-provider framework, GGUF
   engine, new backend or kernel. A cache hit reads the artifact once and
   verifies it during that read. Rehashing all BF16 files on every hit would
   erase the purpose; the explicitly selected artifact is the checkpoint,
   while its recorded source identity is conversion provenance.
5. Publish a unique, exclusively created temporary bundle on the same
   filesystem, sync files and directory, then atomically rename and sync the
   parent. A reader sees an old complete artifact or a new complete artifact.
   Concurrent exporters must verify/reuse an already published identical
   result; never overwrite a user's source checkpoint. Interrupted/corrupt,
   incompatible or stale explicitly selected artifacts fail clearly; do not
   trust file size/mtime as content identity or silently change precision.

First close the inspection/accounting discrepancy, then implement/export/load
on fixtures and the existing 0.6B/1.7B models. Relevant boundaries are loader,
tensor Q8 construction, model weight reader/validator and CLI plumbing. The
math/runtime/KV kernels need no redesign. Export can itself be bounded to
one source+packed tensor; 4B's largest embedding would be about 1.18 GB live
plus runtime, **an estimate**, rather than a resident 4B model.

Proposed gates, fixed before that future implementation:

| Area | Gate |
|---|---|
| Data correctness | All code bytes, FP32 scale/norm bits and tied-table identity equal current on-load INT8 quantization |
| Execution | Bit-identical logits/greedy and seeded continuations, 0.6B/1.7B short/2k, 1/8 threads and scalar/SIMD; reuse/rollback/context errors unchanged |
| Reference/defaults | Existing strict BF16/HF thresholds, CUDA/snapshot regressions and BF16 default untouched; INT8 remains opt-in |
| Artifact safety | Malformed/foreign/truncated/NaN/group/shape/offset/checksum cases rejected before use; interrupted/concurrent publish never exposes partial data or alters source files |
| Ownership | Cache hit performs no BF16 payload reads and no quantization; one owned code/scale allocation per tensor, no whole decoded model |
| Memory | Cache-hit load peak no worse than current matched INT8 +16 MiB; target steady <=payload+128 MiB and peak <=payload+192 MiB; RSS and cgroup peak/OOM/reclaim recorded |
| Startup | >=20% lower warm process-to-first-output and >=10% lower fadvise-cold time on both existing models; 3 fresh processes/condition, cache and I/O verified |
| Inference speed | Decode/prefill within 2% of current INT8 or demonstrably indistinguishable in interleaved runs; no new Stage 4 acceptance claim |
| 4B acceptance | Independent trusted-reference/model quality proof plus real short/2k CPU runs under 8 GiB/no-swap, zero OOM and >=1 GiB measured headroom; estimates alone do not qualify support |

### 4B budget and decision

Section 29's real Qwen3-4B / Instruct-2507 configs parse today: hidden 2560,
query width 4096, FFN 9728, 36 layers, 32 query/8 KV heads, head_dim 128.
Their non-square projections and group-4 attention meet current kernel
conditions; group-128 INT8 divides every input dimension. No architecture
expansion is indicated by inspection. **Actual 4B weights/numerics have not
been loaded or accepted.** A reference requiring more RAM must be prepared
on an appropriately sized system, not by bypassing the constrained run.

Estimated payload is 4,148,752,384 B INT8 versus 8,045,328,384 B BF16.
At 2k, FP32 KV is 576 MiB, named full-prefill workspace 312 MiB plus logits,
attention packing 16 MiB, and the separate runtime/safety/growth allowances
128/256/64 MiB. Total is about **5.185 GiB**. An 8 GiB machine with 2 GiB
reserved for OS has about 0.815 GiB additional process-budget slack. This
is a candidate for a real 2k pilot, not an unlimited-context promise. 4k's
estimate is 6.068 GiB, already beyond that 6 GiB allowance. Growth copies
and allocator/platform variation still need measurement.

Existing matched warm INT8 load medians (section 25) are 0.996 / 2.273 s;
fadvise-cold 1.446 / 3.549 s. A direct artifact reads roughly half the used
BF16 payload and avoids conversion, but adds digest/validation cost. At the
historical 2–3 GB/s disk rate, payload I/O alone is approximately 0.21–0.31 s
for 0.6B, 0.59–0.89 s for 1.7B, 1.38–2.07 s for 4B. These are lower-bound
estimates, not startup predictions. Current and proposed steady-state INT8
payloads are identical. Direct loading's expected peak saving is a tensor
transient (typically tens of MB near final load), not gigabytes.

**GO for a narrowly scoped direct-INT8 artifact milestone**, after explicit
format/digest and accounting gates are reviewed. **NO-GO for claiming 4B/8GiB
support now.** Do not build INT4, new tiling/SIMD, FP16 CPU KV, CPU snapshots,
chunked CPU prefill, the full V9 planner, a scheduler/server, offload or new
model architecture as part of it. Growable FP32 KV remains independently GO;
existing Stage 4 timing misses and rejected bounded prefill remain unchanged.

### Final verification and preservation

Final state: fmt, locked workspace/all-target check, all-feature/all-target
clippy `-D warnings`, **594 workspace tests** with MARIN_REQUIRE_CUDA=1 on
RTX 5050 (zero failures/ignores), line-limit and unsafe checks pass.
Forced-scalar release tensor suite: **50 passed**, zero failures/ignores.
Coverage includes BF16/INT8 numerical tests, thread invariance, retained KV,
rollback, CPU/CUDA/model switching, GPU f32/f16 snapshots and corrupted/foreign
files. The existing CPU snapshot safety test was strengthened, not replaced.
No arithmetic/kernel/default changed. No commit, push or stash change.
Prior README changes are byte-preserved, prior benchmark evidence is retained
apart from the explicitly corrected stale overview row. Raw load callbacks,
verified 512 MiB inspection, affinity output, intentional red/green test,
Cargo dependency proof and all check logs are in
`bench-results/consolidation-audit/`. No exhaustive benchmark matrix, new
quantization experiment or 4B download/run. Older/non-Linux hardware and new
loader gains remain unverified. This is a bounded execution/ownership audit,
not a line-by-line proof of every CUDA instruction or terminal render path.


## 32. Direct packed INT8 loading: gates fixed before implementation

The clean starting revision is c99d38a. Contrary to the documentation-only
claim, that commit includes loader dependency cleanup, CPU snapshot hash
selection and its regression assertion. All three stashes are preserved.

This milestone uses an explicitly exported standalone bundle; ordinary HF
BF16 stays default, INT8 requires explicit selection. No inference kernels,
Qwen3.5 implementation or automatic source-directory cache is in scope.

| Area | Mandatory gate (unchanged from section 31) |
|---|---|
| Prerequisites | Verified 512 MiB inspection must report no fit, using cgroup-aware weights + FP32 KV + full-prefill workspace + safety; default threads respect affinity/quota; explicit overrides remain |
| Data | Every INT8 code and FP32 scale/norm bit equals existing quantization; tied embeddings stored once |
| Execution/quality | Bit-identical logits, greedy and seeded continuations on 0.6B/1.7B short/2k, 1/8 threads and scalar/SIMD; existing HF/BF16 thresholds unchanged |
| Compatibility | CPU/CUDA, switching, KV reuse/rollback, snapshots and existing default/opt-in behavior preserved |
| Safety | Reject malformed, foreign, truncated, NaN, group/shape/offset/checksum errors; atomic publication never exposes partial data or overwrites source; concurrent exporters must verify identical published result |
| Ownership | Direct loads read no BF16 payload and do no quantization; one retained codes/scales allocation per tensor; no full decoded model |
| Memory | Matched peak RSS <= old INT8 +16 MiB; steady target <= payload+128 MiB, peak target <= payload+192 MiB; record cgroup/OOM limitations |
| Startup | >=20% lower warm process-to-first-output and >=10% lower fadvise-cold on both models, three fresh processes per condition with verified I/O/cache conditions |
| Inference | Prefill/decode within 2% of old INT8 or demonstrably indistinguishable in interleaved runs |
| Larger models | Memory estimates and Qwen3.5 research are not model support or numerical acceptance |

Record conversion time/peak RSS, bundle size, focused inference/startup A/B
and exact failures below. A failed performance gate remains failed; a
missing measurement remains unverified. No multi-hour matrix.

### Implementation and ownership

`export-int8 SOURCE OUTPUT [--quant-group N]` exports the existing symmetric
INT8 quantizer once. The standalone bundle contains config/tokenizer and
optional generation/tokenizer config, `model.safetensors` and
`marin-int8.json`. Matrices retain original names as I8 row-major signed
codes; `<name>.scales` is FP32 `[rows, cols/group]`. Norms are exact promoted
FP32 values. Tied embeddings have no duplicate head. Untied heads are tested
separately. BF16 HF loading remains default; both CLIs require explicit
INT8 and the original group for packed input. CUDA uploads these same codes
and scales through its existing quantized path; no new inference kernels.

The manifest identifies layout/quantizer revision, architecture, group,
metadata SHA-256, source tensor SHA-256 provenance and each emitted tensor's
SHA-256. Source identities cover converted tensor values, not unused HF
extras or mutable source paths. There is no automatic source-directory cache,
mtime shortcut, BF16 reread or hidden precision fallback. Integrity is checked
as each typed payload is read. Descriptor bounds/types/shapes, duplicate
keys, overlapping/gapped spans, extra payload/names, finite values, signed
code range and group lengths are checked before execution. Metadata files
are verified before tokenizer construction. New bundles reject symlinked
files. Checksums detect corruption; this is not a signed model distribution.

Weights owns paths/metadata only. On direct CPU load, I8/F32 vectors move
into the existing Q8 owner. Read scratch is bounded to 1 MiB; there is no
full encoded tensor clone or BF16 expansion. Norms remain FP32 on CPU/CUDA.
All 310 source tensors on each real model were compared: every emitted code,
scale bit and norm bit equals old on-load quantization. Identical parameter
bits and identical native kernel/state paths establish unchanged quality;
this milestone introduces no new quantization algorithm.

Export holds one source tensor plus its quantized output, hashes/writes it,
and discards both. A unique temporary sibling directory is created
exclusively, files and directory are synced, then renamed and its parent
synced. Existing output folders are refused. Concurrent exporters that both
started before publication verify an identical winner's metadata and every
payload rather than replacing it. Dropped/failed writers remove only their
own temporary directory; stale hidden directories are not accepted as HF
INT8 models. Final review found that a caller catching a short-write error
could previously finish a partial bundle. Its red test reproduced that
failure. A failed write now permanently blocks retry/publication; the green
test and the full checks pass. The original source checkpoint is untouched.

The only new dependency is `sha2` 0.10.9 for a stable digest, not the
compiler-dependent snapshot hash. [RustCrypto API](https://docs.rs/sha2/0.10.9/sha2/)
and [safetensors format](https://github.com/safetensors/safetensors#format)
were checked. Existing snapshot hashing/identity is preserved; a GPU snapshot
from an HF source conservatively rejects the packed artifact identity even
when quantized values agree. CPU snapshots remain unsupported.

### Prerequisites and admission

Inspection now uses cgroup-aware available RAM and the same retained weights,
full FP32 KV window, full-prefill workspace and CPU safety estimate as CPU
admission. It labels the actual weight format instead of calling INT8 BF16.
A verified 536870912-byte `memory.max`, no-swap child reports **does not fit**
for 1.7B INT8 / 8192 context, with zero OOM events. This is a CPU budget
estimate, not a complete CPU/GPU planner. The norm-byte estimate now counts
actual FP32 storage; old accounting undercounted 131072 / 247808 bytes.

Default CPU threads cap host physical cores by
`std::thread::available_parallelism()`. Normal: available 16/default 8;
one-CPU affinity: available 1/default 1; a real `CPUQuota=100%` child:
available 1/default 1. Explicit 8 remains 8 in each restricted child.
Defaults are resolved once per process, not a live scheduler. Workspace,
KV growth/rollback guards and explicit overrides remain intact.
[Resource limit semantics](https://doc.rust-lang.org/std/thread/fn.available_parallelism.html).

### Measurements

All CPU runs were serial, with no competing CPU benchmark. The reference is
the current HF-to-INT8 path on identical files/group/prompt/backend. Old/new
order alternates in fresh startup processes. The initial warm pilot had
some physical reads and was rejected as warm evidence. Controlled warm
measurements evict competing model pages and read the selected checkpoint;
all final warm measured children report zero physical reads. Cold means
per-file `POSIX_FADV_DONTNEED`, not a machine reboot or global cache flush.
Physical reads were recorded; metadata/executable caches can remain warm.
Three fresh processes per format/cache condition; medians below. Prompt:
`The capital of France is`, greedy, one output token, CPU, one thread,
2048 context. All replies equal ` Paris`.


| Model | Warm HF → direct s | Improvement | Fadvise-cold HF → direct s | Improvement | Matched warm peak RSS HF → direct MB |
|---|---:|---:|---:|---:|---:|
| 0.6B | 1.424 → 0.916 | 35.69% | 1.915 → 1.015 | 46.98% | 704.95 → 675.14 |
| 1.7B | 3.109 → 1.984 | 36.17% | 4.145 → 2.232 | 46.17% | 1898.67 → 1835.18 |

| Model | Fadvise-cold export s | Export peak RSS MB | Physical read/write MB | Full bundle MB | Retained native payload MB |
|---|---:|---:|---:|---:|---:|
| 0.6B | 2.716 | 558.86 | 980.87 / 616.17 | 626.49 | 614.87 |
| 1.7B | 6.177 | 1035.06 | 3071.77 / 1777.57 | 1786.33 | 1774.71 |

Cold startup physical reads: approximately 978 → 615 MB (0.6B) and
3069 → 1775 MB (1.7B). These are process I/O counters, not whole checkpoint
sizes or a storage-bandwidth benchmark. HF files also contain unused tied
heads. Direct peak is about payload +60 MB, below both +128 MiB steady and
+192 MiB peak targets; matched peak falls about 30 / 63 MB. This removes a
transient tensor/conversion, **not another half of steady inference RAM**.
The export figures precede the final failed-writer error guard; the emitted
format, successful export result and direct loading path are identical.

Inference pilot: both INT8 models resident only for timing (this is not the
load-RSS experiment). Alternate old/new order; one warm-up pair; five pairs
at 127 tokens / 1 and 8 threads, three pairs at 2047 / 8 threads; sixteen
greedy decode steps per pass. Identical prompts and generated ids throughout.
Existing opening comparison method regards overlapping min/max ranges as
noise. No performance claim is drawn from correctness runs.

| Model/prompt/threads | Prefill HF → direct s | Decode HF → direct tok/s |
|---|---:|---:|
| 0.6B / 127 / 1 | 1.0755 → 1.0726 | 37.555 → 37.560 |
| 0.6B / 127 / 8 | 0.2483 → 0.2458 | 36.164 → 36.515 |
| 0.6B / 2047 / 8 | 4.3828 → 4.3302 | 19.982 → 20.214 |
| 1.7B / 127 / 1 | 3.6713 → 3.6555 | 12.479 → 12.406 |
| 1.7B / 127 / 8 | 0.6349 → 0.6370 | 12.377 → 12.386 |
| 1.7B / 2047 / 8 | 10.3767 → 10.4424 | 9.663 → 9.582 |

One first-pass 1.7B long-prefill pair was +2.88%, so it was not accepted
alone. Repeated interleaved 2k measurements give +0.63% prefill / -0.83%
decode, within gates and overlapping ranges. An initial 0.6B short single-
thread median was +2.007% decode with separate ranges; a targeted ten-pair
confirmation gives +0.014% and overlapping ranges. It replaces that pilot
as acceptance evidence; both raw records remain. No kernel change or
inference speed improvement is claimed.

Real logits and eight seeded continuations are bit-identical at 127/1,
2047/1 and 2047/8 on both models; short forced-scalar real-model runs also
match. Fixture tests cover 513-token reuse/truncation, 1/8 threads, untied
heads, forced scalar panels and GPU upload. Complete bit equality for every
real tensor, unchanged post-load native storage/kernels and those execution
checks establish quality preservation. An exhaustive real-model scalar 2k
Cartesian matrix, 8k timing and older physical CPUs were **not measured**.
The 2047/1 runs are correctness observations, not a repeated speed gate.

A verified 8 GiB/no-swap child runs 1.7B direct INT8, 2047 prompt / two output
tokens / 2049 context / eight threads. Cgroup peak is **2677231616 bytes**;
all memory.max/high/oom/oom_kill events are zero, swap.current is zero. The
benchmark also records actual KV held **469762048 bytes**, versus full-window
budget **469991424 bytes**. This is a process-budget proof on the 16 GB
host, not proof that every 8 GB desktop has the same OS/other-app headroom.
The legacy benchmark `precision` tag still describes baseline BF16 math;
the separate `quantization` field and typed retained byte count describe
INT8. `kv_reserved_bytes` is the window estimate, not proof of upfront CPU
KV reservation. These existing schema names are not changed here.

### Acceptance

| Original gate | Decision | Evidence/limit |
|---|---|---|
| RAM inspection / thread restriction | PASS | Verified 512 MiB, one-CPU affinity and one-CPU quota; explicit overrides retained |
| Code/scale/norm/tied storage | PASS | All 310 real tensors per model bit-exact; untied-head regression test |
| Correctness / quality | PASS | Complete parameter equivalence with unchanged kernels; real short/2k and seeded checks, scalar fixture/short real checks; no new quality approximation |
| Artifact safety / atomic publish | PASS | Corruption/version/group/metadata/NaN/negative/code/shape/overlap/duplicate/truncation, interrupted/concurrent publish and caught-write-error red/green coverage |
| BF16/HF/defaults / CUDA / snapshots | PASS | Full CUDA-required suite and new packed CLI/GPU/snapshot regressions; strict reference thresholds unchanged |
| Ownership | PASS | Direct typed I8/F32 path into existing Q8; standalone bundles contain no BF16 tensors or source path |
| Memory | PASS | Lower matched RSS, targets met; 8 GiB/no-swap 2k zero OOM; no full-model BF16 mirror |
| Startup | PASS | Warm -35.69/-36.17%, fadvise-cold -46.98/-46.17%; three processes/condition, I/O recorded |
| Inference performance | PASS | Focused final interleaved pilot within 2% or overlapping ranges; initial misses preserved and targeted confirmation explained |
| 4B / Qwen3.5 support | OUT OF SCOPE | Architecture research and estimates below only, no new model implementation or real weight inference |

**GO for direct packed INT8 loading on the validated Linux/RTX 5050 setup.**
BF16 remains default, INT8 opt-in. No new CPU snapshot, INT4, inference
kernel, chunked prefill, planner, scheduler or streaming feature. Non-Linux
directory-sync/publication and older GPUs/CPUs remain unverified. Checksums
are paid during reading; pre-SHA-instruction CPU startup needs its own
measurement before claiming the same startup gains.

Growable FP32 KV remains independently GO. Bounded CPU prefill remains
removed/NO-GO. Section 28's +10.48% TTFT and -3.28% decode misses remain
unresolved; no Stage 4 gate was relaxed or closed by this loader milestone.

Final checks: fmt, locked workspace/all-target check, all-feature/all-target
clippy `-D warnings`, **603 workspace tests** with MARIN_REQUIRE_CUDA=1,
**52 forced-scalar tensor tests**, **5 forced-scalar packed-model tests**,
line limit and unsafe guards pass (zero failures/ignores). The full suite
was rerun after the failed-writer regression fix. No commits, pushes or
stash changes; original three stash hashes remain unchanged. Raw inputs,
startup/conversion/cgroup/timing/parity data, research headers, intentional
red/green logs and checks live in `bench-results/direct-int8/`.

Additional production-model sanity checks on the final build: strict CPU HF
verification passes on both original BF16 models (three prompts, sixteen
steps each, all greedy choices matching). Maximum logit/layer differences:
- 0.6B: 1.91e-05 / 6.28e-07.
- 1.7B: 1.91e-05 / 5.21e-07.

Fresh RTX 5050 processes also generate identical four-token greedy replies
from HF-on-load INT8 and packed INT8 on each real model. These supplement
the full CUDA/snapshot/switching regressions, not GPU performance gates.


## 33. Qwen3.5 text-only architecture plan (research; no implementation)

Qwen3.5-2B is the first architecture target, then 4B. This study fetched
**configs and safetensors headers only**, not the weight payloads; no new
model was loaded or benchmarked. Pin the HF revisions
`15852e8c16360a2fea060d615a32b45270f8a8fc` / `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and Transformers
`db2304e8694db3ecd6f39d353beba361688afe97`. The general Qwen repository now redirects to a
newer family; the specific Qwen3.5 configs, tensor shapes and implementation
are the basis here. Dense 2B/4B are in scope; flagship MoE claims are not.

[Official 2B config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json),
[official 4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json),
[HF model implementation](https://github.com/huggingface/transformers/blob/db2304e8694db3ecd6f39d353beba361688afe97/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
[HF cache implementation](https://github.com/huggingface/transformers/blob/db2304e8694db3ecd6f39d353beba361688afe97/src/transformers/cache_utils.py),
[Gated Delta Networks paper](https://arxiv.org/abs/2412.06464).

### Architecture and state

Both alternate three DeltaNet blocks with one full-attention block: 18+6
layers on 2B, 24+8 on 4B. Hidden/MLP widths are 2048/6144 and 2560/9216.
Vocabulary is 248320 with tied embeddings; full-attention head size is 256,
query/KV heads 8/2 and 16/4. Rotary dimension is only 64. Ordinary RMSNorm
uses `1 + weight`; gated recurrent output norm uses its weight directly.
Full attention has a sigmoid output gate and a doubled query projection, with query/gate split per head.
These details prevent treating Qwen3.5 as a Qwen3 config rename.

DeltaNet projects Q/K/V, z, a and b; a depthwise length-4 causal convolution
precedes Q/K/V splitting. Q/K use L2 normalization; beta is sigmoid(b),
decay is exp(-exp(A_log)*softplus(a+dt_bias)). The state decays, receives a
beta-scaled key/value residual update, and produces the query readout;
SiLU(z) gates its normalized output. Recurrent head dimensions are 128×128,
with 16 value heads on 2B, 32 on 4B and 16 key heads on both. Inspect the
pinned implementation for exact casts and gate ordering before coding.

Use separate **FP32 recurrent matrices**, **convolution history** and
**ordinary attention KV**. A sequence can contain these separate owners
with shared position bookkeeping; none should pretend recurrent state is
paged, per-position KV. Recurrent state is constant-size, but it cannot be
restored to an arbitrary earlier prefix by lowering a length counter. HF
also marks initialized recurrent caches non-croppable. Do not retain an
entire 20/50 MiB state at every token. Start with transactional forward
commit and one before-pass checkpoint; define reset/replay behavior before
connecting arbitrary prefix reuse. Speculation and snapshots remain disabled
for the new architecture until their state contracts are independently
validated; existing Qwen3 behavior is preserved.

### Reuse and minimum additions

Reuse ModelDir/tokenizer, the I8/F32 artifact codec, checked Q8 construction,
CPU SIMD/scalar GEMM/GEMV, MLP/SiLU, sampler, metrics, thread selection,
Linux memory guards and full-attention KV owners. RoPE math needs a separate
rotary dimension/actual head stride; the old full-head call is insufficient.
RMSNorm can reuse its primitive with explicitly prepared effective gamma.
CPU attention already has general dimensions. CUDA accepts head_dim 256,
but its head-128 tensor-core prefill fast path will not apply; first validate
the existing general/split kernels before proposing any new attention kernel.

Minimum new work: a specific nested text-config parser and expected tensor
map; typed reads of existing HF FP32 side tensors (current `read_f32` is
artifact-only); safe scalar depthwise convolution/history update, FP32
DeltaNet recurrence and gate/L2/partial-RoPE handling; a separate hybrid
state owner with checked dimensions and transactional errors. Add explicit
architecture dispatch only when this model exists. No generic graph,
device-tensor framework, all-family registry or planner is required.

Keep I8 row layout/group/quantizer stable. A future explicit architecture
profile/version may allow FP32 convolution/side tensors and text namespaces;
v1 readers must reject it. Do not rewrite stored norms into a different
convention without versioning that policy. The current v1 writer remains
Qwen3-only. No Qwen3.5, FP32 matrix kernel or state implementation was added.

### Header-based memory budgets (estimates, not measured support)

All main text matrix input widths divide group 128. There are 320/426 text
tensors; 297 vision and 15 MTP tensors in each checkpoint are excluded,
verified against actual header names. There is no separate stored LM head.
The proposed policy quantizes every 2D text weight, preserving all other
weights as FP32. This includes FP32 A_log/dt_bias/norms and conservative FP32
convolution weights. It is a memory estimate, not proof this all-matrix INT8
policy is accurate enough. Retaining sensitive a/b matrices in FP32 would
add approximately 3.5/11.7 MB; choose that only after quality evidence.

| Model | Used text source bytes | Proposed native INT8 bytes | FP32 side weights within payload | Recurrent + convolution state | Full-attention FP32 KV at 2k |
|---|---:|---:|---:|---:|---:|
| 2B | 3,763,655,360 | 1,942,260,992 | 2,194,688 | 20,643,840 | 50,331,648 |
| 4B | 8,411,510,272 | 4,340,035,584 | 3,846,144 | 53,477,376 | 134,217,728 |

Persistent FP32 state estimate per linear layer is
`4 * [value_heads*128*128 + (2*key_heads*128 + value_heads*128)*4]` bytes.
KV estimate is `2*full_layers*kv_heads*256*context*4`. Conv state is budgeted
at FP32 even if a future proven implementation keeps it at BF16.

Workspace is a deliberately conservative O(context) live-buffer bound:
`4*T*(6*H + 3*FFN + 2*Kdim + 3*Vdim + 2*full_query_heads*256) + 4*vocab`.
It includes hidden/residual and projection/gating panels, rather than assuming
constant-memory prefill. It does not allocate a T² attention score matrix.
Measured implementation liveness may be smaller; no implementation exists
to prove that yet. Add FP32 attention packing `8*T*256`, runtime 256 MiB,
safety 256 MiB, a whole extra KV window for growth copies and one recurrent/
conv before-pass checkpoint. This replaces the earlier illustrative 64 MiB
growth allowance in the raw first estimate; it is more conservative.

| Model/context | Workspace MiB | Attention KV MiB | Estimated total process GiB incl. all allowances |
|---|---:|---:|---:|
| 2B / 128 | 22.9 | 3.0 | 2.376 |
| 2B / 2048 | 352.9 | 48.0 | 2.790 |
| 2B / 4096 | 704.9 | 96.0 | 3.231 |
| 2B / 8192 | 1408.9 | 192.0 | 4.114 |
| 4B / 128 | 33.9 | 8.0 | 4.691 |
| 4B / 2048 | 528.9 | 128.0 | 5.412 |
| 4B / 4096 | 1056.9 | 256.0 | 6.182 |
| 4B / 8192 | 2112.9 | 512.0 | 7.721 |

On 8 GiB hardware with 2 GiB reserved for OS/other apps, 2B/2k has useful
estimated room. 4B/2k is a candidate, but the conservative 5.412 GiB estimate
leaves only 0.588 GiB beyond the 6 GiB process allowance: it **does not yet
establish the 1 GiB measured headroom gate**. 4B/4k exceeds that allowance.
On 4 GiB, 4B weights alone exceed RAM; NO-GO. 2B short/2k estimates fit only
with a lean OS budget and remain tight, so no general 4 GiB support claim.
Tokenizer peaks, allocator behavior, growth overlap and transactional state
copies need real constrained-process measurements.

The measured Qwen3 INT8 short-decode weight-payload×tok/s proxy is roughly
22 GB/s; it is not an IMC counter or a measurement of DeltaNet bandwidth.
Using 19–22 GB/s and one weight stream plus approximately four FP32 state
passes gives optimistic short-context ceilings around 9–11 tok/s (2B) and
4–5 tok/s (4B). These are **bandwidth ceilings, not performance forecasts**;
KV rereads, recurrence arithmetic, exp/softplus, CPU ISA and serialized state
updates only reduce them. Recurrent KV savings do not make 4B cheap to read.
No CPU/GPU TTFT or throughput was measured for these models.

### Smallest next milestone and proposed gates

1. Pin a trusted HF FP32 eager text-only reference, promoted from the exact
   checkpoint values, with TF32 disabled. Generate intermediates and state
   after every token. Build a tiny four-layer hybrid fixture (three recurrent,
   one full attention), plus real 2B reference samples. Reference generation
   must use a machine that fits it; no bypass of RAM guards.
2. Implement only the CPU reference block/state first. Compare convolution
   history, normalized Q/K, beta/decay, recurrent matrix, gated output and
   logits at 1/2/4/16/128 tokens. Before optimization require normalized
   maximum absolute error <=1e-4, relative RMS <=1e-5 and exact greedy
   continuations on the fixed reference prompts. Empty/full/error cases,
   whole-prefill versus incremental execution, poisoned input and failed
   transactions must preserve both attention and recurrent state correctly.
   Long-prefix equivalence is a numerical tolerance comparison, not a demand
   for bit identity between different accumulation algorithms.
3. Validate full 2B text execution and source loading before wiring a
   versioned INT8 artifact profile. Fix quantization gates before conversion:
   >=95% teacher-forced top-1 agreement, perplexity increase <=3%, no extra
   executable-code failures versus BF16 on identical fixed prompts. Native
   artifact values must equal its approved exporter bit for bit.
4. CPU INT8 pilot at short/2k and 1/8 threads: require >=1.25x BF16 decode,
   <=10% TTFT regression, retained weights <=60% BF16, zero OOM/no swap in
   8 GiB and >=1 GiB measured effective headroom. Test scalar behavior,
   affinity/quota defaults and new state failure/replay semantics. Preserve
   all existing Qwen3 gates. Do not claim 4 GiB support from an 8 GiB test.
5. Only then implement the minimal CUDA recurrent/conv operations behind
   safe checked APIs. Require the same trusted numerical gates with f32
   state, no transfer/allocation per decode step after warm-up, no Qwen3
   regressions, matched prefill/decode no worse than its accepted general
   reference by >5%. Compare with official HF on RTX 5050 as an external
   performance reference; report gaps rather than claiming parity untested.
6. Finally run real 4B text weights at short/2k under 8 GiB/no swap with the
   same numerical, quality and headroom gates. Revisit measured scratch
   liveness if it does not fit; do not resurrect chunked prefill, lower KV
   precision, INT4, offload or a planner to make an estimate look accepted.

**GO to the next narrowly scoped CPU FP32 hybrid-block/state parity milestone
on Qwen3.5-2B. NO-GO for claiming Qwen3.5 support today or 4B/8 GiB acceptance
from these budgets.** Native packed INT8 loading is complete first. No
Qwen3.5 implementation, SIMD kernel, MoE, SSD streaming, scheduler or hardware
planner is part of this change.


## 34. Qwen3.5-2B-Base CPU FP32 hybrid reference

### Gates fixed before implementation (2026-10-13)

First milestone is standalone CPU hybrid decoder blocks and explicit state,
not CLI/model support. Existing Qwen3 execution is untouched. Compare
deterministic FP32 inputs and weights against official HF CPU eager math,
including HF chunked DeltaNet prefill and recurrent decode. No optimized
DeltaNet or quantized Qwen3.5 path.

| Area | Fixed acceptance gate |
|---|---|
| Numerical | For **each** named intermediate/output/state: max absolute error / max(1, max absolute reference) <=1e-4; RMS error / max(1e-6, RMS reference) <=1e-5; all values finite. These define section 33 tolerances, not weaker thresholds. |
| Coverage | Linear + full-attention decoder blocks (norms, projections, gates, residual and MLP), sequence lengths 1/2/4/16/65/128, nonzero initial state, convolution boundaries, full-attention GQA and partial RoPE; HF chunked vs Rust recurrent equivalence. |
| State | Zero initialization, reset/replay, repeated decode, whole prefill vs split passes vs tokenwise, independent sessions; input/shape/context/memory errors leave committed state unchanged. Recurrent state is separate from KV and has no arbitrary truncation API. |
| Safety/memory | Checked sizes, fallible allocations, explicit byte budget for state/workspace/transaction copies before allocation; reject undersized budgets, malformed weights and nonfinite input; no partial state commit. |
| Compatibility | Existing full workspace tests including required CUDA, forced-scalar reference checks, fmt/check/clippy -D warnings, line limit and unsafe checks pass; Qwen3/defaults/packed loader/snapshots unchanged. |
| Real model (later in this session only if ready) | Actual Base text weights; strict HF logit/intermediate thresholds above and exact greedy tokens on fixed prompts; report peak RSS/TTFT/decode separately from estimates; refuse inadequate memory. Missing measurements mean unverified, not GO. |

Source config/header captured **before coding** under
`bench-results/qwen35-base-parity/`: Base revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. Scope reaches only a new
model reference module, its fixtures/tests, public export and documentation;
no existing CLI/backend dispatch or runtime cache contract is extended.

### Actual Base architecture and reference scope

Sources are the [pinned Base config](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/config.json),
[pinned tensor index](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/model.safetensors.index.json),
the remotely range-read safetensors header, and
[official Transformers v5.17.0 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The installed reference source is byte-identical to the official release:
SHA-256 `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
PyTorch 2.13.0+cu130, Transformers 5.17.0, CPU eager, one thread, no FLA or
causal-conv extension. This study uses **Base**, not the chat model from
section 33. No full checkpoint payload was downloaded.

The Base model is dense, tied embeddings, vocabulary 248320, hidden 2048,
FFN 6144, epsilon 1e-6, max positions 262144. Its 24 layers are exactly
`(linear, linear, linear, full) * 6`; ordinary norms use **1 + weight**,
while the DeltaNet gated norm uses weight directly. Every block has
pre-norm/mixer/residual, post-norm/SiLU-gated MLP/residual. Text tensor
names are relative to `model.language_model.`; each block reference accepts
exact names relative to `layers.N.`.

| Block tensor / state | Actual Base shape / semantics |
|---|---|
| Delta `in_proj_qkv.weight` | [6144,2048], ordered Q=2048, K=2048, V=2048 |
| Delta `in_proj_z.weight`, `out_proj.weight` | [2048,2048] each |
| Delta `in_proj_a/b.weight` | [16,2048] each |
| Delta `conv1d.weight` | [6144,1,4], biasless depthwise causal convolution then SiLU |
| Delta `A_log`, `dt_bias`, `norm.weight` | [16], [16], [128]; A_log and gated norm are F32 on disk, dt_bias is BF16 |
| Delta persistent state | convolution [6144,4]; recurrent [16,128,128], FP32, separate owners from attention KV |
| Full `q_proj.weight` | [4096,2048]; query/gate split **within each** of 8 heads, 256 values each |
| Full `k/v_proj.weight`, `o_proj.weight` | [512,2048] each; [2048,2048] |
| Full `q/k_norm.weight` | [256], zero-centered RMSNorm |
| Full attention | 8 query / 2 KV heads, head dimension 256, causal GQA, output multiplied by sigmoid(query gate) before output projection |
| Text RoPE | theta 1e7; rotate first 64 of 256 values; text's three position axes coincide, so no vision mRoPE routing is needed |
| MLP | gate/up [6144,2048], down [2048,6144]; SiLU(gate) * up |

Q/K are L2-normalized with epsilon 1e-6; queries additionally divide by
sqrt(key head dimension). Beta is sigmoid(b), decay log is
`g = -exp(A_log) * softplus(a + dt_bias)`. For each head/token: decay
S by exp(g), compute its key prediction, multiply the residual value by
beta, add the key/delta outer product, then read updated S with Q.
Gated RMSNorm applies per value head before SiLU(z) and output projection.
The reference also handles repeated Q/K heads when value-head count is a
multiple of key-head count; 2B itself has equal counts.

`model::qwen35::ReferenceBlock` is deliberately **not** a `CausalLm`
implementation. It owns genuine FP32 matrices and reuses Marin dot, RMSNorm,
SiLU, attention, partial-prefix RoPE and ordinary growable KV. Its local
state owner explicitly distinguishes recurrent/convolution buffers and a
one-layer attention KV cache. No changes to Qwen3 modules, generation,
packed artifacts, CLI selection, CUDA or snapshot contracts were needed.
The only existing-code extension is the reference module export.

Each block pass validates input, compatibility, context and additional
byte budget before allocation. The budget covers a fallibly allocated
transaction copy, returned output, conservative token workspace, KV window
slack and 1 KiB allocation bookkeeping; cgroup-aware admission additionally
retains the existing 256 MiB CPU margin. Already-owned weights/state are
excluded from the *additional* budget and included in current available
RAM. Caller must admit weight memory before constructing a reference block.
No model-loading budget or full-model fit claim is made by this API.
Commit occurs only after the entire block pass succeeds. Reset zeros Delta
state or clears KV. There is no recurrent arbitrary-prefix truncate API.
This is a correctness reference: full KV and recurrent state are copied
on each pass for atomicity, and tokens are scanned serially. Production
performance and multi-layer transaction ownership require later evidence.

### HF fixtures and measured numerical evidence

`tests/reference/qwen35/generate.py` executes actual HF decoder modules,
not an independent Python reimplementation of Rust. It records projections,
convolution/QKV, gates/decays, norm/RoPE, core attention, gated output,
residuals, MLP stages and final state. Checked-in JSON descriptors plus
`values.f32` retain exact little-endian FP32 bits (about 3.4 MiB payload),
with reference source/payload SHA-256 provenance. No BF16 rounding of
fixture weights and no regenerated expectations from Rust output.

**34 cases** cover linear/full blocks; lengths 1/2/4/16/65/128; nonzero
prefix state; direct HF recurrent single-token cached decode; HF chunked
prefill across its 64-token boundary; 2:1 value/key heads; odd key dimension
and real 128-value/key / 256-attention head dimensions with reduced test
head counts/hidden width. A separate 65-token sequence through three
DeltaNet blocks and one full-attention block matches HF layer outputs.
These are small deterministic fixtures, **not actual 2B weight execution**.

| Execution | Worst normalized max intermediate error | Worst relative RMS intermediate error | Gate |
|---|---:|---:|---|
| Default dispatch | 1.91079e-6 | 7.04187e-6 | <=1e-4 / <=1e-5: PASS |
| Forced scalar (`MARIN_MATMUL_SCALAR=1`) | 1.91079e-6 | 5.57544e-6 | same gates: PASS |

Every convolution/recurrent/KV final-state tensor separately passes these
same tolerances. Returned outputs match the recorded intermediates. Rust
whole-pass vs multi-token continuation + decode vs tokenwise results and
final state are **bit-identical**. Reset/replay is identical; a second
session stays unchanged while the first advances. Tests reject malformed,
missing/extra/nonfinite weights, invalid dimensions/RoPE, ragged/empty/NaN
input, wrong state kind, unsupported activation/bias/ungated/RoPE settings,
overflowing sizes, zero Q/K/input, full context and insufficient
budget. An overflow in MLP after the mixer has actually updated staged
state leaves committed state unchanged. No tolerance was relaxed.

Reproduce fixture/reference checks with:

```bash
OMP_NUM_THREADS=1 /home/mitali/miracles/ralph/.venv/bin/python tests/reference/qwen35/generate.py
cargo test -p model --test qwen35 --locked -- --nocapture
MARIN_MATMUL_SCALAR=1 cargo test -p model --test qwen35 --locked -- --nocapture
```

### Real-model memory: header arithmetic, not runtime measurements

The actual Base header contains **320 text tensors**, 297 vision tensors
and 15 MTP tensors. Text weights have 1,881,825,088 elements and
3,763,655,360 source bytes. Promoting all text weights to FP32 would retain
**7,527,300,352 bytes (7.010 GiB)**; tied embedding/head must be stored once.
Persistent FP32 convolution/recurrent state is **20,643,840 bytes**.
Full-attention FP32 KV is **24,576 bytes per token** across six layers:
50,331,648 bytes at 2k and 201,326,592 at 8k, before allocation slack.
Workspace, transactional state/KV copies, loader transients, runtime and
safety must be added. These figures are distinct from section 33's future
INT8 estimates; no Qwen3.5 quantizer exists.

Full 2B loading/logits, peak RSS, TTFT and decode **remain unverified**.
An apparent raw-weight fit is not admission, and the 8 GiB profile has
limited headroom for a fully FP32 baseline or naive conversion transients.
No memory limit was bypassed. This session stops at the validated block
foundation instead of adding an incomplete full-model backend.

### Remaining implementation before full-model acceptance

1. Add strict text-backbone config/tensor validation and source F32 reads
   (existing source reader supports BF16; its current F32 read is packed-
   artifact-specific). Load only the 320 text tensors, skip vision/MTP,
   and preserve tied embeddings. Budget retained FP32 weights and the
   largest source/promoted tensor transient **before** allocation.
2. Connect embeddings, the exact 24-block schedule, final zero-centered
   norm and tied head into a text reference pass. Own recurrent and KV
   state explicitly, with one **whole-model** atomic commit: per-block
   atomicity here does not protect earlier layers if a later layer fails.
3. Compare real Base HF intermediates/logits and greedy sequences on short
   prose/code prompts, fresh prefill and cached decode, with unchanged
   tolerances. Measure peak RSS, load cost, TTFT and decode serially; test
   inadequate budgets without attempting an unsafe load.

`CausalLm::Cache` currently requires `SequenceCache::truncate`, and prefix
reuse assumes arbitrary retained-prefix rollback. DeltaNet cannot satisfy
that by cropping attention KV. Keep the initial text reference separate
from this contract; later runtime integration must explicitly choose
checkpoint/replay or fresh-run/reset semantics, rather than claim reusable
KV-only state. No scheduler or generic state framework is required now.
No CUDA DeltaNet, Qwen3.5 INT8, vision, MoE, optimization or 4B expansion
should precede real 2B FP32 parity.

A separate compiled probe verifies actual RAM admission on the final
implementation: both state types are admitted normally; a fresh systemd
child with verified `memory.max=134217728` and `memory.swap.max=0` rejects
both with `CpuMemory` before state allocation. Required additional bytes
including the margin are 268436532 (Delta) / 268436480 (full); child
available bytes are 133464064. Child memory peak is 2359296 bytes, with
**zero max/OOM/OOM-kill events**. This probes rejection of the existing
256 MiB safety floor, not real 2B memory usage. Explicit per-call budget
boundary tests also pass, including a zero-byte initial-state budget.
Probe source/output are under `bench-results/qwen35-base-parity/`.

### Final acceptance

| Gate | Result | Evidence |
|---|---|---|
| Actual Base architecture audit | PASS | Pinned config/index/header; installed official HF source verified byte-for-byte; text/vision/MTP and dtype/layout distinctions recorded above |
| Hybrid block numerical parity | PASS | 34 HF cases + four-block hybrid stack; every intermediate and final state passes unchanged 1e-4 normalized-max / 1e-5 relative-RMS gates |
| State lifecycle and equivalence | PASS | Prefill/continuation/repeated decode/tokenwise, zero inputs, reset/replay, session isolation and errors after staged state mutation; exact Rust split-pass equality |
| Reference memory admission | PASS | Checked/fallible major allocations, explicit budget boundary, overflow/context rejection; verified 128 MiB/no-swap refusal with zero OOM |
| Qwen3/packed/CUDA compatibility | PASS | Final full workspace **609 tests / 70 suites**, zero failures/ignores, `MARIN_REQUIRE_CUDA=1` on RTX 5050; legacy BF16/INT8, threading, retained KV, rollback, switching and GPU snapshot fixtures pass |
| Scalar and optimized-build reference | PASS | Six hybrid tests each in release/default and release/forced-scalar; scalar tensor suite **52 tests**, zero failures/ignores; numerical maxima unchanged from debug |
| Source checks | PASS | fmt, workspace/all-targets check, workspace/all-targets/all-features clippy -D warnings, line-limit, unsafe and diff-whitespace checks |
| Full real 2B weights/logits/greedy | UNVERIFIED | Text loader, full 24-layer execution and model-level transaction still require the next coherent increment |
| Real 2B peak RSS/TTFT/decode | UNVERIFIED | No full model run; header-derived FP32/INT8 figures must not be substituted for measurements |

**GO: standalone CPU FP32 hybrid-block/recurrent-state correctness
foundation. NO-GO: full Qwen3.5-2B-Base inference/production acceptance,
pending real-model integration and its original gates.** This is the
user-authorized stopping point at a fully validated block/state milestone.
No numerical tolerance, Qwen3 default, packed-loader format or existing
production cache/backend behavior changed. No CUDA DeltaNet, Qwen3.5 INT8,
vision, MoE, scheduler or optimized recurrent kernels were implemented.

The final suite was rerun after tightening the initial-state bookkeeping
budget and rejecting unsupported activation/bias/gate/RoPE settings. The
first full run also passed; final evidence above uses the final sources.
Raw references, revisions, header arithmetic, cgroup probe and all check
logs are in `bench-results/qwen35-base-parity/`. Existing uncommitted
implementation files are byte-preserved except the one new model-module
export; README is updated and this section appended. HEAD and all three
stash hashes are unchanged. No commit, push, stash manipulation, discarded
work or remaining benchmark/background job.


## 35. End-to-end Qwen3.5-2B text inference

### Gates fixed before implementation

Official Qwen3.5-2B-Base, revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c,
CPU FP32 math with exact source BF16 matrices and source FP32 auxiliary
tensors. Existing BF16 SIMD matmul is reused; no Qwen3.5 quantization.

| Area | Acceptance gate |
|---|---|
| Real execution | Official weights, full 24-layer hybrid text backbone and tied output head generate text through existing CLI; no vision/MTP execution |
| Numerical | Every full layer output and full logit vector: normalized max absolute <=1e-4 and relative RMS <=1e-5, as defined in section 34; exact greedy ids for at least two prose/code prompts, eight steps each; fresh and cached HF FP32 math |
| State | Whole-model error rollback, reset, independent sessions, incremental vs fresh execution; retained prefixes remain mathematically correct (reset/replay allowed), no fake KV-only truncation |
| Memory | Pre-load retained weights + largest transient + state + workspace + 256 MiB safety admission; per-pass full workspace/transaction/KV growth checked; reject inadequate actual cgroup limit without OOM |
| CLI | Existing marin-run completion/verify/bench and normal Marin load dispatch; CPU-only explicit rejection of CUDA/quantization/FP16 for Qwen3.5; Qwen3 defaults unchanged |
| Measurements | Actual load/TTFT/prefill/decode and peak RSS on two short prompts; serialized CPU runs; report bottlenecks and measurement limits, no long-context sweep |
| Compatibility | Full workspace CUDA-required tests, scalar regression, fmt/check/clippy -D warnings, line-limit and unsafe checks pass; packed loading, Qwen3 threading/cache/snapshots preserved |

No optimization acceptance claim; correctness and usable real inference are
the gates. Gates are not weakened following numerical failures.

### Complete execution and ownership

The official **Qwen3.5-2B-Base** checkpoint is now executable by `marin-run`.
All 24 configured layers, embedding lookup, final zero-centered RMSNorm,
tied output head, tokenizer, sampler and prefill/decode run through Marin.
This supersedes section 34's full-model NO-GO. No vision or MTP tensor is
loaded into execution. The original checkpoint remains intact on disk.

`TextConfig` validates the dense text layout, dimensions, SiLU/default
partial RoPE, biasless/gated attention, layer list, context/vocabulary and
tied head. Per-tensor names, exact dimensions and BF16/F32 dtypes are
validated against the real safetensors metadata. Extra/missing text tensors
are rejected. Source FP32 reads now use bounded chunks and reject nonfinite
values; native packed reads retain their checksum/format validation.

Real matrices retain their **original BF16 values** in existing CPU
`Matrix` storage, with SIMD/scalar GEMV and FP32 math. The existing FP32
fixture path remains unchanged. FP32 auxiliaries stay FP32; BF16 norms,
conv weights and small auxiliaries promote exactly. The source contains
320 text tensors, 297 vision and 15 MTP tensors; only text is executed.
Retained text parameters are 1,881,825,088; weights use **3,764,747,520
bytes**, including promoted auxiliary tensors. There is no full FP32
model copy, no Qwen3.5 quantization, and tied embeddings are owned once.

The model-specific cache owns 24 per-layer states and a token ledger.
Appending tokens reuses both recurrent/convolution state and ordinary
attention KV. A pass stages state **once for the whole model**, executes
blocks in place, checks logits, then commits. This removes the block-level
second copy. A last-layer error leaves every earlier layer, KV and token
ledger unchanged. Independent models have explicit cache-owner checks;
independent sessions share only immutable weights. Reset zeros recurrence
and clears KV. Prefix truncation retains token ids and resets state; the
next pass replays that prefix before the suffix. This is correct replay,
not an impossible KV-only recurrent rollback. Edited-prefix reuse counters
reflect logical retained ids, not saved computation. No state framework,
scheduler or checkpoint-per-token history was introduced.

CLI dispatch adds one concrete CPU model/cache variant. Qwen3 CPU/CUDA and
packed INT8 paths retain their original execution. Auto selects CPU for
Qwen3.5; explicit CUDA, quantization and lower KV precision are rejected.
Normal `marin` loading is regression-tested using a tiny ChatML hybrid
fixture. **Base models still use `marin-run`; `marin`'s existing base-model
chat refusal is preserved.** The actual instruct/chat checkpoint has not
been validated in this milestone.

### Real HF parity and diagnosed failures

Checkpoint revision and official source are pinned in section 34.
`tests/reference/qwen35/full.py` loads the exact text tensors into the
official Transformers 5.17.0 `Qwen3_5ForCausalLM`, uses exact BF16 source
values with FP32 embedding/linear/conv math, CPU eager attention, eight
threads and no FLA/causal-conv extension. Tied weights are shared; loading
checks all expected text parameters. This is **HF FP32 math parity**, not
a comparison with default HF BF16 arithmetic.

The first reference attempt exposed HF buffers left on the meta device by
meta construction + `assign=True`. Cached/fresh output differed at the
first full-attention layer while all three preceding DeltaNet layers
agreed; its RoPE cosine values varied between identical calls. Recreating
the official rotary module on CPU and asserting that no buffer remains
meta fixes the harness. No model weight, inference equation or tolerance
changed. A stale pre-build binary also produced a norm-only verification
failure against the new full-tensor reference; final rebuilt verification
below checks full tensors. A shared-budget helper compile typo was fixed
and all checks rerun; none of these failures was suppressed.

The verifier now requires **all embedding/layer output values and every
logit**, rather than allowing top-5-only checks for the hybrid model. It
identifies the first incorrect layer and rejects missing/nonfinite/malformed
full references. Qwen3's existing verification gates remain unchanged.

| Real prompt / execution | Greedy steps | Max absolute logit difference | Max normalized tensor error | Max relative tensor RMS |
|---|---:|---:|---:|---:|
| France, default dispatch | 8/8 exact | 2.10e-5 | 1.194e-6 | 1.993e-6 |
| Fibonacci, default dispatch | 8/8 exact | 1.53e-5 | 1.309e-6 | 1.480e-6 |
| France, forced scalar | 8/8 exact | 3.05e-5 | 2.074e-6 | 3.025e-6 |
| Fibonacci, forced scalar | 8/8 exact | 2.15e-5 | 3.869e-6 | 2.271e-6 |

All 24 layer outputs at every prompt position, initial embeddings and full
248320-element logits meet the original **1e-4 normalized maximum /
1e-5 relative RMS** gates. HF cached-vs-fresh logits and greedy choices
are independently checked at each step with those same gates. Whole-model
fixture tests additionally prove exact native incremental/fresh equality,
reset/replay, edited prefixes, session isolation, foreign-cache rejection,
context/token errors and last-layer failure rollback. Existing 34 hybrid
fixture cases remain passing.

### Actual short-prompt performance and RAM

Ryzen 7 260, Linux, CPU, eight requested threads, context 128, greedy eight
output tokens. Three fresh processes per prompt, serial alternating prose
and code, no competing CPU benchmark. These are ordinary file-cache runs
without controlled eviction; **not a cold-start speedup claim**. Peak RSS
is collected with `wait4`, and load/TTFT/throughput from CLI metrics.
The final shared admission helper only unifies the pre-existing load
estimate with inspection; matrix/math/state execution is unchanged.

| Prompt | Tokens | Model load s | TTFT s | Prefill tok/s | Decode tok/s | Peak RSS GB |
|---|---:|---:|---:|---:|---:|---:|
| prose | 5 | 3.18 | 0.6646 | 7.5 | 6.07 | 3.898 |
| code | 4 | 3.23 | 0.5451 | 7.3 | 5.92 | 3.898 |

Peak RSS is about **3.90 GB / 3.63 GiB**, distinct from section 34's
7.01 GiB fully promoted FP32 weight estimate. Fixed FP32 recurrent/conv
state is 20,643,840 bytes; six full-attention caches add 24,576 bytes per
active token. At context 128, physical state storage holds 23,789,568 bytes.
Telemetry/inspection label hybrid state explicitly and report actual text
parameters, rather than using Qwen3's KV/parameter formulas.

Admission retains the existing 256 MiB safety floor. Before loading it
checks retained weights + conservative largest-tensor transient + initial
state/workspace; inspection uses the greater of that peak and the full
context execution budget. The short-window conservative budget is
**5,072,758,024 bytes**, above measured RSS; it must not be called actual
RAM usage. Per-pass checks include staged state, KV growth, full input/
output workspace and ledger allocation. Fallible major allocations are
retained. No memory cap was bypassed.

Verified systemd children: `memory.max=536870912` / swap 0 rejects before
weights are read, with zero OOM/OOM-kill events. `memory.max=8589934592` /
swap 0 loads the real model and generates the exact France continuation,
with zero OOM events. These short, cached-file 8 GiB checks do not prove
4 GiB fit or cold-page-cache peak behavior.

Decode streams BF16 matrices; effective weight traffic inferred from
retained bytes * observed decode rate is about 22 GB/s, **not a hardware
counter measurement**. Prefill currently scans tokens through GEMV inside
each block, rather than batching projections into existing GEMM. This
explains its low prefill rate and means extra requested CPU threads do not
currently accelerate this Qwen3.5 path. Qwen3 multithreaded prefill is
preserved. Loader parsing/finite validation and weight reads contribute
to the measured load time; their individual shares are unprofiled.
Per-stage `--profile`/`trace`, long contexts, instruct/chat model parity,
untied heads, FP16 state, Qwen3.5 INT8 and CUDA DeltaNet remain unsupported
or unverified. CPU snapshots remain unsupported as before.

### Working commands and actual output

```bash
cargo build --release -p cli --bins --locked
target/release/marin-run /home/mitali/models/Qwen3.5-2B-Base \
  --backend cpu --threads 8 --ctx 128 --temp 0 --max-tokens 8 --metrics \
  -p "The capital of France is"
```

Actual base-model continuation (all three runs, also HF):

```text
 Paris.
A. True
B
```

For `-p "def fibonacci(n):"`, eight-token output is:

```text

    if n == 0:
```

These are truncated continuations from a base model, not instruction/chat
answers or an executable-code-quality claim. Strict real verification:

```bash
target/release/marin-run verify /home/mitali/models/Qwen3.5-2B-Base \
  --backend cpu --reference bench-results/qwen35-end-to-end/hf-reference.json
MARIN_MATMUL_SCALAR=1 target/release/marin-run verify \
  /home/mitali/models/Qwen3.5-2B-Base --backend cpu \
  --reference bench-results/qwen35-end-to-end/hf-reference.json
```

Regenerate the real/tiny references with `tests/reference/qwen35/full.py`
in the existing HF environment. Raw source revision, full vectors, parity
logs, timing/resource rows and verified cgroup probes are retained under
`bench-results/qwen35-end-to-end/`.

### Final acceptance decision

| Gate | Status | Exact evidence |
|---|---|---|
| Real full backbone / existing CLI | PASS | Official Base checkpoint, all 24 layers and tied head; actual France/Fibonacci generation shown above |
| HF full layer/logit numerical parity | PASS | Every prompt-position layer value and full logits; original normalized-max/RMS gates; default and scalar; 16/16 greedy ids per dispatch |
| State correctness and atomicity | PASS | Whole-model incremental/fresh, prefix replay, reset, session isolation, foreign-model cache, context/token errors and failure after earlier layers updated staged state |
| RAM admission | PASS | Shared inspection/load peak estimate, per-pass guards, verified 512 MiB refusal / 8 GiB real generation with swap disabled and zero OOM |
| Short performance / RAM measurement | PASS | Three serialized fresh processes per real prompt, actual model load/TTFT/prefill/decode/wait4 peak RSS; no speedup or cold-cache claim |
| Existing Qwen3 / CUDA / INT8 / KV / snapshots | PASS | Final **615 workspace tests / 71 suites**, **zero failures/ignores**, `MARIN_REQUIRE_CUDA=1` on RTX 5050; packed artifact, switching, thread invariance, retained KV, GPU snapshot regressions included |
| Scalar / block regressions | PASS | Real model full-vector verification; **52 scalar tensor tests**, six scalar hybrid reference tests; original hybrid fixtures preserved |
| Source quality checks | PASS | fmt, workspace/all-targets check, workspace/all-targets/all-features clippy -D warnings, line-limit, unsafe and whitespace guards |
| CUDA DeltaNet / Qwen3.5 quantization / multimodal / MoE | OUT OF SCOPE | Explicitly unsupported, no implementation or changed defaults |
| Long context / real instruct checkpoint / cold cache | UNVERIFIED | No claims from short Base-model measurements |

**GO: real end-to-end Qwen3.5-2B-Base text inference on CPU, BF16
source matrices with FP32 math/state.** This is a complete model and
existing CLI path, not an isolated-block acceptance. Keep the measured
reference as the baseline before attempting batched prefill projections
using existing GEMM. No new SIMD/CUDA DeltaNet kernel or quantization
should be inferred from this GO.

HEAD and all three stash hashes are unchanged; no commit or push. Prior
uncommitted work is retained, with necessary source-reader/block/CLI
integration extensions. No unrelated refactor or discarded work. All
benchmark/check jobs completed. README links this acceptance and gives
the working model command.

## 36. Qwen3.5 CPU batched prefill (2026-10-13)

### Acceptance gates recorded before implementation

Same official 2B-Base checkpoint and pinned HF FP32-math reference as §35.
Serial release-build comparisons on Ryzen 7 260, eight threads, identical
5/128/512-token inputs, eight greedy output tokens. No cache eviction claim.

| Gate | Requirement |
|---|---|
| Correctness | Existing normalized maximum ≤1e-4 and relative RMS ≤1e-5 for full layer/logit vectors; exact greedy ids; existing block/state, rollback/reset/isolation and prefill→decode tests preserved |
| Prefill | At least 2× throughput and 50% lower TTFT on both 128 and 512 tokens versus measured tokenwise baseline |
| Short/decode | No >5% median TTFT or decode regression on the five-token reference, three serialized runs; long-prompt decode reported separately |
| Memory | Peak RSS increase ≤128 MiB at 512 tokens; explicit admission covers live batched workspace and attention packing; no full-context reservation or unbounded chunking |
| Compatibility | Full workspace checks, scalar reference, Qwen3 CPU/CUDA/INT8/KV/snapshot regressions pass |

Profile first, batch existing matrix projections/MLP across the whole prompt,
retain sequential convolution/DeltaNet updates and one whole-model transaction.
No bounded prefill, quantization or new kernel is part of this experiment.

### Implementation and controlled measurements

Dense BF16 source projections and MLP now call the existing Matrix batched
SIMD/blocked GEMM and thread dispatch for passes of at least 16 tokens.
FP32 fixtures and smaller passes retain tokenwise GEMV, preserving the
existing exact reference/session tests. This conservative dispatch boundary
is not a measured optimum. Whole-input norms/activations reuse the norm
buffer; MLP up is released before down projection. Delta projections and
output projection batch across tokens; causal convolution and delta-rule
updates remain sequential. Full-attention projection, normalization, RoPE
and causal attention use the existing batched attention path. One staged
whole-model transaction remains; edited prefixes still reset/replay.
There are no new inference kernels or bounded-prefill chunks.

A first fully batched reference experiment changed the rounding of batched
attention/GEMM relative to tokenwise execution and failed an existing exact
fixture assertion. It was rejected for reference/small passes; no existing
test or HF tolerance was changed. New batched BF16 tests compare state and
logits under the original numerical gates, require exact greedy agreement,
and cover prefill→decode, session isolation and a last-layer failure after
staged earlier layers have executed.

Serial fresh-process release runs, eight threads, context capacity 1,024,
eight greedy output tokens; same official weights and hardware as §35.
Five-token France prompt: three runs per implementation, median below.
128/512-token ` hello` repetitions: one uninstrumented pair each, plus
separate instrumented runs. These are controlled length probes, not a
representative long-form quality corpus. The before 512-token run briefly
overlapped a focused test build; its 62.12 s result agrees with the isolated
instrumented 62.69 s result. No CPU benchmarks ran concurrently.

| Prompt tokens | TTFT before → after | Prefill tok/s before → after | Decode tok/s before → after | Peak RSS bytes before → after |
|---|---:|---:|---:|---:|
| 5 (median of 3) | 0.6658 → 0.6634 s | 7.5 → 7.5 | 6.00 → 6.11 | 3,898,847,232 → 3,899,002,880 |
| 128 | 15.74 → 1.41 s | 8.1 → 90.6 | 6.08 → 6.03 | 3,906,420,736 → 3,922,337,792 |
| 512 | 62.12 → 5.48 s | 8.2 → 93.5 | 5.96 → 5.86 | 3,931,742,208 → 3,966,226,432 |

128/512 prefill improves **11.19× / 11.40×**; TTFT falls **91.04% / 91.18%**.
512 peak increases **32.89 MiB**, below the predeclared 128 MiB ceiling.
Warm-file-cache process load is 3.03–3.53 s before and 2.71–3.15 s after;
this is neither a cold-cache result nor an intended loader improvement.
The France output remains ` Paris.\nA. True\nB`; both length probes emit
exactly eight ` hello` tokens in both implementations.

### Profile and memory accounting

An observer-timestamp build, retained separately from the final binary,
measured stage intervals on the actual model. Temporary profiling edits
were restored; normal CLI stage profiling remains unsupported for Qwen3.5.
Timers include allocation/copy costs within each interval; they do not
separately count allocator calls or hardware memory traffic. `perf` is not
installed, so cache misses, cycles and measured bandwidth are unavailable.

| Stage group | 5 tokens baseline | 128 before → after | 512 before → after |
|---|---:|---:|---:|
| MLP gate/up/down projections | 401.7 ms | 10.011 → 0.508 s | 39.976 → 1.903 s |
| Delta/full mixer projections including output | 194.4 ms | 4.837 → 0.265 s | 19.777 → 1.050 s |
| Delta recurrence | 14.0 ms | 0.345 → 0.307 s | 1.416 → 1.274 s |
| Delta convolution and associated row movement | 9.3 ms | 0.221 → 0.214 s | 0.883 → 0.810 s |

At 512, batched full attention including KV write/read took 72.7 ms;
Delta gated normalization took 107.2 ms. Other intervals and LM head,
embedding, transaction staging/admission are included in end-to-end TTFT.
Recurrence plus convolution now consumes about **38%** of the instrumented
5.50 s TTFT; MLP about 35%. These are the next measured bottlenecks,
not a reason to add another kernel in this milestone.

Retained model weights are unchanged at 3,764,747,520 bytes. Persistent
state remains 20,643,840 bytes plus 24,576 bytes per active attention
position. Workspace admission now includes eight widest FP32 activation
rows per input token, conservative GEMM panels per configured worker,
logits/token bookkeeping and packed attention for the entire visible
prefix, including padding. Whole-model staging/KV growth and the existing
256 MiB safety margin remain separate. No complete FP32/BF16 duplicate
model and no maximum-context KV reservation are introduced. New admission
tests check batched/prefix workspace growth and overflow.

### Numerical validation (fixed gates)

Original 16 short-prompt greedy steps and all layer/logit values still pass;
scalar dispatch also passes the original 16 steps. New HF length probes
compare all embedding/24-layer values for every prompt position plus full
248,320-element logits for prefill and one cached decode:

| Tokens | Max logit absolute error | Max layer absolute error | Worst normalized max / relative RMS | Greedy |
|---|---:|---:|---:|---:|
| 128 | 1.29e-5 | 9.06e-6 | 1.395e-6 / 1.840e-6 | 2/2 |
| 512 | 2.03e-5 | 5.40e-5 | 2.125e-6 / 2.603e-6 | 2/2 |

HF uses pinned Transformers 5.17.0, official checkpoint, FP32 math on BF16
source values, rebuilt real rotary buffers and official cached/fresh state
validation. Reference generation keeps only last-position head logits,
without changing their calculation. Full vectors and serial timing/profile
logs are retained in `bench-results/qwen35-prefill/`.

A separate real engineering-prose prompt, truncated/re-encoded to exactly
128 tokens, improves **15.15 → 1.40 s** TTFT and **8.5 → 91.3 tok/s**
prefill. Decode **6.01 → 6.00 tok/s**; RSS **3,906,945,024 → 3,921,969,152
bytes**. Both implementations produce the same eight greedy tokens. Its
HF full-vector comparison passes two steps: maximum logit error 2.12e-5,
maximum layer error 4.36e-6, normalized maximum 9.687e-7 and relative RMS
1.108e-6. Scalar real 128-token verification also passes two greedy steps
and full vectors (2.590e-6 normalized maximum / 3.081e-6 relative RMS).

Real 512-token generation under an enforced **8 GiB memory.max**, zero
swap, succeeds with cgroup memory.peak **3,976,540,160 bytes** and zero OOM/
OOM-kill events. A verified **512 MiB** cap refuses the conservative 5.07 GB
load budget before loading weights; cgroup peak **130,551,808 bytes**, zero
OOM events. Cgroup usage is distinct from process RSS and includes page
cache. One-thread prefill is measured separately below; it is not an
exhaustive thread matrix. Long-context quality, latency tails, cold-cache
loading, allocation counts and hardware bandwidth/counters remain
unverified. Ordinary desktop background activity was not disabled.

The optimized one-thread 128-token probe measured **4.07 s TTFT,
31.4 tok/s prefill, 6.12 tok/s decode**. Eight-thread prefill is 2.89×
faster on this probe. No one-thread tokenwise A/B or exhaustive thread/
context sweep was run.

### Final acceptance

| Gate | Status | Evidence |
|---|---|---|
| HF intermediate/logit and greedy parity | PASS | 22/22 default greedy steps across short, controlled 128/512 and real 128-token prose; original full-vector tolerances unchanged; 18/18 scalar steps including real 128-token prefill |
| State/reset/reuse/rollback/isolation | PASS | Existing exact fixture tests retained; new BF16 whole-vs-tokenwise/split-prefill, cached append, reset, edited-prefix replay and last-layer failure tests; 29 focused model/reference tests per dispatch |
| ≥2× prefill / ≥50% TTFT reduction | PASS | 128/512: 11.19×/11.40× throughput, 91.04%/91.18% lower TTFT; real prose independently 15.15→1.40 s |
| Short TTFT / decode ≤5% regression | PASS | Three-run medians: TTFT 0.6658→0.6634 s; decode 6.00→6.11 tok/s; measured long-probe decode within 2% |
| Peak-RSS increase ≤128 MiB | PASS | 512-token increase 32.89 MiB; unchanged retained weight ownership; explicit workspace/packing admission |
| Low-RAM admission / OOM | PASS | Enforced 8 GiB/no-swap real 512-token success; 512 MiB refusal before weight load; both zero OOM/OOM-kill |
| Qwen3 / CUDA / packed INT8 / KV / snapshots | PASS | Full workspace **618 tests / 71 suites**, zero failures/ignores, `MARIN_REQUIRE_CUDA=1` on RTX 5050; no changes to those execution paths |
| Scalar fallback | PASS | Real short and 128-token full-vector HF parity, **29** focused model tests and **52** tensor tests under `MARIN_MATMUL_SCALAR=1` |
| Source checks | PASS | fmt, workspace/all-targets check, all-targets/all-features clippy -D warnings, line-limit, unsafe and whitespace checks |
| Beyond 512 / cold cache / hardware counters | UNVERIFIED | Focused pilot only; no latency-tail, cache-miss or measured bandwidth claim |
| INT8 Qwen3.5 / CUDA DeltaNet / scheduler / multimodal / MoE | OUT OF SCOPE | No implementation |

**GO: the validated batched CPU prefill path for Qwen3.5-2B-Base.**
The strongest speed evidence is at 128/512 tokens; smaller passes preserve
reference/GEMV behavior. The next measured bottleneck is sequential
DeltaNet recurrence/convolution, alongside remaining dense MLP work.
Investigate those only in a subsequent measured milestone. No more kernels,
quantization paths or prefill planners were added here.

All benchmark/validation jobs completed. HEAD and all three stash hashes
remain unchanged. Existing uncommitted work is preserved; no commit/push.

## 37. Qwen3.5 CPU packed INT8 (2026-10-13)

### Gates fixed before implementation

BF16 remains default. INT8 uses the existing symmetric max-abs quantizer,
Q8 kernels and immutable packed format. Decay/beta projections (a/b),
convolution, norms, A_log/dt_bias, recurrent state and attention KV retain
validated FP32 values/math; source BF16 values are promoted exactly.

| Gate | Requirement |
|---|---|
| Kernel correctness | Against HF using the exact dequantized artifact values: original full layer/logit normalized maximum ≤1e-4 and relative RMS ≤1e-5; exact greedy ids. BF16 verification unchanged |
| Quantization quality | ≥95% teacher-forced top-1 agreement with BF16 across ≥256 prose/code positions; mean loss increase ≤0.03 nats and perplexity increase ≤3%; independent greedy differences reported separately |
| RAM | ≥35% reduction in fresh-process peak RSS on short and 512-token probes; packed weights remain packed, no full BF16 inference copy; safe enforced budget/refusal |
| Decode | ≥20% median improvement at short prompts, three serialized pairs |
| Prefill/TTFT | No >5% TTFT regression at 128/512/1024, identical ids and eight threads; one focused pair per longer length, investigate failures |
| Startup | Warm-file-cache direct load no slower than BF16; report one-off conversion time/RSS and artifact size separately; no cold-cache claim without controlled eviction |
| State/compatibility | Reset/replay/rollback/isolation, split prefill→decode, scalar/threads, packed corruption/incompatibility/partial publication and existing Qwen3/CUDA/snapshot regressions pass |
| Source checks | fmt/check/clippy -D warnings/full workspace/line-limit/unsafe checks pass |

Quantization error against original BF16 is a quality measurement, not a
relaxation of the numerical implementation gates. Group 128 is the initial
pilot (existing default); a smaller existing group is permitted only as a
measured quality tradeoff, with the same acceptance thresholds. No new
kernels, precision changes to state, chunking or generic model framework.

### Implementation and ownership

The official `Qwen3.5-2B-Base` checkpoint (revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`) has 24 text blocks: 18
DeltaNet and six full-attention. The export quantizes 151 dense matrices,
including embedding/tied output, QKV, Z, output, attention and MLP projections.
It stores 151 scale tensors and 169 protected FP32 tensors (471 tensors total).
The protected values include a/b projections, convolution, norms, A_log and
dt_bias; recurrent and attention state stay FP32. Existing Q8 GEMV/GEMM and
batched prefill operate on the packed matrices. The loader validates every
text tensor, shape, dtype, scale and group size, and binds the packed manifest
to the model architecture. The direct path constructs Q8 matrices from INT8
codes and scales; it never loads or retains a full BF16 weight model. BF16
remains the default and source folders still allow opt-in inline quantization.
The original Qwen3 packed format and atomic publication procedure are reused.

The retained BF16 source weights occupy **3,764,747,520 bytes**; retained Q8
tensor payload occupies **1,945,763,072 bytes**. The exported
`model.safetensors` is **1,945,824,752 bytes**, including its header; the full
bundle with config, tokenizer and metadata is **1,958,790,217 bytes**. The
one-off export took **8.242 s** and peaked at **1,655,894,016 bytes RSS**
with source files already in the OS page cache. These figures are measured
separately from warm direct loading. Export rejects an existing destination
and publishes only a complete, validated temporary artifact.

### End-to-end performance and memory

Ryzen 7 260, eight threads, CPU backend, group 128, `--temp 0`, eight
generated tokens, `--ctx 1152`. Each entry uses the same token IDs for BF16
and INT8; 128/512/1024-token prompts repeat ` hello`. Three serialized
fresh-process BF16/INT8 pairs give the short-prompt medians; each longer
length is one serialized pair. OS file cache was warm. Times below are seconds,
throughput is tokens/s, and RSS is fresh-process peak bytes.

| Prompt tokens | BF16 → INT8 TTFT | BF16 → INT8 prefill | BF16 → INT8 decode | BF16 → INT8 peak RSS | BF16 → INT8 model load |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.730 → **0.438** | 6.9 → **11.4** | 5.52 → **9.30** | 3,898,724,352 → **2,079,879,168** | 3.86 → **1.90** |
| 128 | 1.55 → **1.47** | 82.7 → **86.8** | 5.50 → **9.68** | 3,921,551,360 → **2,092,797,952** | 3.88 → **1.94** |
| 512 | 6.01 → **5.41** | 85.1 → **94.7** | 5.38 → **9.45** | 3,966,111,744 → **2,138,320,896** | 3.98 → **1.98** |
| 1024 | 11.61 → **10.47** | 88.2 → **97.8** | 5.39 → **9.00** | 4,030,210,048 → **2,205,065,216** | 3.87 → **1.98** |

Short decode improves **68.5%** and peak RSS falls **46.7%**. The longer
TTFT changes are **−5.2%, −10.0%, −9.8%**, respectively, with no 5%
regression. The short-prompt generated text is identical for BF16 and INT8:
` Paris.\nA. True\nB`. Older §35–36 figures used different runs and are
historical baselines, not the paired comparison for these gates.

Under an enforced **4 GiB `memory.max`**, zero swap, the real 1024-token
INT8 run generates eight tokens, has cgroup `memory.peak` **2,213,138,432
bytes**, and records zero OOM/OOM-kill events. A separate **512 MiB** cap
rejects the estimated **2.57 GB** execution budget before loading weights;
cgroup peak is **128,700,416 bytes**, with zero OOM/OOM-kill events. Cgroup
usage and process RSS are different measures; resident file-cache charging
depends on the host's cache state. This is an enforced process budget test,
not a claim that a complete desktop fits in 4 GiB.

### Numerical and quality validation

The official HF Transformers 5.17.0 implementation receives the *same*
packed INT8 codes dequantized to FP32 for reference execution, with protected
FP32 tensors unchanged. This tests Marin's Q8 computation and hybrid state,
while the following teacher-forced comparison separately tests quantization
loss against the original BF16 checkpoint. HF cached/fresh transitions were
checked, and the original normalized maximum **1e-4**, relative RMS
**1e-5**, and exact greedy gates were unchanged. Every embedding, all 24
layer outputs at every prefill position, and all 248,320 output logits were
compared:

| Prompt | Greedy matches | Max logit absolute error | Max layer absolute error | Worst normalized max / relative RMS |
|---|---:|---:|---:|---:|
| Short prose + code | 16/16 | 1.72e-5 | 5.09e-7 | 1.473e-6 / 1.770e-6 |
| 128 tokens | 2/2 | 1.19e-5 | 8.09e-6 | 1.927e-6 / 1.969e-6 |
| 512 tokens | 2/2 | 1.34e-5 | 3.30e-5 | 2.250e-6 / 2.371e-6 |
| 1024 tokens | 2/2 | 1.57e-5 | 7.35e-5 | 2.713e-6 / 2.708e-6 |

All **22/22** default-dispatch greedy choices match. Forced scalar Q8
matches **16/16** short greedy choices, worst normalized maximum **2.798e-6**
and relative RMS **2.867e-6**. The original BF16 short-prompt HF run still
matches **16/16**. A deterministic 32-wide hybrid fixture compares direct
packed with inline Q8 at one and eight threads and checks repeated decode,
split prefill, prefix replay, reset, session isolation and late whole-model
rollback. Both CLIs select packed INT8 only when requested; the CLI `verify`
command accepts the matching quantization flags. Tests reject corrupted,
mislabelled, missing and wrong-group artifacts, as well as bare INT8 tensors
without a packed manifest.

Teacher forcing uses identical checkpoint/tokenizer, the same 64-token
prompt and next 128 scored tokens for each prose and code sample. The
top-1 agreement is **128/128** prose and **126/128** code, **254/256 =
99.22%** overall. Relative to BF16, mean loss changes by **−0.002147** and
**−0.000267 nats**, and perplexity by **−0.214%** and **−0.027%**,
respectively. These small samples pass the preset 95%, +0.03-nat and +3%
gates; they do not establish broad instruction or executable-code quality.
No real-model group-size change was needed.

### Checks, failures and decision

The first quantized CLI verification exposed that `verify` did not parse
weight quantization flags; it now uses the existing quantization argument
set. An initial combined long HF-reference job ended with SIGTERM before
publishing a result; its cause was not established. Streaming JSON output
and separate 128/512/1024-token reference jobs then completed and passed.
No acceptance threshold was changed. Warm-cache startup was measured;
controlled cold-disk startup, latency tails, hardware memory-bandwidth
counters and broad quality/code-execution evaluation remain unverified.

`cargo fmt --all --check`, workspace/all-targets `cargo check`, and
workspace/all-targets/all-features `cargo clippy -- -D warnings` pass.
`MARIN_REQUIRE_CUDA=1 cargo test --workspace --all-features --locked`
passes **622 tests across 73 suites**, zero failed/ignored, on the available
RTX 5050. Focused final all-features model tests and forced-scalar model/
tensor tests pass; the tensor scalar suite has **52 tests**. Rust line-limit,
unsafe-code and `git diff --check` checks pass. Existing Qwen3 CPU/CUDA,
packed loader, KV and snapshot regressions are included in the workspace
suite. Raw commands, measurements, HF outputs and check logs are retained in
ignored `bench-results/qwen35-int8/`.

| Original gate | Decision | Evidence |
|---|---|---|
| Matched-HF numerical/greedy correctness | **PASS** | 22/22 real-model steps, maximum normalized 2.713e-6 and RMS 2.708e-6; BF16 and scalar checks pass |
| BF16 quality ≥95%, loss/PPL bounds | **PASS** | 254/256 top-1; both corpus loss and PPL changes below bounds |
| Peak RSS ≥35% lower and safe admission | **PASS** | 46.7% short and 46.1% 512-token reductions; 4 GiB execution, 512 MiB early refusal, zero OOM |
| Short decode ≥20% faster | **PASS** | 5.52 → 9.30 tok/s median, +68.5% |
| 128/512/1024 TTFT regression ≤5% | **PASS** | −5.2% / −10.0% / −9.8% in focused paired runs |
| Warm direct startup ≤BF16 | **PASS** | 1.90 vs 3.86 s short median; one-off 8.242 s export reported separately |
| State, format, scalar, thread, CLI and Qwen3/CUDA compatibility | **PASS** | Fixture/CLI tests, strict scalar HF, RTX 5050 workspace suite |
| Source and full test checks | **PASS** | fmt, check, clippy, 622 workspace tests, scalar suites, line and unsafe checks |
| Controlled cold-disk startup and broad quality | **UNVERIFIED** | Outside fixed gate; warm cache and two 128-position corpora only |

**GO: opt-in packed CPU INT8 for Qwen3.5-2B-Base, group 128.** BF16 stays
default. The larger remaining prefill cost is sequential DeltaNet recurrence/
convolution and dense work; §36 profiled those before INT8. No new bottleneck
profile is claimed from these end-to-end measurements. Qwen3.5 CUDA DeltaNet,
vision, Instruct checkpoints and CPU snapshots were not validated here.
No commit, push or stash change was made.

## 38. Qwen3.5 CUDA and 4B milestone (in progress)

### Phase 1 gates fixed before implementation

Target: the official Qwen3.5-2B-Base text backbone on the RTX 5050, BF16
weights and FP32 activations/state. Use the existing CLI with `--backend cuda`.
No host-side recurrent or attention state in the inference path.

| Gate | Requirement |
|---|---|
| HF numerical parity | Against the same BF16 source weights and official HF FP32 reference, compare embedding, all 24 layer outputs and full logits: normalized maximum ≤1e-4 and relative RMS ≤1e-5; greedy IDs exact on prose and code; investigate first mismatched layer |
| State | CUDA convolution/recurrent state and full-attention KV remain device-resident; fresh session, reset, repeated decode, prefill→decode and edited-prefix replay agree with fresh execution; failures preserve the last committed session state |
| Usability | Real 2B model loads and generates text through `marin-run --backend cuda`; explicit GPU admission refuses an impossible budget before weight upload; default BF16/CPU fallback behavior remains correct |
| Compatibility | Existing Qwen3 CPU/CUDA, Qwen3.5 CPU BF16/INT8, packed loading, scalar CPU, KV reuse and snapshots pass their relevant regression tests |

Phase 2 performance and phase 3 real-4B gates will be recorded **before**
their respective implementations, after the prior phase passes. No speed or
8 GiB feasibility claim is accepted from synthetic fixtures alone.

### Phase 1 result and phase 2 gates fixed before optimization

On the RTX 5050, the original Qwen3.5-2B-Base BF16 checkpoint runs the
complete 24-layer hybrid text path on CUDA. Convolution and recurrent state
use device buffers, the six full-attention blocks use device paged KV, and a
pass stages device state until final logits have synchronized and passed
finite-value validation. Short prose/code HF verification passes **16/16**
greedy steps; 128/512-token HF full-layer/full-logit verification passes
**4/4** more. Worst normalized maximum/RMS at 512 tokens:
**3.055e-6 / 2.639e-6**. Tiny hybrid fixture full-vector verification
passes **8/8** greedy steps. A first 128-token run missed the fixed RMS gate
at layer 1 (**1.054e-5**); a controlled precise GPU matmul run passed
(**1.937e-6** RMS), identifying tensor-core projection rounding as the cause.
The tiny fixture also caught an incorrect +1 applied to DeltaNet's norm
weight; the focused GPU-vs-CPU intermediate/state test now passes. Device
session tests cover whole/tokenwise, replay, reset, independent sessions and
late failure after staged state updates. A 3 GiB competing GPU reservation
left **4,000,645,120 bytes** free; explicit CUDA refused the estimated
**4,470 MB** requirement before uploading weights. An initial real short
generation produced ` Paris.\nA. True\nB`: **21.5 ms TTFT**, **55.27
decode tok/s**, **4.55 s load**, and **4.76 GB device-wide VRAM in use**.
These are single-run phase-1 figures, not phase-2 comparison medians.

**Phase 1 GO:** the fixed strict HF, real-generation, device-resident state
and rollback gates pass. Remaining stage checks run at final integration.

| Phase 2 gate | Requirement |
|---|---|
| Profile first | Measure real 5/128/512-token prefill and decode, kernel launches, allocations, host/device transfers, synchronization, VRAM and GPU power where available; distinguish device-wide readings from process ownership |
| Correctness | Preserve phase-1 HF full-layer/logit tolerances, exact greedy IDs and transactional state tests after every accepted optimization |
| Speed | Accept a change only with ≥15% gain in a measured hot path or ≥10% end-to-end TTFT/decode gain on three serialized paired runs, without >5% regression on another representative workload |
| Memory and power | Peak VRAM must not rise >5% for a speed-only change; report measured power/energy rather than inferring it from speed; no per-token host copy of recurrent/KV state |
| Comparisons | Same model, precision, tokens, context and hardware for CPU and available llama.cpp baselines; label any unmatched format/configuration explicitly |

Phase 3 gates will be written after phase 2 passes. No 4B support is inferred
from the 2B fixture or config shape alone.

### Phase 2 focused measurements and rejected experiments

The RTX 5050 Laptop GPU has an enforced **40 W** power limit. All runs below
use the official BF16 2B checkpoint, context 1152, eight output tokens,
one warmup and three serialized measured runs. `--profile --sample-gpu` adds
one separate instrumented pass. Times are median end-to-end CLI measurements;
VRAM is device-wide `nvidia-smi` usage, not this process's allocation.

| Prompt | First baseline TTFT / decode | Grouped-projection trial | Unchanged baseline rerun | Baseline-rerun prefill / peak VRAM |
|---|---:|---:|---:|---:|
| 5 | 30.94 ms / 60.60 tok/s | 29.08 ms / 62.88 tok/s | 28.92 ms / 62.44 tok/s | 175.15 tok/s / 4.78 GB |
| 128 | 442.14 ms / 41.42 tok/s | 453.06 ms / 46.38 tok/s | 451.99 ms / 54.84 tok/s | 283.34 tok/s / 4.81 GB |
| 512 | 1742.68 ms / 30.19 tok/s | 1654.77 ms / 50.26 tok/s | 1590.13 ms / 48.09 tok/s | 322.04 tok/s / 4.86 GB |

The baseline rerun moves as much as the trial: at 128 tokens the unchanged
baseline is faster than the trial, and at 512 its TTFT is **3.9% lower**.
The grouped Q/K/V and MLP launch fusion was therefore **reverted**. A scalar
16-token tile trial changed 512 TTFT by only **2.6%** and regressed 128 TTFT
by **3.3%**; it was reverted. Applying existing BF16 tensor-core MMA to the
hybrid prefill failed the fixed 128-token HF RMS gate at layer 1
(**1.054e-5** versus **1e-5**); a three-part MMA accumulation trial missed
more (**1.434e-5**). The final path retains FP32-accumulating scalar GPU
matmul for strict parity. Neither incorrect fast path is in the source.

On the unchanged baseline rerun, 512-token profiled prefill spends **755.7
ms** in MLP, **346.4 ms** in Q/K/V, and **358.8 ms** in DeltaNet recurrence.
Device-state staging costs **0.23 ms** once. Decode executes **399 kernel
launches** and **60–61 device allocations** per pass, with **zero model-state
host uploads** and about **993 kB** of final logits downloaded. At 128 tokens,
GPU waiting is **450.4 ms** in profiled prefill versus **0.19 ms** download;
the bottleneck is GPU execution, not host synchronization or copying.
At 512 tokens the sampler saw mean **35.5 W** and the 40 W power cap on every
busy sample; short-run power has only two samples and cannot support an
energy claim. The trial's 512-token device-wide peak was **4.88 GB** versus
**4.86 GB** for the nearby baseline rerun. All raw JSON and logs are in the
ignored `bench-results/qwen35-cuda/` directory.

A separate, serialized **matched BF16 CPU comparison** uses the same model,
1152-token context, prompt and output lengths, three runs and one warmup;
CPU prefill has eight explicit threads. This compares backends rather than
the rejected optimization:

| Prompt | CPU TTFT / prefill / decode | CUDA baseline-rerun TTFT / prefill / decode | CPU process RSS / CUDA process RSS |
|---|---:|---:|---:|
| 5 | 689.81 ms / 7.25 / 5.75 tok/s | 28.92 ms / 175.15 / 62.44 tok/s | 3.90 / 1.27 GB |
| 128 | 1719.85 ms / 74.44 / 5.23 tok/s | 451.99 ms / 283.34 / 54.84 tok/s | 3.93 / 1.27 GB |
| 512 | 5868.76 ms / 87.24 / 5.69 tok/s | 1590.13 ms / 322.04 / 48.09 tok/s | 3.99 / 1.27 GB |

CUDA also occupies **4.78–4.86 GB device-wide VRAM** in these runs; its low
process RSS does not mean the GPU weights disappeared. CPU and GPU runs are
separate, so clock/power state still limits exact speedup attribution.

The available host has no `llama-cli`, `llama-bench`, or comparable Qwen3.5
GGUF, so a matched llama.cpp figure is **unverified**. Historical §37 CPU
measurements use different runs and must not be treated as paired results.

**Phase 2 NO-GO:** the fixed ≥15% hot-path or ≥10% stable end-to-end speed
gate was not met by a correct optimization. The measured 40 W cap and strict
HF tolerance leave dense projection and recurrent GPU work as the next
optimization target. Phase 3 is **BLOCKED by phase 2's sequential gate**;
no 4B checkpoint was downloaded and no 4B inference, parity, or 8 GiB
feasibility claim is made. The [official 4B checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B/tree/main)
totals **9.34 GB** including vision shards, larger than this GPU's ~8 GiB
physical VRAM. Its [32-layer text configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
resembles the validated 2B hybrid topology, but text-only weight bytes and
admission have not been measured. A future 4B CUDA attempt needs explicit
text-only admission budgeting; whether BF16 fits or a validated lower-precision
weight path is required remains an open measurement.

The full `MARIN_REQUIRE_CUDA=1 cargo test --workspace --all-features --locked`
run passes **624 tests in 73 suites** with zero failures or ignores, including
Qwen3 CUDA snapshots and backend switching, Qwen3.5 CUDA state rollback,
CPU INT8, scalar and CLI tests. A first suite run exposed the stale Qwen3.5
CLI expectation that CUDA must fail; after updating that test it exposed
`marin`'s implicit FP16 GPU KV choice. Qwen3.5 now selects FP32 state for
implicit GPU use, while explicit FP16 remains rejected. Workspace/all-targets
`cargo check` and `cargo clippy -- -D warnings`, Rust formatting, line-limit,
unsafe checks and `git diff --check` pass.

## 39. Real Qwen3.5-4B text milestone (gates fixed before implementation)

The previous 2B GPU speed experiment is deferred, not a dependency for 4B.
Target the official post-trained `Qwen/Qwen3.5-4B` checkpoint and its text
backbone only. Reuse the validated Qwen3.5 CPU/CUDA hybrid execution. The
official index lists 426 `model.language_model.*` tensors, 297 vision tensors
and 15 MTP tensors; the exact retained text bytes must be read from shard
headers before memory admission is evaluated. BF16/FP32 is the numerical
reference. INT8 is opt-in and accepted only with its own quality checks.

| Gate | Fixed requirement |
|---|---|
| Layout and admission | Validate all expected 32-layer shapes, dtypes and names before inference; skip vision/MTP tensors; account separately for retained weights, recurrent/conv state, paged attention KV, workspace, staging/transient buffers and safety margin. Explicit 4B CPU/GPU requests must refuse an impossible RAM/VRAM budget before uncontrolled allocation. |
| BF16 HF parity | Compare embedding, every layer output and full final logits against official HF on the same text tokens, at short and at least one 128-token prompt: normalized maximum ≤1e-4, relative RMS ≤1e-5, and exact greedy token IDs. Diagnose the first mismatched layer; never relax these tolerances. |
| Session semantics | For the real model, whole versus split prefill→decode, repeated decode, edited-prefix replay, reset, independent sessions and late whole-model failure rollback agree with fresh execution. Preserve existing 2B and Qwen3 behavior. |
| Post-trained utility | Generate actual chat and coding responses using the official tokenizer/chat template, not Base-model continuation. Confirm role/end-token handling and deterministic greedy output against HF. Report output verbatim and any template limitation. |
| INT8 if needed | Reuse the packed Q8 format/kernels. Matched-quant HF execution must meet the same numerical gate and exact greedy IDs; versus original BF16, ≥95% top-1 agreement on matched prose/code teacher-forced positions, mean loss increase ≤0.03 nat and perplexity increase ≤3%. Never keep a full BF16 copy after direct packed loading. |
| Performance and memory | Report real 128/512/2k prompt TTFT, prefill/decode tok/s, load time, peak process RSS and process-attributable CUDA allocation where measurable; include device-wide `nvidia-smi` only with that label. Serialize runs under the 40 W GPU limit. If a context cannot fit or is prohibitively slow, state the actual limit rather than extrapolate a pass. Compare a matched reference engine only if available. |
| Regression checks | Full workspace/CUDA tests, scalar CPU, 2B HF parity, fmt, check, Clippy warnings denied, line-limit and unsafe checks pass. |

### Checkpoint, memory ownership and CLI behavior

The official post-trained `Qwen/Qwen3.5-4B` config (local SHA-256
`ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`)
specifies 32 ordered blocks: 24 linear-attention/DeltaNet and eight full
attention, hidden width 2560, MLP width 9216, 16 query/four KV heads of
width 256, and a tied 248320-row output embedding. Safetensors headers show
426 text tensors occupying **8,411,510,272 bytes**; 297 vision and 15 MTP
tensors are ignored. The CPU's BF16/FP32 retained-weight estimate is
**8,413,425,664 bytes** after promotion of protected vectors. The existing
Q8 exporter took **15.39 s** once and produced a **4,351,791,712-byte**
`model.safetensors` with **4,351,709,184 bytes** of retained INT8 codes,
scales and protected FP32 weights. Packed direct loading reads one matrix at
a time. The CUDA path uploads those Q8 matrices without a whole-model BF16
copy; the small protected a/b projections originate in BF16 and are checked
for exact BF16 representability before GPU upload. Recurrent/convolution
state, full-attention KV and sensitive vectors retain FP32 computation.

At 2k positions, fixed recurrent/convolution state is about **53 MB**;
CPU state held **188 MB** and CUDA state held **221 MB** with a **570 MB**
full-window upper bound in the measured run. CPU admission separately
includes loaded weights, twice the state for rollback, full-prefill
workspace, transient loading and a **256 MiB** safety margin. `inspect`
estimates **10.01 GB BF16** versus **5.66 GB Q8** at 2048 context. CUDA
admission includes weights, staged state, bounded GPU workspace/paged KV
and a **512 MiB** safety margin. Explicit BF16 CUDA at 128 context refused
before loading: **7235 MB free, 9164 MB needed** on the 8.11 GB RTX 5050.
CUDA loading also checks host staging headroom: the largest text tensor,
twice that size for inline INT8 conversion, plus **512 MiB**. This protects
the transient upload path under a restricted host-memory budget.
Under a 512 MiB zero-swap cgroup, the packed CUDA request refused cleanly
before upload: **1.17 GB needed, 400 MB available** (exit 1, no OOM kill).
With `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, BF16
CPU refused its 10.01 GB estimate against 8.48 GB available, while packed
CPU INT8 completed both a short generation and a real 2048-token prompt;
the latter peaked at **4,960,686,080 bytes RSS**, with no OOM kill.

The official tokenizer's `enable_thinking=False` chat template exactly
matched Marin's rendered system/user prompts. A real 4B coding reply exposed
that `text_config.eos_token_id=248044` names `<|endoftext|>`, while the chat
tokenizer ends assistant turns with `<|im_end|>` (`248046`). The CLI now adds
the tokenizer's chat end token to configured stops for chat-capable models;
Base completion behavior and Qwen3 stop tokens remain unchanged. The real
CUDA Q8 answer ended cleanly and contained this executable result:

```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

Executing the extracted function gave `factorial(0)==1`,
`factorial(5)==120`, and `ValueError` for `-1`. The BF16 CPU chat answer
began `A mutex (mutual exclusion lock) is a synchronization primitive...`.
This is a focused utility check, not a broad instruction-quality evaluation.

### Official HF parity and INT8 quality

Transformers **5.17.0**, CPU eager attention, exact official text weights,
FP32 reference math and cached-versus-fresh decode checks were used.
BF16 Marin CPU matched **4/4** short and **1/1** 128-token greedy choices,
all embedding/32 layer outputs and all 248320 final logits; worst short
normalized maximum/RMS were **6.249e-6/1.898e-6**, and at 128 positions
**4.761e-6/3.531e-6**. Official-template chat and coding prompts added
**16/16** exact greedy choices with worst normalized maximum **3.972e-6**
and relative RMS **3.125e-6**. The largest absolute logit difference in
those chat cases was **5.77e-5**; the fixed gate is normalized, not an
absolute-logit cutoff.

For the exact dequantized packed artifact, CPU and CUDA each matched **4/4**
short and **1/1** 128-position HF greedy choices, with every full layer/logit
tensor checked. Worst normalized maximum/RMS were **5.662e-6/3.760e-6**
on CPU and **5.296e-6/3.288e-6** on CUDA. The first unmodified HF Q8
reference attempt was OOM-killed because expanding 4B Q8 matrices into
anonymous FP32 weights exceeds this host's RAM. The test reference now uses
temporary file-backed FP32 dequantized matrices and completed; production
Marin inference never creates that expansion.

Teacher forcing used the same tokenizer, 64-token lead-in and 128 scored
positions each from the §37 prose/code corpora. INT8 versus BF16 selected
the same first token at **127/128 prose** and **128/128 code** positions:
**255/256 = 99.61%** overall, above the original 95% gate. Mean loss
changed **−0.0027 / +0.0000 nats** and perplexity **−0.27% / +0.00%**;
both remain within +0.03 nat/+3% gates. Independent long greedy paths can
still diverge after an altered choice.

### Serialized end-to-end measurements

Ryzen 7 260, eight CPU threads, RTX 5050 Laptop GPU at 40 W, group-128
INT8, FP32 hybrid state, four generated tokens, one measured run and no
warmup per row. The same built-in prompt token IDs and context
(`prompt + 5`) were used. Load elapsed includes file-cache effects; these
are **not** controlled cold-disk comparisons. One run per row gives no
latency distribution.

| Prompt | Backend/weights | Load s | TTFT s | Prefill tok/s | Decode tok/s | Peak process RSS GB |
|---:|---|---:|---:|---:|---:|---:|
| 128 | CPU BF16 | 8.09 | 3.21 | 39.9 | 2.79 | 8.65 |
| 128 | CPU Q8 | 3.47 | 2.95 | 43.4 | 5.20 | 4.57 |
| 128 | CUDA Q8 | 3.01 | 0.84 | 152.1 | 27.27 | 0.91 |
| 512 | CPU BF16 | 8.26 | 12.91 | 39.7 | 2.73 | 8.73 |
| 512 | CPU Q8 | 3.50 | 11.56 | 44.3 | 4.75 | 4.66 |
| 512 | CUDA Q8 | 2.99 | 3.68 | 139.3 | 35.91 | 0.91 |
| 2048 | CPU BF16 | 8.07 | 53.11 | 38.6 | 2.33 | 9.02 |
| 2048 | CPU Q8 | 3.55 | 47.25 | 43.3 | 4.29 | 4.96 |
| 2048 | CUDA Q8 | 3.05 | 15.94 | 128.4 | 38.03 | 0.91 |

CPU Q8 reduced 128/512/2k TTFT by **8.2%/10.5%/11.0%** and 2k RSS by
**45.0%** relative to BF16; 2k decode improved **84%**. The 8 GiB
zero-swap 2k CPU Q8 run separately measured **51.16 s TTFT**, **40.0
prefill tok/s** and **4.07 decode tok/s** at the same 4.96 GB process peak.
For CUDA Q8, device-wide peak use was **5.42/5.56/5.69 GB** at
128/512/2k; device-wide use grew by about **4.42 GB** after loading. A
separate 2k run sampled `nvidia-smi --query-compute-apps` 46 times and
observed this `marin-run` PID at **4696 MiB (4.92 GB) maximum process
VRAM**, distinct from total-device use. The 512/2k samples saw the 40 W
power cap active for **100% of busy samples**; 2k mean power was **38.2 W**.
The tool's 2k **152.92 J per generated token** includes the entire prompt
and only four output tokens, so it is not a steady decode-energy figure.
At 2k the GPU prompt uses **2092 launches**, mostly repeated sequential
recurrent/convolution work; optimizing that is separate from this capability
milestone. No matched llama.cpp Qwen3.5-4B GGUF/binary was available
locally, so no comparable reference-engine throughput claim is made.

Commands used for the real model:

```bash
marin-run export-int8 ~/models/Qwen3.5-4B ~/models/Qwen3.5-4B-Q8 --quant-group 128
marin-run verify ~/models/Qwen3.5-4B --backend cpu --reference bench-results/qwen35-4b/hf-chat-bf16.json
marin-run verify ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 --reference bench-results/qwen35-4b/hf-int8-128.json
marin --model ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 --ctx 256 --max-tokens 160 --temp 0 run 'Write a Python function factorial(n) that returns 1 for n=0 and raises ValueError for negative n.'
```

The ignored `bench-results/qwen35-4b/` folder contains the HF vectors,
quality scores and per-row JSON. The 2B CUDA optimization NO-GO in §38 is
unchanged. Qwen3.5 vision/MTP, BF16 on this 8 GB GPU, Qwen3.5 snapshots,
long-context HF parity beyond 128 positions, wide quality sampling and a
controlled cold-disk load study remain outside the demonstrated capability.

Real-checkpoint state regression used the group-128 packed 4B artifact in
optimized model tests (`MARIN_QWEN35_4B_Q8=... MARIN_REQUIRE_CUDA=1 cargo
test --release -p model --lib qwen35:: --all-features --locked --
--test-threads=1`): **9/9 passed in 17.82 s**, including CPU and CUDA
whole/tokenwise execution, edited-prefix replay, reset, session isolation
and induced late-layer failure rollback. The fixture test retains its
65-token case; the real-checkpoint case uses six tokens to keep the focused
regression affordable. A debug build of the longer real-model case was
stopped after exceeding ten minutes; no result is inferred from it.

### Stage 39 acceptance

| Original gate | Decision and evidence |
|---|---|
| Layout and admission | **PASS.** Official 426-tensor text backbone validates and excludes 312 non-text tensors. Q8 2k CPU fits an 8 GiB zero-swap cgroup at 4.96 GB peak RSS; BF16 CPU and GPU refuse budgets they cannot meet. A 512 MiB host cgroup refuses CUDA staging cleanly. |
| BF16 HF parity | **PASS.** CPU matches embedding, all 32 layer outputs, full logits, and greedy IDs at short and 128 tokens within 1e-4/1e-5 normalized gates; official-template chat/code add 16/16 exact choices. |
| Real-model session semantics | **PASS.** Release-mode group-128 4B CPU/CUDA state suite 9/9, including replay, reset, isolation and forced late failure rollback. The standard full suite retains its longer fixture cases. |
| Post-trained utility | **PASS for focused prose/code prompts.** Official-template rendering matches HF; 4B GPU Q8 chat generated executable factorial code and stopped at `<|im_end|>`; BF16 CPU answered the mutex prompt. Broad utility evaluation remains unmeasured. |
| Opt-in packed INT8 | **PASS.** Matched packed-HF CPU/CUDA full-tensor and greedy parity pass. Against BF16, 255/256 teacher-forced top-1 choices match, with loss and perplexity changes within the fixed gates. Direct Q8 loading retains 4.35 GB of quantized weights without a full BF16 duplicate. |
| Performance and memory | **PASS for measured capability.** Serialized 128/512/2k measurements above include TTFT, prefill/decode, load, process RSS, process VRAM and power. Q8 CPU and CUDA fit the measured budgets. Controlled cold-disk startup and matched llama.cpp comparisons were unavailable and are not claimed. |
| Regression checks | **PASS.** Final `MARIN_REQUIRE_CUDA=1 cargo test --workspace --all-features --locked` passed. Real 2B BF16 HF recheck passed 2/2 cases and 16/16 steps on both scalar CPU and CUDA (worst normalized max 3.869e-6 / 9.620e-7). `cargo fmt --all --check`, workspace all-target/all-feature `cargo check`, Clippy `-D warnings`, line-limit, unsafe and `git diff --check` passed. |

**GO: real Qwen3.5-4B text inference on CPU BF16 and opt-in packed INT8,
and on this 8 GB RTX 5050 with opt-in packed INT8.** The 8 GiB host-memory
claim applies to packed INT8. BF16 CUDA is correctly refused on this device.
No claim is made for vision/MTP, 2k HF tensor parity, a broad quality suite,
or a matched reference-engine speedup.

## 40. Minimal V9 execution planner (gates fixed before implementation)

This milestone plans one explicitly named model folder. That folder is the
available artifact: a packed Q8 folder is used only with matching explicit
`--quantize int8`; a BF16 folder stays BF16 by default. Discovering sibling
artifacts by filename is excluded because the current packed manifest does
not prove that a sibling was exported from the requested checkpoint without
rehashing the source weights. The planner must share the actual load path,
not predict a different configuration than inference executes.

| Gate | Fixed requirement |
|---|---|
| Determinism and overrides | Given the same model metadata, available RAM/VRAM and flags, report the same backend, artifact, precision, threads and context. Explicit backend, quantization, thread count and context survive unchanged or fail with a clear reason. Never select INT8 without explicit opt-in. No explicit context is reduced. |
| Memory safety | Use the existing Qwen3/Qwen3.5 weight, state/KV, full-prefill workspace, staging and safety-margin estimates. CUDA auto selection requires the *whole requested context window* to fit, not merely the startup slab; otherwise choose CPU only if CPU admission passes. Explicit CUDA may retain its established startup-only warning, but the dry run must label the unsafe full-window fit and never claim it is safe. Unknown memory is reported as unknown, not as a verified fit. |
| Dry run | `marin-run plan MODEL` reports selected backend, artifact/format, thread count, context, CPU/GPU budget components, available memory, fallback or rejection reason. It does not load weight tensors or generate tokens. The reported plan agrees with an actual load under unchanged conditions. |
| Constrained hardware | Check Qwen3-0.6B and Qwen3.5-2B/4B at 4/8/16 GiB RAM scopes, restricted affinity, and 8/6/4 GiB available GPU where controllable. Verify appropriate admission/refusal and no OOM kill. Distinguish a simulated budget calculation from an enforced cgroup or actual GPU limit. |
| Numerical and session regression | Fixed Qwen3 and Qwen3.5 HF vectors, scalar CPU, CUDA, packed INT8, prefix reuse and rollback retain their existing tolerances and greedy IDs. Run real inference with the selected plan. |
| Focused 4B utility and cost | One coding task, a multi-turn session, a longer prompt and at least 32 sustained decode tokens on packed 4B; report output correctness, TTFT, decode rate, RSS/VRAM and planning overhead where measurable. A llama.cpp comparison is made only if a matching Qwen3.5-4B binary/GGUF is available locally; never extrapolate from a different model or format. |
| Project checks | `cargo fmt`, workspace check/test, all-feature Clippy warnings denied, line-limit, unsafe and final diff checks pass. Document failures and a GO/NO-GO without weakening these gates. |

### Planner behavior and constrained evidence

The implementation uses the loader's same metadata, `Footprint` and
`Backend::select` path for `marin plan`, `marin-run plan` and real model
loading. It does not search by sibling filename: the named folder remains
the artifact, and packed INT8 still needs an explicit matching opt-in.
Automatic CUDA now requires the full requested GPU window plus the
existing 512 MiB margin. Explicit CUDA retains the existing startup-only
mode but the dry run labels its full-window OOM risk. CPU default context
fit still uses the existing Qwen3 calculation; Qwen3.5 now finds the
largest feasible default context by the same whole-prefill/state estimate.
An explicit `--ctx` is refused rather than reduced. CPU thread reporting
uses the same affinity/cgroup-capped default as execution; explicit counts
are preserved. No weight tensors are read during planning; packed tensor
digests remain validated during the actual load.

All RAM scopes below used `systemd-run --user --scope -p MemoryMax=... -p
MemorySwapMax=0`, with real model metadata. These are admission results,
not simulated model weights. Free bytes include current cgroup use.

| Scope and request | Planner result | Actual execution |
|---|---|---|
| 4 GiB, Qwen3-0.6B BF16 CPU, explicit 2k | **2.07 GB needed**, 4.21 GB available: admit | Existing HF/scalar path rechecked separately, 48/48 greedy IDs. |
| 4 GiB, Qwen3-1.7B BF16 CPU, default 2k | Fit **1484** tokens and print reduction; explicit 2k refused (4.40 > 4.21 GB) | Separate invocation fit **1483** as free memory shifted; generated four tokens, **3.53 GB peak RSS**, no OOM kill. |
| 4 GiB, Qwen3.5-2B-Base BF16 CPU, explicit 2k | Refuse **5.07 > 4.18 GB** | No allocation attempted. |
| 4 GiB, Qwen3.5-4B Q8 CPU, explicit 2k | Refuse **5.66 > 4.18 GB** | No allocation attempted. |
| 4 GiB, Qwen3.5-4B Q8 CUDA, explicit 2k | Admit: **4.92 GB full GPU window + 512 MiB**, **6.72 GB free VRAM**; host stage **1.17 < 4.18 GB** | At 256 context, eight tokens generated; **912 MB peak RSS**, **4.55 s model load**, no OOM kill. |
| 8 GiB, Qwen3.5-4B BF16 CPU, explicit 2k | Refuse **10.01 > 8.48 GB** | No allocation attempted. |
| 8 GiB, Qwen3.5-4B Q8 CPU, explicit 2k | Admit **5.66 < 8.48 GB** | §39's enforced 8 GiB 2k run peaked at **4.96 GB RSS**. |
| 16 GiB scope, Qwen3.5-4B BF16 CPU, explicit 2k | Admit **10.01 < 10.70 GB actually free**; host physical RAM, rather than the scope, was limiting | BF16 4B HF chat verification below executed on this host. |

Additional final-binary CPU plans under enforced 8/16 GiB, zero-swap
scopes admitted Qwen3-0.6B BF16 at **2.07 GB** and Qwen3.5-2B BF16 at
**5.07 GB** in both scopes; 16 GiB also admitted packed 4B Q8 at **5.66
GB**. These are dry runs, not extra inference measurements.

With `taskset -c 0`, the default plan reported **one** CPU prefill
thread; explicit `--threads 8` reported **eight**. With CUDA hidden under
the 4 GiB scope, automatic Q8 reported both rejection causes: no visible
GPU and **5.66 GB CPU need vs 4.18 GB available**. On the live RTX 5050,
4B BF16 auto chose CPU because the GPU needed about **9.29 GB** for its
full window while only **6.72 GB** was free; packed Q8 auto chose CUDA.
Using the real Q8 2k budget (**4.92 GB + 512 MiB**) against hypothetical
4/6/8 GB free VRAM gives reject/fit/fit respectively. These last three
VRAM outcomes are **budget calculations, not enforced GPU quotas**; this
driver exposes no VRAM quota control. CPU/GPU dry runs on the packed 4B
artifact took medians **0.418/0.778 s** over three process invocations.
That is total dry-run time, not additional model-load latency.

### Real-model workload and numerical regression

After planning, the official-template GPU Q8 coding prompt generated a
recursive factorial function with zero/negative guards; the same
function's `0→1`, `5→120` and negative-`ValueError` checks passed in
§39. A piped two-turn `marin` session replied **`OK`**, then **`17`**
to a remembered-number question. The 4 GiB GPU scope also completed
real generation. Serialized `marin-run bench` runs used the same packed
4B artifact, RTX 5050 at 40 W, one run, no warmup, and **32** output
tokens; JSON is in ignored `bench-results/v9/`.

| Prompt | TTFT | Prefill | Decode | Peak process RSS | Peak device-wide VRAM |
|---:|---:|---:|---:|---:|---:|
| 512 | 4.006 s | 127.8 tok/s | 40.34 tok/s | 913 MB | 6.09 GB |
| 2048 | 16.736 s | 122.4 tok/s | 35.52 tok/s | 913 MB | 6.22 GB |

The 2k pass used **2092 GPU launches**; sustained decode's p50/p95
inter-token latency was **27.85/33.11 ms**. These are single-run,
file-cache-warm observations. Different output lengths and power history
make §39's four-token decode row an unsuitable controlled before/after
speed comparison. Process-specific VRAM was not resampled here; §39
measured **4696 MiB** on a matching 2k 4B Q8 process. No local
`llama-cli`, `llama-bench` or matching Qwen3.5-4B GGUF was present,
so no llama.cpp speed comparison is claimed.

Final release-mode HF verification after planner integration passed:
Qwen3-0.6B scalar CPU **48/48** greedy IDs; Qwen3.5-2B BF16 scalar CPU
**16/16** with worst normalized max/RMS **3.869e-6/3.025e-6**;
Qwen3.5-4B BF16 CPU official chat/code **16/16** with worst
**3.972e-6/3.125e-6**; and packed 4B Q8 CUDA at 128 positions **1/1**
with **3.879e-6/3.288e-6**. The matched full-layer/logit gates remain
1e-4/1e-5. The earlier 4B real-checkpoint state suite and workspace
rollback tests remain in force.

The final release binary again planned packed 4B Q8 at 256 tokens on
CUDA (**4.64 GB GPU full-window budget + 512 MiB margin** vs **6.72 GB**
free) and then generated ` Paris.` in two tokens with automatic backend
selection. That real run loaded in **3.75 s**, reported **913 MB** peak
process RSS and **45.58 tok/s** for the single measured decode interval;
its five-token prompt is only a load/selection smoke test, not a stable
throughput benchmark.

### Stage 40 acceptance

| Fixed gate | Decision and evidence |
|---|---|
| Determinism and overrides | **PASS.** One shared plan feeds dry-run and load. Explicit context and quantization are retained or refused; CPU affinity defaults and explicit thread overrides tested. BF16 remains default, and the packed-artifact integration test rejects missing INT8 opt-in. |
| Memory safety | **PASS for measured hardware.** CPU estimates include weights, full KV/recurrent state, rollback, full-prefill workspace and 256 MiB safety; Qwen3.5 additionally budgets its known load transient. CUDA auto requires its full GPU window plus 512 MiB, and host staging is checked. Explicit CUDA reports startup-only window risk. 4/8/16 GiB admission/refusal and real 4 GiB CPU/GPU runs showed no OOM kill. |
| Dry run | **PASS.** Both CLIs report the same artifact/format, backend, threads and context used by load without reading weight tensors. Final-binary Q8 auto CUDA plan and real generation agreed. CPU/GPU dry-run medians were 0.418/0.778 s; this is total process time. |
| Constrained hardware | **PASS for enforceable RAM and actual GPU, PARTIAL for hypothetical GPU quotas.** Enforced 4/8/16 GiB RAM and one-core affinity checks passed; no-GPU and live 6.72 GB free-VRAM fallback/selection passed. 4/6/8 GB VRAM thresholds were calculated from actual model budgets but could not be enforced on this driver. |
| Numerical and session regression | **PASS.** Final release HF checks above retained exact greedy IDs and strict normalized logits across Qwen3/Qwen3.5 BF16, scalar CPU and Q8 CUDA. Full workspace tests covered CUDA, CPU, packed INT8, prefix reuse, rollback and snapshots. |
| Focused 4B utility and cost | **PASS with comparator unavailable.** Executable coding output, two-turn recall, 512/2k prompts and 32-token decode completed; table above records TTFT, rate, RSS and device-wide VRAM. A matched llama.cpp 4B binary/GGUF was absent locally, so no comparison is claimed. |
| Project checks | **PASS.** The full `MARIN_REQUIRE_CUDA=1 cargo test --workspace --all-features --locked` suite passed after planner logic was complete. A later output-label wording change passed focused `plan_cli` tests, `cargo fmt --all --check`, workspace check and all-target/all-feature Clippy `-D warnings`. Line-limit, unsafe and `git diff --check` passed. |

**GO for the minimal V9 static execution planner on the measured CPU and
RTX 5050 configurations.** It is an admission estimate, not a reservation:
concurrent processes can change free RAM/VRAM between planning and load.
No claim is made for enforceable GPU quotas, matched llama.cpp throughput,
or runtime re-planning after another process consumes memory.

### Next milestone: MoE residency pilot design, not implementation

[IBM Granite 3.0 1B-A400M Instruct](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct)
is a small first candidate: its [official config](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/blob/main/config.json)
specifies 24 layers, 32 local experts and top-8 routing, hidden size 1024,
4k maximum context, and tied embeddings. The [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoe/modeling_granitemoe.py)
uses FP32 router logits, top-k softmax, and separate gate/up/down expert
projections. Its embedding, attention and RMS-norm structure may reuse
Marin's tensor operations, but Granite's scaling factors, exact router,
expert weights and model loader require separate HF parity work. Top-8
across 24 layers means up to 192 selected expert uses per decoded token;
at BF16, the config dimensions imply about **3 MiB per expert**, **2.25 GiB**
for all routed experts and **576 MiB/token** of cold expert reads if none
are cached. These are arithmetic estimates, not measured I/O. Small total
size alone does not imply useful disk-streaming speed.

The smallest pilot is (1) exact all-resident BF16 text inference with
layer/logit and greedy-token HF parity; (2) measure per-layer expert IDs,
bytes and reuse on a fixed short/long conversation; (3) only then store
experts separately and compare all-resident to a fixed-budget, per-layer
LRU cache under enforced 4/8 GiB RAM. Keep embeddings, attention, norms
and router resident; never approximate routing or silently change
precision. Report cold/warm hit rates, disk bytes/token, TTFT, decode
tok/s, peak RSS and OOM behavior. [Colibrì](https://github.com/uv-genai/colibri)
demonstrates this dense-resident, routed-expert-on-disk design with a
per-layer LRU and OS page cache; its GLM-5.2 speed figures are its own
measurements, not predictions for Granite or Marin. Do not add prefetch,
learned hotness, mixed precision or a generic MoE scheduler to the pilot.
