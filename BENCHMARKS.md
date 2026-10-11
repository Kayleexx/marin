# Benchmarks

What Marin measures on a laptop, what each optimization changed, and how to repeat every number. Every claim comes from a saved run; numbers from different sessions are never mixed unless the text says so.

**Contents:** [Method](#method) · [Results at a glance](#results-at-a-glance) · [The power cap](#the-power-cap) · [Starting point](#starting-point) · [Optimizations 1–22](#optimizations) · [Not adopted](#tried-and-not-adopted) · [Repeating a measurement](#repeating-a-measurement)

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

`marin --quantize none|int8` (with `--quant-group N`, default 128) uses the same quantizer and GPU kernels as `marin-run`. bf16 stays the default; the Model tab names the weight format; an explicit `--quantize int8` on the CPU is refused.

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
