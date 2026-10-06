# Benchmarks

Numbers for the current engine, so later changes can be judged against them. Nothing here is tuned yet.

## Setup

- Model: Qwen3-1.7B-Base, bf16 weights, f32 math, greedy decoding, batch size 1
- Machine: AMD Ryzen 7 260, 16 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), NVIDIA driver 610.57.04
- Prompt of 32, 128 or 512 tokens, then exactly 64 generated tokens
- GPU: 1 warmup run, then the median of 5 runs, with `--profile --sample-gpu`. Other programs (desktop, editor) were using about 1.3 GB of the GPU's memory.
- CPU: 1 warmup run and the median of 3 runs for 32 and 128 tokens; 2 runs and no warmup for 512 tokens (each run takes over a minute). Ordinary desktop programs (browser, editor) were open during the CPU runs.
- Build: commit 723c9bf3d4d9 plus the uncommitted observability work (`marin 0.0.0 (723c9bf3d4d9-dirty)` in the saved results)

`--json FILE` saves the full result. `--profile` adds one extra pass that times every stage. That pass is slower than a normal run, so it is used for shares of time, never for speed.

## GPU results

| prompt tokens | first token | prefill | decode | time between tokens (p50 / p95) | sampling per token | weight bandwidth (est.) |
|---|---|---|---|---|---|---|
| 32 | 125 ms | 264 tok/s | 49.6 tok/s | 20.0 / 21.1 ms | 3.7 ms | 171 GB/s |
| 128 | 351 ms | 369 tok/s | 55.7 tok/s | 17.7 / 19.4 ms | 3.6 ms | 192 GB/s |
| 512 | 1477 ms | 348 tok/s | 49.1 tok/s | 20.4 / 20.9 ms | 3.7 ms | 169 GB/s |

The 32-token row ran first on a GPU that had been idle, and it is the least steady: first token ranged from 92 to 132 ms and prefill from 249 to 365 tok/s. Its mean GPU clock was 1515 MHz, against 1736 and 1936 MHz for the 128 and 512-token runs, and its decode speed is below the 55.7 tok/s of the 128-token row. Treat it as a lower bound until it is repeated with the GPU warmed up.

Loading takes about 2.4 s, of which copying the weights to the GPU is 0.38 s (about 9 GB/s). The GPU's memory in use grew by 3.5 GB and peaked at about 5.0 GB for the whole device, which includes the desktop's own share. The process itself peaked at 840 MB of RAM. Every run kept one of the 16 cores busy.

While running, the GPU reported (mean / max, from `nvidia-smi`):

| prompt tokens | load | power | clock | temperature |
|---|---|---|---|---|
| 32 | 75% / 92% | 28.1 / 38.8 W | 1515 / 1935 MHz | 53 / 54 C |
| 128 | 79% / 100% | 35.0 / 40.4 W | 1736 / 2175 MHz | 55 / 58 C |
| 512 | 87% / 100% | 38.1 / 40.5 W | 1936 / 2377 MHz | 57 / 59 C |

Before the workspace work every forward pass launched 479 kernels and made 283 small GPU allocations, for the prompt pass and the one-token passes alike (see "Fixed workspace and fewer launches" for what it is now), and downloads 608 KB of results in about 0.12 ms. The two kinds of pass differ in how long the host waits for the queued GPU work: 119, 335 and 1469 ms for the prompt pass at 32, 128 and 512 tokens, against 12 to 15 ms for a one-token pass.

```bash
marin-run bench ~/models/Qwen3-1.7B-Base --backend cuda --prompt-tokens 128 --tokens 64 --runs 5 --profile --sample-gpu --json gpu-p128.json
```

`scripts/bench_baseline.sh` runs the whole protocol for both backends.

### Where the time goes (GPU)

Prefill (reading the prompt) is dominated by the MLP, then the attention projections:

| prompt tokens | MLP | q/k/v projections | attention output proj | attention |
|---|---|---|---|---|
| 32 | 67% | 17% | 8% | 2% |
| 128 | 70% | 16% | 9% | 4% |
| 512 | 64% | 14% | 7% | 14% |

Decode (one token at a time) is mostly MLP too, and attention grows with the prompt:

| prompt tokens | MLP | q/k/v projections | LM head | attention | attention output proj |
|---|---|---|---|---|---|
| 32 | 47% | 14% | 13% | 5% | 6% |
| 128 | 45% | 13% | 12% | 11% | 6% |
| 512 | 37% | 11% | 10% | 26% | 5% |

### What this suggests (GPU)

- Decode reads all 3.4 GB of weights for every token, so it is limited by memory traffic: the estimated weight bandwidth is 169 to 192 GB/s. Smaller weights (quantization) are the most direct way to speed it up. The GPU's rated memory speed was not checked here, so how close this is to the limit is not yet known.
- Choosing the next token takes about 3.7 ms, roughly a fifth of each decode step, which is a lot for such a small job. It is the cheapest thing to fix.
- Prefill matrix work now runs on the tensor cores (see the prefill section); the remaining prefill cost at long prompts is attention.
- Attention time grows with context length, so long prompts will need a faster attention kernel.
- The GPU is busy almost all of the time. Under 5% of a decode step is outside the timed stages, and the host spends 12 to 15 ms of each 15 to 18 ms step waiting for the GPU, so launch overhead is not the problem yet.
- GPU load averages 79% to 87% in the longer runs and peaks at 100%, with power around 40 W. Whether the GPU is power-limited was not checked.

## What this GPU can do, and how close the engine gets

`marin-run probe` measures the GPU itself with no model involved (four runs of 4 s, desktop programs open). The GPU is held to a software power limit during all of these (the driver reports the cap active in 65 to 68% of busy samples, 100% during decode), so "sustained" is the median of the windows after the opening burst.

| what | opening burst | sustained |
|---|---|---|
| memory read (1 GB buffer) | 320 to 344 GB/s | 277 to 302 GB/s |
| fp32 math (fused multiply-add) | 7.0 to 11.2 TFLOP/s | 7.0 to 11.2 TFLOP/s |
| bf16 tensor cores | 15 to 28 TFLOP/s | 28.1 to 28.3 TFLOP/s |
| copy 608 KB (the logits) GPU to host | 96 µs (6.3 GB/s) | |
| copy 4 bytes GPU to host | 6.7 µs | |

The fp32 figure follows the clock (7.0 TFLOP/s in the run where the GPU averaged 1640 MHz, 11.1 at about 2050 MHz). The measured memory ceiling can be under-read by a couple of percent: the engine's LM head reaches 351 GB/s, above the best probe window of 344.

What the engine reaches on the same machine (`bench --profile`, 128-token prompt):

| stage | decode (weights read per second) | prefill (math per second) |
|---|---|---|
| MLP | 295 GB/s | 1.19 TFLOP/s |
| q/k/v projections | 214 GB/s | 1.13 TFLOP/s |
| attention output proj | 237 GB/s | 1.14 TFLOP/s |
| LM head | 351 GB/s | 0.36 TFLOP/s (one token) |

- Decode weight stages sit at or near the memory ceiling (MLP and LM head), with the smaller q/k/v and output projections lower. All weight stages together read at about 284 GB/s.
- Prefill matrix stages reach about 10% of the fp32 ceiling and about 4% of the tensor-core ceiling.

## What choosing tokens costs

The engine copies the full logits vector to the host after every pass and the sampler runs on the CPU. For this model that is 151,936 values of 4 bytes, 607,744 bytes, copied in about 0.1 ms per decode token; the GPU wait before the copy is timed separately (11 to 14 ms).

Greedy against the chat defaults (`--temp 0.7 --top-k 20 --top-p 0.8`), 128-token prompt, alternated three times each on a warmed GPU:

| | decode | time between tokens (p50) | sampling per token | GPU clock |
|---|---|---|---|---|
| greedy | 61.5 tok/s | 16.0 ms | 0.14 ms | 1360 MHz |
| chat defaults | 56.8 tok/s | 17.1 ms | 3.69 ms | 1624 MHz |

The 3.7 ms of host sampling costs only 1.1 ms of time between tokens, because the GPU rests during it and then runs at a higher clock (the GPU wait fell from 14.1 to 11.7 ms). Speeds in this table were taken in one session and are not comparable with the other tables, since the GPU's clock and power state differ between sessions.

## Idle time between tokens

Does feeding the GPU quickly wear it down? `bench --token-pause-ms N` makes the benchmark wait N ms after every token, as a slower consumer would, with greedy sampling, a 128-token prompt and 64 generated tokens. Three rounds, interleaved, on a warmed GPU on AC power with a 40 W power limit; the figures are averages of the three rounds, which agreed closely.

| pause | engine speed | time between tokens (p50) | GPU load | GPU clock | GPU power | power cap active | energy per token | tokens/s on the wall clock |
|---|---|---|---|---|---|---|---|---|
| 0 ms | 61.4 tok/s | 15.9 ms | 89% | 1383 MHz | 36.4 W | 100% | 0.78 J | 46.8 |
| 1 ms | 64.8 tok/s | 15.2 ms | 85% | 1446 MHz | 36.3 W | 100% | 0.79 J | 45.9 |
| 3 ms | 70.2 tok/s | 13.6 ms | 76% | 1641 MHz | 36.6 W | 100% | 0.82 J | 45.0 |
| 8 ms | 77.5 tok/s | 12.9 ms | 62% | 1890 MHz | 36.5 W | 100% | 0.95 J | 38.9 |

- More idle time raises the clock the GPU runs at while it works, so the engine's own speed goes up (+26% at 8 ms). The average power does not change: it stays at the power limit whatever share of the time the GPU is busy.
- Energy per token is the best with no pause, and the wall-clock speed is the best too. Energy per token here is mean GPU power times the time of the whole measured loop (prompt processing, warmup and pauses included) divided by the tokens generated.
- The shorter the time per token, the fewer joules each token costs. The GPU handing a token to the CPU and getting the next one back is not what uses the energy: the 3.4 GB of weights read for every token is.

## Cost of processing several tokens in one pass

One pass over `k` tokens (a short prompt on an empty context, `prompt_forward_ms` of the benchmark runs, 15 runs each):

| tokens in one pass | pass time | compared with 1 token | per token |
|---|---|---|---|
| 1 | 12.1 ms | 1.00× | 12.1 ms |
| 2 | 12.9 ms | 1.06× | 6.4 ms |
| 4 | 17.5 ms | 1.45× | 4.4 ms |
| 8 | 27.3 ms | 2.26× | 3.4 ms |
| 16 | 47.3 ms | 3.91× | 3.0 ms |
| 32 | 84.1 ms | 6.95× | 2.6 ms |

The weights are read once per pass, so a pass that handles two tokens costs almost the same as one. The cost starts to grow from about four tokens, where the matrix kernel's arithmetic begins to matter. This is the cost side of checking several guessed tokens in one pass; how many guesses are right was not measured.

## Decode speed against context length

Greedy decoding after a prompt of the given length (64 tokens, 5 runs, 3 at 4096, warmed GPU, `marin-run compare` between the two builds, same session). Before: one block per (token, head), so a decode step used 16 blocks on a 20-multiprocessor GPU, each scanning the whole context alone. After: the context is split across about 80 blocks per layer, each block reads a key/value row once for both query heads that share it, and a small second kernel merges the pieces (exact softmax, same result up to rounding).

| context | decode before | decode after | time between tokens p50 before | after | attention per token before | after | joules per token before | after |
|---|---|---|---|---|---|---|---|---|
| 128 | 60.0 tok/s | 59.2 tok/s (same) | 16.3 ms | 16.1 ms | 1.98 ms | 1.77 ms | 0.81 | 0.83 |
| 1024 | 44.9 tok/s | 52.7 tok/s (+17.5%) | 22.2 ms | 19.0 ms | 9.17 ms | 4.64 ms | 3.01 | 2.87 |
| 2048 | 33.4 tok/s | 41.4 tok/s (+24.2%) | 29.9 ms | 24.1 ms | 16.47 ms | 9.54 ms | 6.47 | 6.02 |
| 4096 | 23.1 tok/s | 34.2 tok/s (+48.2%) | 43.3 ms | 29.2 ms | 30.64 ms | 14.14 ms | 16.31 | 14.63 |

- Joules per token include the prompt pass, so they grow with the context. Prefill speed did not regress (4096: 172.9 to 189.1 tok/s). GPU power stayed at the 40 W cap in every run.
- The weight stages do not change with context. All the slowdown is attention. At 4096 tokens the KV cache read per token is about 0.94 GB (stored as 32-bit floats), which at the measured 300 GB/s would take about 3 ms; attention now takes 14 ms, down from 31 ms, so about 4x the ideal remains.
- The new kernel added 28 launches and 56 small allocations per pass (507 launches, 339 allocations); the next section removes the allocations.
- Storing keys and values as 16-bit floats (half the bytes) was tried and rejected: it fails `marin-run verify` at the existing tolerances (largest logit difference 2.8e-3 against 5e-4 allowed, one layer output 1.05e-4 off against 1e-4), although the short greedy outputs matched. The tolerances were not loosened and the f32 cache stays.

## Prompt processing (prefill) with tensor cores

Prefill of a prompt of the given length plus one more token (5 runs, warmed GPU, `marin-run compare` between builds, same session). Before: the matrix kernel used ordinary 32-bit multiply-adds, one weight row per warp. After: for passes of 16 or more tokens (and when the row length is a multiple of 16) a tensor-core kernel runs (`mma.sync` bf16 with f32 accumulation, 64 rows x 64 tokens per block). The activations are f32, so each is split into a bf16 high part and a bf16 low part and both are multiplied; one bf16 rounding would lose too much (checked: dropping the low part makes the kernel test fail). Passes of 1 to 15 tokens, including every decode step, keep the old kernel.

| prompt tokens | first token before | after | prefill before | after | matrix stages (MLP / QKV / output proj) before | after |
|---|---|---|---|---|---|---|
| 128 | 352 ms | 127 ms | 364 tok/s | 1009 tok/s (+177%) | 1.2 / 1.2 / 1.2 TFLOP/s | 5.2 / 4.0 / 4.2 TFLOP/s |
| 512 | 1576 ms | 519 ms | 325 tok/s | 986 tok/s (+204%) | 1.1 / 1.1 / 1.1 TFLOP/s | 6.5 / 6.7 / 6.2 TFLOP/s |
| 1024 | 3499 ms | 1469 ms | 293 tok/s | 697 tok/s (+138%) | | |
| 2048 | 8531 ms | 4351 ms | 240 tok/s | 471 tok/s (+96%) | 1.1 / 1.0 / 1.1 TFLOP/s | 7.7 / 6.7 / 8.7 TFLOP/s |

- The matrix stages are 5 to 8 times faster. The whole prefill gains less as the prompt grows because attention is not a matrix kernel here: it still uses the old one-block-per-(token, head) kernel and is now 81% of a 2048-token prefill (3.4 s of 4.3 s). It is the next prefill limit.
- Crossover (15 runs each): prompts of 16 tokens are the same speed within noise (50 vs 46 ms), 48 tokens take 129 ms before and 51 ms after, 96 tokens 270 ms and 83 ms.
- Decode is unchanged within noise at every context length (checked against the previous table with the same script).
- Correctness: `marin-run verify` passes with unchanged numbers, but its three prompts are shorter than 16 tokens and never reach the new kernel. Instead, greedy output on five long prompts (about 400 to 1500 tokens) is identical between the old and new GPU build, and two of them also match the CPU build. Kernel tests compare against the CPU matrix code with the same 1e-4 tolerance across shapes that cross the tile edges.
- With the prompt pass about twice as fast and the power cap still the limit (39 W mean at 2048 tokens), the energy for a 2048-token prompt roughly halves (169 J to 86 J per generated token, prompt included). At 128 tokens the run is too short for the 500 ms power sampler to say anything.

## Fixed workspace and fewer launches

Decode (64 tokens after the given prompt, warmed GPU, 5 runs, 3 at 4096). Three changes, each measured against the one before in the same session:

1. Fixed workspace: every buffer a pass needs is allocated once and reused by all 28 layers. A one-token pass uses a workspace that lives with the sequence's cache; a prompt pass allocates one for the pass. The normalisation writes into a second buffer instead of copying first, and the split-attention scratch lives in the KV cache.
2. Grouped matrix launches: query/key/value are one launch and gate/up are one launch (one block of rows after another, same arithmetic, same bits). Passes of 16 or more tokens use the tensor-core kernel and run them one by one.
3. Prompts longer than 512 tokens run in pieces of 512, which keeps the workspace at about 55 MB however long the prompt is. Without it the fixed workspace raised peak GPU memory at a 4096-token prompt from 5700 to 6100 MB. Each token's result does not depend on how the prompt is split, and the logits agree to 1e-4.

| | before | fixed workspace | + grouped launches | + prompt pieces |
|---|---|---|---|---|
| device allocations per one-token pass | 339 | 0 | 0 | 0 |
| kernel launches per one-token pass | 507 | 507 | 423 | 423 |
| decode at 128 | 61.3 tok/s | 61.7 | 62.0 | 62.1 |
| decode at 1024 | 54.2 tok/s | 53.8 | 54.4 | 54.3 |
| decode at 4096 | 39.5 tok/s | 39.6 | 39.6 | 39.6 |
| peak GPU memory at 4096 | 5700 MB | 6103 | 6096 | 5691 |

- Every difference in decode speed is inside the run-to-run ranges of a single comparison. The grouped launches were checked again with 3 interleaved pairs of 12 runs: 61.54 tok/s before, 61.98 after, better in all three pairs (+0.7%). The query/key/value stage reads 254 GB/s instead of 222; the stage times are from the profiled pass, so they overstate what a real decode step gains.
- Energy per token (power times time) and power did not change: 0.63, 1.41 and 9.69 J at 128, 1024 and 4096 before; 0.62, 1.45 and 9.69 after. Prefill speed did not change either.
- CUDA graphs were measured, not adopted. A throwaway prototype captured one decode pass and replayed it (positions frozen, so only the timing is meaningful): 62.75 tok/s against 62.0 for normal launching, interleaved, 3 pairs, +1.2%. A real version would need the position (rope, key/value write, attention length and split grid) read from device memory, a separate stream and cudarc's event tracking turned off. The 84 launches removed by grouping were worth about 1.3 microseconds each, so all 423 remaining launches cost about 0.55 ms of a 16 ms step. Further fusing (residual add with norm, query and key norm, the two rotations) can save at most a few tenths of a percent each and was not done.
- Verification: `marin-run verify` numbers, the greedy and seeded outputs of five short prompts, and the greedy output of five long prompts (about 400 to 1500 tokens) are identical to the previous build.

## Weight quantization

`--quantize int8` or `int4` stores the weights of the 7 projection matrices of every layer as integer codes with one f32 scale per group of 128 columns (`--quant-group`). The embedding table and the output head (the same tensor on this model) are int8 whenever quantization is on, whatever the layers use. The codes are made at load time from the bf16 file, on the CPU, in about 1.5 s extra. GPU only; the CPU backend refuses it. The decode kernels read the codes and apply the scale to each group's partial sum; the prompt kernels feed the codes to the tensor cores unchanged and scale the partial sums per group, so no weight is ever rounded twice.

**Accuracy** (`marin-run quality`: Qwen3-1.7B-Base reads two texts of 2048 tokens, the first 64 as a prompt and the other 1984 one token at a time; a text of Vim release notes and a text of this repository's Rust code, neither likely to be memorised). "Same choice" is how often the quantized model's first choice is the bf16 model's first choice. Loss is nats per token; the bf16 model scores 1.527 on the notes and 0.828 on the code.

| | notes: loss change | same choice | code: loss change | same choice | weights |
|---|---|---|---|---|---|
| int8, groups of 128 | +0.0003 (+0.03%) | 98.2% | +0.0011 (+0.11%) | 98.9% | 1774 MB |
| int8, groups of 32 | +0.0021 (+0.21%) | 98.3% | +0.0011 (+0.11%) | 99.0% | 1936 MB |
| int4, groups of 128 | +0.137 (+14.6%) | 84.0% | +0.154 (+16.6%) | 85.6% | 1070 MB |
| int4, groups of 64 | +0.115 (+12.2%) | 85.9% | +0.101 (+10.7%) | 88.2% | 1124 MB |
| int4, groups of 32 | +0.087 (+9.1%) | 86.9% | +0.078 (+8.2%) | 90.1% | 1231 MB |

(The percentages are the change in perplexity.) bf16 weights are 3441 MB. int8 changes the loss by 0.2% at most and the first choice in only 1 to 2% of positions: it is the setting to use. int4 is clearly lossy: even with groups of 32 the loss rises by 8 to 9%. The loss of 4-bit weights comes mostly from the layers, not the head: putting only the head at 4 bits (layers int8) costs +8.4% and +5.9%, while at 8 bits the head costs about +0.1%; that is why the head stays int8.

**Speed** (decode of 128 tokens after a 128-token prompt, the same binary for every row, 3 rounds of 12 runs interleaved, warmed GPU, GPU at its 40 W power cap in every run):

| | decode | weights | peak GPU memory | energy per token (prompt included) |
|---|---|---|---|---|
| bf16 | 56.8 tok/s | 3441 MB | 5052 MB | 0.73 J |
| int8, groups of 128 | 93.6 tok/s (+65%) | 1774 MB | 3404 MB | 0.45 J |
| int8, groups of 32 | 87.8 tok/s | 1936 MB | 3475 MB | 0.48 J |
| int4, groups of 128 | 108.4 tok/s (+91%) | 1070 MB | 2712 MB | 0.39 J |
| int4, groups of 64 | 106.5 tok/s | 1124 MB | 2707 MB | 0.40 J |
| int4, groups of 32 | 103.5 tok/s | 1231 MB | 2763 MB | 0.41 J |

Over longer contexts (`marin-run compare`, the standard decode baseline, bf16 and the quantized runs in one session): at 1024 tokens of context decode goes from 49.3 to 75.5 tok/s (int8) and 86.3 (int4); at 4096 tokens from 36.6 to 49.8 and 53.5, because attention reads the same f32 key/value cache whatever the weights are. Prompt processing is unchanged (687 to 695 tok/s at 1024 tokens, 248 to 256 at 4096) and time to first token with it. Peak memory at 4096 tokens of context drops from 5990 MB to 4266 MB (int8) and 3591 MB (int4), so about 1.7 GB (int8) or 2.4 GB (int4) more context fits. bf16 decode in this session (56 to 58 tok/s at 128 tokens) is slower than in the section above (62): the GPU's memory read speed measured 263 GB/s sustained now against 277 to 302 then, and the same bf16 binary as before measured 57.1 tok/s now against 57.4 for the previous build, interleaved.

Decode kernels, what the tuning did (per-stage rates from the profiled pass, so only their relative size means anything): the first int8 kernel, one warp per row, read the MLP at 192 GB/s against 247 for bf16; taking 4 rows per warp so the activation reads are shared and the weight loads are in flight together raised it to 231; issuing two strides of loads before using any raised it to 282 (the memory limit, which the profiled pass overstates). Converting the codes with the 2^23 bit trick instead of the integer-to-float instruction helped int4 most. int4 stays below the memory limit (about 180 GB/s of weights in the MLP, held back by the instruction count of unpacking nibbles) and was not improved by 16 columns per lane (+2%, but 3% slower for int8, so dropped), by 4 strides instead of 2 (same speed), or by scaling each group by its largest value over 8 instead of 7 (mixed: +0.9 points on the notes, -4.3 on the code, kept out). The embedding and output head at int8 took the head from 2.43 ms to 1.07 ms of a decode step (profiled pass).

Not done: queries, keys and values (and gate/up) still run as separate launches when quantized, which would save about 1% by the measured cost of a launch; 4-bit weights are the plain round-to-nearest ones, with no calibration (GPTQ-style) or per-layer choice of width, which is where the 8-9% loss of 4-bit weights could be attacked; the key/value cache is still f32.

## Guessing tokens ahead (measurement only)

Speculative decoding would guess the next few tokens cheaply, run one pass over the last token plus the guesses, and keep the guesses the model agrees with. This section only asks whether the cheapest guesser, copying from earlier text, would be right often enough. Nothing is implemented in the engine: no pass is ever run with real guesses.

**Method.** `marin-run guess` runs Qwen3-1.7B (chat template, the `marin` chat prompt) on 40 prompts and saves every token, once greedy and once with the chat sampler (temperature 0.7, top-k 20, top-p 0.8, fixed seed), up to 400 tokens each. `marin-run guess-report` then replays each reply as if it were being generated with guessing. Under greedy decoding a guess is accepted exactly when it equals the next token the model chose, so a replay gives the true acceptance. For the sampled replies the same test matches speculative sampling with a single guessed token (it is accepted with the probability the model gives it). The guesser looks for the last `n` tokens (longest match first, up to 8) earlier in the prompt plus the reply so far and proposes the tokens that followed, at most `d` of them (`scripts/guess_prompts.sh` builds the prompts):

| class | prompts | what they are |
|---|---|---|
| chat | 12 | general questions, no context (51 to 57 prompt tokens) |
| chat-doc | 6 | summarize or answer from a document (README, benchmarks, design notes; 0.6 to 4.2k tokens) |
| code-write | 8 | write a function or script from a description (56 to 75 prompt tokens) |
| code-edit | 6 | a source file of this repository plus "add a method / add docs, show the complete file" (1.4 to 3.0k tokens) |
| code-explain | 4 | explain or review one source file (1.4 to 3.0k tokens) |
| long | 4 | 2.3 to 3.6k token documents or files, summarize or document |

Pass times come from `marin-run pass-cost`: for each size it reads the context again, runs passes of that size for 150 ms so the power-capped GPU settles, then times 6 more; 5 rounds, sizes rotated, median. A speculative pass of `1+d` tokens is priced from these curves at the nearest of 128, 1024 and 4096 tokens of context. On top of the pass, every extra token in the pass is charged 0.15 ms for its logits (copying 0.1 ms, measured; one more output-head row 0.05 ms, an estimate) and every settled token 0.15 ms for choosing it. The guesser's own measured time is added. The baseline is the same tokens one per pass.

**How often the guesses are right** (BF16, greedy, guess after 3 matching tokens, up to 4 tokens). "guess" is the share of passes that had a guess; "right@k" is how often the first k guessed tokens were all right, among passes that guessed at least k:

| class | tokens | guess | right@1 | right@2 | right@4 | tokens settled per pass |
|---|---|---|---|---|---|---|
| chat | 3059 | 6% | 56% | 38% | 14% | 1.08 |
| chat-doc | 1518 | 15% | 51% | 32% | 18% | 1.18 |
| code-write | 2682 | 9% | 56% | 38% | 18% | 1.12 |
| code-explain | 1547 | 14% | 50% | 36% | 20% | 1.17 |
| long | 1596 | 20% | 53% | 36% | 20% | 1.26 |
| **code-edit** | 2394 | 60% | 92% | 89% | 83% | **3.10** |

With the chat sampler the picture is the same (code-edit 3.63 tokens per pass, the others 1.06 to 1.18). Copying only pays when the reply repeats text that is already in view, as when the model writes out a file it was given. It almost never does in a conversation or when writing new code. With a looser rule (guess after 2 matching tokens) only 44 to 53% of first guesses are right in those classes, too few to pay for a longer pass.

**What a longer pass costs** (time compared with a one-token pass, same session and same weights; at 128, 1024 and 4096 tokens of context):

| tokens in the pass | bf16 | int8 |
|---|---|---|
| 2 | 1.19 / 1.12 / 1.57 | 1.42 / 1.46 / 1.96 |
| 4 | 1.44 / 1.51 / 1.91 | 1.79 / 2.01 / 2.49 |
| 8 | 1.90 / 2.09 / 2.91 | 2.60 / 2.97 / 3.81 |

The one-token pass took 19.1, 22.3 and 30.9 ms for bf16 and 12.3, 14.8 and 22.9 ms for int8 in these runs (the GPU's clock was lower than in the sections above; only ratios within one run are used). Longer passes are dearer for int8 because its smaller weights leave less memory time to hide the extra arithmetic. At long contexts every extra token costs about 7 to 8 ms for either weight type: during a multi-token pass attention uses the old kernel, one block per (token, head), and each guessed token re-reads the whole key/value cache, 0.94 GB at 4096 tokens. Passes of 9 tokens cost much more than 8 (the matrix kernel works on tokens in tiles of 8, so a ninth token starts a second tile), so guesses should stop at 7. An early version of this measurement that rotated the sizes inside one round, with 12 and 16 token passes among them, measured 2 tokens at 1.4 to 2 times and was discarded: the heavy passes lowered the clock for the passes after them. A real decode loop mixes sizes, so its costs can land between the two.

**Projected speed-up** (replay cost model above, guess after 3 matching tokens, up to 4 tokens, greedy / chat sampler):

| class | bf16 | int8 |
|---|---|---|
| chat | 1.05x / 1.03x | 1.03x / 0.99x |
| chat-doc | 1.06x / 1.06x | 0.97x / 0.97x |
| code-write | 1.06x / 1.05x | 1.03x / 1.05x |
| code-explain | 1.02x / 1.01x | 0.94x / 0.97x |
| long | 1.03x / 0.95x | 0.90x / 0.90x |
| **code-edit** | **1.95x / 2.11x** | **1.57x / 1.64x** |
| all 40 prompts | 1.16x / 1.15x | 1.06x / 1.07x |

- Only the code-edit class gains, 1.6 to 2.1 times, and the gain is smaller for int8. Everything else is within a few percent of 1 and the long-context and int8 cases are slightly below it.
- The matching rule matters. Guessing after a single matching token turns the same chat classes into slowdowns (bf16 greedy 0.94, 0.81 and 0.81 for chat-doc, code-explain and long; int8 0.73, 0.63 and 0.61). A match of 3 or 4 tokens costs chat nothing. Longer guesses (7 tokens) add a little for code-edit (2.1x bf16, 1.7x int8) and lose more elsewhere.
- Long context eats the gain. The same code-edit guesses priced at 128, 1024 and 4096 tokens of context: bf16 2.30x, 2.19x, 1.81x; int8 1.86x, 1.75x, 1.46x. At 4096 tokens, bf16 chat, chat-doc and code-write drop to 1.0x. Speculation does not remove the attention cost, it multiplies it by the pass size until the multi-token attention shares one read of the key/value cache between the tokens of the pass.
- Overhead of the guesser is 0.6 to 0.8 microseconds per generated token (a decode step is 10 to 30 ms) and under 0.4 MB of memory for a 4k-token context.

Limits of this measurement: it is a replay with a cost table, not a decode loop, and the cost of getting logits for every position is partly estimated (tripling that estimate moves the code-edit result from 1.95x to 1.92x for bf16 and from 1.57x to 1.54x for int8). The corpus is 40 prompts of mostly whole-file rewriting; real agent traffic (tool calls, diffs, long edits of known text) was not sampled. A verification pass would use different kernels from the one-token decode path, so its logits agree to rounding rather than bit for bit, and a tie between two tokens could resolve differently from plain decoding.

Reproduce:

```bash
scripts/guess_prompts.sh /tmp/prompts
marin-run guess ~/models/Qwen3-1.7B --prompts /tmp/prompts --save replies-bf16.json
marin-run pass-cost ~/models/Qwen3-1.7B --text LONG_TEXT_FILE --sizes 1,2,3,4,5,6,8,9 --rounds 5 --warmup 0 --save costs-bf16.json
marin-run guess-report replies-bf16.json --costs costs-bf16.json --min-match 3 --draft 4
# the same with --quantize int8 on guess and pass-cost
```

## Guessing tokens ahead, end to end

The replay above projected a gain for code edits. This section runs it for real: an experimental decode loop (`marin-run spec-bench`, GPU only, off everywhere else) guesses tokens by copying from the prompt and the reply so far, feeds the last token plus the guesses in one pass that returns the logits for every position, keeps the guesses the model agrees with and rolls the KV cache back over the rest. Greedy decoding; Qwen3-1.7B chat template, `marin` system prompt.

**Method.** 16 prompts in 8 scenarios, 2 prompts each: whole-file edits ("add a method, show the complete file"), diff-style edits ("reply with only a unified diff"), fresh code, and chat, each at a short context (52 to 462 prompt tokens) and a long one (3.6 to 3.9k tokens; `scripts/spec_prompts.sh`). Each prompt is read once, then every mode decodes 300 tokens from that same state (the cache is rolled back to the end of the prompt between modes), so prompt time is the same for all modes and left out of the speeds. Modes are interleaved, in a different order each round, after one thrown-away round, then 2 timed rounds. GPU power is read by `nvidia-smi` around each decode and energy per token is mean power times decode time over tokens. `same` counts replies identical to plain decoding's: all 4 of 4 in every row of every run below.

**Final rule: guess only after 8 matching tokens, then guess 7 tokens** (decode speed in tokens per second, then the gain; `tok/pass` is tokens settled per model pass; J/token is energy per token):

| scenario | bf16 plain | bf16 guessing | gain | int8 plain | int8 guessing | gain | tok/pass (bf16 / int8) | J/token (bf16 / int8) |
|---|---|---|---|---|---|---|---|---|
| edit, whole file, short | 57.5 | 120.5 | **2.10x** | 92.6 | 148.0 | **1.60x** | 3.21 / 3.21 | -52% / -38% |
| edit, whole file, 4k | 39.5 | 68.7 | **1.74x** | 53.9 | 78.0 | **1.45x** | 3.68 / 3.82 | -42% / -31% |
| edit, diff, short | 57.2 | 86.7 | **1.51x** | 92.4 | 146.4 | **1.59x** | 1.89 / 3.21 | -31% / -37% |
| edit, diff, 4k | 39.4 | 70.1 | **1.78x** | 53.7 | 60.4 | **1.12x** | 3.77 / 1.70 | -44% / -11% |
| fresh code, short | 60.6 | 61.6 | 1.02x | 101.9 | 106.1 | 1.04x | 1.05 / 1.09 | -2% / -4% |
| fresh code, 4k | 39.2 | 40.2 | 1.03x | 53.8 | 53.3 | 0.99x | 1.08 / 1.10 | -3% / +1% |
| chat, short | 61.1 | 61.3 | 1.00x | 102.3 | 101.5 | 0.99x | 1.01 / 1.02 | 0% / +1% |
| chat, 4k | 40.7 | 40.4 | 0.99x | 55.7 | 54.3 | 0.97x | 1.00 / 1.00 | +1% / +3% |

Prompt processing is unchanged (the pass over the prompt is the ordinary one). Output tokens are identical to plain decoding in every run.

- Where the prompt or the reply copies text that is already in view, it is a real win, close to what the replay projected (replay for whole-file edits: bf16 1.95x, int8 1.57x). It is smaller on int8 and at 4k context, for the reasons found in the pass-cost tables: a longer pass costs relatively more with int8's cheaper weights, and each guessed token re-reads the whole key/value cache. Where nothing is copied the result is within 0.97x to 1.04x. The few slowdowns are 1 to 3%.
- The 40 W cap holds in every run (39.7 to 40.0 W mean). The GPU does not use less power while guessing; it finishes sooner, so energy per token falls with the speed-up.
- The match rule decides whether this is safe. Guessing after only 3 matching tokens (up to 7 guessed) on int8 loses 25% on fresh code at 4k, 13% on chat at 4k and 5% on short chat; on bf16 15% and 9%. A failed 8-token pass at 4k context takes about 100 ms against 20 ms for a normal step. Making the guess length depend on the match (at most one token fewer than matched) and the match at least 5 tokens fixed most of it (int8 0.96x to 1.05x); 8 matching tokens was the safest and gave up none of the whole-file gains. Pausing guesses for a while after wrong ones (tried with 3 to 5 matching tokens) did not help and was removed.
- Cost of the machinery, per generated token: the guesser takes 1.0 to 1.7 microseconds at short context and 4.3 to 5.6 at 4k (most of it building the index over the prompt, once), the cache rollback 0.0 to 0.05 microseconds (it is only a length), against 11 to 28 ms per decode step. The index and history take under 0.4 MB. Verification passes keep a workspace per size on the GPU (about 25 MB at most), and copy each verified position's logits (608 KB) to the host.
- Latency: time to first token is unchanged. Tokens now arrive in bursts: in the edit scenarios the slowest 5% of passes take 45 to 105 ms against 11 to 28 ms for a plain step, but those passes carry 3 to 4 tokens on average. Where nothing is copied the pass times are those of plain decoding.

Limits: 2 timed rounds of 300 tokens on 16 prompts, one model, greedy only (the sampled path, which the loop supports by sampling each position and comparing, was not timed), and the prompts are whole-file or diff edits of files from this repository. Per-run ranges are in the saved files; in the whole-file int8 rows they are within 10 percent of the median. Power was read with one-second averaging around runs of 3 to 6 seconds, so energy is only as exact as that.

That experiment ran a loop kept in the benchmark tool; it has since been replaced by the engine's own path (next section), which reproduced it.

## Guessing tokens ahead in the engine

The guessing loop now lives in the `runtime` crate (`generate` with `GenerateParams::speculation`). It is off unless asked for: `marin-run MODEL --speculate`, or `marin-run spec-bench`. `marin` never turns it on. It needs the GPU; on the CPU the request is ignored and decoding is plain. Rule: index the prompt and the reply so far by every run of 8 tokens; when the last 8 tokens exactly repeat an earlier run, guess the (at most 7) tokens that followed it, feed the last token plus the guesses in one pass that returns the logits of every position, take from the first row on while each guess equals the token the sampler picks (the first row that differs supplies the next token), and roll the KV cache back over the guesses that were rejected. The sampler is called once per token delivered, in order, so seeded sampling draws the same random numbers as plain decoding.

`marin-run spec-bench` calls the same `generate` for four modes (greedy and the chat sampler, each plain and guessing), each time from the start of the prompt, interleaved and in a different order each round after one thrown-away round, then 2 timed rounds. 12 prompts in 8 scenarios (`scripts/spec_prompts.sh`; one prompt per 4k scenario to keep prefill time down), up to 300 tokens each. Decode speed excludes the prompt pass; energy is mean GPU power from the first token to the end times decode time over tokens. Every guessing reply was identical to the plain one with the same seed: `same` is full for all 96 guessing replies (48 per weight type), sampled replies included.

decode tokens/s plain -> guessing, gain over plain, tokens settled per model pass, energy per token change:

| scenario | bf16 greedy | bf16 chat sampler | int8 greedy | int8 chat sampler |
|---|---|---|---|---|
| edit, whole file, short | 46.8 -> 107.9, **2.30x**, 3.20, -57% | 48.7 -> 91.2, 1.87x, 3.20, -47% | 83.0 -> 143.2, **1.73x**, 3.20, -42% | 79.5 -> 113.2, 1.42x, 3.23, -31% |
| edit, whole file, 4k | 32.1 -> 58.3, 1.81x, 3.36, -45% | 32.1 -> 56.1, 1.75x, 3.74, -43% | 43.4 -> 68.9, 1.59x, 3.40, -37% | 46.0 -> 62.2, 1.35x, 3.48, -27% |
| edit, diff, short | 48.3 -> 76.8, 1.59x, 1.89, -34% | 49.7 -> 79.8, 1.60x, 2.24, -28% | 76.5 -> 135.9, 1.78x, 3.20, -44% | 77.3 -> 96.3, 1.25x, 2.07, -15% |
| edit, diff, 4k | 32.1 -> 52.7, 1.64x, 2.69, -39% | 32.3 -> 50.7, 1.57x, 2.69, -37% | 43.1 -> 62.1, 1.44x, 2.53, -31% | 44.4 -> 56.5, 1.27x, 2.53, -22% |
| fresh code, short | 49.4 -> 50.2, 1.02x, 1.05, -2% | 50.9 -> 51.5, 1.01x, 1.01, -1% | 90.3 -> 94.9, 1.05x, 1.09, -5% | 87.1 -> 86.3, 0.99x, 1.05, +1% |
| fresh code, 4k | 33.6 -> 36.1, 1.07x, 1.13, -7% | 34.7 -> 58.7, 1.69x, 3.44, -41% | 51.6 -> 53.1, 1.03x, 1.15, -3% | 49.9 -> 51.4, 1.03x, 1.08, -3% |
| chat, short | 49.8 -> 50.0, 1.00x, 1.01, 0% | 51.9 -> 52.2, 1.01x, 1.01, -1% | 83.1 -> 83.1, 1.00x, 1.01, 0% | 84.3 -> 84.5, 1.00x, 1.02, 0% |
| chat, 4k | 33.8 -> 33.7, 1.00x, 0.99, 0% | 34.4 -> 34.8, 1.01x, 0.99, -1% | 46.9 -> 45.5, 0.97x, 0.99, +3% | 48.7 -> 46.7, 0.96x, 0.99, +4% |

(Plain decode speeds are lower than in the sections above because the GPU's clock was lower in these sessions; compare only within a column.)

- Edits gain 1.25x to 2.30x on every row, in both weight types and both samplers; the gain is smaller at 4k context (each guessed token re-reads the whole key/value cache) and for int8 (its cheaper weights leave less time to hide the extra arithmetic). The 4k fresh-code row with the chat sampler gained 1.69x because that sampled reply happened to copy the reference code it was shown; greedy did not (1.07x).
- Outside edits the result is within 0.99x to 1.07x except int8 chat at 4k: 0.97x greedy and 0.96x sampled (3 to 4% slower, 3 to 4% more energy). There, repeated 8-token runs in a long document are followed by different text, and a wrong 8-token pass at 4k context costs about 5 plain steps. 1 to 2 of its roughly 160 passes were guesses and none was right.
- The chat sampler's gain is smaller than greedy's on the same tokens per pass (whole-file short: 3.20 for both, 2.30x against 1.87x on bf16): choosing a token on the CPU takes about 3.7 ms with top-k and top-p, a per-token cost guessing does not remove. Faster sampling would raise these gains; it was not touched.
- GPU power stayed at the 40 W cap (38.9 to 40.1 W mean in every run), so energy per token follows speed: down 31% to 57% on the whole-file edits for greedy, 15% to 47% sampled.
- Overhead per generated token: finding matches and indexing 0.4 to 0.6 microseconds at short context and 1.0 to 1.9 at 4k (the prompt index is built once), rollback 0.00 to 0.02 microseconds, against 11 to 30 ms for a decode step. GPU memory in use is unchanged at the 10 MB resolution the tool prints (the per-size workspaces for the checking passes are about 25 MB at most).
- Latency: time to first token is unchanged. Tokens settled by one pass arrive together, so the 95th percentile gap between tokens is longer where guessing is used (21 to 111 ms in the edit scenarios against 12 to 35 ms for plain decoding) while the median is near zero.

Limits: 2 timed rounds of at most 300 tokens, 12 prompts, one model; whole-file and diff edits of files from this repository; per-run ranges are in the saved files and some are wide (diff-style short edits: 0.98x to 3.00x per run). Power was read around runs of 2 to 15 seconds with one-second averaging.

Reproduce:

```bash
scripts/spec_prompts.sh /tmp/specprompts
marin-run spec-bench ~/models/Qwen3-1.7B --prompts /tmp/specprompts --rounds 2
marin-run spec-bench ~/models/Qwen3-1.7B --quantize int8 --prompts /tmp/specprompts --rounds 2
marin-run ~/models/Qwen3-1.7B-Base --speculate -p "TEXT" --metrics   # one reply, with what guessing did
```

## Shared-read attention for guessed tokens

Checking guessed tokens ran the attention kernel built for prompts: one block per (new token, query head), each scanning the whole context alone. At 4096 tokens of context every guessed token therefore re-read the key/value cache, and every query head read its own copy of the rows it shares with its neighbour. `marin-run pass-cost --stages` profiles a pass of each size stage by stage (profiled passes, steady GPU, median of 3 rounds of 6, same session for each pair of columns).

**Profile before the change** (key/value cache is f32, 28 layers, 8 kv heads, 16 query heads, 0.94 GB held at 4096 tokens):

| tokens in the pass | attention at 4096: bf16 | int8 | per token: bf16 | KV bytes the kernel asks for | whole pass: bf16 | int8 |
|---|---|---|---|---|---|---|
| 1 (decode kernel) | 15.9 ms | 13.9 ms | 15.9 ms | 0.94 GB | 31.6 ms | 22.8 ms |
| 2 | 37.4 ms | 40.0 ms | 18.7 ms | 3.8 GB | 49.9 ms | 50.8 ms |
| 4 | 44.6 ms | 44.9 ms | 11.2 ms | 7.5 GB | 60.3 ms | 57.8 ms |
| 7 | 58.7 ms | 60.8 ms | 8.4 ms | 13.2 GB | 83.8 ms | 80.2 ms |
| 8 | 65.3 ms | 66.4 ms | 8.2 ms | 15.0 GB | 92.1 ms | 86.0 ms |

- Attention was 71% (bf16) and 77% (int8) of an 8-token pass at 4096 tokens, which made the pass 2.9x (bf16) or 3.8x (int8) a normal step. At 128 tokens of context attention is 3 to 4 ms of 34 to 38 and none of this matters.
- A 2-token pass was the worst per token (19 to 20 ms per token): it left the split decode kernel for one with only 32 blocks on a 20-multiprocessor GPU. The kernel asked for 15 GB for 8 tokens against 0.94 GB held, 230 GB/s, so repeated reads mostly went to memory, not cache.
- A rejected guess costs the same as an accepted one, so it wastes the pass minus one normal step: 61 ms for 8 tokens at 4096 on bf16, 63 ms on int8, against 18 ms at 128 tokens.

**The change.** For 2 to 8 new tokens (when the rows, tokens times query heads sharing a kv head, number at most 32) a new kernel splits the context into blocks of 64 positions like the decode kernel does. Each block holds the queries of all the new tokens for one kv head in shared memory and reads each key row and value row once, scoring and weighting it for every (token, head) row; row `t` only sees positions up to `start + t`. A row that sees nothing in a chunk contributes nothing to the merge. The existing merge kernel combines the partial softmaxes, one block per row. The kernel is built for 8, 16 and 32 rows so a small pass does not pay for loops sized for many (that alone took the 2-token pass from 24 to 14 ms). One-token decode, prompt passes (more than 8 tokens) and the tensor-core prefill are unchanged; so are the cache layout and rollback. Tests compare every size from 2 to 8 tokens, contexts crossing the block size, and head layouts with 1, 2 and 4 query heads per kv head with an f64 reference (1e-4), and break when the mask is wrong.

**Profile after** (same session as the before-and-after columns):

| tokens in the pass | attention at 4096: bf16 | int8 | per token: bf16 | whole pass: bf16 | int8 | wasted if rejected: bf16 | int8 |
|---|---|---|---|---|---|---|---|
| 1 | 15.7 ms | 13.9 ms | 15.7 ms | 32.7 ms | 23.5 ms | 0 | 0 |
| 2 | 14.5 ms | 13.1 ms | 7.2 ms | 35.6 ms | 30.9 ms | 2.9 ms (was 18.3) | 7.4 ms (was 28.0) |
| 4 | 13.7 ms | 12.4 ms | 3.4 ms | 44.2 ms | 38.5 ms | 11.5 ms (28.7) | 14.9 ms (35.0) |
| 7 | 21.6 ms | 19.5 ms | 3.1 ms | 58.2 ms | 52.2 ms | 25.5 ms (52.2) | 28.6 ms (57.5) |
| 8 | 21.6 ms | 20.9 ms | 2.7 ms | 60.0 ms | 55.9 ms | 27.3 ms (60.5) | 32.3 ms (63.2) |

- Attention for 2 to 4 tokens now costs about what one token does, and for 8 tokens 1.4 to 1.5 times; per token it fell from 8 to 3 ms (bf16, 7 to 8 tokens). Each key and value row is read once per pass: 0.94 GB plus about 0.24 GB of partial results written and read again, against 15 GB. That is still only about 55 GB/s for the 8-token pass, well under the 300 GB/s the memory can do, so the kernel is bound by its own work (the dot products and shuffles for each row) and not by memory; there is room left.
- Whole passes of 2 to 8 tokens cost 27 to 39% less at 4096 tokens and are unchanged at 128 (the weight stages dominate there). A whole 8-token pass is now 1.8x (bf16) and 2.4x (int8) a normal step instead of 2.9x and 3.8x.

**Real replies, old kernel against new kernel** (`marin-run spec-bench`, the production guessing path, 12 prompts, 2 timed rounds after a thrown-away one, both binaries run back to back on the same prompts). Gain over plain decoding with the same sampler, old -> new, then the energy per token against plain decoding and the 95th-percentile gap between tokens (guessing runs; plain decoding for comparison). Every guessing reply was identical to the plain reply with the same seed (`same` full for all 48 guessing replies per weight type).

| scenario | bf16 greedy | bf16 chat sampler | bf16 J/token, gap p95 (greedy) | int8 greedy | int8 chat sampler | int8 J/token, gap p95 (greedy) |
|---|---|---|---|---|---|---|
| edit, whole file, short | 2.31x -> **2.50x** | 1.95x -> 2.09x | -60%; 44.5 ms (was 50.4; plain 26.3) | 1.77x -> **1.99x** | 1.43x -> 1.56x | -50%; 39.7 ms (48.6; plain 16.3) |
| edit, whole file, 4k | 1.85x -> **2.46x** | 1.71x -> 2.25x | -59%; 66.4 ms (108.2; plain 36.4) | 1.56x -> **2.18x** | 1.36x -> 1.89x | -54%; 61.5 ms (105.3; plain 26.4) |
| edit, diff, short | 1.81x -> 1.91x | 1.44x -> 1.53x | -39%; 42.9 ms (48.9; plain 27.0) | 1.51x -> 1.64x | 1.24x -> 1.29x | -32%; 38.4 ms (46.3; plain 16.0) |
| edit, diff, 4k | 1.69x -> **2.13x** | 1.54x -> 1.86x | -53%; 65.5 ms (108.8; plain 36.3) | 1.40x -> **1.87x** | 1.26x -> 1.63x | -46%; 60.8 ms (110.1; plain 26.6) |
| fresh code, short | 1.02x -> 1.03x | 1.00x -> 1.01x | -3%; 25.5 ms (plain 24.9) | 1.05x -> 1.06x | 1.00x -> 1.00x | -5%; 15.1 ms (plain 15.4) |
| fresh code, 4k | 1.07x -> 1.10x | 1.69x -> 2.15x | -9%; 34.7 ms (plain 33.9) | 1.05x -> 1.10x | 1.01x -> 1.04x | -9%; 24.3 ms (plain 24.2) |
| chat, short | 1.01x -> 1.01x | 1.00x -> 1.00x | -1%; 24.6 ms (plain 25.1) | 1.01x -> 1.00x | 0.99x -> 1.00x | 0%; 15.0 ms (plain 14.6) |
| chat, 4k | 1.01x -> 1.00x | 0.99x -> 1.01x | 0%; 34.5 ms (plain 34.3) | 0.96x -> **0.98x** | 0.99x -> 0.99x | +2%; 24.9 ms (plain 24.5) |

- 4k edits improved the most: whole-file 1.85x -> 2.46x (bf16) and 1.56x -> 2.18x (int8); diff-style 1.69x -> 2.13x and 1.40x -> 1.87x. Short-context edits gained 5 to 12% too (the 2 to 8 token passes are a little cheaper there as well). Energy per token for 4k edits fell from -46%/-36% to -59%/-54% against plain decoding (GPU power stays at the 40 W cap, so it follows speed).
- The 95th-percentile gap between tokens in 4k edits fell from about 108 ms to 61 to 66 ms (plain decoding: 26 to 36 ms). What a rejected guess wastes fell by 57 to 84% for passes of 2 to 4 tokens and by about half for 7 to 8 tokens (table above).
- The int8 long-chat slowdown is smaller, not gone: 0.96x -> 0.98x greedy (energy +4% -> +2%), 0.99x sampled. What is left is the one or two wrong 8-token guesses per reply at 4k tokens, each still about 30 ms of wasted work, mostly in the weight stages. Nothing else outside edits moved by more than 1 to 2%: bf16 chat and fresh code are 1.00x to 1.03x greedy.
- Plain decoding was unchanged between the two binaries (4k plain bf16 30.0 against 30.4 tok/s, int8 40.9 against 41.2; short 44.1 against 44.0 and 69.7 against 70.7).

Limits: 2 timed rounds, 12 prompts; one model; the GPU's clock differs between sessions, so only ratios inside a pair of runs are compared. Prompt processing (prefill) attention was not changed.

Reproduce:

```bash
marin-run pass-cost ~/models/Qwen3-1.7B --text LONG_TEXT_FILE --contexts 128,4096 --sizes 1,2,4,7,8 --stages
marin-run pass-cost ~/models/Qwen3-1.7B --quantize int8 --text LONG_TEXT_FILE --contexts 128,4096 --sizes 1,2,4,7,8 --stages
marin-run spec-bench ~/models/Qwen3-1.7B --prompts /tmp/specprompts --rounds 2
```

## CPU results

| prompt tokens | first token | prefill | decode | time between tokens (p50 / p95) | sampling per token | weight bandwidth (est.) |
|---|---|---|---|---|---|---|
| 32 | 3855 ms | 8.3 tok/s | 3.72 tok/s | 264.2 / 281.7 ms | 3.8 ms | 12.8 GB/s |
| 128 | 15184 ms | 8.4 tok/s | 3.86 tok/s | 258.1 / 266.5 ms | 3.7 ms | 13.3 GB/s |
| 512 | 63459 ms | 8.1 tok/s | 3.67 tok/s | 272.5 / 275.6 ms | 3.7 ms | 12.6 GB/s |

Loading takes about 3 s and the process holds about 3.5 GB. Every run kept exactly one of the 16 cores busy (`cpu busy during runs 1.0 of 16`).

```bash
marin-run bench ~/models/Qwen3-1.7B-Base --backend cpu --prompt-tokens 128 --tokens 64 --runs 3 --profile --json cpu-p128.json
```

### Where the time goes (CPU, 128-token run)

| phase | MLP | q/k/v projections | attention output proj | LM head | attention |
|---|---|---|---|---|---|
| prefill | 74% | 16% | 8% | 0.3% | 0.7% |
| decode | 64% | 12% | 6% | 16% | 1.6% |

At 512 tokens attention grows to 3.4% of prefill. `marin-run trace` shows the same split layer by layer: all 28 layers cost the same, about 6.3 ms of MLP and 1.1 ms of q/k/v each in a one-token pass.

### What this suggests (CPU)

- The CPU path uses one core. Splitting the matrix multiplies across cores is the first thing to try.
- Prefill reads a prompt at only about twice the speed of decode (8.3 vs 3.8 tok/s). A matrix multiply that reuses each weight across many prompt tokens should be much faster than one that reads the weights again for every token.
- Decode reads the weights at an estimated 13 GB/s. That figure assumes every weight is read once per token. The memory's rated speed was not measured here, so it is not yet known how close this is to the limit.

## Comparing before and after

Save a result before a change and another after it, with identical options, then:

```bash
marin-run compare before.json after.json
```

It refuses results from different setups, and calls a change "same" when the old and new min..max ranges overlap, so noise is not reported as a win. With few runs on a very small model, a gap can still slip past that test, so take more `--runs` before believing a small difference.
