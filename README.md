# Marin

<img width="532" height="331" alt="image" src="https://github.com/user-attachments/assets/ad91e0dd-1ff3-424b-8d2b-91652ac77165" />

Marin is a small language-model engine written from scratch in Rust, built to run well on ordinary laptops and desktops. No Python, no outside inference library. The goal is to understand every step of running a model, measure it, and make it faster on weak hardware. A chat assistant on top gives the engine real work to be judged by.

- **Models:** Qwen3 on CPU/CUDA; Qwen3.5-2B-Base and post-trained Qwen3.5-4B text inference on CPU/CUDA with FP32 hybrid state.
- **Checked against Hugging Face**, token for token.
- **Fast where it matters:** tensor-core prefill, tuned attention, int8 / int4 weights, a chat cache that survives between turns, a KV cache that grows with the conversation (stored as f16 on the GPU), prefix snapshots across restarts, opt-in guessing ahead, and SIMD matmul kernels on the CPU.
- **Measured:** every optimization has before/after numbers, and the commands to repeat them, in [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

You need Rust (the version is pinned in `rust-toolchain.toml`) and a model folder from Hugging Face (`config.json`, `tokenizer.json`, `model.safetensors`). The GPU path also needs an NVIDIA GPU and CUDA.

```bash
cargo install --path crates/cli --locked
```

| Program | For |
|---|---|
| `marin` | the chat assistant |
| `marin-run` | the low-level tool: run base models, check correctness, measure |

```bash
marin --model ~/models/Qwen3-1.7B        # chat, type / for actions
marin run "explain a mutex in one line"  # one answer, then exit
marin-run MODEL -p "The capital of France is"
```

Marin uses the GPU when CUDA works and the model fits, and the CPU otherwise; `--backend cpu` or `--backend cuda` forces a choice.

### Plan before loading

```bash
marin --model ~/models/Qwen3.5-4B-Q8 --quantize int8 plan
marin-run plan ~/models/Qwen3.5-4B-Q8 --quantize int8 --ctx 2048
```

`plan` reads model metadata and hardware limits, then shows the selected
backend, artifact format, CPU threads, context and memory budget without
loading weight tensors. `marin` uses its 4096-token chat default;
`marin-run` uses 2048, so pass `--ctx` when comparing either dry run with
a command. Automatic CUDA selection requires the full context window to
fit; explicit CUDA still reports when only startup fits. If no context was
specified, a CPU default may be reduced to fit and the reduction is printed;
an explicit `--ctx` is never reduced. The named folder remains the artifact:
INT8 requires explicit `--quantize int8`, and a packed folder is never
silently substituted for a BF16 folder. The dry run checks metadata and
budgets; inference verifies packed tensor payloads while loading.
[Planner gates and constrained-hardware results](BENCHMARKS.md#40-minimal-v9-execution-planner-gates-fixed-before-implementation).

### Options worth knowing

| Option | What it does |
|---|---|
| `--quantize int8` | smaller weights on CPU and GPU (`marin-run` also takes GPU `int4`). CPU INT8 uses grouped signed bytes with f32 scales and fused SIMD computation; `--quant-group` defaults to 128. On the GPU: decode 33–63% faster, about 1.7 GB less memory, same first-token time. Rounding can change wording; bf16 stays the default. [CPU gates and results](BENCHMARKS.md#25-cpu-int8-weights-stage-3), [GPU results](BENCHMARKS.md#16-int8-weights-in-chat) |
| `--kv-type f32\|f16` | KV cache precision. `marin` uses f16 on the GPU (half the memory, twice the context, about half the first-token time at 8k–16k) and f32 on the CPU. `marin-run` and `verify` stay on f32. [Details](BENCHMARKS.md#12-kv-cache-precision-f16) |
| `--snapshot FILE` | save the conversation's cache when the session ends and reuse the shared start next time. GPU only; nothing is saved unless you ask. [Details](BENCHMARKS.md#13-prefix-snapshot) |
| `--threads N` | CPU threads for prompt passes (default: physical cores capped by affinity/cgroup availability; explicit counts override this, `1` is the single-thread path). Decode stays on one thread: memory bandwidth, not arithmetic, limits it. Output is bit-identical for any thread count. [Details](BENCHMARKS.md#24-cpu-threads-for-prompts-stage-2) |
| `--ctx N` | the longest conversation allowed (not a reservation, see below) |
| `--speculate` (`marin-run`) | guess ahead by copying repeated text. [Details](BENCHMARKS.md#5-guessing-ahead-opt-in-gpu-only) |

### Load packed INT8 directly

Export once, then explicitly select INT8 on either backend:

```bash
marin-run export-int8 ~/models/Qwen3-1.7B ~/models/Qwen3-1.7B-int8
marin --model ~/models/Qwen3-1.7B-int8 --quantize int8
marin-run ~/models/Qwen3-1.7B-int8 --backend cpu --quantize int8 -p "Hello"
```

The standalone bundle stores existing Q8 codes/scales and FP32 norms, with
versioned metadata and SHA-256 verification during loading. It never loads
BF16 weights or requantizes on a direct load. Export refuses existing output
folders; keep published bundles immutable. Non-default groups need the same
`--quant-group` at export and load. BF16 remains the default for HF folders;
a packed folder without explicit INT8 fails clearly. GPU snapshots work
within the same artifact identity; HF-source snapshots are conservatively
incompatible. CPU snapshots remain unsupported.

[Measured results, gates and limitations](BENCHMARKS.md#32-direct-packed-int8-loading-gates-fixed-before-implementation).
On Qwen3-0.6B/1.7B, direct loading cuts measured warm time to first output
by **35.7% / 36.2%**, fadvise-cold by **47.0% / 46.2%**, and load peak RSS
by about **30 / 63 MB**, with identical INT8 values and outputs. Bundles are
**626 MB / 1.786 GB** including tokenizer/metadata. Focused inference timing
gates pass. [Acceptance and limitations](BENCHMARKS.md#acceptance).

### Qwen3.5-2B-Base text inference

```bash
marin-run ~/models/Qwen3.5-2B-Base --backend cpu --threads 8 --ctx 128 \
  --temp 0 --max-tokens 8 --metrics -p "The capital of France is"
```

The complete 24-layer hybrid backbone retains official BF16 matrices by default and
uses FP32 math, recurrent/convolution state and growable attention KV.
It matches HF's FP32 layer/logit reference and greedy tokens on real prose
and code prompts. Measured short-prompt peak RAM is about **3.90 GB**,
TTFT **0.55–0.66 s**, and decode **5.9–6.1 tok/s** on the reference CPU.
[Exact gates, measurements and limitations](BENCHMARKS.md#35-end-to-end-qwen35-2b-text-inference).

BF16 remains default; opt-in CPU INT8 reuses the same Q8 kernels
and packed loader as Qwen3. On the reference CPU, short-prompt decode rises
from **5.52 to 9.30 tok/s**, peak RSS falls from **3.90 to 2.08 GB**, and warm
load falls from **3.86 to 1.90 s**. Convolution, norms, decay/beta projections,
recurrent state and attention KV keep FP32 values/math. Vision, MoE and untied
heads remain unsupported.

The BF16 text checkpoint also runs on CUDA with device-resident recurrent,
convolution and attention state. On the RTX 5050, strict HF verification
passes at 128/512 tokens (worst normalized max **3.055e-6**, relative RMS
**2.639e-6**). Use `--backend cuda` with the same command above; an explicit
GPU request refuses a model that exceeds the available VRAM. The CUDA speed
optimization experiments remain deferred; 4B support is measured separately.
[CUDA measurements and limitations](BENCHMARKS.md#38-qwen35-cuda-and-4b-milestone-in-progress).

```bash
marin-run export-int8 ~/models/Qwen3.5-2B-Base ~/models/Qwen3.5-2B-Base-Q8 --quant-group 128
marin-run ~/models/Qwen3.5-2B-Base-Q8 --backend cpu --quantize int8 --quant-group 128 \
  --threads 8 --ctx 1152 --temp 0 --max-tokens 8 -p "The capital of France is"
```

The artifact requires matching opt-in INT8/group flags. Original BF16 folders
also accept `--quantize int8`; exporting once avoids conversion on subsequent
loads. Export refuses an existing destination and publishes atomically.
[Quality, strict matched-HF parity, 4 GiB evidence and limits](BENCHMARKS.md#37-qwen35-cpu-packed-int8-2026-10-13).

Prefill of at least 16 tokens batches dense projections and
MLP through existing CPU GEMM/threading, with sequential recurrent updates.
At 128/512 tokens, measured prefill rose from 8.1/8.2 to **90.6/93.5 tok/s**
and TTFT fell from 15.74/62.12 to **1.41/5.48 s**, with a **33 MiB** peak-RAM
increase at 512. Short passes retain GEMV. [Gates, HF parity and workload limits](BENCHMARKS.md#36-qwen35-cpu-batched-prefill-2026-10-13).
Appending reuses state; edited prefixes
reset and replay retained tokens. Whole-model errors leave committed state
unchanged. Qwen3.5 snapshots remain unsupported on either backend. Base models use `marin-run`;
the post-trained 4B model works with `marin`'s non-thinking chat template.
Qwen3 defaults and CPU/CUDA/INT8 behavior are retained.

### Qwen3.5-4B text chat on constrained hardware

The official post-trained 4B checkpoint has 32 hybrid text layers. CPU BF16
is the default and passes strict Hugging Face layer/logit and greedy-token
checks. Its 8.41 GB text weights exceed this RTX 5050's total VRAM, so an
explicit BF16 CUDA request refuses to load. Exporting once to Marin's existing
Q8 format makes the same text model usable on CPU and this 8 GB GPU:

```bash
marin-run export-int8 ~/models/Qwen3.5-4B ~/models/Qwen3.5-4B-Q8 --quant-group 128
marin --model ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 \
  --ctx 256 --max-tokens 160 --temp 0 run "Explain a mutex in one sentence."
marin-run ~/models/Qwen3.5-4B-Q8 --backend cpu --quantize int8 \
  --ctx 2053 --temp 0 -p "The capital of France is"
```

The packed folder needs explicit `--quantize int8`; its stored group size is
validated at load. There is no whole-model BF16 expansion at inference time.
A 2k prompt peaked
at **4.96 GB process RAM** on CPU and **4696 MiB process VRAM** on the
RTX 5050. CPU BF16 versus INT8 decode was **2.33 → 4.29 tok/s**, while
CUDA INT8 reached **38.03 tok/s**; 2k TTFT was **53.11 / 47.25 / 15.94 s**
respectively. The CPU Q8 run also completed inside an enforced 8 GiB,
zero-swap memory scope. On 256 matched prose/code positions, INT8 agreed with
BF16 on **255** first choices. The official non-thinking chat template and
assistant end marker are handled by `marin`; a generated factorial function
passed executable checks for zero, five and negative input. Vision and MTP
weights are skipped; Qwen3.5 snapshots remain unsupported. These are focused
measurements, with no matched llama.cpp 4B baseline or controlled cold-disk
startup claim. [Gates, HF parity and measurements](BENCHMARKS.md#39-real-qwen35-4b-text-milestone-gates-fixed-before-implementation).


### `marin-run` subcommands

| Command | What it answers |
|---|---|
| `export-int8 MODEL PACKED` | Convert BF16 once to a standalone packed INT8 bundle |
| `inspect MODEL` | How big is this model and how much memory will it need? |
| `verify MODEL --reference FILE` | Does the output match Hugging Face? |
| `bench MODEL [--profile] [--sample-gpu]` | How fast is it, and where does the time go? |
| `trace MODEL` | How long does each stage of each layer take? |
| `compare before.json after.json` | Did the change help, or is it noise? |
| `probe` | What can this GPU do on its own (memory and math speed)? |
| `quality MODEL --text FILE` | How much accuracy does quantization or a KV format cost? |
| `turns MODEL` | How much sooner is a chat turn's first token when the cache is kept? |
| `restart MODEL --mode save\|restore\|cold` | How much sooner is the first token after a restart with a snapshot? |
| `spec-bench`, `guess`, `guess-report`, `pass-cost` | Guessing-ahead experiments |

## How it works

A model turns text into token numbers, runs them through a stack of identical layers, and picks the next token. Marin does each step itself.

1. **Tokenizer** splits text into tokens.
2. **Layers** normalize, run attention, then an MLP. Attention looks back at earlier tokens, whose keys and values live in the **KV cache** so past work is never repeated.
3. **Sampler** picks the next token from the model's scores.

Reading the prompt is **prefill** (big passes, limited by arithmetic and, for long prompts, attention). Writing the answer is **decode**, one token per pass, limited by how fast weights can be read from memory, which is why smaller weights decode faster.

### Features

- **Chat cache kept between turns.** Marin remembers which token ids its KV cache holds, finds the longest exactly shared start on the next turn, cuts the cache back to it, and reads only what is new. The record is written only after a reply ends cleanly and cleared on any error. After 4,000 tokens of history the first token arrives in about 0.25 s instead of 9 s, with identical replies. [Details](BENCHMARKS.md#10-keeping-the-cache-between-chat-turns)

  ```text
  cached : A B C D E F
  turn 2 : A B C D E F G H I   -> keep 6, read G H I
  edited : A B C X Y           -> keep 3, read X Y
  ```

- **KV cache that grows (GPU).** The cache is kept in pages of 64 positions (14 MiB for Qwen3-1.7B), allocated eight at a time as the conversation needs them. A short chat with `--ctx 8192` holds about 117 MB of KV instead of 1.88 GB, at a cost of 0–4% speed at long contexts. A conversation that fills the GPU stops with a clear out-of-memory error. [Details](BENCHMARKS.md#11-paged-kv-cache)
- **Snapshots across restarts (GPU, opt-in).** After 4,000 tokens the first token arrives in 0.2–0.4 s instead of 6.5–7 s. The file is checked against the exact model, tokenizer, KV type and weight format; anything wrong with it means the prompt is simply read in full. [Details](BENCHMARKS.md#13-prefix-snapshot)
- **Guessing ahead (GPU, opt-in).** When a reply repeats text it was just shown, Marin guesses up to 7 tokens by copying, checks them in one pass, and rolls back wrong guesses, so the output equals normal decoding. Code edits run 1.5–2.5x faster; chat and fresh code are unchanged.

  ```bash
  marin-run MODEL --speculate -p "..."
  ```

- **CPU kernels and threads.** CPU weights use bf16 pairs by default, or opt-in grouped INT8, with run-time-dispatched AVX2 + FMA kernels: a register-blocked GEMM for prompts and a streaming GEMV for decode, plus vector attention. INT8 decode fuses dequantization; prefill expands only small panels per worker. Prompt passes use threads (physical cores capped by affinity/cgroup availability by default); decode stays on one thread. CPUs without AVX2 retain scalar computation and threaded prefill. Both `marin` and `marin-run` accept `--backend cpu --quantize int8`; CPU INT4 is refused. [Matmul](BENCHMARKS.md#22-cpu-matmul-kernels-stage-1-one-thread), [attention](BENCHMARKS.md#23-cpu-attention-and-rope-one-thread), [threads](BENCHMARKS.md#24-cpu-threads-for-prompts-stage-2), [INT8](BENCHMARKS.md#25-cpu-int8-weights-stage-3)

## Speed

Qwen3-1.7B on a laptop GPU held to about 40 W (the power limit decides a lot, so every change is checked end to end). All numbers are before/after on the same machine.

| What | Result |
|---|---|
| int8 vs bf16 decode | 65% faster, half the weight memory |
| Prompt, first token (f16 cache, tensor-core prefill attention) | 4,000 tokens: 8.0 s → 2.8 s (energy 318 → 109 J); 8,000 tokens: 25 s → 6 s |
| Decode with 16-byte cache loads (f16 cache) | +13 / +34 / +65 / +95% (bf16) at 2k / 4k / 8k / 16k tokens of context |
| Chat turn after 4.1k tokens of history | first token 35x (bf16) / 37x (int8) sooner with the kept cache |
| Short chat, `--ctx 8192` | 117 MB of KV instead of 1.88 GB; GPU peak 1.8 GB lower |
| f16 KV cache | KV memory halved; 32k context fits (f32: 16k); first token about half at 8k–16k |
| Restart after 4,000 tokens, `--snapshot` | first token 0.2–0.4 s instead of 6.5–7 s |

On the CPU (Ryzen 7 260, 8 cores), Qwen3-0.6B: on **one thread** a 127-token prompt reads at 109 tok/s (was 26) and decode runs at 19 tok/s (was 11); on **8 threads** prompts read at 400–600 tok/s (a 1,023-token prompt: first token 2.1 s, was 49 s). Qwen3-1.7B: prefill 7.7 → 34 tok/s on one thread, 155 on 8; decode 3.6 → 6.7. On one thread llama.cpp reaches 210 / 18.8 (0.6B, AVX-512) and 102 / 19.4 (AVX2-only build); on 8 threads 1,061 / 19.8.

CPU INT8 (group 128, three measured rounds after a warm-up) raises one-thread decode at 127 tokens from **18.5 to 34.5 tok/s on 0.6B** and **6.1 to 11.4 on 1.7B**. Peak RAM falls **1.30 to 0.73 GB** / **3.56 to 1.90 GB**. The 1/8-thread, 127/2,047-token matrix passes its speed and RAM gates; weights are 51.6% of BF16. HF BF16 files are converted at load time; the optional standalone packed artifact avoids repeating conversion. Relative quality gates pass on prose/code (97.559–98.730% top-1 agreement), with no new semantic regression in eight fixed prompts per model.

CPU Stage 3 passes its original gates on both Qwen3 models, including strict HF parity and RTX 5050 CUDA/snapshot regressions. INT8 stays opt-in and BF16 stays the default. Stage 4 growable FP32 CPU KV is **GO**. Bounded prefill is **NO-GO and removed**; whole-prompt workspace budgeting and RAM safety checks remain. The original +77% TTFT failure was not reproduced conclusively, and final focused checks retain two narrow timing misses. Complete Stage 4 remains **NO-GO**. A bounded CPU INT4 study found faster external Q4 decode but failed pilot quality, with higher long-context RSS and slower 0.6B prefill; native CPU INT4 is **NO-GO for now**. FP16 KV and native INT4 are not implemented. [Study, memory audit and next steps](BENCHMARKS.md#28-stage-4-closure-and-stage-5-cpu-int4-feasibility-2026-10-12). [Final fallback evidence and limitations](BENCHMARKS.md#27-stage-4-bounded-prefill-regression-investigation). [Stage 3 acceptance](BENCHMARKS.md#25-cpu-int8-weights-stage-3).

Methods, tables and commands to repeat each number are in [BENCHMARKS.md](BENCHMARKS.md).

## Repository layout

| Crate | What it does |
|---|---|
| `tensor` | math on CPU data: SIMD matmul, norms, rotary embeddings, quantization |
| `loader` | reads the config, tokenizer and safetensors weights |
| `model` | the Qwen3 network on CPU and GPU |
| `cuda` | GPU kernels and the paged KV cache behind a safe API |
| `kv` | the CPU KV cache |
| `sampler` | greedy, temperature, top-k, top-p |
| `runtime` | the generate loop: prefill, decode, stop conditions, cache reuse, guessing ahead |
| `chat` | chat formatting and conversation history |
| `metrics` | timers, memory, CPU and GPU readings |
| `cli` | the `marin` and `marin-run` programs and the terminal interface |

`tests/fixtures` holds tiny models so the test suite runs without downloading weights; `tests/reference` holds Hugging Face reference output.

## Correctness

Output is compared with Hugging Face in float32: same tokens, logits within about 3e-5, on the CPU and on the GPU. Quantized models and the f16 cache cannot match that exactly, so they are checked against their own higher-precision version with written pass/fail gates (see BENCHMARKS.md). Every kernel change is tested against a plain reference. Kept-cache, snapshot and guessing-ahead output must equal a fresh read token for token, and the tests break the logic on purpose to prove they would notice.

## Development

```bash
cargo fmt --all
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
```
