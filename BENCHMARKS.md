# Marin performance and correctness

This is the living summary. It records accepted behavior, comparable results,
and open limits; the dated experiments, fixed gates, failed candidates, raw
numeric traces and reproduction details remain intact in the [archive](docs/benchmarks/archive/INDEX.log).
The original 5,752 lines were preserved byte-for-byte across four files.
Update this page for new conclusions and archive long measurement logs.

## Scope and method

Development machine: Ryzen 7 260, 16 GB RAM, RTX 5050 Laptop GPU with 8 GB
VRAM and an approximately 40 W operating limit. Early GPU experiments used
slightly different reported device labels and power conditions; their numbers
are historical, not interchangeable with the recent RTX 5050 runs. Unless
specified, rates are tokens/second, TTFT includes prefill and the first
sample, and RSS is process peak. `nvidia-smi` device-wide VRAM includes other
programs; process VRAM is named explicitly. BF16 is the default weight format.
INT8 always requires `--quantize int8`; packed artifacts require a matching
group size (default 128). CPU KV defaults to FP32; Qwen3 GPU chat defaults to
FP16 KV, while Qwen3.5 and Granite use FP32 state.

Most historical speed rows have one to five runs and warm OS file cache.
Only matched runs in the same session support a before/after percentage.
The 40 W cap changes clocks as work changes. Estimates and simulated budgets
are labeled; no model support or quality is inferred from header arithmetic.

### Acceptance gates for this consolidation and server

- Archive every original benchmark line and retain fixed tolerances, failures,
  commands and decision history. Keep this living summary at 200–350 lines;
  README links must resolve to this page or an archived section.
- Preserve all Qwen3, Qwen3.5 and Granite execution paths, defaults, scalar
  fallback, state rollback, snapshots and numerical NO-GO decisions. Change
  source only for a verified issue, with a focused regression test.
- `marin serve` must use the same loader/planner as `marin run`, load one model
  once, and serve sequential OpenAI-style text requests over loopback. A new
  cache per request must prevent state leakage. Streaming chunks must rebuild
  the same greedy text as a non-streaming request and the CLI.
- Bad input, too-long prompts, disconnects and cancelled generations must
  leave the following request usable. Three repeated real Qwen3.5-4B Q8 CUDA
  requests must show no second-to-third growth over 128 MiB in process RSS
  or process VRAM, stay within the selected memory budget, and avoid reload.
- Run full workspace/CUDA and forced-scalar regressions, fmt, check, Clippy
  with warnings denied, line-limit and unsafe checks. Record unverified gates.

## Current support and decisions

| Model/path | Status | Correctness and limitation |
|---|---|---|
| Qwen3 0.6B/1.7B CPU BF16 | **GO, default** | Strict HF checks, scalar fallback, KV reuse and multithreaded prefill. |
| Qwen3 CPU packed INT8 | **GO, opt-in** | Stage 3 numerical, quality, RAM and speed gates passed for both sizes. Direct Q8 load does not retain a full BF16 copy. |
| Qwen3 CUDA BF16/INT8 | **GO** | GPU KV paging, optional FP16 KV and snapshots tested; INT8 changes some wording. |
| Qwen3.5-2B text CPU BF16/INT8 | **GO** | Full 24-layer HF parity; INT8 matched quantized HF and 254/256 BF16 top-1 choices. |
| Qwen3.5-2B text CUDA BF16 | **GO** | Device-resident hybrid state, strict HF logits and greedy parity. Measured optimization prototypes remain deferred. |
| Qwen3.5-4B text CPU BF16/INT8, CUDA INT8 | **GO** | Full 32-layer backbone, official non-thinking chat, strict HF parity and real coding output. BF16 CUDA does not fit this GPU. Vision and MTP are unsupported. |
| Granite 3.0 1B-A400M CPU BF16 | **GO** | All-resident MoE with strict HF router/layer/logit parity and batched prefill. |
| Granite CPU INT8 | **NO-GO for parity**, opt-in pilot | Decode/RSS improve, but code and long-prompt strict RMS gate fails; BF16 remains reference. |
| Granite CUDA BF16 | **NO-GO for default**, explicit experimental | Real inference and greedy IDs pass, but code-step-2 strict RMS gate fails. |
| Granite 4.0 H-Tiny residency | **BLOCKED / no implementation** | Header and transfer pilot only; no real routes or H-Tiny CPU baseline. |
| CPU growable FP32 KV | **GO independently** | Full-context reservation removed; bounded prefill removed as NO-GO. Full Stage 4 still misses narrow TTFT/decode gates. |
| Native CPU INT4 | **NO-GO** | Q4 pilot missed fixed quality/memory gates; no production CPU INT4 path. |
| Minimal V9 planner | **GO** | Dry run shares admission logic with load; explicit context and INT8 opt-in preserved. It is an estimate, not a reservation. |

Fixed numerical gates are not interchangeable: Qwen3.5 hybrid HF checks use
normalized max ≤1e-4, relative RMS ≤1e-5 and exact greedy IDs. Granite
CPU Q8 and CUDA BF16 retained their stricter pinned trace gates, including
1e-5 RMS; the measured misses were not waived. See the exact gate language
and first divergent operations in the [Granite archive](docs/benchmarks/archive/04-granite-moe.log).

## Current measured baselines

The rows below identify the workload so a short decode figure is not mistaken
for a long-context figure. `~` denotes rounded values from the archived runs.
Historical rows are not a new matched cross-model benchmark matrix.

| Model / format / backend | Prompt; threads | TTFT | Prefill | Decode | Peak process RSS / VRAM |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B BF16 CPU | 256; 8 | 0.27 s | — | 20.0 | 1.25 GB RSS |
| Qwen3-0.6B INT8 CPU | 256; 8 | 0.24 s | — | 36.6 | 0.70 GB RSS |
| Qwen3-1.7B BF16 CPU | 256; 8 | 0.75 s | — | 7.2 | 3.41 GB RSS |
| Qwen3-1.7B INT8 CPU | 256; 8 | 0.63 s | — | 13.0 | 1.82 GB RSS |
| Qwen3.5-2B BF16 CPU | 512; 8 | 6.01 s | 85.1 | 5.38 | 3.97 GB RSS |
| Qwen3.5-2B INT8 CPU | 512; 8 | 5.41 s | 94.7 | 9.45 | 2.14 GB RSS |
| Qwen3.5-2B BF16 CUDA | 512 | 1.59 s | 322.0 | 48.1 | 1.27 GB RSS; 4.8–4.9 GB device-wide VRAM |
| Qwen3.5-4B BF16 CPU | 512; 8 | 12.91 s | 39.7 | 2.73 | 8.73 GB RSS |
| Qwen3.5-4B INT8 CPU | 512; 8 | 11.56 s | 44.3 | 4.75 | 4.66 GB RSS |
| Qwen3.5-4B INT8 CUDA | 512 | 3.68 s | 139.3 | 35.91 | 0.91 GB RSS; 5.56 GB device-wide VRAM |
| Granite 3.0 BF16 CPU | 512; 1 | 3.95 s | 129.5 | 26.4 | 2.78 GB RSS |
| Granite 3.0 INT8 CPU, experimental | 512; 1 | 4.20 s | 121.9 | 44.9 | 1.49 GB RSS |
| Granite 3.0 BF16 CUDA, experimental | 512 | 1.36 s | 376.7 | 148.7 | 0.31 GB RSS; 3.79 GB device-wide VRAM |

Qwen3 rows come from constrained-RAM Stage 4 runs with 32 output tokens,
zero swap and exact-output companions. Qwen3.5-2B rows come from focused
paired BF16/INT8 runs with eight output tokens (CPU) and a separate matched
BF16 CPU/CUDA comparison; they must not be used for a cross-backend speedup
ratio. Qwen3.5-4B rows use one run, four output tokens and warm file cache.
Granite rows are two-run medians with eight output tokens. See the archives
for exact command lines, ranges, context limits and raw-result paths.

## Persistent server gate: Qwen3.5-4B INT8 CUDA

**GO for sequential local text requests.** One release-build process loaded
`/home/mitali/models/Qwen3.5-4B-Q8` in 4.41 s, then answered three requests
without another model load. Configuration: RTX 5050, `--backend cuda
--quantize int8 --ctx 256 --temp 0`, 47 prompt tokens, up to 24 output tokens,
the same one-sentence greeting each time. A separate non-streaming request
reported nine completion tokens. This is one focused run with warm file cache,
not a throughput distribution or a multi-client benchmark.

| Request | First visible token | Total response | Current RSS after | Process VRAM after |
|---|---:|---:|---:|---:|
| Load, before requests | — | 4.41 s | 268.6 MiB | 4,248 MiB |
| 1 | 0.327 s | 0.516 s | 269.1 MiB | 4,440 MiB |
| 2 | 0.438 s | 0.722 s | 269.2 MiB | 4,440 MiB |
| 3 | 0.327 s | 0.737 s | 269.2 MiB | 4,472 MiB |

The process RSS high-water mark was 870.8 MiB. From request 2 to 3,
resident RSS rose <1 MiB and process VRAM rose 32 MiB, below the fixed
128 MiB gate. The greeting was identical in streaming, non-streaming and
`marin run`: `Hello! How can I help you today?` The focused fixture test
also exercises a bad model name, A→B→A request isolation, disconnect recovery
and CLI parity. The pinned one-case 128-token Qwen3.5-4B Q8 CUDA HF verifier
also passed: exact token choice, max logit difference 1.31e-5, max layer
difference 1.18e-5, normalized max 3.879e-6 and relative RMS 3.288e-6.
This verifies the shared inference engine; it is not a separate HTTP-level HF
trace. The stream sends nine content pieces; the eight post-first
pieces divided by remaining response time imply about 20–42 pieces/s here,
but this is a short-run proxy, not Marin's formal decode TPS metric. Power
and clocks were not sampled for these requests.

Reproduce the serving setup and a streamed request:

```bash
marin --model ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 \
  --ctx 256 --temp 0 serve --port 8000
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B-Q8","messages":[{"role":"user","content":"Say hello in one short sentence."}],"temperature":0,"max_tokens":24,"stream":true}'
```

The server accepts one request at a time on loopback. It has no batching,
remote authentication, persistent conversation store, vision or tool calls.
OpenAI clients must send string-message chat completions within the selected
context and the 1 MiB request-body limit. A graceful client half-close can be
treated as cancellation, so clients should keep the connection open to read
the response. A long concurrent-client or adverse-network load test remains
unverified.

### Gates fixed before session reuse and bounded queuing

- Opt-in bearer session IDs are scoped to one loaded model process. Reuse only
  when the complete retained token ledger is an exact prefix of the next
  rendered prompt, with at least one new token. A changed or shorter prompt
  must report zero reused tokens and match fresh greedy text exactly. IDs
  from another server instance must not access state.
- A complete Qwen3.5-4B CUDA INT8 two-turn chat must produce identical greedy
  text with reuse on/off and exact streamed/non-streamed output. Reused turn 2
  must skip at least 90% of turn 1's prompt tokens and cut matched, warmed
  turn-2 TTFT by at least 20% across three runs. Decode throughput must stay
  within 10%; retain the original HF tolerances and exact token choices.
- Only one session cache may be resident. New, anonymous, reset or incompatible
  requests evict before allocating a replacement. Across three repeated
  turn pairs, second-to-third process RSS and VRAM growth must each stay below
  128 MiB. No extra model-weight copy or unbudgeted parallel cache is allowed.
- One request executes while at most two wait in FIFO order. A further
  connection gets HTTP 503; a queued request older than 10 s or a partially
  read request older than 10 s fails clearly. At 300 s, active-generation
  cancellation must be requested and take effect at the next generation
  boundary. Disconnect, timeout and inference failure must never publish a
  partially updated session; the next request remains usable. TCP write
  half-close must still receive a valid response.

### Opt-in server sessions: measured result

RTX 5050 (8 GB), Qwen3.5-4B packed INT8, CUDA, `--ctx 512`, temperature 0,
32 generated tokens, one 177-token Rust prompt followed by a coding request;
three sequential matched streaming pairs after model load. No competing
benchmark ran. `marin_session_id: "new"` enabled reuse; omitting it forced a
fresh prefill. The full prompt and assistant answer were identical on both
paths. Detailed per-run values are in the [server archive](docs/benchmarks/archive/04-server-sessions.log).

| Metric (median of three) | Reuse off | Reuse on | Gate |
|---|---:|---:|---|
| Turn 1 TTFT | 1.243 s | 1.262 s | Measured |
| Turn 1 complete response | 2.203 s | 3.436 s | Rebuild costs 1.23 s |
| Turn 2 TTFT | 1.616 s | 0.422 s | **PASS**, −73.9% |
| Turn 2 decode event rate | 31.9/s | 32.4/s | **PASS**, +1.6% |
| Turn 2 complete response | 2.588 s | 2.969 s | Rebuild costs 0.38 s net |
| Reused positions | 0 | 170/177 first-turn prompt tokens | **PASS**, 96.0% |

All three turn-2 greedy strings matched exactly. VmRSS remained about 269.4
MiB after model load and across three turn pairs; VmHWM was about 871 MiB.
Process-attributable `nvidia-smi` VRAM readings were 4440–4472 MiB, a 32 MiB
spread. The one-slot cache evicts before creating another; failure or
cancellation never publishes a partial cache. FIFO, 503, read/queue 408,
stream/non-stream equivalence and TCP write half-close passed fixture tests.

**Historical GO for next-turn TTFT, NO-GO for total latency.** The official
template diverges before assistant text, and recurrent state cannot roll
back without replay. Socket-close detection before a non-streamed final
write remains limited by TCP write half-close. Remote networking, a 300 s
wall-clock timeout and long-duration load were not tested; the deadline had
only a 250 ms equivalent unit test. Prior checks and traces are in the
[server archive](docs/benchmarks/archive/04-server-sessions.log).

## End-to-end session checkpoint gates (set before implementation)

On the same Qwen3.5-4B packed INT8 artifact, RTX 5050, context 512 and
temperature zero, use three matched two-turn coding pairs and three chat
pairs, with the exact returned assistant text in turn two. Compare reuse off,
the replay-based server, and the candidate on a warm model with no competing
benchmark. Accept only if greedy text and the pinned HF logit tolerances stay
unchanged; template boundaries, changed-prefix fallback, rollback,
cancellation, isolation and streaming/non-streaming equality pass. The median
whole-pair response time must improve at least 10% over reuse off and 20%
over replay-based reuse. Turn-two TTFT must remain at least 20% below fresh;
decode event rate must stay within 5% of replay-based reuse. Process RSS and
VRAM after three pairs may grow less than 128 MiB; checkpoint allocation must
respect the existing admission slack. Reject the candidate if any of these
gates fails. Archive raw timings and the measured breakdown separately.

### Measured checkpoint result — GO for Qwen3.5 CUDA sessions

The server now advances only the new rendered history tokens, checkpoints
Qwen3.5 GPU recurrent/convolution state and attention KV at that boundary,
then generates from a writable cache. It publishes the checkpoint after a
successful reply. This preserves the official template without a full
post-reply replay. Other backends retain the prior refresh behavior.
RTX 5050, Qwen3.5-4B Q8 CUDA, context 512, 32 output tokens; three matched
streamed coding and chat pairs per variant, warm model, temperature zero.
Raw runs and GPU-stage breakdown are in the
[checkpoint archive](docs/benchmarks/archive/05-server-checkpoints.log).

| Median two-turn metric | Coding replay ON / fresh / checkpoint | Chat replay ON / fresh / checkpoint | Gate |
|---|---:|---:|---|
| Whole pair | 6.478 / 4.705 / **3.501 s** | 6.809 / 4.957 / **3.650 s** | **PASS**: checkpoint −46.0%/−46.4% vs replay, −25.6%/−26.4% vs matched fresh baseline |
| Turn-two TTFT | 0.474 / 1.763 / **0.496 s** | 0.481 / 1.916 / **0.512 s** | **PASS**: −72.8%/−74.0% vs current fresh runs |
| Turn-two decode SSE events/s | 39.4 / 39.1 / **38.8** | 40.3 / 40.0 / **39.1** | **PASS**: −1.5%/−2.9% vs replay |
| Earlier positions skipped | 170 / 0 / **170** | 186 / 0 / **186** | **PASS**: current-turn history prefill excluded from count |

Current fresh-run pair medians were 4.818 s coding and 5.073 s chat; the
table retains the separately measured replay-era fresh baselines to expose
clock variation. Greedy strings matched on/off in all pairs. The pinned 4B
CUDA Q8 HF check passed unchanged: exact token, max logit 1.31e−5, max layer
1.18e−5, normalized max 3.879e−6, relative RMS 3.288e−6. A real-model
checkpoint test checked exact restored recurrent/KV state and continuation;
real HTTP changed-prefix fallback, stream-disconnect rollback, session eviction
and retry matched fresh output. After three coding pairs RSS stayed at
275,960 KiB and process VRAM varied 4504–4536 MiB; sampled peak was 4536
MiB. No energy or continuous VRAM peak was measured. Admission retains 512
MiB GPU slack; low-free-VRAM requests fall back to the replay path. The extra
cache exists only during an active request, so long-context checkpoint fit
remains a memory-limit case rather than a promise of speed at every context.
Full CUDA-required workspace tests, forced-scalar server/model tests, fmt,
check, Clippy `-D warnings`, line-limit and unsafe checks passed.

## Results that changed the runtime

| Change | Matched evidence | Decision |
|---|---|---|
| CPU SIMD GEMM/GEMV and attention | Qwen3 prefill rose roughly 3–5× over the original scalar path; decode 1.5–1.8× before attention tuning. | **GO**; scalar fallback retained. |
| Eight-thread CPU prefill | Qwen3 prompt throughput about 5× one-thread at 511–1023 tokens; decode stays one-thread, near memory bandwidth. | **GO**; affinity-aware default. |
| Qwen3 CPU INT8 | 1.48–1.87× decode, weights 51.6% of BF16, roughly 0.56/1.65 GB less RSS on 0.6B/1.7B. | **GO, opt-in**. |
| Direct packed INT8 loading | Warm first output −35.7%/−36.2% and fadvise-cold −47.0%/−46.2% on Qwen3 0.6B/1.7B; bit-identical to old INT8. | **GO**; BF16 source never retained on warm load. |
| CPU growable KV | Upfront context-sized allocation removed; 0.6B INT8 8k under a 3 GiB zero-swap cap. | **GO for growth**, Stage 4 performance closure remains NO-GO. |
| Qwen3.5-2B batched CPU prefill | 128/512 prefill 8.1/8.2 → 90.6/93.5 tok/s; TTFT 15.74/62.12 → 1.41/5.48 s. | **GO**; recurrent updates remain sequential. |
| Qwen3.5-2B CPU INT8 | Short decode 5.52 → 9.30 tok/s; RSS 3.90 → 2.08 GB; 512 TTFT 6.01 → 5.41 s. | **GO, opt-in**. |
| Qwen3.5-4B CPU INT8 | 2k decode 2.33 → 4.29 tok/s, RSS 9.02 → 4.96 GB; CUDA Q8 2k decode 38.03 tok/s. | **GO for measured 8 GB host/8 GB GPU fit**. |
| Granite batched MoE prefill | 512 CPU BF16 30.5 → 131.3 tok/s one-thread and 388.2 tok/s eight-thread. | **GO**; all experts remain resident. |
| Granite CPU INT8 | 512 decode 26.4 → 44.9 tok/s and RSS 2.78 → 1.49 GB, but strict HF RMS fails. | **NO-GO for acceptance**. |
| Granite CUDA BF16 | 512 decode 148.7 tok/s, but strict HF code-step-2 RMS fails. | **NO-GO for default**. |

GPU Qwen3 attention, paged KV, FP16 KV and optional snapshots yielded
substantial context and latency gains. Details remain in the
[GPU archive](docs/benchmarks/archive/01-gpu-foundations.log). Rejected
bounded prefill, native CPU INT4, prefill packing, speculative draft-model
and Granite caching experiments remain in their archives, including original
thresholds and negative numbers. No performance win is inferred from a
reverted prototype.

## Bottlenecks and limits

- Qwen3/Qwen3.5 CPU decode is largely weight-bandwidth-bound. The 4B CPU Q8
  path is usable under 8 GiB but only about 4–5 tok/s at 512–2k contexts.
- Qwen3.5-4B CUDA 2k prefill takes 15.94 s in the short four-output run;
  2,092 GPU launches and sequential recurrent/convolution work dominate.
  The 40 W cap affects comparisons. GPU optimization candidates that missed
  their fixed gate remain deferred.
- Qwen3.5 vision/MTP, Qwen3.5 snapshots, and Granite 3.0 strict INT8/CUDA
  numerical parity are not accepted capabilities. No H-Tiny execution path
  or larger-than-VRAM expert cache exists.
- Static memory admission can become stale if another process consumes RAM
  or VRAM after planning. CUDA GPU quotas and broad quality testing are not
  demonstrated. Early Qwen3 figures and later model runs use different
  conditions and must not be combined into synthetic speedup claims.

## Reproduce current measurements

Use the same model artifact, context, prompt length, output length, thread
count, power mode and warmup on both sides. `bench --json` records the setup;
`compare` checks ranges. Raw historical files under ignored
`bench-results/` are optional local evidence; the tracked archive preserves
published tables and commands. No reference JSON is implicitly produced by
these commands.

```bash
# BF16 reference, then packed INT8 on the same CPU and input
marin-run bench ~/models/Qwen3.5-4B --backend cpu --threads 8 \
  --prompt-tokens 512 --tokens 32 --ctx 2048 --runs 3 --warmup 1 \
  --json bf16.json
marin-run bench ~/models/Qwen3.5-4B-Q8 --backend cpu --threads 8 \
  --quantize int8 --prompt-tokens 512 --tokens 32 --ctx 2048 \
  --runs 3 --warmup 1 --json int8.json
marin-run compare bf16.json int8.json

# CUDA on the same packed artifact; report process and device-wide memory separately
marin-run plan ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 --ctx 2048
marin-run bench ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 \
  --prompt-tokens 512 --tokens 32 --ctx 2048 --runs 3 --warmup 1 \
  --sample-gpu --json cuda.json

# Correctness requires an independently generated, matching HF reference file
marin-run verify ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 \
  --reference bench-results/qwen35-4b/hf-int8-128.json

# Scalar and multithreaded CPU checks
MARIN_MATMUL_SCALAR=1 marin-run bench ~/models/Qwen3-0.6B \
  --backend cpu --threads 1 --prompt-tokens 128 --tokens 16
marin-run bench ~/models/Qwen3-0.6B --backend cpu --threads 8 \
  --prompt-tokens 128 --tokens 16
```

Full checks: `cargo fmt --all -- --check`, `cargo check --workspace
--all-targets --locked`, `cargo clippy --workspace --all-targets
--all-features --locked -- -D warnings`, `MARIN_REQUIRE_CUDA=1 cargo test
--workspace --all-features --locked`, `scripts/check-line-limit.sh`, and
`scripts/check-unsafe.sh`. Real-checkpoint ignored tests need their named
model/reference environment variables and are reported separately.

## Archive map

The four files below concatenate byte-for-byte into the pre-consolidation
BENCHMARKS.md (SHA-256
`6cf53334a6d2e46d66d60f4203607023e35320cc274ad9a456b58a23fb8f90d0`).
Section numbers and their original gates remain unchanged inside them:

- [1–21: GPU foundations and first CPU audit](docs/benchmarks/archive/01-gpu-foundations.log)
- [22–31: CPU SIMD, INT8, KV, INT4 pilots and consolidation](docs/benchmarks/archive/02-cpu-stages.log)
- [32–40: packed loader, Qwen3.5 and V9 planner](docs/benchmarks/archive/03-loader-qwen35-planner.log)
- [41–46: Granite MoE, numerical NO-GOs and H-Tiny pilot](docs/benchmarks/archive/04-granite-moe.log)
