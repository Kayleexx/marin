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
  commands and decision history. Keep this living summary brief and readable;
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

**GO for local text requests.** One release process loaded
`~/models/Qwen3.5-4B-Q8` in 4.41 s and answered three requests with no reload
(RTX 5050, `--ctx 256 --temp 0`, 47 prompt tokens, one warm run):

| Request | First visible token | Total response | Current RSS after | Process VRAM after |
|---|---:|---:|---:|---:|
| Load, before requests | — | 4.41 s | 268.6 MiB | 4,248 MiB |
| 1 | 0.327 s | 0.516 s | 269.1 MiB | 4,440 MiB |
| 2 | 0.438 s | 0.722 s | 269.2 MiB | 4,440 MiB |
| 3 | 0.327 s | 0.737 s | 269.2 MiB | 4,472 MiB |

RSS high-water mark 870.8 MiB; request 2 to 3 grew RSS <1 MiB and VRAM 32 MiB (gate 128 MiB). The greeting was identical streamed, non-streamed and
in `marin run`; the fixture test covers a bad model name, A→B→A isolation, disconnect recovery and CLI parity. The pinned 128-token Q8 CUDA HF check
passed: exact token, max logit 1.31e-5, max layer 1.18e-5, normalized max
3.879e-6, relative RMS 3.288e-6 (shared engine, not an HTTP-level trace).
Streamed pieces/s is a short-run proxy, not decode TPS; power was not sampled. Loopback only: no authentication, vision or tools; string messages
within the context and a 1 MiB body. A graceful client half-close can count as cancellation.
Command lines are in the [server archive](docs/benchmarks/archive/04-server-sessions.log) and the README.

### Gates for session reuse and bounded queuing

Fixed before implementation; text archived in the [server archive](docs/benchmarks/archive/04-server-sessions.log). In short: exact-prefix reuse only, one resident session cache, one request running with two waiting in FIFO order, 503 beyond that, 408 after 10 s, 300 s cancellation, and no partial session ever published.

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

## End-to-end session checkpoint gates

Fixed before implementation ([archive](docs/benchmarks/archive/05-server-checkpoints.log)): identical greedy text and HF tolerances, whole-pair latency at least 10% below reuse off and 20% below replay reuse, turn-two TTFT at least 20% below fresh, decode within 5%, RSS/VRAM growth under 128 MiB.

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

## V7/V7c multi-request serving — GO for Qwen3.5 CUDA

Up to 3 request threads ([scheduling](docs/benchmarks/archive/06-server-scheduling.log),
[batched decode](docs/benchmarks/archive/07-batched-decode.log), [memory](docs/benchmarks/archive/08-batched-memory.log));
GPU passes go through a gate one at a time, prompts in 64-token slices (bit-identical), decode steps of up to 3
requests share one pass with caches updated in place (a failed pass marks the batch for replay, 5.7e-6 off).

Qwen3.5-4B Q8 CUDA, 128 tokens/client, V7 sequential → batched: 1 / 2 / 3 decoders 29.5 / 29.1 / 29.0 → 29.3 / 44.5 / **54.5 events/s (1.88x)**; long + 2 short:
short TTFT p50 / p95 rejected (408) → 0.47 / 0.87 s; 3 short: mean completion 2.05 → 1.60 s with no VRAM growth (gate 128).

## V8/V9 prompt prefill — batching NO-GO, shared prefix GO

Prefill is a flat 7.1–7.7 ms/token (82% matmul), so one pass over two prompts cost −8% to +13% against a
>= 10% gate: not built ([archive](docs/benchmarks/archive/09-batched-prefill.log)). All default-system requests
start with the same 31 token IDs; restoring a stored cache costs 0.45 ms against 239 ms, bit-identical. One
immutable cache is built at start (200 ms, 87 MiB): solo short TTFT 0.333 → 0.112 s, 3 short clients p95
0.99 → 0.30 s, no match unchanged, VRAM +0–64 MiB ([archive](docs/benchmarks/archive/10-shared-prefix.log)).

## Coding V0 and the unified terminal — GO for the interface, NO-GO as a reliable reader

Read-only code mode (`list_files`, `read_file`, `search`; confined, bounded) ([archive](docs/benchmarks/archive/11-coding-v0.log)). Qwen3.5-4B Q8, the same 25 tasks re-run on
the current tree (V10-A): 15 pass (60%), median task 26.8 s; failures are confident (a call site named as the definition). Use it to find `path:line`, then read the line.

V10 puts chat and code behind one `marin` ([archive](docs/benchmarks/archive/12-unified-terminal.log)): in-process load, planner fit in the model list, `/mode` `/model` `/status`
`/settings` `/clear` on a pipe, a server on port 8000 reused by `marin code`; Ctrl-C now takes 0.01–0.43 s.
Persistent checkpoints — GO, opt-in, Qwen3.5 GPU only ([archive](docs/benchmarks/archive/15-checkpoint-feasibility.log)): replaying 1k tokens costs 7.7 s, 4k 32.8 s;
a resume with one question takes 9.3 s instead of 25.6 s at 2,265 tokens. `/bench` and `/verify` run the `marin-run` code on the loaded model and agree within 1%
([archive](docs/benchmarks/archive/16-terminal-bench-verify.log)): first token 920–950 ms, prefill 134–139, decode 40.0–40.6 tok/s.

## Context cost on an 8 GB GPU — O(state) passes, GO ([archive](docs/benchmarks/archive/18-inference-state-memory.log))

A Qwen3.5 GPU pass used to stage the whole cache. It now runs in place: keys and values are only appended, so a failed pass is undone by cutting
them back, and only the 53 MB of DeltaNet state is marked and put back.
Matched, interleaved before → after (idle machine, 4B-Q8, decode tok/s; process VRAM peak): 128 tokens 40.0 → 40.1 (4,472 → 4,376 MiB),
2k 36.7 → 39.4 (4,696 → 4,568), 8k 27.2 → 35.3 (5,496 → 4,984), 14k 22.3 → 32.0 (6,296 → 5,368); prefill and first token unchanged.
The planner admits ~26–28k tokens instead of 15.4k (estimate within 5–50 MiB of the measured peak, plus the 512 MiB safety). A 19-turn chat
to 27k tokens reuses every turn (11–43 s each); before, turns after 9.8k replayed everything (123–180 s). `Outcome.replayed` reports tokens a cache
had to read again; `reused` no longer counts them. `marin serve --ctx 8192` runs 2 requests at once instead of 1. **V5 tiering stays NO-GO** (37 ms/token at 8k).

## Memory pressure after loading — host marks and waiting, GO ([archive](docs/benchmarks/archive/19-vram-pressure.log))

A mark needed the 53 MB plus the 512 MiB safety free, and the server copied the whole history KV every turn; below that the next turn read the
whole history again. The recurrent-state mark now goes to host memory when the GPU is short (same bytes and logits as a GPU mark), and the server
returns to a mark instead of a copy (peak at 5k tokens 5,176 → 4,824 MiB). Turns with 340–600 MiB free: chat 29–42 s (2.9–4.3k tokens replayed) →
6.2–7.3 s, replayed 0; chat at 10k tokens 100–109 → 9.8–10.0 s; server 34–50 → 7.4–8.1 s. Uncontended turns and decode are unchanged (within 0.3%).
Two requests that do not fit together both failed with 500; now the older finishes and the younger gets a retryable 503 (`Gate::wait_for_memory`).
`marin_replayed_tokens` reports a re-read. Below ~150 MiB free a single pass cannot run.

## Speculative decoding (V8) — PARTIAL GO: opt-in, Qwen3.5 CUDA ([archive](docs/benchmarks/archive/20-speculation.log))

Copying guesses from earlier text existed for Qwen3 only. Qwen3.5 can now check up to 4 tokens in one pass (1.06–1.48x one token's cost; 6 or more cost
2.6–3.4x) and, when a guess is wrong, cut back inside that pass by resuming from its mark. Rule: 4 matching tokens, at most 3 guesses.
Interleaved plain vs `--speculate`, 300-token replies, 4B-Q8: code edits 39.9 → 81.2 tok/s (2.04x, 1.76–2.24x paired, 1.88x isolated re-run),
repeated text 1.38x, lines with a shared start and different ends 1.11x, chat 1.02x, long context 1.00x; 40 W in every mode, joules per token on edits 1.00 → 0.49.
Greedy output identical in 64 of 64 runs; sampled identical in 26 of 28 (a pass of rows equals single steps only to float rounding). Peak VRAM unchanged
(+53 MB resident mark); under a second program holding the GPU both finished identically. The server does not guess. A draft model and a controller
were not built: copying already uses the cheap check and nothing measured loses.

## Model evaluation (V10-A) — routing NO-GO, one model ([archive](docs/benchmarks/archive/21-model-evaluation.log))

Gates were written first; 17 held-out single-shot tasks per configuration (greedy, official templates, no parser repair; numbers, JSON and tool calls checked by
rule, lookups and explanations read against references). Thinking is in both templates but Marin always renders non-thinking, so it was tested through `marin-run -p`.

| Configuration | Pass | Lookup / explain / long / tool / chat | Malformed calls | Median s | Tokens written | Load, VRAM |
|---|---:|---|---:|---:|---:|---|
| Qwen3.5-4B Q8, non-thinking (today) | 13/17 | 4/4, 1/2, 2/3, 2/4, 4/4 | 2 | 11.2 | 3,233 | 3.5 s, 6.0 GB |
| Qwen3.5-4B Q8, thinking | 12/17 | 4/4, 1/2, 2/3, 1/4, 4/4 | 2 | 25.2 | 13,570 | 3.7 s, 6.0 GB |
| Qwen3-1.7B BF16, non-thinking | 8/17 | 4/4, 0/2, 2/3, 0/4, 2/4 | 3 | 4.1 | 3,340 | 2.7 s, 6.0 GB |
| Qwen3-1.7B BF16, thinking | 9/17 | 4/4, 0/2, 2/3, 0/4, 3/4 | 2 | 10.9 | 11,826 | 2.4 s, 6.4 GB |

Thinking costs 2.2x the time and 4.2x the tokens for no gain (12 vs 13, noise), and two replies were cut at 2,048 tokens. The 1.7B model is 2.7x faster only through a
fast prefill and fails the quality gate (light work 6/8 vs 8/8; tool format 0/4). Every model wrote malformed calls, so strict parsing is not met by any: the tool client's
repair stays. Repository tasks (4B, tools): dev 9/17, held-out 6/8. Not run: Qwen3-0.6B (the 1.7B already fails), Qwen3.5-2B-Base (a base model), Granite (experimental pilot). VRAM is the all-programs reading (about 0.8 GB is the desktop).
Smallest policy that the data allows: keep the 4B non-thinking as the only chat/code model; offer thinking only as an explicit per-request opt-in. Not implemented.

## Results that changed the runtime

| Change | Matched evidence | Decision |
|---|---|---|
| CPU SIMD GEMM/GEMV, attention, eight-thread prefill | Qwen3 prefill roughly 3–5× the scalar path (about 5× one-thread at 511–1023 tokens); decode 1.5–1.8×, one thread near memory bandwidth. | **GO**; scalar fallback, affinity-aware default. |
| Qwen3 CPU INT8 | 1.48–1.87× decode, weights 51.6% of BF16, roughly 0.56/1.65 GB less RSS on 0.6B/1.7B. | **GO, opt-in**. |
| Direct packed INT8 loading | Warm first output −35.7%/−36.2% and fadvise-cold −47.0%/−46.2% on Qwen3 0.6B/1.7B; bit-identical to old INT8. | **GO**; BF16 source never retained on warm load. |
| CPU growable KV | Upfront context-sized allocation removed; 0.6B INT8 8k under a 3 GiB zero-swap cap. | **GO for growth**, Stage 4 performance closure remains NO-GO. |
| Qwen3.5-2B CPU | Batched prefill 128/512: 8.1/8.2 → 90.6/93.5 tok/s (TTFT 15.74/62.12 → 1.41/5.48 s); INT8 decode 5.52 → 9.30 tok/s, RSS 3.90 → 2.08 GB. | **GO**; INT8 opt-in. |
| Qwen3.5-4B CPU INT8 | 2k decode 2.33 → 4.29 tok/s, RSS 9.02 → 4.96 GB; CUDA Q8 2k decode 38.03 tok/s. | **GO for measured 8 GB host/8 GB GPU fit**. |
| Granite | Batched MoE prefill 512: 30.5 → 131.3 (388.2 eight-thread) tok/s. CPU INT8 decode 26.4 → 44.9 tok/s, RSS 2.78 → 1.49 GB; CUDA BF16 148.7 tok/s; both fail strict HF RMS. | **GO** BF16 CPU prefill; INT8 and CUDA **NO-GO** for acceptance. |

GPU Qwen3 attention, paged KV, FP16 KV and snapshots gave large context and latency gains ([GPU archive](docs/benchmarks/archive/01-gpu-foundations.log)). Rejected
bounded prefill, CPU INT4, prefill packing, draft-model and Granite caching experiments keep their thresholds and negative numbers in their archives.

## Bottlenecks and limits

- Qwen3/Qwen3.5 CPU decode is weight-bandwidth-bound; the 4B CPU Q8 path is usable under 8 GiB at about 4–5 tok/s (512–2k contexts).
- Qwen3.5-4B CUDA 2k prefill takes 15.94 s (2,092 launches, sequential recurrent work); the 40 W cap affects comparisons; missed-gate candidates stay deferred.
- Qwen3.5 vision/MTP, Qwen3.5 snapshots, and Granite 3.0 strict INT8/CUDA
  numerical parity are not accepted capabilities. No H-Tiny execution path
  or larger-than-VRAM expert cache exists.
- Admission is planned once; after loading, a pass that finds no GPU memory is waited out or refused (503), not predicted.
  CUDA GPU quotas and broad quality testing are not demonstrated. Early Qwen3 figures and later model runs use different
  conditions and must not be combined into synthetic speedup claims.

## Reproduce current measurements

Use the same model artifact, context, prompt length, output length, thread count, power mode and warmup on both sides. `bench --json` records the setup;
`compare` checks ranges; the tracked archive preserves published tables and commands. The pressure-run drivers are in git history (`git show 5a4a8ee:docs/benchmarks/archive/19-harness/`).

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

### From a clean checkout

Tracked and runnable with only the repository: `scripts/check-line-limit.sh`, `scripts/check-unsafe.sh`, the tiny-model fixtures and references under
`tests/`, and the workloads in `benchmarks/`: `bench_baseline.sh MODEL_DIR` (Qwen3 bf16 CPU and CUDA matrix), `guess_prompts.sh DIR` and `spec_prompts.sh DIR` (prompt
folders for `marin-run guess` and `spec-bench`; the prompts built from the untracked spec are skipped, with a note, when it is absent), `server_load.py PORT PID SCENARIO` (multi-client load against `marin serve`) with its `long-prompt.txt`.
Hugging Face references are generated, not tracked: `scripts/gen_reference.py`, `scripts/gen_chat_reference.py`, `scripts/gen_chat_golden.py` (Qwen3 and chat
template) and `tests/reference/{qwen35,granite}/*.py` (need torch and transformers). The pinned 128-step Qwen3.5-4B INT8 reference
(`bench-results/qwen35-4b/hf-int8-128.json`, 226 MB) is not in the repository and its exact invocation was not recorded; regenerate with
`tests/reference/qwen35/full.py MODEL OUT --steps 128` against the 4B BF16 checkpoint and compare it against the figures above before trusting it.
Lasting trade-offs are in [`docs/decisions/`](docs/decisions/).

## Archive map

Every log is indexed in the [archive index](docs/benchmarks/archive/INDEX.log). The four
oldest logs (01–04) concatenate byte-for-byte into the pre-consolidation BENCHMARKS.md (SHA-256
`6cf53334a6d2e46d66d60f4203607023e35320cc274ad9a456b58a23fb8f90d0`); never edit them.
