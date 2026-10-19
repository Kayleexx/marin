# Marin

Marin is a local Rust inference engine for ordinary PCs. It runs selected
Qwen3, Qwen3.5 and Granite text models on CPU or CUDA, measures the work, and
chooses a backend only when its memory estimate fits. BF16 is the default;
packed INT8 needs explicit opt-in. The [performance summary](BENCHMARKS.md)
separates accepted paths from experiments, and the
[archive](docs/benchmarks/archive/INDEX.log) keeps the full measurement history.

## Install and run

Build with the Rust version in `rust-toolchain.toml`. CUDA inference needs an
NVIDIA driver and CUDA toolkit; CPU inference works without them.

```bash
cargo install --path crates/cli --locked
marin --model ~/models/Qwen3-1.7B
marin run "Explain a mutex in one sentence."
marin-run ~/models/Qwen3-1.7B-Base --backend cpu --temp 0 \
  -p "The capital of France is"
```

`marin` is the chat interface for supported post-trained chat models;
`marin-run` exposes base-model continuation, benchmarking and verification.
The model folder needs `config.json`, tokenizer files and complete
safetensors weights. The default backend uses CUDA when the planned window
fits, then CPU; `--backend cpu` or `--backend cuda` forces a choice. The
planner estimates memory before loading, but cannot reserve RAM or VRAM
against other processes.

```bash
marin --model ~/models/Qwen3.5-4B-Q8 --quantize int8 --ctx 2048 plan
marin-run plan ~/models/Qwen3.5-4B-Q8 --quantize int8 --ctx 2048
```

The two commands have different default context sizes, so pass `--ctx` when
comparing them. An explicit context is never silently reduced. The selected
folder stays the selected artifact; Marin does not search for a sibling Q8
file or silently change precision.

## Supported models and precision

| Model | Accepted execution | Notes |
|---|---|---|
| Qwen3 0.6B/1.7B | CPU/CUDA BF16, opt-in INT8 | CPU SIMD and scalar fallback; threaded prefill; GPU paged KV and snapshots. |
| Qwen3.5-2B-Base text | CPU BF16/INT8, CUDA BF16 | Full hybrid recurrent and attention state; base-model continuation through `marin-run`. |
| Qwen3.5-4B post-trained text | CPU BF16/INT8, CUDA INT8 on 8 GB RTX 5050 | Official non-thinking chat template; 8 GB host fit measured for CPU Q8. BF16 CUDA does not fit this GPU. |
| Granite 3.0 1B-A400M | CPU BF16 | All-resident MoE; use `marin-run` and the model's chat template. |

Granite CPU INT8 and CUDA BF16 can be selected explicitly for experiments,
but both miss strict Hugging Face numerical tolerances. Neither is an
accepted default. Native CPU INT4, Granite 4 H-Tiny, Qwen3.5 vision/MTP,
Qwen3.5 snapshots and larger-than-VRAM expert caching are not supported.
See the [decision table](BENCHMARKS.md#current-support-and-decisions).

Export Q8 once, then load it directly. The artifact stores codes, scales,
protected tensors and versioned metadata; direct inference does not create
a full BF16 weight copy. Export refuses an existing output folder and loading
checks its integrity and group size.

```bash
marin-run export-int8 ~/models/Qwen3.5-4B ~/models/Qwen3.5-4B-Q8 \
  --quant-group 128
marin --model ~/models/Qwen3.5-4B-Q8 --backend cuda \
  --quantize int8 --ctx 256 run "Write a Python factorial function."
```

The same packed folder also runs on CPU with `--backend cpu`. Without
`--quantize int8`, Marin refuses it instead of silently switching formats.
Qwen3.5 keeps recurrent/convolution and attention state at its validated
FP32 precision. Qwen3 GPU chat uses FP16 KV by default; CPU KV is FP32.

## Serving text requests

`marin serve` loads one chat model once and accepts requests on
`127.0.0.1` only. Qwen3.5 on CUDA runs up to three requests at once (fewer when
free GPU memory is short): each keeps its own state, prompts are fed in
64-token slices, and decode steps and short prompts run ahead of a long
prefill, so a short request is not stuck behind a long one. Decode steps of
concurrent requests share one GPU pass (up to three). Every other
backend runs one request at a time. It supports OpenAI-style `/v1/models` and
`/v1/chat/completions` with string messages, non-streaming JSON or SSE token
streaming. Anonymous requests use a fresh inference cache. It does not batch prompts,
execute tools, accept multimodal content or provide network authentication;
keep it on loopback. Snapshots are unavailable in server mode and `--snapshot`
is rejected. Opt-in sessions retain one conversation prefix in memory; a new
session or anonymous request evicts the previous one.

```bash
marin --model ~/models/Qwen3.5-4B-Q8 --backend cuda --quantize int8 \
  --ctx 512 serve --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B-Q8","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":32,"marin_session_id":"new"}'
```

Save the returned `marin_session_id` and send it with the full conversation,
including Marin's prior assistant text, on the next request. For example:

```json
{"model":"Qwen3.5-4B-Q8","marin_session_id":"<returned 32-character ID>","messages":[{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Now explain why."}],"temperature":0,"max_tokens":32}
```

`marin_reused_tokens` reports exact token-prefix reuse; a changed prefix
starts fresh. IDs are bearer secrets scoped to this server process; an evicted
or unknown ID returns 404. `"new"` resets the one retained slot. Sessions
reuse attention KV and Qwen3.5 recurrent/convolution state. To preserve the
official multi-turn template, Qwen3.5 CUDA checkpoints the stable history
before generation and avoids replaying it after the reply. Other backends
still rebuild that history after each opt-in turn.

Add `"stream":true` for `text/event-stream` chunks ending in `[DONE]`;
the final chunk includes the session ID and reuse count. TCP write half-close
still permits a response. At most two requests wait beyond those running;
further connections receive 503. A session ID in use by another request returns 404. Queued and partial-read requests time out after
10 seconds; generation requests cancellation after 300 seconds, effective
at the next model boundary. The 1 MiB body and selected context limits still
apply. A failed or cancelled turn never publishes partial state; a session
may be evicted and need `"new"` again.

## Useful controls

| Option | Effect |
|---|---|
| `--backend cpu\|cuda` | Force an execution backend; automatic selection is the default. |
| `--quantize int8` | Opt in to grouped INT8 weights; `--quant-group` defaults to 128. |
| `--ctx N` | Explicit maximum context; never silently reduced. |
| `--threads N` | CPU prompt threads; default respects physical cores and affinity/cgroup restrictions. Decode remains single-threaded. |
| `--kv-type f32\|f16` | Qwen3 GPU KV format; CPU, Qwen3.5 and Granite retain their supported precision. |
| `--snapshot FILE` | Opt-in Qwen3 GPU prefix snapshot, checked against model and format. CPU and Qwen3.5 snapshots are unsupported. |

`marin-run bench`, `verify`, `trace`, `quality`, `compare` and `probe` are
measurement tools. Reproducible commands and current baselines are in
[BENCHMARKS.md](BENCHMARKS.md#reproduce-current-measurements). The full
historical record, including failed numerical gates and rejected
optimizations, remains in the archive.

## Verification

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
MARIN_REQUIRE_CUDA=1 cargo test --workspace --all-features --locked
scripts/check-line-limit.sh
scripts/check-unsafe.sh
```

CUDA-required checks need a CUDA-capable machine. Real-checkpoint Hugging
Face tests also need the named local model and reference files. The full
source and detailed architecture goals are in [marin_spec_v0.4.md](marin_spec_v0.4.md).
