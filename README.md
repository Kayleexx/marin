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
marin                                  # chat; the first run finds your models and asks which one
marin code                             # ask about the project in this folder (read-only)
marin run "Explain a mutex in one sentence."   # one answer, then exit (also reads a pipe)
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
execute tools (`marin code` reads files itself), accept multimodal content or provide network authentication;
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
the final chunk includes the session ID, reuse count and token usage; `/v1/models` reports the context window. TCP write half-close
still permits a response. At most two requests wait beyond those running;
further connections receive 503. A session ID in use by another request returns 404. Queued and partial-read requests time out after
10 seconds; generation requests cancellation after 300 seconds, effective
at the next model boundary. The 1 MiB body and selected context limits still
apply. A failed or cancelled turn never publishes partial state; a session
may be evicted and need `"new"` again.

## One terminal for chat and code

`marin` is the everyday entry: it loads the model in its own process, so nothing needs
starting first. On the first run it looks in `~/models` (or `MARIN_MODELS`), shows the
chat models with what the planner says about each on this machine (fits the GPU, CPU
only, too big) and asks which to use; with none it asks for a folder. A single model is
just used. The choice is saved in `~/.config/marin/config.json` (XDG, or `MARIN_HOME`)
and the next launch asks nothing. Flags always win over saved values; the saved ones are
the model, the backend picked in `/model`, and `temp`/`max_tokens` changed in the UI. A
damaged or outdated config is reported once and replaced. A packed 8-bit model needs a
`--quantize int8` opt-in as before: picking it from the list (or answering yes to the one
question for a saved model) is that opt-in, remembered with the model.

Type `/` for actions and keep typing to search them: `/mode code`, `/mode chat`,
`/model`, `/status`, `/bench`, `/verify`, `/settings`, `/save`, `/resume`, `/sessions`, `/clear`, `/help`, `/exit`. Piped input takes the same
commands as lines (`printf '/mode code\nwhere is main?\n' | marin`); other lines are
questions. Each mode keeps its own conversation, `/clear` forgets the current one, a reply stopped
with Ctrl-C is not kept (in either mode), and a
different model starts both over. On Qwen3.5 GPU the chat history is
read once and checkpointed, so a long conversation does not replay itself each turn. `marin run` with no text on a terminal opens the same chat;
with text or a pipe it answers once, as before.

**Checking the model you are talking to.** `/bench [PROMPT [OUTPUT]] [runs=N] [profile] [quick]` times the
loaded model on a fixed prompt (default 128 tokens in, 64 out, 3 runs after a warmup) and prints first-token time, prefill and
decode speed, token latency, and memory; `/verify [FILE]` compares it with a float32 Hugging Face reference and prints
`result: PASS` only if every case matches. Both are the same code as `marin-run bench` and `marin-run verify`, run on the model
already in memory: it is not loaded again, the conversation, its cache and any checkpoint are not touched, and Ctrl-C stops
either one. The benchmark uses a cache of its own (a few hundred MB at most, shown in its report) and a workload that fits this
session's window; it keeps the load time of startup apart from what it measures, and says nothing it did not measure. Peak
process memory is the highest since `marin` started. `/verify` without a file looks for a reference made for this model (same
weights storage, same layers, same tokenization of its prompts) in the model's folder, `tests/reference`, `tests/fixtures/*`
and `bench-results/*` of the current directory; a reference made for something else is explained, never run, and no reference
means no verdict. Ctrl-C takes effect at the next token or case: a long prompt pass (about 7 ms per token on Qwen3.5-4B) finishes
first. `marin-run` is unchanged for scripts and reproducible measurements.

**Saving a conversation** is opt-in: nothing is written until you ask.

- `/save [name]` keeps both conversations as they stand, finished questions only (a cancelled or
  failed one is never kept). Without a name it updates the session this one came from, or makes
  `session-<time>`. `/sessions` lists what is saved, `/resume [name]` replaces the current
  conversations with a saved one (the newest without a name) and continues in the mode it was left in.
- Sessions are JSON files in `$XDG_STATE_HOME/marin/sessions` (default `~/.local/state/marin/sessions`,
  or `$MARIN_HOME/sessions`): the folder is mode 700, the files 600, each is written whole and renamed
  into place, carries a format version, and is limited to 2 MiB; at most 100 are kept. A damaged, newer
  or oversized file is named and skipped; a file that is not a Marin session is never overwritten.
- **A saved code session contains the source code and tool output the model read from the project.**
  Treat the file as private as the project; delete it when you no longer need it.
- `/save` keeps only text, never the cache. Resuming reads the history again through the normal path
  (on Qwen3.5 GPU the next turns are cheap again after one read). It works only with the model it was
  saved with (name, configuration and tokenizer) and, for code mode, while the project folder exists;
  otherwise it says why and changes nothing. Sessions need a model loaded by `marin` itself, so they are
  not available in `marin code --port`.
- Two runs cannot silently overwrite each other. A session carries a revision, and `/save` replaces it
  only if your conversation started from that revision; otherwise it says another run saved since and
  keeps theirs. Saving to a name that exists, when this conversation did not come from it, is refused. A
  save waits a few seconds for a run that is saving at the same moment. The sessions folder must belong
  to you, be mode 700, and sit on a path nobody else can redirect or change (links of other users and
  group- or world-writable folders without the sticky bit are refused).
- **`/checkpoint [name]` also keeps the model's processed state** of a saved session, so that `/resume`
  skips reading the history again. It is separate from `/save` and off unless you ask. Qwen3.5 on the
  GPU only; at least 256 tokens; the session must be saved and unchanged. The file is `<name>.state`
  beside the session: about 51 MiB plus 64 KiB per token (82 MiB at 500 tokens, 307 MiB at 4,096),
  at most 1 GiB each and 4 GiB together, mode 600. **It is derived from your conversation, code
  included.** `/resume` uses it only if the session revision, model, weights, precision and state
  layout match and every block of the file passes its hash; otherwise it says why and reads the
  history instead. Saving the session again removes its checkpoint. On Qwen3.5-4B-Q8 a 2,265-token
  resume takes 9.3 s instead of 25.6 s ([measurements](docs/benchmarks/archive/15-checkpoint-feasibility.log)).

**Code mode** (`marin code [PATH]`, or `/mode code`) is read-only: the model uses three
tools, shown as quiet `›` lines, and cannot write files, run commands or leave the project.

- Tools: `list_files` (a folder, up to 3 levels), `read_file` (numbered lines, 250 at a
  time, with the line to continue from) and `search` (literal text, `a|b` for either,
  `a.*b` for in order, case-insensitive unless the pattern has capitals). Results are
  bounded by the context size; binary, huge and unreadable files are reported, not shown.
  Symlinks never lead out of the project and `..` or absolute paths outside it are
  refused. File text cannot pose as tool markup. Code mode decodes greedily and wants a
  window of at least 2048 tokens (`marin code` starts at 8192, reduced if memory is short).
- When the window fills, old tool output is shortened first, then the oldest questions are
  dropped, with a note. A failed or cancelled question leaves no trace. Ctrl-C stops an
  answer within a fraction of a second; `--log FILE` records every request and tool call.
- A 4B model is unreliable here: it answers single lookups well, but sometimes breaks the
  call format, wanders on questions that span files, and can say code is missing when it did
  not look in the right place. Check its `path:line` citations ([BENCHMARKS.md](BENCHMARKS.md)).
- `marin code` still works as a client: with `--port N` it uses the `marin serve` there
  instead of loading a model, and without `--port` it reuses a server already listening on
  port 8000 only if it serves the same model with a large enough window (otherwise it says
  why and loads locally; a flag the server cannot honor, such as `--backend`, loads locally). `marin serve` is unchanged; it is
  the headless API and never reads the saved backend or sampling values.

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
Face tests also need the named local model and reference files.
