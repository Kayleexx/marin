# Marin

Marin is a private assistant that runs on your own computer. You talk to a language model in your terminal, and
nothing leaves your machine.

**Marin runs on its own inference runtime.** It does not wrap llama.cpp, Python or any other engine. The model loader,
the CPU and GPU math, the cache, the sampler and the server are all written in Rust in this repository (the GPU kernels
are CUDA). It runs on a normal CPU and uses an NVIDIA GPU when you have one.

It runs selected Qwen3, Qwen3.5 and Granite models. It never downloads anything and never sends data anywhere.

## Install

You need Rust (the version in `rust-toolchain.toml`). For GPU use you also need an NVIDIA driver and the CUDA toolkit.
The CPU mode needs neither.

```bash
cargo install --path crates/cli --locked
```

This gives you two programs: `marin` for everyday use and `marin-run` for measuring and checking models.

## Get a model

Marin reads a normal Hugging Face model folder. Put your models in `~/models` (or set `MARIN_MODELS`), then run:

```bash
marin                               # lists your models, shows what fits this machine, and remembers your choice
marin model ~/models/Qwen3-1.7B     # or pick one yourself
marin plan                          # shows the memory plan without loading anything
```

For an 8 GB GPU, turn Qwen3.5-4B into a smaller 8-bit copy once, then use it:

```bash
marin-run export-int8 ~/models/Qwen3.5-4B ~/models/Qwen3.5-4B-Q8 --quant-group 128
marin --model ~/models/Qwen3.5-4B-Q8 --quantize int8
```

Marin never changes your model, precision or backend behind your back. If something does not fit, it tells you why
instead of shrinking it.

## Use it

```bash
marin                                          # chat; type / to see actions
marin run "Explain a mutex in one sentence."   # one answer, then exit (also reads from a pipe)
marin code                                     # ask questions about the project in this folder
marin serve --port 8000                        # a small local API on 127.0.0.1
```

In the chat, **Ctrl-C** stops a reply (a stopped question is not kept). Type **/** for `/save`, `/resume`, `/checkpoint`,
`/mode code`, `/model`, `/status`, `/settings`, `/bench`, `/verify`, `/clear` and `/exit`. If the conversation gets longer
than the context window, Marin drops the oldest messages and tells you.

**Saving.** Nothing is saved unless you ask. `/save` asks for a name and writes the text of the conversation to
`~/.local/state/marin/sessions` (private files). `/resume` reads it back. A saved code session contains source code the
model read, so treat it like the project itself. `/checkpoint` (Qwen3.5 on GPU) also keeps the model's state so a resume
is faster; if anything does not match, Marin simply reads the text again.

**Server.** `marin serve` answers `/v1/models` and `/v1/chat/completions` in the OpenAI style. It listens on loopback only
and has no login, so keep it that way. Send `"marin_session_id":"new"` on the first turn and the returned id after that to
reuse the cache. `--idle-unload SECONDS` frees the GPU when idle and reloads on the next request.

## Background awareness (optional, off by default)

A tiny helper that notices a few things you approve and offers help, without ever loading a model or touching the GPU.
It never runs a command, edits a file or sends data anywhere. A suggestion only reaches a model when you accept it and
press Enter in a normal chat. Idle, it uses 4 to 7 MB and no measurable CPU.

Everything is a slash action in the normal chat (type `/`), and each also exists as `marin assist ...`:

| Action | What it does |
|---|---|
| `/awareness` | `enable`, `disable`, `status`, `desktop on`, `browser on`, `allow-app NAME`, `allow-site HOST`, `autostart on` |
| `/context` | what the helper holds right now (memory only, at most 32 items for 30 minutes) |
| `/suggestions` | `list`, `accept N`, `dismiss N`, `mute N`, `snooze N` |
| `/why` | the rule behind a suggestion and exactly what accepting would put in your input line |

Nothing is watched until you choose each source:

- **Failed commands:** `marin assist allow` in a git repo, then `marin assist watch -- cargo test`. The last 4 KiB of output is kept, secrets filtered.
- **Browser tabs (Firefox only):** `marin assist browser install` writes the link Firefox needs (a small manifest in
  `~/.mozilla/native-messaging-hosts` and a launcher in Marin's folder; `browser remove` deletes both). Then add the
  extension in `extras/browser/`. **Normal Firefox only runs signed extensions**, so a general user has two options:
  load it temporarily (`about:debugging`, This Firefox, Load Temporary Add-on; it disappears when Firefox restarts), or
  sign it yourself as an unlisted add-on (`web-ext sign` with your own addons.mozilla.org key) and install the `.xpi`.
  Developer Edition, Nightly and ESR can also allow unsigned add-ons. Marin does not ship a signed build, and this has
  only been tested with a temporary load in Firefox 155. Then `/awareness browser on` and `/awareness allow-site docs.rs`.
  Only the title and address (host and path, no query) of allowed sites are sent, never page text. The extension is
  set to `incognito: not_allowed` and also skips private tabs, so private windows are not observed. The link accepts
  only the Marin extension's id, and the helper drops any site you did not allow.
- **Windows (experimental, X11 only):** `/awareness desktop on` and `/awareness allow-app firefox`. It listens to the X
  server's focus events (no polling), reads a title only for apps you allowed, and has not been verified on a real X11
  session. Wayland does not let programs watch other windows, so there you use `marin assist share "title" --url URL`
  (text may come from a pipe) to hand something over yourself.

A suggestion appears only for a reason you can read with `/why`: a watched command failed, or an allowed page about the same
error was opened within 15 minutes of it, or you shared something. Merely switching apps does nothing. Repeats are counted,
not repeated; at most one suggestion per repo every 5 minutes and 6 per hour; `dismiss` silences it for a day and `mute`
silences that repo or site until you unmute it. `/awareness disable` stops the helper and drops everything it held. It keeps
running after the terminal closes; start it at login only if you ask: `/awareness autostart on`.

Linux and other Unix systems only. Windows and macOS are not supported in v0.1: the helper uses Unix sockets and has no adapters for them. Only one helper runs at a time, and if it is killed nothing is left running.

## Models and speed

| Model | Where it runs | Status |
|---|---|---|
| Qwen3 0.6B, 1.7B | CPU, GPU | works well |
| Qwen3.5-4B | CPU, GPU (8-bit on an 8 GB card) | works well; coding is experimental |
| Qwen3.5-2B base | CPU, GPU | via `marin-run` |
| Granite 3.0 1B-A400M | CPU | via `marin-run`; 8-bit and GPU are experimental |

You need at least 4 GB of free RAM for Qwen3 0.6B or 1.7B on the CPU. For Qwen3.5-4B, 16 GB of RAM and an 8 GB NVIDIA
GPU are comfortable. On a Ryzen 7 260 with an RTX 5050 laptop GPU (about 40 W), Qwen3.5-4B in 8-bit does:

| | Writing | Reading the prompt | First word (512-token prompt) |
|---|---:|---:|---:|
| GPU | 31 tokens/s (about 45 for the first 20 s) | about 500 tokens/s | 1.0 s |
| CPU, 8 threads | 4.8 tokens/s | 44 tokens/s | 11.6 s |

Qwen3-1.7B on the CPU writes about 13 tokens/s and Qwen3-0.6B about 37. Follow-up turns reuse the cache, so they are much
cheaper than the first one.

**Compared with llama.cpp** (same machine, same model, 8-bit): Marin writes about 15% faster on the GPU, uses far less
system memory (0.9 GB against 4.8 GB), and finishes short answers sooner. It reads long prompts more slowly (about 3.5 to 4
times slower on the GPU), so the first word of a long prompt arrives later. On the CPU the two are close when writing.
Details, caveats and how we measured: [BENCHMARKS.md](BENCHMARKS.md).

## Things to know

- **Coding mode is experimental.** It can list, read and search files, and nothing else. On 25 questions about this repo
  it found the right place 14 to 16 times. Its misses sound confident, so use it to find `path:line`, then read the line.
- **Guessing ahead (`--speculate`)** is optional, GPU only, and not used by the server. Greedy answers stay identical.
- **Sessions are text.** The server forgets its sessions when it restarts.
- **Low GPU memory.** A request that finds no memory is undone and retried or refused with a 503, never half done.
- Marin estimates memory but cannot reserve it against other programs. Anything on the same machine can reach the
  server on loopback.

## Uninstall

```bash
marin assist awareness disable; marin assist awareness autostart off; marin assist browser remove   # only if you used awareness
cargo uninstall cli                      # removes marin and marin-run
rm -r ~/.config/marin ~/.local/state/marin
```

Marin writes nothing else outside those two folders (`$MARIN_HOME` replaces both).

## Develop

```bash
cargo fmt --all && cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked     # real-model tests need MARIN_REQUIRE_CUDA=1 MARIN_QWEN35_Q8_DIR=...
scripts/check-line-limit.sh && scripts/check-unsafe.sh
```

The rules: safe Rust outside the `cuda` crate, no panicking code in production paths, and no file over 425 lines. CI runs
the same checks. The coding benchmark is `scripts/code_eval.py`; see [BENCHMARKS.md](BENCHMARKS.md) to run it.

## License

Apache-2.0, see [LICENSE](LICENSE).
