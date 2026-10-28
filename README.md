# Marin

Marin is a private assistant that runs on your own computer. You talk to a language model in your terminal, and
nothing leaves your machine. It is written in Rust, runs on a normal CPU, and uses an NVIDIA GPU when you have one.

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

## Background assistant (optional, off by default)

A tiny helper that can suggest a fix when a command you chose to watch fails in a repo you allowed. It uses simple
rules, never loads a model, and sleeps when idle (about 7 MB).

```bash
marin assist allow                 # in a git repo
marin assist start
marin assist watch -- cargo test   # runs your command as usual
marin assist inbox                 # then: why 1 | accept 1 | dismiss 1 | stop | deny
```

It keeps only the last 4 KiB of output, with secrets filtered. Run `marin assist --help` for the rest.

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
| GPU | 32 to 36 tokens/s | 319 tokens/s | 1.6 s |
| CPU, 8 threads | 4.8 tokens/s | 44 tokens/s | 11.6 s |

Qwen3-1.7B on the CPU writes about 13 tokens/s and Qwen3-0.6B about 37. Follow-up turns reuse the cache, so they are much
cheaper than the first one. More numbers and how we measured them: [BENCHMARKS.md](BENCHMARKS.md).

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
marin assist stop; marin assist deny     # only if you used the assistant
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
