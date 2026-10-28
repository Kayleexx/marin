#!/usr/bin/env python3
"""The 25-question repository benchmark for `marin code`, run on this repository.

  python3 -I scripts/code_eval.py run   --model DIR [--quantize int8] --tag TAG [--out DIR] [IDS...]
  python3 -I scripts/code_eval.py score --tag TAG [--out DIR]

`run` asks each question with `marin code .` (temperature 0, 8192-token window) and keeps the transcript and the
`--log` events as OUT/TAG-ID.txt and OUT/TAG-ID.jsonl. `score` grades the final answers: a run passes when its answer
contains every fact in `all`, at least one of `any`, and nothing in `none` (case-insensitive regular expressions);
unanswerable questions pass when the answer admits it and names no file as the place. t1-t5 and a1-a12 are the
development set (17); b1-b8 are the held-out set (8), written and fixed before their first run.

The answers depend on this repository's files, so run it on the commit you are checking and compare tags from the
same commit. It needs a built `marin` (cargo install --path crates/cli --locked, or target/release/marin) and a
Qwen3.5-4B chat model; the published numbers used the packed 8-bit export on an 8 GB GPU (see BENCHMARKS.md).
Replies are sampled greedily but tool order can differ between runs, so expect a task or two of run-to-run noise."""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

TASKS = {
    't1': 'Where is the function max_active defined, and what does it decide?',
    't2': 'Explain how a streamed chat completion request travels through the server: from the HTTP connection to the tokens being sent back. Name the main functions and files.',
    't3': 'Clients sometimes get a 404 "session not found, evicted or in use". Which code produces that error, and under what conditions does it happen?',
    't4': 'What time limits does the server enforce on requests, and where are they defined?',
    't5': 'Where does Marin store user accounts and passwords for the server login?',
    'a1': 'Where is the constant SLICE defined and what is its value?',
    'a2': 'Which function checks that a session id is valid, and in which file?',
    'a3': 'What does the Interrupt type do when Ctrl-C is pressed while a reply is being generated?',
    'a4': 'What is the default port of marin serve?',
    'a5': 'What does Store::evict do and where is it called?',
    'a6': 'Which crates make up the workspace?',
    'a7': 'What context size does the interactive chat use by default?',
    'a8': 'Where is the shared prompt prefix built and what does it contain?',
    'a9': 'Which tools does the coding assistant give the model?',
    'a10': 'Where is the Kubernetes deployment configuration for the server?',
    'a11': 'Why might a request whose user message starts with a newline not use the shared prompt prefix?',
    'a12': 'Which file decides how many requests run at the same time, and what limits does it apply?',
    'b1': 'Where is the function that renders chat messages into the model prompt text?',
    'b2': 'What does the /clear command do in marin code?',
    'b3': 'How many rounds of tool use does marin code allow for one question?',
    'b4': 'How many entries does list_files show at most, and where is that limit set?',
    'b5': 'How does marin code decide that a tool call is malformed?',
    'b6': 'What is GPU_SLACK_BYTES and where is it used?',
    'b7': 'Which file implements export_int8?',
    'b8': 'Where is the Windows installer script?',
}

RUBRIC = {
    "t1": dict(all=[r"sched\.rs", r"\b26\b"]),
    "t2": dict(all=[r"completion\.rs", r"request\.rs"], any=[r"generate_reusing", r"gate"]),
    "t3": dict(all=[r"completion\.rs", r"take"], none=[r"expired"]),
    "t4": dict(all=[r"queue\.rs", r"READ_LIMIT|10 ?s", r"RUN_LIMIT|300", r"WAIT_LIMIT"]),
    "t5": dict(any=[r"not find|did not find|no (reference|code|evidence|mention)|could not|cannot find|no such|does not (store|contain|have|appear)|not (stored|present)"], none=[r"stored in `?crates"]),
    "a1": dict(all=[r"gate\.rs", r"\b64\b"]),
    "a2": dict(all=[r"valid_id", r"session\.rs"]),
    "a3": dict(all=[r"signal\.rs", r"cancel"], any=[r"second|again|exit"]),
    "a4": dict(all=[r"8000"]),
    "a5": dict(all=[r"session\.rs", r"completion\.rs"]),
    "a6": dict(all=[r"cli", r"cuda", r"model", r"runtime"]),
    "a7": dict(all=[r"4096|4,096"]),
    "a8": dict(all=[r"prefix\.rs", r"request\.rs|default_prefix"]),
    "a9": dict(all=[r"list_files", r"read_file", r"search"]),
    "a10": dict(any=[r"not find|did not find|no (reference|code|evidence|mention|kubernetes|deployment)|could not|cannot find|no such|does not (contain|have|appear)|not (present|part|found)"], none=[r"\.ya?ml`"]),
    "a11": dict(all=[r"token"], any=[r"prefix\.rs", r"request\.rs", r"starts_with"]),
    "a12": dict(all=[r"sched\.rs", r"MAX_ACTIVE|\b3\b"], any=[r"memory|VRAM"]),
    # Held-out set: written and fixed before the first run; the harness was not changed after seeing it.
    "b1": dict(all=[r"qwen3\.rs", r"render"]),
    "b2": dict(all=[r"clear"], any=[r"conversation|history"]),
    "b3": dict(all=[r"\b8\b"]),
    "b4": dict(all=[r"\b100\b", r"tools\.rs"]),
    "b5": dict(all=[r"call\.rs"], any=[r"never closed|unknown tool|needs a|parse"]),
    "b6": dict(all=[r"GPU_SLACK_BYTES"], any=[r"engine\.rs", r"engine_session\.rs", r"sched\.rs", r"fit\.rs"]),
    "b7": dict(all=[r"export\.rs"]),
    "b8": dict(any=[r"not find|did not find|no (reference|code|evidence|mention|windows|installer)|could not|cannot find|no such|does not (contain|have|appear|exist)|not (present|part|found)"], none=[r"\.ps1|\.msi|install\.bat"]),
}

def answer(text):
    lines = text.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if l.startswith("  ›") or l.startswith("  ! ") or l.startswith("  · "):
            start = i + 1
    return "\n".join(lines[start:])

def metrics(path):
    m = dict(tools=0, invalid=0, repeated=0, repaired=0, prompt=0, completion=0, replies=0, tool_ms=0, reply_ms=0, bad_tool=0)
    if not os.path.exists(path): return m
    for l in open(path):
        j = json.loads(l)
        e = j.get("event")
        if e == "tool":
            m["tools"] += 1; m["tool_ms"] += j["ms"]; m["bad_tool"] += 0 if j["ok"] else 1
        elif e == "invalid": m["invalid"] += 1
        elif e == "repeated": m["repeated"] += 1
        elif e == "repaired": m["repaired"] += j["calls"]
        elif e == "reply":
            m["replies"] += 1; m["prompt"] += j["prompt_tokens"]; m["completion"] += j["completion_tokens"]; m["reply_ms"] += j["ms"]
    return m


def run(args):
    ids = args.ids or list(TASKS)
    os.makedirs(args.out, exist_ok=True)
    home = tempfile.mkdtemp(prefix="marin-eval-")
    env = dict(os.environ, MARIN_HOME=home, NO_COLOR="1")
    exe = args.marin or shutil.which("marin") or "target/release/marin"
    cmd = [exe, "--model", args.model, "--ctx", "8192", "--temp", "0", "--max-tokens", "700"]
    if args.quantize:
        cmd += ["--quantize", args.quantize]
    try:
        for tid in ids:
            base = os.path.join(args.out, f"{args.tag}-{tid}")
            with open(base + ".txt", "w") as out:
                subprocess.run(cmd + ["code", ".", "--log", base + ".jsonl"], input=TASKS[tid] + "\n", text=True,
                               stdout=out, stderr=subprocess.STDOUT, env=env, timeout=args.timeout)
            print("ran", tid, flush=True)
    finally:
        shutil.rmtree(home, ignore_errors=True)

def score(args):
    d, tag = args.out, args.tag
    rows = []
    for tid in sorted(RUBRIC, key=lambda t: (t[0], int(t[1:]))):
        p = os.path.join(d, f"{tag}-{tid}.txt")
        if not os.path.exists(p): continue
        text = open(p).read()
        ans = answer(text)
        r = RUBRIC[tid]
        ok = all(re.search(x, ans, re.I) for x in r.get("all", [])) \
            and (not r.get("any") or any(re.search(x, ans, re.I) for x in r["any"])) \
            and not any(re.search(x, ans, re.I) for x in r.get("none", []))
        if "error:" in text: ok = False
        m = metrics(os.path.join(d, f"{tag}-{tid}.jsonl"))
        secs = (m["tool_ms"] + m["reply_ms"]) / 1000
        rows.append((tid, ok, m, secs))
        print(f"{tag} {tid:4s} {'PASS' if ok else 'fail'}  tools {m['tools']:2d} (bad {m['bad_tool']}) invalid {m['invalid']} "
              f"repaired {m['repaired']} repeated {m['repeated']}  requests {m['replies']}  prompt {m['prompt']:5d} gen {m['completion']:4d}  {secs:5.1f} s")
    n = len(rows)
    if not n:
        sys.exit("no runs found: run the questions first")
    dev = [r for r in rows if r[0][0] in "ta"]; held = [r for r in rows if r[0][0] == "b"]
    print(f"\n{sum(r[1] for r in rows)}/{n} pass (development {sum(r[1] for r in dev)}/{len(dev)}, held-out {sum(r[1] for r in held)}/{len(held)}); "
          f"tool calls {sum(r[2]['tools'] for r in rows)}, invalid {sum(r[2]['invalid'] for r in rows)}, "
          f"repaired {sum(r[2]['repaired'] for r in rows)}, repeated {sum(r[2]['repeated'] for r in rows)}; "
          f"total {sum(r[3] for r in rows):.0f} s, prompt tokens {sum(r[2]['prompt'] for r in rows)}, generated {sum(r[2]['completion'] for r in rows)}")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--quantize")
    r.add_argument("--marin"); r.add_argument("--timeout", type=int, default=600); r.add_argument("ids", nargs="*")
    s = sub.add_parser("score")
    for x in (r, s):
        x.add_argument("--tag", required=True); x.add_argument("--out", default="eval-out")
    args = ap.parse_args()
    {"run": run, "score": score}[args.cmd](args)

if __name__ == "__main__":
    main()
