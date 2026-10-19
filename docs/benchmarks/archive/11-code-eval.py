#!/usr/bin/env python3
"""Scores `marin code` runs on Marin's own repository. usage: 11-code-eval.py TASK_DIR TAG [TAG...]
Each run is TASK_DIR/TAG-ID.txt (the transcript) and TAG-ID.jsonl (`--log`). A run passes when its final answer
contains every fact in `all`, at least one of `any`, and nothing in `none` (case-insensitive regular expressions);
unanswerable questions pass when the answer admits it and names no file as the place."""
import json, re, sys, os

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

def main():
    d, tags = sys.argv[1], sys.argv[2:]
    rows = []
    for tag in tags:
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
            rows.append((tag, tid, ok, m, secs))
            print(f"{tag} {tid:4s} {'PASS' if ok else 'fail'}  tools {m['tools']:2d} (bad {m['bad_tool']}) invalid {m['invalid']} repaired {m['repaired']} repeated {m['repeated']}  "
                  f"requests {m['replies']}  prompt {m['prompt']:5d} gen {m['completion']:4d}  {secs:5.1f} s")
    n = len(rows)
    if n:
        print(f"\n{sum(r[2] for r in rows)}/{n} pass; tool calls {sum(r[3]['tools'] for r in rows)}, "
              f"invalid {sum(r[3]['invalid'] for r in rows)}, repaired {sum(r[3]['repaired'] for r in rows)}, repeated {sum(r[3]['repeated'] for r in rows)}; "
              f"median {sorted(r[4] for r in rows)[n // 2]:.1f} s, prompt tokens {sum(r[3]['prompt'] for r in rows)}, generated {sum(r[3]['completion'] for r in rows)}")

main()
