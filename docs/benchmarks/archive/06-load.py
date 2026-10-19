#!/usr/bin/env python3
"""Multi-client load generator for marin serve. usage: load.py PORT PID SCENARIO [RUNS]"""
import json, sys, time, threading, subprocess, http.client, os, statistics

PORT, PID, SCEN = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MODEL = "Qwen3.5-4B-Q8"
SRC = open("/home/mitali/miracles/marin/docs/benchmarks/archive/06-long-prompt.txt").read()
def long_prompt(chars):
    return "Review this Rust code and list three bugs briefly.\n\n" + SRC[:chars]
SHORT = "Say hello in one short sentence."
SHORT2 = "Give me one tip for writing readable Rust, in one sentence."

def client(name, delay, prompt, max_tokens, out, t0, extra=None):
    time.sleep(max(0, delay - (time.time() - t0)))
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True}
    if extra: body.update(extra)
    start = time.time(); first = None; events = 0; text = ""; status = None
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=400)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse(); status = r.status
        if status != 200:
            msg = r.read().decode()[:100]; text = msg
        else:
            for line in r:
                if not line.startswith(b"data: "): continue
                d = line[6:].strip()
                if d == b"[DONE]": break
                j = json.loads(d)
                ch = j.get("choices", [{}])[0].get("delta", {}).get("content")
                if ch:
                    events += 1; text += ch
                    if first is None: first = time.time()
        c.close()
    except Exception as e:
        status = f"ERR {e}"
    end = time.time()
    out[name] = dict(start=start - t0, ttft=(first - start) if first else None, total=end - start,
                     events=events, status=status, text=text,
                     rate=((events - 1) / (end - first)) if first and events > 1 and end > first else None)

def rss_kib():
    for l in open(f"/proc/{PID}/status"):
        if l.startswith("VmRSS"): return int(l.split()[1])
def vram_mib():
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for l in o.splitlines():
        p, m = l.split(",")
        if int(p) == PID: return int(m)
def gpu_util():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [float(x) for x in o.strip().split(",")]

def sampler(stop, rows):
    while not stop.is_set():
        try: rows.append((rss_kib(), vram_mib() or 0, *gpu_util()))
        except Exception: pass
        time.sleep(0.5)

LONG_CHARS = int(os.environ.get("LONG_CHARS", "4500"))
SCENARIOS = {
  # name: list of (client, delay, prompt-fn, max_tokens)
  "solo_short": [("s1", 0, lambda: SHORT, 32)],
  "solo_long": [("L", 0, lambda: long_prompt(LONG_CHARS), 64)],
  "three_interactive": [("a", 0, lambda: SHORT, 32), ("b", 0.05, lambda: SHORT2, 32), ("c", 0.1, lambda: SHORT, 32)],
  "decode1": [("d1", 0, lambda: "Write a long numbered list of tips for writing readable Rust code.", 128)],
  "decode2": [("d1", 0, lambda: "Write a long numbered list of tips for writing readable Rust code.", 128), ("d2", 0.05, lambda: "Write a long numbered list of tips for planning a trip to Kyoto.", 128)],
  "decode3": [("d1", 0, lambda: "Write a long numbered list of tips for writing readable Rust code.", 128), ("d2", 0.05, lambda: "Write a long numbered list of tips for planning a trip to Kyoto.", 128), ("d3", 0.1, lambda: "Write a long numbered list of tips for learning to cook.", 128)],
  "long_plus_two": [("L", 0, lambda: long_prompt(LONG_CHARS), 64), ("a", 1.0, lambda: SHORT, 32), ("b", 2.0, lambda: SHORT2, 32)],
}
def run_once():
    out = {}; t0 = time.time(); ths = []
    rows = []; stop = threading.Event(); st = threading.Thread(target=sampler, args=(stop, rows)); st.start()
    for name, delay, pf, mt in SCENARIOS[SCEN]:
        t = threading.Thread(target=client, args=(name, delay, pf(), mt, out, t0)); t.start(); ths.append(t)
    for t in ths: t.join()
    wall = time.time() - t0; stop.set(); st.join()
    return out, wall, rows

runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
results = []
for i in range(runs):
    out, wall, rows = run_once()
    toks = sum(v["events"] for v in out.values())
    peak_rss = max((r[0] for r in rows), default=0) / 1024; peak_vram = max((r[1] for r in rows), default=0)
    util = statistics.mean(r[2] for r in rows) if rows else 0
    print(f"--- {SCEN} run {i+1}: wall {wall:.2f}s events {toks} agg {toks/wall:.1f} ev/s  rss_peak {peak_rss:.0f}MiB vram_peak {peak_vram}MiB gpu_util_avg {util:.0f}% pw_avg {statistics.mean(r[3] for r in rows) if rows else 0:.0f}W")
    for k, v in sorted(out.items()):
        print(f"  {k}: status={v['status']} start={v['start']:.2f} ttft={v['ttft'] if v['ttft'] is None else round(v['ttft'],3)} total={v['total']:.2f} events={v['events']} rate={v['rate'] and round(v['rate'],1)}")
    results.append(dict(out=out, wall=wall))
json.dump(results, open(os.environ.get("OUT", f"/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad/{SCEN}.json"), "w"))
