#!/usr/bin/env python3
"""A client dropping 2 s into a three-way batched decode. usage: 10-cancel.py PORT REFERENCE_DECODE3_JSON"""
import json, sys, time, threading, http.client
PORT = int(sys.argv[1]); M = "Qwen3.5-4B-Q8"
P = {"d1": "Write a long numbered list of tips for writing readable Rust code.", "d2": "Write a long numbered list of tips for planning a trip to Kyoto.", "d3": "Write a long numbered list of tips for learning to cook."}
out = {}
def client(name, delay, cut):
    time.sleep(delay)
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    c.request("POST", "/v1/chat/completions", json.dumps({"model": M, "messages": [{"role": "user", "content": P[name]}], "temperature": 0, "max_tokens": 128, "stream": True}), {"Content-Type": "application/json"})
    r = c.getresponse(); txt = ""; t0 = time.time()
    for line in r:
        if not line.startswith(b"data: "): continue
        d = line[6:].strip()
        if d == b"[DONE]": break
        for ch in json.loads(d).get("choices", []): txt += ch.get("delta", {}).get("content") or ""
        if cut and time.time() - t0 > cut: c.close(); break
    out[name] = txt
ths = [threading.Thread(target=client, args=a) for a in (("d1", 0, None), ("d2", 0.05, None), ("d3", 0.1, 2.0))]
[t.start() for t in ths]; [t.join() for t in ths]
ref = {k: v["text"] for k, v in json.load(open(sys.argv[2]))[0]["out"].items()}
print("survivors equal reference:", out["d1"] == ref["d1"], out["d2"] == ref["d2"], "| cancelled client got", len(out["d3"]), "of", len(ref["d3"]), "chars")
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
c.request("POST", "/v1/chat/completions", json.dumps({"model": M, "messages": [{"role": "user", "content": P["d3"]}], "temperature": 0, "max_tokens": 128}), {"Content-Type": "application/json"})
print("same prompt afterwards equals reference:", json.loads(c.getresponse().read())["choices"][0]["message"]["content"] == ref["d3"])
