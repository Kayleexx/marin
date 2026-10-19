#!/usr/bin/env python3
"""Behaviour checks against a running Qwen3.5-4B server. usage: 10-checks.py PORT  (prints JSON; diff two servers)"""
import json, sys, time, threading, http.client
PORT = int(sys.argv[1]); M = "Qwen3.5-4B-Q8"
CODE = (("Explain Rust ownership, borrowing, lifetimes and Send/Sync with a practical example. " * 8).strip(), "Now write one small Rust function that illustrates borrowing.")
CHAT = (("Explain how to plan a two-day visit to Kyoto while minimizing travel time and keeping costs low. " * 8).strip(), "Now adapt that plan for rainy weather and a vegetarian traveler.")

def ask(messages, session=None, stream=False, max_tokens=24, cut=None, system=None):
    if system: messages = [{"role": "system", "content": system}] + messages
    body = {"model": M, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": stream}
    if session: body["marin_session_id"] = session
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=200)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if not stream:
        raw = r.read().decode()
        try: j = json.loads(raw)
        except Exception: return r.status, raw, None, None
        if r.status != 200: return r.status, raw[:80], None, None
        return 200, j["choices"][0]["message"]["content"], j.get("marin_session_id"), j.get("marin_reused_tokens")
    text, sid, reused, t0 = "", None, None, time.time()
    for line in r:
        if not line.startswith(b"data: "): continue
        d = line[6:].strip()
        if d == b"[DONE]": break
        j = json.loads(d)
        for ch in j.get("choices", []): text += ch.get("delta", {}).get("content") or ""
        sid = j.get("marin_session_id", sid); reused = j.get("marin_reused_tokens", reused)
        if cut and time.time() - t0 > cut: c.close(); return r.status, text, sid, reused
    return r.status, text, sid, reused

def user(t): return {"role": "user", "content": t}
def asst(t): return {"role": "assistant", "content": t}
out = {}
out["stateless_hello"] = ask([user("Say hello in one short sentence.")])[:2]
out["newline_leading"] = ask([user("\nweird newline start")])[:2]
out["custom_system"] = ask([user("Say hello in one short sentence.")], system="You are a terse assistant.")[:2]
for name, (first, second) in (("code", CODE), ("chat", CHAT)):
    s1, a1, sid, r1 = ask([user(first)], session="new", stream=True)
    s2, a2, _, r2 = ask([user(first), asst(a1), user(second)], session=sid, stream=True)
    s3, a3, _, r3 = ask([user(first), asst(a1), user(second)])
    out[f"pair_{name}"] = {"turn1": [s1, a1], "turn2": [s2, a2], "turn2_reused": r2, "turn2_equals_stateless": a2 == a3}
res = {}
def conc(name, messages, session=None):
    res[name] = ask(messages, session=session, stream=True)
sids = {}
for k, text in (("A", CODE[0]), ("B", CHAT[0])):
    s, a, sid, _ = ask([user(text)], session="new", stream=True); sids[k] = (sid, text, a)
ths = [threading.Thread(target=conc, args=(k, [user(t), asst(a), user("Continue briefly.")], sid)) for k, (sid, t, a) in sids.items()]
[t.start() for t in ths]; [t.join() for t in ths]
fresh = {k: ask([user(t), asst(a), user("Continue briefly.")], stream=True)[1] for k, (sid, t, a) in sids.items()}
out["isolation"] = {k: [res[k][0], res[k][0] != 200 or res[k][1] == fresh[k]] for k in res}
sid, text, a = sids["B"]
s, a, sid, _ = ask([user(text)], session="new", stream=True)
ths = [threading.Thread(target=conc, args=(f"same{i}", [user(text), asst(a), user("Continue briefly.")], sid)) for i in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
out["same_session_concurrent"] = sorted(res[f"same{i}"][0] for i in range(2))
ask([user("Say hello in one short sentence.")], stream=True, cut=0.15)
time.sleep(0.5)
out["after_disconnect"] = ask([user("Say hello in one short sentence.")])[:2]
codes = {}
def five(i): codes[i] = ask([user(f"Write a long numbered list of tips about topic {i}.")], max_tokens=200)[0]
ths = [threading.Thread(target=five, args=(i,)) for i in range(5)]
[t.start() for t in ths]; [t.join() for t in ths]
out["five_simultaneous"] = sorted(codes.values())
print(json.dumps(out, indent=1, sort_keys=True))
