import json, os, subprocess, sys, time, urllib.request, threading
ROOT = "/home/mitali/miracles/marin"
S = os.path.dirname(os.path.abspath(__file__)) + "/"
BIN = os.environ.get("MARIN", ROOT + "/target/release/marin")
BALLAST = ROOT + "/target/release/deps/zz_ballast-545db35539485d0c"
target, turns, pause, rep, ctx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sessions = int(os.environ.get("SESSIONS", "1"))
port = 8190
def free(): return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"]).split()[0])
def mine(pid):
    o = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"]).decode()
    for l in o.splitlines():
        a = l.split(", ")
        if a[0] == str(pid): return int(a[1])
    return 0
log = open(S + f"srv.{target}.{ctx}.log", "w")
srv = subprocess.Popen([BIN, "--model", os.path.expanduser("~/models/Qwen3.5-4B-Q8"), "--quantize", "int8", "--backend", "cuda", "--ctx", ctx, "serve", "--port", str(port)], stdout=log, stderr=log)
for _ in range(90):
    time.sleep(1)
    if "listening" in open(S + f"srv.{target}.{ctx}.log").read(): break
print(open(S + f"srv.{target}.{ctx}.log").read().strip().splitlines()[-1][:200])
peak = [0]; stop = [False]
def watch():
    while not stop[0]:
        peak[0] = max(peak[0], mine(srv.pid)); time.sleep(0.4)
threading.Thread(target=watch, daemon=True).start()
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r: d = json.load(r); code = r.status
    except urllib.error.HTTPError as e: d = json.loads(e.read() or b"{}"); code = e.code
    return code, d, time.time() - t
state = [{"id": "new", "msgs": []} for _ in range(sessions)]
ballast = None
flag = S + "ballast.flag"
def run_session(i, n, out):
    st = state[i]
    text = os.environ.get("PROMPT") or f"Note {n} of conversation {i}: the lighthouse keeper climbed the spiral stairs each evening, trimmed the lamp and watched the ships. " * rep
    if os.environ.get("NOSESSION"): st["msgs"] = []
    st["msgs"].append({"role": "user", "content": text})
    code, d, t = post({"model": "Qwen3.5-4B-Q8", "messages": st["msgs"], "max_tokens": int(os.environ.get("MAXTOK", "8")), "temperature": 0, **({} if os.environ.get("NOSESSION") else {"marin_session_id": st["id"]})})
    if code == 200:
        if not os.environ.get("NOSESSION"): st["id"] = d["marin_session_id"]
        st["msgs"].append(d["choices"][0]["message"])
        out[i] = f"s{i} t{n+1}: 200 {t:6.1f}s prompt {d['usage']['prompt_tokens']:5} reused {d['marin_reused_tokens']:5}"
    else:
        st["msgs"].pop()
        out[i] = f"s{i} t{n+1}: {code} {t:6.1f}s {str(d)[:140]}"
for n in range(turns):
    if n == pause and target != "none":
        mib = free() - int(target)
        open(flag, "w").close()
        ballast = subprocess.Popen([BALLAST, "--nocapture"], env={**os.environ, "MIB": str(mib), "FLAG": flag}, stdout=subprocess.PIPE, text=True)
        for l in ballast.stdout:
            if "BALLAST held" in l: break
        print(f"ballast {mib} MiB, free now {free()} MiB, server holds {mine(srv.pid)} MiB")
    out = [None] * sessions
    ts = [threading.Thread(target=run_session, args=(i, n, out)) for i in range(sessions)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(" | ".join(out), f"| server {mine(srv.pid)} MiB peak {peak[0]}", flush=True)
stop[0] = True
if ballast:
    os.remove(flag); ballast.wait()
srv.terminate(); srv.wait()
tail = [l for l in open(S + f"srv.{target}.{ctx}.log").read().splitlines() if "error" in l.lower() or "fail" in l.lower() or "panic" in l.lower()]
print("server log issues:", tail[-5:])
