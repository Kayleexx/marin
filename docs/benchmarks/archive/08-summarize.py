import json, statistics as st, re, os
def p95(x): x=sorted(x); return x[min(len(x)-1, int(round(0.95*(len(x)-1))))]
def vram(tag, sc): return max(int(m) for m in re.findall(r"vram_peak (\d+)MiB", open(f"{tag}-{sc}.txt").read()))
def row(tags, sc, inter=None):
    tags=[t for t in tags if os.path.exists(f"{t}-{sc}.json")]
    R=[r for t in tags for r in json.load(open(f"{t}-{sc}.json"))]
    sel=lambda r:[v for k,v in r['out'].items() if inter is None or k in inter]
    agg=[sum(v['events'] for v in r['out'].values())/r['wall'] for r in R]
    tt=[v['ttft'] for r in R for v in sel(r) if v['ttft'] is not None]
    comp=[v['total'] for r in R for v in sel(r)]
    spread=[(max(v['total'] for v in sel(r))-min(v['total'] for v in sel(r)))/st.mean(v['total'] for v in sel(r))*100 for r in R]
    per=[v['rate'] for r in R for v in sel(r) if v['rate']]
    return f"agg {st.median(agg):5.1f} (n={len(R)}) ttft p50 {st.median(tt):.2f} p95 {p95(tt):.2f} | compl mean {st.mean(comp):5.2f} max {max(comp):5.2f} spread {st.median(spread):4.1f}% | per-stream {st.median(per) if per else 0:4.1f} | vram {max(vram(t,sc) for t in tags)}"
sets={"V7   ":["v7d2","td"],"prev ":["g3","tg1","tg2","tg3"],"fixed":["v8","v82","tv1","tv2","tv3"]}
for sc,inter in (("solo_long",None),("decode1",None),("decode2",None),("decode3",None),("three_interactive",None),("long_plus_two",{"a","b"})):
    print("##",sc)
    for n,t in sets.items(): print(" ",n,row(t,sc,inter))
