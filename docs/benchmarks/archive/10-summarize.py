import json, statistics as st, re
def p95(x): x=sorted(x); return x[min(len(x)-1, int(round(0.95*(len(x)-1))))]
def vram(tag, sc): return max(int(m) for m in re.findall(r"vram_peak (\d+)MiB", open(f"{tag}-{sc}.txt").read()))
def row(tags, sc, inter=None):
    R=[r for t in tags for r in json.load(open(f"{t}-{sc}.json"))]
    sel=lambda r:[v for k,v in r['out'].items() if inter is None or k in inter]
    agg=[sum(v['events'] for v in r['out'].values())/r['wall'] for r in R]
    tt=[v['ttft'] for r in R for v in sel(r) if v['ttft'] is not None]
    comp=[v['total'] for r in R for v in sel(r)]
    wall=[r['wall'] for r in R]
    return dict(agg=round(st.median(agg),1), ttft_p50=round(st.median(tt),3), ttft_p95=round(p95(tt),3), compl_mean=round(st.mean(comp),3), wall=round(st.median(wall),2), vram=max(vram(t,sc) for t in tags), n=len(R))
sets={"base":["b1","b2","b3"],"new":["n1","n2","n3"]}
for sc,inter in (("solo_short",None),("three_interactive",None),("custom",None),("decode3",None),("long_plus_two",{"a","b"})):
    for n,t in sets.items(): print(f"{sc:18s}{n:5s}",row(t,sc,inter))
def texts(tag,sc): return [{k:v['text'] for k,v in r['out'].items()} for r in json.load(open(f"{tag}-{sc}.json"))]
for sc in ("solo_short","three_interactive","custom","decode3","long_plus_two"):
    ref=texts("b1",sc)[0]
    ok=all(r==ref for t in ("b1","b2","b3","n1","n2","n3") for r in texts(t,sc))
    print("texts identical across all runs, base and new:",sc,ok)
