import json, sys
d=json.load(open(sys.argv[1]))
C={1:24.77,2:27.23,3:29.95,4:33.15}   # pass-cost medians at 1024 context (ms)
MARK=0.5   # D2D mark / rewind of the 53 MB state
def draft_for(ctx, n, dmax):
    if len(ctx)<n: return []
    key=ctx[-n:]
    for i in range(len(ctx)-n-1, -1, -1):
        if ctx[i:i+n]==key:
            return ctx[i+n:i+n+dmax]
    return []
def run(p, n, dmax, refeed=True):
    ctx=list(p["prompt"]); rep=[r for r in p["replies"] if r["mode"]=="greedy"][0]["tokens"]
    ctx.append(rep[0]); g=1; cost=C[1]; plain=C[1]*len(rep); passes=0; acc=0; drafted=0; rolled=0
    # first token came from the prompt pass (same for both); count only decode passes
    cost=0; plain=C[1]*(len(rep)-1)
    while g<len(rep):
        dr=draft_for(ctx,n,min(dmax,len(rep)-g-1)) if dmax>0 else []
        if not dr:
            cost+=C[1]; ctx.append(rep[g]); g+=1; continue
        a=0
        while a<len(dr) and dr[a]==rep[g+a]: a+=1
        cost+=C[1+len(dr)]+MARK; drafted+=len(dr); acc+=a; passes+=1
        if a<len(dr):
            cost+=C[a+1]+MARK; rolled+=1   # rewind + refeed t and accepted drafts
        take=min(a+1,len(rep)-g); ctx.extend(rep[g:g+take]); g+=take
    return cost, plain, passes, drafted, acc, rolled
print("class        n d  speedup  passes drafted accepted rolled")
for cls in ["chat","code_edit","repetitive","long_context"]:
    for n in (2,3,4,6):
        for dm in (1,2,3):
            tc=tp=pa=dr=ac=ro=0
            for p in d["prompts"]:
                if p["class"]!=cls: continue
                c,pl,a,b,e,r=run(p,n,dm); tc+=c; tp+=pl; pa+=a; dr+=b; ac+=e; ro+=r
            print(f"{cls:13}{n} {dm}  {tp/tc:5.2f}x  {pa:5} {dr:6} {ac:7} {ro:6}")
