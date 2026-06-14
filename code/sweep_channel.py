#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/sweep_channel.py
SWEEP the deposit channel: discover many candidate axes unsupervised and CAUSALLY validate each automatically
(Olli). Generalises place_causal/body_causal into a pipeline with NO hand-crafted frame/readout per axis.

 1. Diverse COMMON-noun bank; lexical sig (noun@L8) + transport sig (verb@L12); fit deposit map R; channel U_k.
 2. k-means the words IN the channel -> K candidate clusters (each = one discovered axis, cluster-vs-rest).
 3. For each cluster C:
    - TRANSPORT: cos(R*u_C_lex, u_C_transport) -- does the axis ride the shared deposit map?
    - CAUSAL (automated, learned readout): frame "The {N} is"; the cluster's natural continuation signature
      s_C = mean next-token-logits over C-words minus the global mean (a vocab-space direction, LEARNED from
      behaviour, not hand-specified). Patch ONLY the u_C component of a NON-C noun's residual @L8 (set to C-mean
      vs rest-mean) and measure how far the next-token logits move along s_C. Matched RANDOM-direction control.
      Effect_C >> Effect_rand and >0 => the discovered input axis causally produces the cluster's output behaviour.
 Report a table; flag which discovered axes are causally validated.

Usage: .venv/Scripts/python.exe palimpsest/code/sweep_channel.py [--K 8]
Output: palimpsest/data/sweep_channel.json
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np, torch
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass
ROOT=Path(__file__).resolve().parent.parent.parent
P=argparse.ArgumentParser(); P.add_argument("--model",default="EleutherAI/pythia-410m-deduped"); P.add_argument("--seed",type=int,default=0)
P.add_argument("--llex",type=int,default=8); P.add_argument("--ltr",type=int,default=12); P.add_argument("--k",type=int,default=30); P.add_argument("--K",type=int,default=8); P.add_argument("--ns",type=int,default=8)
A=P.parse_args(); DEV="cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"DEPOSIT CHANNEL SWEEP | {A.model} | K={A.K} clusters | {DEV}")
model=AutoModelForCausalLM.from_pretrained(A.model,dtype=torch.float32,low_cpu_mem_usage=True).to(DEV).eval()
tok=AutoTokenizer.from_pretrained(A.model)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
for p in model.parameters(): p.requires_grad_(False)
NL=len(model.gpt_neox.layers); D=model.config.hidden_size
def tid(s):
    t=tok(s,add_special_tokens=False)["input_ids"]; return t[0] if len(t)==1 else None
def enc(s,first=False): return tok(s if first else " "+s,add_special_tokens=False)["input_ids"]
def has(w): return tid(" "+w) is not None
rng=random.Random(A.seed)
def cos(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-9))
# diverse COMMON nouns (no proper nouns -> "The {N} is" stays grammatical)
RAW="""man woman king queen boy girl father mother brother sister uncle aunt actor actress lord lady doctor teacher
dog cat bird horse lion cow fish bear wolf mouse sheep goat rabbit deer frog
car box table chair door wall book lamp cup desk clock knife ship bottle plate spoon brick
city town country village river island forest mountain valley harbor lake field road bridge
eye heart face head hand foot arm leg ear nose mouth finger knee neck chest bone skin lip
bread apple wine meat milk rice cheese soup cake egg fruit
idea hope fear truth peace law mind life love power faith doubt anger joy fact plan
tree flower grass leaf root seed bush""".split()
WORDS=sorted(set(w for w in RAW if has(w))); print(f"  {len(WORDS)} common nouns")
N2POOL=[w for w in ["car","box","table","chair","door","book","cup","desk"] if has(w)]; VBR=[w for w in ["hurt","blamed","saw","watched","washed","followed"] if has(w)]
TAILS=["near the table","by the window","in the room","on the shelf"]
@torch.no_grad()
def resid_batch(seqs,pos,L):
    out=[]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; hs=model(torch.tensor(b,device=DEV),output_hidden_states=True).hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos[i+j]].cpu())
    return torch.stack(out)
def lex_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc(rng.choice(TAILS)) for _ in range(n)],[1]*n,A.llex).mean(0)
def tr_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(n)],[5]*n,A.ltr).mean(0)
print("  collecting signatures + fitting deposit map (R on TRAIN words only; transport tested on HELD-OUT) ...")
Xlex=torch.stack([lex_sig(w,A.ns) for w in WORDS]); Xtr=torch.stack([tr_sig(w,A.ns) for w in WORDS])
mu_l=Xlex.mean(0); mu_t=Xtr.mean(0); XL=Xlex-mu_l; XT=Xtr-mu_t
perm=list(range(len(WORDS))); random.Random(1).shuffle(perm); ntr=int(0.7*len(perm)); TR=set(perm[:ntr]); TE=set(perm[ntr:])
trI=sorted(TR); lam=1e-1*XL[trI].pow(2).sum(0).mean(); R=torch.linalg.solve(XL[trI].T@XL[trI]+lam*torch.eye(D),XL[trI].T@XT[trI])
U,S,Vh=torch.linalg.svd(R); Uk=U[:,:A.k]; C=(XL@Uk); Cs=(C-C.mean(0))/(C.std(0)+1e-6)
def kmeans(Xk,kk,iters=40):
    g=torch.Generator().manual_seed(0); cen=Xk[torch.randperm(len(Xk),generator=g)[:kk]].clone()
    for _ in range(iters):
        d=((Xk[:,None,:]-cen[None])**2).sum(-1); a=d.argmin(1)
        for c in range(kk):
            if (a==c).any(): cen[c]=Xk[a==c].mean(0)
    return a,cen
assign,cen=kmeans(Cs,A.K)

# clean frame logits for the learned readout
FRAME=lambda w: enc("The",True)+[tid(" "+w)]+enc("is")   # "The N is ___"
@torch.no_grad()
def frame_logits(words):
    out=[]
    for i in range(0,len(words),64):
        b=[FRAME(w) for w in words[i:i+64]]; out.append(model(torch.tensor(b,device=DEV)).logits[:,-1,:].float().cpu())
    return torch.cat(out)
Lall=frame_logits(WORDS); Lmean=Lall.mean(0)
Fr=resid_batch([FRAME(w) for w in WORDS],[1]*len(WORDS),A.llex)   # noun-pos residual in the frame

# patch readout machinery
def make_hook(state):
    def hook(m,inp,oo):
        if not state["on"]: return
        hs=oo[0] if isinstance(oo,tuple) else oo; u=state["u"].to(hs.device)
        proj=(hs[:,1,:]@u).unsqueeze(1); hs[:,1,:]=hs[:,1,:]-proj*u+state["c"]*u
        return (hs,)+tuple(oo[1:]) if isinstance(oo,tuple) else hs
    return hook
STATE={"on":False,"u":None,"c":0.0}
model.gpt_neox.layers[A.llex-1].register_forward_hook(make_hook(STATE))
@torch.no_grad()
def patched_logits(words,u,c):
    STATE["on"]=True; STATE["u"]=u; STATE["c"]=c; out=[]
    for i in range(0,len(words),64):
        b=[FRAME(w) for w in words[i:i+64]]; out.append(model(torch.tensor(b,device=DEV)).logits[:,-1,:].float().cpu())
    STATE["on"]=False; return torch.cat(out)

g=torch.Generator().manual_seed(7)
print(f"\n  SWEEP -- {A.K} discovered axes (cluster-vs-rest), transport + automated causal patch:")
print(f"   {'axis (top words)':>34} {'n':>3} {'transport':>9} {'causal':>7} {'rand':>6} {'verdict':>10}")
res={"model":A.model,"K":A.K,"axes":[]}
for c in range(A.K):
    members=[i for i in range(len(WORDS)) if int(assign[i])==c]; rest=[i for i in range(len(WORDS)) if int(assign[i])!=c]
    if len(members)<3 or len(rest)<3: continue
    # name by closeness to centroid
    d=((Cs[members]-cen[c])**2).sum(1); top=[WORDS[members[int(j)]] for j in torch.argsort(d)[:5]]
    # axis direction (discovered, full data) for the causal patch
    uL=(Xlex[members].mean(0)-Xlex[rest].mean(0)); uL=uL/(uL.norm()+1e-9)
    # TRANSPORT measured on HELD-OUT words only (R never saw them): honest cos(R*uL_te, uT_te)
    mTE=[i for i in members if i in TE]; rTE=[i for i in rest if i in TE]
    if len(mTE)>=2 and len(rTE)>=2:
        uLte=(Xlex[mTE].mean(0)-Xlex[rTE].mean(0)); uTte=(Xtr[mTE].mean(0)-Xtr[rTE].mean(0))
        transp=cos((uLte@R)/((uLte@R).norm()+1e-9), uTte/(uTte.norm()+1e-9))
    else: transp=float('nan')
    # learned continuation signature
    sC=Lall[members].mean(0)-Lmean; sC=sC/(sC.norm()+1e-9)
    # patch on rest (non-C) targets, set u-component to C-mean vs rest-mean
    tgt=[WORDS[i] for i in rest]; uu=uL
    c_hi=float((Fr[members]@uu).mean()); c_lo=float((Fr[rest]@uu).mean())
    Lhi=patched_logits(tgt,uu,c_hi); Llo=patched_logits(tgt,uu,c_lo)
    eff=float(((Lhi-Llo)@sC).mean())
    # random control matched
    ur=torch.randn(D,generator=g); ur=ur/ur.norm(); cr_hi=float((Fr[members]@ur).mean()); cr_lo=float((Fr[rest]@ur).mean())
    Rhi=patched_logits(tgt,ur,cr_hi); Rlo=patched_logits(tgt,ur,cr_lo); effr=float(((Rhi-Rlo)@sC).mean())
    causal_ok = eff>0.3 and eff>3*abs(effr); transp_ok = (transp>0.5) if not (transp!=transp) else None
    ok = causal_ok and (transp_ok is not False)
    res["axes"].append({"cluster":c,"top_words":top,"n":len(members),"transport":transp,"causal_effect":eff,"random_effect":effr,"validated":bool(ok)})
    print(f"   {('['+', '.join(top)+']')[:34]:>34} {len(members):>3} {transp:>9.2f} {eff:>7.2f} {effr:>6.2f} {('VALID' if ok else '--'):>10}")
nv=sum(1 for a in res["axes"] if a["validated"])
print(f"\n  => {nv}/{len(res['axes'])} discovered axes causally validated (transport>0.5, effect>0.3 & >3x random).")
res["n_validated"]=nv
Path(str(ROOT/"palimpsest"/"data"/"sweep_channel.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: each row = an axis DISCOVERED unsupervised in the deposit channel, auto-validated -- transports via")
print("  the shared map AND patching it moves the output along that cluster's own continuation (vs ~0 for random).")
print("  A systematic mine of the channel's causal semantic contents, no hand-specified features.")
print("\n wrote palimpsest/data/sweep_channel.json")
