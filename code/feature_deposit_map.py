#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/feature_deposit_map.py
IS THE TRANSPORT RE-ENCODING ONE SHARED LINEAR DEPOSIT MAP, OR FEATURE-SPECIFIC? (Olli.) feature_slot/_form/
_trajectory showed the subject feature deposited at the verb is RE-ENCODED into a distinct direction. Question: is
that re-encoding a SINGLE linear operation R (a generic "write the subject's lexical signature into the consuming
position") applied to everything, or content-specific machinery?

Word-level test (well-powered, unlike a few feature dirs): for ~120 nouns, get each word's
  - LEXICAL signature  x_lex(w) = mean residual at the NOUN position of "The {w} <tail>" @L_lex
  - TRANSPORT signature x_tr(w) = mean residual at the VERB position of "The {w} near the {N2} {V}" @L_tr
center across words (isolate the noun-dependent component), fit a ridge map R: x_lex -> x_tr on TRAIN words, and
test on HELD-OUT words: cos(R x_lex, x_tr) and R^2. A shuffled-pairing null calibrates chance. If R generalizes to
unseen words, ONE linear deposit map governs the re-encoding. Then check the map ALSO reproduces the feature-
direction re-encoding: cos(R l_F, t_F) for number/gender/animacy (lexical dir l_F -> transport dir t_F). Singular
spectrum of R = is the deposit low-rank (a few channels) or full.

Usage: .venv/Scripts/python.exe palimpsest/code/feature_deposit_map.py
Output: palimpsest/data/feature_deposit_map.json
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np, torch
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass
ROOT=Path(__file__).resolve().parent.parent.parent
P=argparse.ArgumentParser(); P.add_argument("--model",default="EleutherAI/pythia-410m-deduped"); P.add_argument("--seed",type=int,default=0)
P.add_argument("--llex",type=int,default=8); P.add_argument("--ltr",type=int,default=12); P.add_argument("--k",type=int,default=10)
A=P.parse_args(); DEV="cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"FEATURE DEPOSIT MAP | {A.model} | lex L{A.llex} -> transport L{A.ltr} | {DEV}")
model=AutoModelForCausalLM.from_pretrained(A.model,dtype=torch.float32,low_cpu_mem_usage=True).to(DEV).eval()
tok=AutoTokenizer.from_pretrained(A.model)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
for p in model.parameters(): p.requires_grad_(False)
NL=len(model.gpt_neox.layers); D=model.config.hidden_size
def tid(s):
    t=tok(s,add_special_tokens=False)["input_ids"]; return t[0] if len(t)==1 else None
def enc(s,first=False): return tok(s if first else " "+s,add_special_tokens=False)["input_ids"]
rng=random.Random(A.seed)
def cos(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-9))
# word inventory with feature tags
NUMP=[("key","keys"),("book","books"),("door","doors"),("car","cars"),("box","boxes"),("dog","dogs"),("cat","cats"),("boy","boys"),("girl","girls"),("table","tables"),("bird","birds"),("tree","trees"),("wall","walls"),("road","roads"),("house","houses"),("chair","chairs"),("lamp","lamps"),("cup","cups"),("hand","hands"),("king","kings")]
MASC=["man","king","boy","father","son","brother","uncle","prince","actor","priest","duke","lord","waiter","nephew"]
FEM =["woman","queen","girl","mother","daughter","sister","aunt","princess","actress","lady","nun","duchess","widow","niece"]
ANIM=["man","woman","dog","cat","boy","girl","king","queen","bird","horse","lion","cow","fish","child"]
INAN=["car","box","table","chair","door","wall","book","lamp","rock","cup","road","window","stone","desk","clock","knife"]
# build a tagged unique word set
def mk_words():
    W={}
    for s,p in NUMP:
        if tid(" "+s): W.setdefault(s,{})["number"]="sg"
        if tid(" "+p): W.setdefault(p,{})["number"]="pl"
    for w in MASC:
        if tid(" "+w): W.setdefault(w,{})["gender"]="m"
    for w in FEM:
        if tid(" "+w): W.setdefault(w,{})["gender"]="f"
    for w in ANIM:
        if tid(" "+w): W.setdefault(w,{})["anim"]="a"
    for w in INAN:
        if tid(" "+w): W.setdefault(w,{})["anim"]="i"
    return {w:t for w,t in W.items() if tid(" "+w)}
WORDS=mk_words(); WL=sorted(WORDS); print(f"  {len(WL)} unique noun tokens")
N2POOL=[w for w in INAN if tid(" "+w)]; VBR=[w for w in ["hurt","blamed","saw","watched","washed","cleaned","defended","scratched","followed","pushed"] if tid(" "+w)]
TAILS=["near the table","by the window","in the room","on the shelf","beside the wall"]
@torch.no_grad()
def resid_batch(seqs,pos,L):
    out=[]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; o=model(torch.tensor(b,device=DEV),output_hidden_states=True)
        hs=o.hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos[i+j]].cpu())
    return torch.stack(out)
# per-word signatures
def lex_sig(w):
    seqs=[enc("The",True)+[tid(" "+w)]+enc(rng.choice(TAILS)) for _ in range(A.k)]; return resid_batch(seqs,[1]*A.k,A.llex).mean(0)
def tr_sig(w):
    seqs=[enc("The",True)+[tid(" "+w)]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(A.k)]
    return resid_batch(seqs,[5]*A.k,A.ltr).mean(0)
print("  collecting per-word lexical & transport signatures ...")
Xlex=torch.stack([lex_sig(w) for w in WL]); Xtr=torch.stack([tr_sig(w) for w in WL])
mu_l=Xlex.mean(0); mu_t=Xtr.mean(0); XL=Xlex-mu_l; XT=Xtr-mu_t   # noun-dependent components

# fit ridge map R: XL -> XT on train words, test on held-out
idx=list(range(len(WL))); rng.shuffle(idx); cut=int(0.75*len(idx)); tr,te=idx[:cut],idx[cut:]
def fit_R(rows):
    Xc=XL[rows]; Yc=XT[rows]; lam=1e-1*Xc.pow(2).sum(0).mean(); G=Xc.T@Xc+lam*torch.eye(D); return torch.linalg.solve(G,Xc.T@Yc)
R=fit_R(tr)
def evalR(R,rows):
    pred=XL[rows]@R; cs=[cos(pred[i],XT[rows][i]) for i in range(len(rows))]
    r2=float(1-(XT[rows]-pred).pow(2).sum()/XT[rows].pow(2).sum()); return float(np.mean(cs)),r2
ctr,r2tr=evalR(R,tr); cte,r2te=evalR(R,te)
# null: shuffle pairing
perm=te[:]; rng.shuffle(perm); predN=XL[te]@R; cnull=float(np.mean([cos(predN[i],XT[perm][i]) for i in range(len(te))]))
# baseline: identity (is transport already ~ lexical without a map?)
cid=float(np.mean([cos(XL[te][i],XT[te][i]) for i in range(len(te))]))
print(f"\n  ONE linear deposit map R (lexical signature -> transport signature):")
print(f"   train: cos {ctr:.2f} R^2 {r2tr:.2f} | HELD-OUT words: cos {cte:.2f} R^2 {r2te:.2f}")
print(f"   baselines: identity(no map) cos {cid:.2f} | shuffled-pair null cos {cnull:.2f}")
verdict=("ONE SHARED LINEAR DEPOSIT MAP (generalises to unseen words)" if cte>0.5 and cte>cid+0.15 and cte>cnull+0.3
         else "NOT a single linear map (poor held-out generalisation)")
print(f"   -> {verdict}")
res={"model":A.model,"llex":A.llex,"ltr":A.ltr,"nwords":len(WL),
     "map":{"train_cos":ctr,"train_r2":r2tr,"heldout_cos":cte,"heldout_r2":r2te,"identity_cos":cid,"null_cos":cnull,"verdict":verdict}}

# does the SAME word-map reproduce the FEATURE-direction re-encoding?
def fdir(g_a,g_b,L):
    Xa=resid_batch(*g_a,L); Xb=resid_batch(*g_b,L); w=Xa.mean(0)-Xb.mean(0); return w/(w.norm()+1e-9)
def seqs_lex(pool): return [enc("The",True)+[tid(" "+rng.choice(pool))]+enc("near the table") for _ in range(220)],[1]*220
def seqs_tr(pool):
    return [enc("The",True)+[tid(" "+rng.choice(pool))]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(220)],[5]*220
SGp=[s for s,_ in NUMP if tid(" "+s)]; PLp=[p for _,p in NUMP if tid(" "+p)]
feats={"number":(PLp,SGp),"gender":([w for w in MASC if tid(" "+w)],[w for w in FEM if tid(" "+w)]),
       "animacy":([w for w in ANIM if tid(" "+w)],[w for w in INAN if tid(" "+w)])}
print(f"\n  does the word-map R also reproduce the FEATURE re-encoding? cos(R*lexdir, transportdir):")
res["feature_reencode"]={}
for f,(pa,pb) in feats.items():
    lF=fdir(seqs_lex(pa),seqs_lex(pb),A.llex); tF=fdir(seqs_tr(pa),seqs_tr(pb),A.ltr)
    mapped=(lF-mu_l*0)@R  # apply R to the (centered) lexical direction
    mapped=mapped/(mapped.norm()+1e-9)
    c_map=cos(mapped,tF); c_raw=cos(lF,tF)
    res["feature_reencode"][f]={"cos_R_l_vs_t":c_map,"cos_raw_l_vs_t":c_raw}
    print(f"   {f:>8}: raw cos(l,t) {c_raw:+.2f} -> after R {c_map:+.2f}")
# rank of R (effective channels of the deposit)
sv=torch.linalg.svdvals(R); er=float((sv.sum()**2)/(sv.pow(2).sum()))  # participation ratio
print(f"\n  deposit map R singular spectrum: participation-ratio (effective rank) ~ {er:.0f} of {D}")
res["R_effrank"]=er
Path(str(ROOT/"palimpsest"/"data"/"feature_deposit_map.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: held-out cos >> identity & null => ONE shared linear map writes any lexical signature into the")
print("  consuming position (a generic deposit op). If R also lifts cos(l,t) toward 1 for the feature dirs, the")
print("  same op governs feature re-encoding. Effective rank = how many channels the deposit uses.")
print("\n wrote palimpsest/data/feature_deposit_map.json")
