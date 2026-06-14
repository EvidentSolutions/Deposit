#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/feature_channel_discover.py
DISCOVERY (not confirmation) of the deposit-channel contents (Olli). feature_channel CONFIRMED hand-picked axes
(number/gender/animacy/...). Here we analyse the channel UNSUPERVISED: project a diverse noun bank into the ~40-dim
deposit channel and ask what organises it WITHOUT pre-specifying features, plus how much of the channel our NAMED
axes actually explain (i.e. how much is UNNAMED).

 (1) Fit deposit map R (lex@L8->transport@L12), SVD -> input channel U_k.
 (2) Project a diverse noun bank into the channel; standardize.
 (3) NAMED COVERAGE: fraction of channel-word variance spanned by our named feature directions (number/gender/
     animacy/person-animal/concrete-abstract). 1 - that = UNNAMED channel variance.
 (4) UNSUPERVISED: PCA + k-means on words-in-channel; print clusters with representative words and their coarse
     category mix -- does structure emerge that we did NOT pre-specify?
 (5) RESIDUAL AXES: project out the named directions; PCA the residual; list extreme words of the top residual
     axes -> name (post-hoc) what unnamed dimensions the channel still carries.

Usage: .venv/Scripts/python.exe palimpsest/code/feature_channel_discover.py
Output: palimpsest/data/feature_channel_discover.json
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np, torch
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass
ROOT=Path(__file__).resolve().parent.parent.parent
P=argparse.ArgumentParser(); P.add_argument("--model",default="EleutherAI/pythia-410m-deduped"); P.add_argument("--seed",type=int,default=0)
P.add_argument("--llex",type=int,default=8); P.add_argument("--ltr",type=int,default=12); P.add_argument("--k",type=int,default=30); P.add_argument("--ns",type=int,default=10)
A=P.parse_args(); DEV="cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"DEPOSIT CHANNEL DISCOVERY | {A.model} | k={A.k} | {DEV}")
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
# diverse noun bank with coarse category tags (for post-hoc interpretation only, NOT used to find structure)
BANK={
 "man":"person_m","king":"person_m","boy":"person_m","father":"person_m","brother":"person_m","uncle":"person_m","prince":"person_m","actor":"person_m","lord":"person_m",
 "woman":"person_f","queen":"person_f","girl":"person_f","mother":"person_f","sister":"person_f","aunt":"person_f","princess":"person_f","actress":"person_f","lady":"person_f","nun":"person_f",
 "dog":"animal","cat":"animal","bird":"animal","horse":"animal","lion":"animal","cow":"animal","fish":"animal","bear":"animal","wolf":"animal","mouse":"animal","sheep":"animal",
 "car":"object","box":"object","table":"object","chair":"object","door":"object","wall":"object","book":"object","lamp":"object","cup":"object","desk":"object","clock":"object","knife":"object","ship":"object","window":"object",
 "time":"abstract","idea":"abstract","plan":"abstract","hope":"abstract","fear":"abstract","truth":"abstract","peace":"abstract","law":"abstract","mind":"abstract","life":"abstract","love":"abstract","power":"abstract","faith":"abstract",
 "London":"place","Paris":"place","Rome":"place","Berlin":"place","China":"place","river":"place","city":"place","village":"place","mountain":"place","island":"place","forest":"place",
 "hand":"body","eye":"body","head":"body","heart":"body","arm":"body","foot":"body","face":"body",
 "bread":"food","apple":"food","wine":"food","meat":"food","milk":"food","rice":"food",
}
PL_OF={"dog":"dogs","cat":"cats","car":"cars","box":"boxes","book":"books","boy":"boys","girl":"girls","table":"tables","bird":"birds","door":"doors","wall":"walls","king":"kings","hand":"hands","eye":"eyes","apple":"apples"}
WORDS=[w for w in BANK if has(w)]; print(f"  {len(WORDS)} diverse nouns across {len(set(BANK.values()))} coarse categories")
N2POOL=[w for w in ["car","box","table","chair","door","wall","book","lamp","cup","desk"] if has(w)]
VBR=[w for w in ["hurt","blamed","saw","watched","washed","cleaned","defended","scratched","followed","pushed"] if has(w)]
TAILS=["near the table","by the window","in the room","on the shelf","beside the wall"]
@torch.no_grad()
def resid_batch(seqs,pos,L):
    out=[]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; hs=model(torch.tensor(b,device=DEV),output_hidden_states=True).hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos[i+j]].cpu())
    return torch.stack(out)
def lex_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc(rng.choice(TAILS)) for _ in range(n)],[1]*n,A.llex).mean(0)
def tr_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(n)],[5]*n,A.ltr).mean(0)
print("  collecting signatures + fitting deposit map ...")
Xlex=torch.stack([lex_sig(w,A.ns) for w in WORDS]); Xtr=torch.stack([tr_sig(w,A.ns) for w in WORDS])
mu_l=Xlex.mean(0); XL=Xlex-mu_l; XT=Xtr-Xtr.mean(0)
lam=1e-1*XL.pow(2).sum(0).mean(); R=torch.linalg.solve(XL.T@XL+lam*torch.eye(D), XL.T@XT)
U,S,Vh=torch.linalg.svd(R); Uk=U[:,:A.k]
C=(XL@Uk)  # [nwords, k] channel coordinates of each word
C=(C-C.mean(0))/(C.std(0)+1e-6)  # standardize channel dims
print(f"  channel = top-{A.k} input singular vectors; projected {len(WORDS)} words.")
res={"model":A.model,"k":A.k,"nwords":len(WORDS)}

# (3) named coverage
def fdir(pa,pb):
    pa=[w for w in pa if has(w)]; pb=[w for w in pb if has(w)]
    a=torch.stack([lex_sig(w,6) for w in pa]).mean(0)-mu_l; b=torch.stack([lex_sig(w,6) for w in pb]).mean(0)-mu_l
    w=(a-b); w=Uk.T@w; return w/(w.norm()+1e-9)   # named direction IN channel coords
P_=[w for w,c in BANK.items() if c.startswith("person")]; AN=[w for w,c in BANK.items() if c=="animal"]; OB=[w for w,c in BANK.items() if c=="object"]; AB=[w for w,c in BANK.items() if c=="abstract"]
named={"gender":fdir([w for w,c in BANK.items() if c=="person_m"],[w for w,c in BANK.items() if c=="person_f"]),
       "animacy":fdir(P_+AN,OB),"person-animal":fdir(P_,AN),"concrete-abstract":fdir(OB+AN,AB)}
B=torch.stack(list(named.values()),1); Qn,_=torch.linalg.qr(B)  # named subspace in channel coords (standardized space approx)
# project standardized C onto named subspace (re-standardize named dirs into C's space: use C directly)
Cn=C@Qn; cover=float(Cn.pow(2).sum()/C.pow(2).sum())
print(f"\n(3) NAMED COVERAGE: our {len(named)} named axes span {cover*100:.0f}% of the channel's word-variance"
      f" -> {100-cover*100:.0f}% is UNNAMED.")
res["named_coverage"]=cover

# (4) unsupervised k-means on channel coords
def kmeans(X,kk,iters=30):
    g=torch.Generator().manual_seed(0); cen=X[torch.randperm(len(X),generator=g)[:kk]].clone()
    for _ in range(iters):
        d=((X[:,None,:]-cen[None])**2).sum(-1); a=d.argmin(1)
        for c in range(kk):
            m=a==c
            if m.any(): cen[c]=X[m].mean(0)
    return a,cen
KK=6; assign,cen=kmeans(C,KK)
print(f"\n(4) UNSUPERVISED k-means ({KK} clusters) on words-in-channel -- structure we did NOT pre-specify:")
res["clusters"]=[]
for c in range(KK):
    members=[WORDS[i] for i in range(len(WORDS)) if int(assign[i])==c]
    if not members: continue
    cats={}
    for w in members: cats[BANK[w]]=cats.get(BANK[w],0)+1
    catstr=",".join(f"{k}:{v}" for k,v in sorted(cats.items(),key=lambda x:-x[1]))
    print(f"   cluster {c} (n={len(members)}): {', '.join(members[:10])}{'...' if len(members)>10 else ''}")
    print(f"             category mix -> {catstr}")
    res["clusters"].append({"id":c,"members":members,"category_mix":cats})

# (5) residual axes: project out named, PCA residual, extreme words
Cr=C - C@Qn@Qn.T
Ur,Sr,_=torch.linalg.svd(Cr-Cr.mean(0), full_matrices=False)
print(f"\n(5) RESIDUAL AXES (after removing named): top unnamed channel dimensions, extreme words:")
res["residual_axes"]=[]
scores_full=Cr@ (Vh[:0].T if False else torch.eye(A.k))  # placeholder
pcs=(Cr-Cr.mean(0))  # project onto residual right singular vectors
RV=torch.linalg.svd(pcs,full_matrices=False)[2]  # [k,k] right vectors
for ax in range(min(4,A.k)):
    proj=pcs@RV[ax]
    order=torch.argsort(proj)
    lo=[WORDS[int(i)] for i in order[:6]]; hi=[WORDS[int(i)] for i in order[-6:]]
    var=float((proj.pow(2).sum()/pcs.pow(2).sum()))
    print(f"   resid-axis {ax} (var {var*100:.0f}% of residual): [-] {', '.join(lo)}   <->   [+] {', '.join(hi)}")
    res["residual_axes"].append({"axis":ax,"neg":lo,"pos":hi,"var_frac":var})
Path(str(ROOT/"palimpsest"/"data"/"feature_channel_discover.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: named coverage << 100% => the channel carries structure beyond our axes. Clusters/residual-axes that")
print("  track a coherent word-grouping we did NOT pre-specify = the method DISCOVERING channel contents, to be")
print("  named post-hoc (then validated). This is the broad-analysis tool, not just confirmation of known dims.")
print("\n wrote palimpsest/data/feature_channel_discover.json")
