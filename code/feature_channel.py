#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/feature_channel.py
WHAT ELSE DOES THE ~30-DIM DEPOSIT CHANNEL TRANSPORT? (Olli.) feature_deposit_map found one shared low-rank (~30)
linear map R that writes a noun's lexical signature into the consuming (verb) position. The INPUT channel = top
left singular vectors of R (the lexical directions that actually get deposited). We probe what lies inside it.

 (1) Fit R (lexical@L8 -> transport@L12, per-word, centered); SVD -> input channel U_k (k=30).
 (2) FEATURE BATTERY: for each candidate lexical property, measure (a) channel-energy = ||U_k^T l_F||^2/||l_F||^2
     (fraction of the feature direction inside the deposit channel; random ~ k/D ~ 0.03), and (b) transport quality
     cos(R l_F, t_F). Knowns: number/gender/animacy. New: definiteness, person-vs-animal, concreteness
     (concrete/abstract), proper-vs-common. High energy+cos => carried; low => FILTERED OUT (not transported).
 (3) WORD IDENTITY: does the channel carry the exact noun, or only abstract features? eta^2 (between-word/total
     variance) at the noun (lexical) vs verb (transport) -- if it collapses, identity is filtered.
 (4) Interpret the top individual channels by which feature direction each best aligns with.

Usage: .venv/Scripts/python.exe palimpsest/code/feature_channel.py
Output: palimpsest/data/feature_channel.json
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
print(f"DEPOSIT CHANNEL CONTENTS | {A.model} | lex L{A.llex}->tr L{A.ltr} | k={A.k} | {DEV}")
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
NUMP=[("key","keys"),("book","books"),("door","doors"),("car","cars"),("box","boxes"),("dog","dogs"),("cat","cats"),("boy","boys"),("girl","girls"),("table","tables"),("bird","birds"),("tree","trees"),("wall","walls"),("road","roads"),("house","houses"),("chair","chairs"),("lamp","lamps"),("cup","cups"),("hand","hands"),("ship","ships")]
PERSON=[w for w in ["man","woman","king","queen","boy","girl","father","mother","brother","sister","actor","priest","lord","lady"] if has(w)]
ANIMAL=[w for w in ["dog","cat","bird","horse","lion","cow","fish","sheep","bear","wolf","mouse","goat"] if has(w)]
OBJECT=[w for w in ["car","box","table","chair","door","wall","book","lamp","rock","cup","desk","clock","knife","ship"] if has(w)]
MASC=[w for w in ["man","king","boy","father","brother","uncle","prince","actor","lord","duke"] if has(w)]
FEM =[w for w in ["woman","queen","girl","mother","sister","aunt","princess","actress","lady","nun"] if has(w)]
ABSTRACT=[w for w in ["time","idea","plan","hope","fear","truth","peace","law","mind","life","love","power","faith","doubt"] if has(w)]
PROPER=[w for w in ["John","Mary","London","Paris","Rome","Peter","Anna","Berlin","James","Sarah","David","China"] if has(w)]
N2POOL=[w for w in OBJECT if has(w)]; VBR=[w for w in ["hurt","blamed","saw","watched","washed","cleaned","defended","scratched","followed","pushed"] if has(w)]
TAILS=["near the table","by the window","in the room","on the shelf","beside the wall"]
@torch.no_grad()
def resid_batch(seqs,pos,L):
    out=[]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; hs=model(torch.tensor(b,device=DEV),output_hidden_states=True).hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos[i+j]].cpu())
    return torch.stack(out)
def lex_samples(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc(rng.choice(TAILS)) for _ in range(n)],[1]*n,A.llex)
def tr_samples(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(n)],[5]*n,A.ltr)

# ----- (1) fit R on a broad word set, SVD -> channel -----
WL=sorted(set([s for s,_ in NUMP]+[p for _,p in NUMP]+PERSON+ANIMAL+OBJECT+ABSTRACT))
WL=[w for w in WL if has(w)]
print(f"  {len(WL)} words; collecting per-word signatures (ns={A.ns}) ...")
Xlex=torch.stack([lex_samples(w,A.ns).mean(0) for w in WL]); Xtr=torch.stack([tr_samples(w,A.ns).mean(0) for w in WL])
mu_l=Xlex.mean(0); mu_t=Xtr.mean(0); XL=Xlex-mu_l; XT=Xtr-mu_t
lam=1e-1*XL.pow(2).sum(0).mean(); R=torch.linalg.solve(XL.T@XL+lam*torch.eye(D), XL.T@XT)
U,S,Vh=torch.linalg.svd(R); Uk=U[:,:A.k]   # input channel (lexical dirs that get deposited)
er=float((S.sum()**2)/(S.pow(2).sum())); print(f"  R effective rank ~{er:.0f}; using top-{A.k} input channel.")
res={"model":A.model,"k":A.k,"eff_rank":er,"battery":{},"singvals_top":[float(x) for x in S[:40]]}

# feature lexical & transport directions
def fdir(pa,pb,L,pos):
    if pos==1: A1=[enc("The",True)+[tid(" "+rng.choice(pa))]+enc("near the table") for _ in range(220)]; B1=[enc("The",True)+[tid(" "+rng.choice(pb))]+enc("near the table") for _ in range(220)]
    else:
        A1=[enc("The",True)+[tid(" "+rng.choice(pa))]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(220)]
        B1=[enc("The",True)+[tid(" "+rng.choice(pb))]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(220)]
    wa=resid_batch(A1,[pos]*220,L).mean(0); wb=resid_batch(B1,[pos]*220,L).mean(0); w=wa-wb; return w/(w.norm()+1e-9)
SG=[s for s,_ in NUMP]; PL=[p for _,p in NUMP]
BATTERY={
 "number(sg/pl)":(SG,PL),"gender(m/f)":(MASC,FEM),"animacy(anim/inan)":(PERSON+ANIMAL,OBJECT),
 "person-vs-animal":(PERSON,ANIMAL),"concrete-vs-abstract":(OBJECT+ANIMAL,ABSTRACT),"common-vs-proper":(OBJECT+PERSON,PROPER),
}
def chan_energy(l): m=Uk.T@l; return float((m@m)/(l@l+1e-9))
# random null energy
g=torch.Generator().manual_seed(1); rnd=[torch.randn(D,generator=g) for _ in range(20)]; null_e=float(np.mean([chan_energy(r) for r in rnd]))
print(f"\n  FEATURE BATTERY -- inside the deposit channel? (random null energy ~ {null_e:.2f})")
print(f"   {'feature':>22} {'chan-energy':>11} {'transport cos(Rl,t)':>20} {'verdict':>14}")
for name,(pa,pb) in BATTERY.items():
    pa=[w for w in pa if has(w)]; pb=[w for w in pb if has(w)]
    if len(pa)<3 or len(pb)<3: print(f"   {name:>22}  (insufficient words)"); continue
    lF=fdir(pa,pb,A.llex,1); tF=fdir(pa,pb,A.ltr,5)
    e=chan_energy(lF); mapped=lF@R; cmap=cos(mapped,tF)
    v=("CARRIED" if e>4*null_e and cmap>0.5 else ("filtered" if e<2*null_e else "partial"))
    res["battery"][name]={"chan_energy":e,"transport_cos":cmap,"verdict":v}
    print(f"   {name:>22} {e:>11.2f} {cmap:>20.2f} {v:>14}")

# ----- (3) WORD IDENTITY: eta^2 between-word variance at noun vs verb -----
def eta2(words,nper,which):
    samp={w:(lex_samples(w,nper) if which=='lex' else tr_samples(w,nper)) for w in words}
    allX=torch.cat([samp[w] for w in words]); gm=allX.mean(0)
    sst=(allX-gm).pow(2).sum()
    ssb=sum(len(samp[w])*((samp[w].mean(0)-gm).pow(2).sum()) for w in words)
    return float(ssb/sst)
idw=rng.sample(WL,min(16,len(WL)))
e_lex=eta2(idw,12,'lex'); e_tr=eta2(idw,12,'tr')
print(f"\n  WORD IDENTITY (eta^2 between-word/total variance): noun(lexical) {e_lex:.2f} -> verb(transport) {e_tr:.2f}")
print(f"   -> {'identity largely FILTERED (only abstract features transport)' if e_tr<0.5*e_lex else 'identity partly transported'}")
res["word_identity_eta2"]={"lexical":e_lex,"transport":e_tr}

# ----- (4) interpret top channels by best-aligned feature -----
print(f"\n  TOP CHANNELS -- which feature each top input singular vector best aligns with:")
fdirs={name:fdir([w for w in pa if has(w)],[w for w in pb if has(w)],A.llex,1) for name,(pa,pb) in BATTERY.items() if len([w for w in pa if has(w)])>=3 and len([w for w in pb if has(w)])>=3}
res["top_channels"]=[]
for i in range(min(6,A.k)):
    u=U[:,i]; best=max(fdirs.items(), key=lambda kv: abs(cos(u,kv[1])));
    res["top_channels"].append({"channel":i,"singval":float(S[i]),"best_feature":best[0],"cos":abs(cos(u,best[1]))})
    print(f"   ch{i} (s={float(S[i]):.1f}): best-aligned = {best[0]} (|cos| {abs(cos(u,best[1])):.2f})")
Path(str(ROOT/"palimpsest"/"data"/"feature_channel.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: high chan-energy + transport cos => the channel carries that property; energy ~ null => it is")
print("  FILTERED OUT (the deposit does not transport it). eta^2 collapse => exact word identity is dropped, only")
print("  abstract grammatical/semantic features ride the channel.")
print("\n wrote palimpsest/data/feature_channel.json")
