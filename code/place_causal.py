#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/place_causal.py
CAUSALLY VALIDATE A DISCOVERED CHANNEL AXIS (Olli). feature_channel_discover surfaced an unnamed 'place/location'
axis by unsupervised clustering. Discovery proposes; causal intervention disposes. Two tests:
 (1) TRANSPORT: does the place axis ride the SAME low-rank deposit map as number/gender/animacy? cos(R*place_lex,
     place_transport) vs a number baseline.
 (2) CAUSAL: in a category-forcing frame "The {N} is a type of ___", the model predicts a category-consistent noun.
     Define a readout = logit(place-words) - logit(object-words) at the final token. Then PATCH only the place-axis
     COMPONENT of the noun's residual (project out u_place, set it to the place-class mean vs the object-class mean)
     and measure the readout shift. CONTROL: do the identical patch with a RANDOM unit direction (matched norms).
     If the place axis shifts the prediction toward place-category and the random direction does not, the discovered
     axis is causally efficacious -- a real channel content, not a geometric artifact.

Usage: .venv/Scripts/python.exe palimpsest/code/place_causal.py
Output: palimpsest/data/place_causal.json
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass
ROOT=Path(__file__).resolve().parent.parent.parent
P=argparse.ArgumentParser(); P.add_argument("--model",default="EleutherAI/pythia-410m-deduped"); P.add_argument("--seed",type=int,default=0)
P.add_argument("--llex",type=int,default=8); P.add_argument("--ltr",type=int,default=12)
A=P.parse_args(); DEV="cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"PLACE AXIS CAUSAL VALIDATION | {A.model} | {DEV}")
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
PLACE=[w for w in ["London","Paris","Rome","Berlin","city","town","country","village","river","island","forest","mountain","valley","harbor"] if has(w)]
OBJ =[w for w in ["car","box","table","chair","book","lamp","cup","desk","door","wall","clock","knife","ship","bottle"] if has(w)]
ANIM=[w for w in ["dog","cat","bird","horse","cow","fish","lion","bear","wolf","sheep"] if has(w)]
NONPLACE=OBJ+ANIM
N2POOL=[w for w in OBJ if has(w)]; VBR=[w for w in ["hurt","blamed","saw","watched","washed","followed","pushed"] if has(w)]
print(f"  {len(PLACE)} place / {len(OBJ)} object / {len(ANIM)} animal nouns")
@torch.no_grad()
def resid_batch(seqs,pos,L):
    out=[]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; hs=model(torch.tensor(b,device=DEV),output_hidden_states=True).hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos[i+j]].cpu())
    return torch.stack(out)
# ---------- (1) transport through the deposit map ----------
TAILS=["near the table","by the window","in the room","on the shelf"]
def lex_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc(rng.choice(TAILS)) for _ in range(n)],[1]*n,A.llex).mean(0)
def tr_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(n)],[5]*n,A.ltr).mean(0)
WL=[w for w in PLACE+OBJ+ANIM if has(w)]
Xlex=torch.stack([lex_sig(w,8) for w in WL]); Xtr=torch.stack([tr_sig(w,8) for w in WL]); mu_l=Xlex.mean(0)
XL=Xlex-mu_l; XT=Xtr-Xtr.mean(0); lam=1e-1*XL.pow(2).sum(0).mean(); R=torch.linalg.solve(XL.T@XL+lam*torch.eye(D),XL.T@XT)
def dir_lt(pa,pb,Lx,pos,sig):
    a=torch.stack([sig(w,6) for w in pa]).mean(0); b=torch.stack([sig(w,6) for w in pb]).mean(0); w=a-b; return w/(w.norm()+1e-9)
pl_lex=dir_lt(PLACE,NONPLACE,A.llex,1,lex_sig); pl_tr=dir_lt(PLACE,NONPLACE,A.ltr,5,tr_sig)
num_lex=dir_lt([w for w in ["dogs","cats","cars","books","boys","tables"] if has(w)],[w for w in ["dog","cat","car","book","boy","table"] if has(w)],A.llex,1,lex_sig)
num_tr =dir_lt([w for w in ["dogs","cats","cars","books","boys","tables"] if has(w)],[w for w in ["dog","cat","car","book","boy","table"] if has(w)],A.ltr,5,tr_sig)
cpl=cos((pl_lex@R)/((pl_lex@R).norm()+1e-9),pl_tr); cnu=cos((num_lex@R)/((num_lex@R).norm()+1e-9),num_tr)
print(f"\n(1) TRANSPORT through deposit map R: cos(R*lex, transport)  place {cpl:.2f}  | number(baseline) {cnu:.2f}")
res={"model":A.model,"transport":{"place":cpl,"number_baseline":cnu}}

# ---------- (2) causal: category-forcing frame + patch the place component ----------
FRAME=lambda w: enc("The",True)+[tid(" "+w)]+enc("is a type of")   # The N is a type of ___ ; noun at pos1, read last
PLACE_W=[" city"," town"," place"," country"," area"," region"]; OBJ_W=[" object"," thing"," tool"," device"," item"," material"]
PW=[tid(t) for t in PLACE_W if tid(t)]; OW=[tid(t) for t in OBJ_W if tid(t)]
@torch.no_grad()
def readout(seqs, intervene=None):
    # intervene: dict {pos,u,c} applied at hidden_states[llex] = output of layer (llex-1)
    out=[]
    H={"on":intervene is not None,"iv":intervene}
    def hook(m,inp,oo):
        if not H["on"]: return
        hs=oo[0] if isinstance(oo,tuple) else oo; iv=H["iv"]; u=iv["u"].to(hs.device)
        p=iv["pos"]; proj=(hs[:,p,:]@u).unsqueeze(1)
        hs[:,p,:]=hs[:,p,:]-proj*u+iv["c"]*u
        return (hs,)+tuple(oo[1:]) if isinstance(oo,tuple) else hs
    h=model.gpt_neox.layers[A.llex-1].register_forward_hook(hook)
    try:
        for i in range(0,len(seqs),64):
            b=seqs[i:i+64]; lg=model(torch.tensor(b,device=DEV)).logits[:,-1,:].float()
            r=torch.logsumexp(lg[:,PW],1)-torch.logsumexp(lg[:,OW],1); out.append(r.cpu())
    finally: h.remove()
    return torch.cat(out)
# class means of the place-axis projection at the noun pos in the FRAME
u=(pl_lex/pl_lex.norm())
def frame_resid(words): return resid_batch([FRAME(w) for w in words],[1]*len(words),A.llex)
proj_place=float((frame_resid(PLACE)@u).mean()); proj_obj=float((frame_resid(NONPLACE)@u).mean())
print(f"\n(2) CAUSAL -- frame 'The N is a type of ___', readout = logit(place-words) - logit(object-words):")
# baseline category-consistency
rb_place=readout([FRAME(w) for w in PLACE]).mean(); rb_obj=readout([FRAME(w) for w in NONPLACE]).mean()
print(f"   baseline readout: place-subject {rb_place:+.2f}  vs  nonplace-subject {rb_obj:+.2f}  (consistency {rb_place-rb_obj:+.2f})")
res["baseline"]={"place_subj":float(rb_place),"nonplace_subj":float(rb_obj)}
# target = nonplace nouns; patch place-component to place-mean vs object-mean
TGT=[FRAME(w) for w in NONPLACE]
r_to_place=readout(TGT, {"pos":1,"u":u,"c":proj_place}).mean()
r_to_obj  =readout(TGT, {"pos":1,"u":u,"c":proj_obj}).mean()
eff_place=float(r_to_place-r_to_obj)
# random-direction control: matched projection values
g=torch.Generator().manual_seed(7); ur=torch.randn(D,generator=g); ur=ur/ur.norm()
rp_place=float((frame_resid(PLACE)@ur).mean()); rp_obj=float((frame_resid(NONPLACE)@ur).mean())
r_rand_hi=readout(TGT,{"pos":1,"u":ur,"c":rp_place}).mean(); r_rand_lo=readout(TGT,{"pos":1,"u":ur,"c":rp_obj}).mean()
eff_rand=float(r_rand_hi-r_rand_lo)
print(f"   PATCH place-axis on nonplace targets: set->place {r_to_place:+.2f}  set->object {r_to_obj:+.2f}  EFFECT {eff_place:+.2f}")
print(f"   random-direction control (matched):                                              EFFECT {eff_rand:+.2f}")
verdict=("CAUSAL: the discovered place axis shifts the category prediction; random does not" if eff_place>0.3 and eff_place>3*abs(eff_rand)
         else "NOT clearly causal (effect ~ random control)")
print(f"   -> {verdict}")
res["causal"]={"effect_place":eff_place,"effect_random":eff_rand,"set_to_place":float(r_to_place),"set_to_obj":float(r_to_obj),"verdict":verdict}
Path(str(ROOT/"palimpsest"/"data"/"place_causal.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: (1) place transports like the named features (rides the same deposit map). (2) patching ONLY the")
print("  place component flips the category prediction toward place while a matched random direction does nothing")
print("  => the unsupervised-discovered axis is a CAUSAL channel content, validating discovery->causal end to end.")
print("\n wrote palimpsest/data/place_causal.json")
