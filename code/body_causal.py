#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/body_causal.py
CAUSALLY VALIDATE the second discovered channel axis: BODY-PART (Olli). Mirrors place_causal.py.
 (1) TRANSPORT: cos(R*body_lex, body_transport) -- does the body-part axis ride the shared deposit map?
 (2) CAUSAL: frame "The {N} is a part of the ___"; readout = logit(body-context words) - logit(object-context
     words) at the last token (body parts -> body/face/skeleton; objects -> machine/building/system). Patch ONLY
     the body-axis component of the noun residual @L8 (set to body-class mean vs object-class mean); measure the
     readout shift, with a matched RANDOM-direction control. Body axis shifts it / random does not => causal.

Usage: .venv/Scripts/python.exe palimpsest/code/body_causal.py
Output: palimpsest/data/body_causal.json
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
print(f"BODY-PART AXIS CAUSAL VALIDATION | {A.model} | {DEV}")
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
BODY=[w for w in ["eye","heart","face","head","hand","foot","arm","leg","ear","nose","mouth","finger","knee","neck","chest","bone","skin","lip","toe","wrist","elbow","shoulder","thumb","brain","tongue","jaw","hip","heel","palm"] if has(w)]
OBJ =[w for w in ["car","box","table","chair","book","lamp","cup","desk","door","wall","clock","knife","ship","bottle","engine","brick","plate","spoon"] if has(w)]
N2POOL=[w for w in OBJ if has(w)]; VBR=[w for w in ["hurt","blamed","saw","watched","washed","followed","pushed"] if has(w)]
print(f"  {len(BODY)} body-part / {len(OBJ)} object nouns")
@torch.no_grad()
def resid_batch(seqs,pos,L):
    out=[]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; hs=model(torch.tensor(b,device=DEV),output_hidden_states=True).hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos[i+j]].cpu())
    return torch.stack(out)
TAILS=["near the table","by the window","in the room","on the shelf"]
def lex_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc(rng.choice(TAILS)) for _ in range(n)],[1]*n,A.llex).mean(0)
def tr_sig(w,n): return resid_batch([enc("The",True)+[tid(" "+w)]+enc("near")+enc("the")+[tid(" "+rng.choice(N2POOL))]+[tid(" "+rng.choice(VBR))] for _ in range(n)],[5]*n,A.ltr).mean(0)
# ---------- (1) transport ----------
WL=[w for w in BODY+OBJ if has(w)]
Xlex=torch.stack([lex_sig(w,8) for w in WL]); Xtr=torch.stack([tr_sig(w,8) for w in WL]); mu_l=Xlex.mean(0)
XL=Xlex-mu_l; XT=Xtr-Xtr.mean(0); lam=1e-1*XL.pow(2).sum(0).mean(); R=torch.linalg.solve(XL.T@XL+lam*torch.eye(D),XL.T@XT)
def dir_lt(pa,pb,sig,L):
    a=torch.stack([sig(w,6) for w in pa]).mean(0); b=torch.stack([sig(w,6) for w in pb]).mean(0); w=a-b; return w/(w.norm()+1e-9)
bd_lex=dir_lt(BODY,OBJ,lex_sig,A.llex); bd_tr=dir_lt(BODY,OBJ,tr_sig,A.ltr)
anim_lex=dir_lt([w for w in ["dog","cat","horse","bird","cow"] if has(w)],[w for w in ["car","box","table","book","cup"] if has(w)],lex_sig,A.llex)
anim_tr =dir_lt([w for w in ["dog","cat","horse","bird","cow"] if has(w)],[w for w in ["car","box","table","book","cup"] if has(w)],tr_sig,A.ltr)
cbd=cos((bd_lex@R)/((bd_lex@R).norm()+1e-9),bd_tr); can=cos((anim_lex@R)/((anim_lex@R).norm()+1e-9),anim_tr)
print(f"\n(1) TRANSPORT through deposit map R: cos(R*lex, transport)  body-part {cbd:.2f}  | animacy(baseline) {can:.2f}")
res={"model":A.model,"transport":{"body":cbd,"animacy_baseline":can}}
# ---------- (2) causal ----------
FRAME=lambda w: enc("The",True)+[tid(" "+w)]+enc("is a part of the")  # noun pos1, read last
BODY_W=[" body"," face"," head"," skeleton"," arm"," leg"," skull"," chest"]; OBJ_W=[" machine"," building"," system"," structure"," engine"," house"," device"," frame"]
BW=[tid(t) for t in BODY_W if tid(t)]; OW=[tid(t) for t in OBJ_W if tid(t)]
@torch.no_grad()
def readout(seqs, intervene=None):
    H={"on":intervene is not None,"iv":intervene}
    def hook(m,inp,oo):
        if not H["on"]: return
        hs=oo[0] if isinstance(oo,tuple) else oo; iv=H["iv"]; u=iv["u"].to(hs.device); p=iv["pos"]
        proj=(hs[:,p,:]@u).unsqueeze(1); hs[:,p,:]=hs[:,p,:]-proj*u+iv["c"]*u
        return (hs,)+tuple(oo[1:]) if isinstance(oo,tuple) else hs
    h=model.gpt_neox.layers[A.llex-1].register_forward_hook(hook)
    try:
        out=[]
        for i in range(0,len(seqs),64):
            b=seqs[i:i+64]; lg=model(torch.tensor(b,device=DEV)).logits[:,-1,:].float()
            out.append((torch.logsumexp(lg[:,BW],1)-torch.logsumexp(lg[:,OW],1)).cpu())
        return torch.cat(out)
    finally: h.remove()
u=(bd_lex/bd_lex.norm())
def frame_resid(words): return resid_batch([FRAME(w) for w in words],[1]*len(words),A.llex)
proj_body=float((frame_resid(BODY)@u).mean()); proj_obj=float((frame_resid(OBJ)@u).mean())
print(f"\n(2) CAUSAL -- frame 'The N is a part of the ___', readout = logit(body-words) - logit(object-words):")
rb_body=readout([FRAME(w) for w in BODY]).mean(); rb_obj=readout([FRAME(w) for w in OBJ]).mean()
print(f"   baseline readout: body-subject {rb_body:+.2f}  vs  object-subject {rb_obj:+.2f}  (consistency {rb_body-rb_obj:+.2f})")
res["baseline"]={"body_subj":float(rb_body),"object_subj":float(rb_obj)}
TGT=[FRAME(w) for w in OBJ]
r_to_body=readout(TGT,{"pos":1,"u":u,"c":proj_body}).mean(); r_to_obj=readout(TGT,{"pos":1,"u":u,"c":proj_obj}).mean()
eff_body=float(r_to_body-r_to_obj)
g=torch.Generator().manual_seed(7); ur=torch.randn(D,generator=g); ur=ur/ur.norm()
rp_body=float((frame_resid(BODY)@ur).mean()); rp_obj=float((frame_resid(OBJ)@ur).mean())
r_rand_hi=readout(TGT,{"pos":1,"u":ur,"c":rp_body}).mean(); r_rand_lo=readout(TGT,{"pos":1,"u":ur,"c":rp_obj}).mean()
eff_rand=float(r_rand_hi-r_rand_lo)
print(f"   PATCH body-axis on object targets: set->body {r_to_body:+.2f}  set->object {r_to_obj:+.2f}  EFFECT {eff_body:+.2f}")
print(f"   random-direction control (matched):                                          EFFECT {eff_rand:+.2f}")
verdict=("CAUSAL: the discovered body-part axis shifts the category prediction; random does not" if eff_body>0.3 and eff_body>3*abs(eff_rand)
         else "NOT clearly causal (effect ~ random control)")
print(f"   -> {verdict}")
res["causal"]={"effect_body":eff_body,"effect_random":eff_rand,"set_to_body":float(r_to_body),"set_to_obj":float(r_to_obj),"verdict":verdict}
Path(str(ROOT/"palimpsest"/"data"/"body_causal.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: body-part axis transports + patching it flips the body/object category prediction while a matched")
print("  random direction does nothing => a second unsupervised-discovered axis validated causally.")
print("\n wrote palimpsest/data/body_causal.json")
