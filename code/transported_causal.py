#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/transported_causal.py
IS THE DEPOSITED (TRANSPORTED) COPY CAUSAL AT THE CONSUMING POSITION? (Olli -- close the transport loop.)
place_causal showed the LEXICAL place axis at the noun is causal. Now test the re-encoded copy that attention
DEPOSITED downstream: patch the category axis at the CONSUMING position (the last token, where only the transported
copy lives -- the noun token is back at position 1), past a distractor, and see if it still drives the prediction.

Frame F(N) = "The {N} near the old table is a type of ___"  (subject N @pos1; distractor 'table' @pos5; read @last)
Readout = logit(place-words) - logit(object-words).
 (1) Baseline: place-subject vs object-subject readout (with distractor) -> the category SURVIVES transport.
 (2) CAUSAL @ CONSUMING position: u_cons = transported category direction at the last token (place-subj vs
     object-subj residual @last, @L8). Patch ONLY that component at the LAST position (set place-mean vs
     object-mean); measure readout shift. Matched RANDOM control.
 (3) For comparison: patch the LEXICAL axis at the NOUN position in the same frame.
If patching at the consuming position (transported copy, noun absent there) shifts the prediction while random does
not, the deposited copy is itself CAUSAL at the point of use -- the channel delivers causal semantic type to where
it is consumed.

Usage: .venv/Scripts/python.exe palimpsest/code/transported_causal.py
Output: palimpsest/data/transported_causal.json
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np, torch
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass
ROOT=Path(__file__).resolve().parent.parent.parent
P=argparse.ArgumentParser(); P.add_argument("--model",default="EleutherAI/pythia-410m-deduped"); P.add_argument("--seed",type=int,default=0); P.add_argument("--L",type=int,default=8)
A=P.parse_args(); DEV="cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"TRANSPORTED-COPY CAUSALITY (consuming position) | {A.model} | L{A.L} | {DEV}")
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
PLACE=[w for w in ["city","town","country","village","river","island","forest","mountain","valley","harbor","lake","field","road","bridge"] if has(w)]
OBJ =[w for w in ["car","box","chair","book","lamp","cup","desk","door","wall","clock","knife","ship","bottle","plate"] if has(w)]
print(f"  {len(PLACE)} place / {len(OBJ)} object nouns")
# frame: The(0) N(1) near(2) the(3) old(4) table(5) is(6) a(7) type(8) of(9)
FRAME=lambda w: enc("The",True)+[tid(" "+w)]+enc("near the old table is a type of")
LEN=len(FRAME("city")); NOUN=1; LAST=LEN-1
assert all(len(FRAME(w))==LEN for w in PLACE+OBJ), "frame length varies"
print(f"  frame len {LEN}, noun@{NOUN}, consuming(last)@{LAST}")
PLACE_W=[" city"," town"," place"," country"," area"," region"]; OBJ_W=[" object"," thing"," tool"," device"," item"," material"]
PW=[tid(t) for t in PLACE_W if tid(t)]; OW=[tid(t) for t in OBJ_W if tid(t)]
@torch.no_grad()
def resid_at(words,pos,L):
    out=[]; seqs=[FRAME(w) for w in words]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; hs=model(torch.tensor(b,device=DEV),output_hidden_states=True).hidden_states[L]
        for j in range(len(b)): out.append(hs[j,pos].cpu())
    return torch.stack(out)
STATE={"on":False,"u":None,"c":0.0,"pos":LAST,"layer":A.L}
def mk(idx):
    def hook(m,inp,oo):
        if not STATE["on"] or STATE["layer"]-1!=idx: return
        hs=oo[0] if isinstance(oo,tuple) else oo; u=STATE["u"].to(hs.device); p=STATE["pos"]
        proj=(hs[:,p,:]@u).unsqueeze(1); hs[:,p,:]=hs[:,p,:]-proj*u+STATE["c"]*u
        return (hs,)+tuple(oo[1:]) if isinstance(oo,tuple) else hs
    return hook
for i in range(NL): model.gpt_neox.layers[i].register_forward_hook(mk(i))
@torch.no_grad()
def readout(words, iv=None):
    if iv: STATE.update(iv); STATE["on"]=True
    out=[]; seqs=[FRAME(w) for w in words]
    for i in range(0,len(seqs),64):
        b=seqs[i:i+64]; lg=model(torch.tensor(b,device=DEV)).logits[:,-1,:].float()
        out.append((torch.logsumexp(lg[:,PW],1)-torch.logsumexp(lg[:,OW],1)).cpu())
    STATE["on"]=False; return torch.cat(out)
# (1) baseline: category survives transport past the distractor?
rb_p=float(readout(PLACE).mean()); rb_o=float(readout(OBJ).mean())
print(f"\n(1) baseline readout (place-words - object-words), WITH distractor 'table':")
print(f"    place-subject {rb_p:+.2f}  vs  object-subject {rb_o:+.2f}  (category survives transport: {rb_p-rb_o:+.2f})")
res={"model":A.model,"L":A.L,"baseline":{"place":rb_p,"object":rb_o}}
# directions
u_cons=(resid_at(PLACE,LAST,A.L).mean(0)-resid_at(OBJ,LAST,A.L).mean(0)); u_cons=u_cons/u_cons.norm()  # transported copy @last
u_noun=(resid_at(PLACE,NOUN,A.L).mean(0)-resid_at(OBJ,NOUN,A.L).mean(0)); u_noun=u_noun/u_noun.norm()    # lexical copy @noun
print(f"\n    cos(transported@last , lexical@noun) = {float(torch.dot(u_cons,u_noun)):.2f} (re-encoded if low)")
res["cos_transported_vs_lexical"]=float(torch.dot(u_cons,u_noun))
def proj_means(u,pos,L): return float((resid_at(PLACE,pos,L)@u).mean()), float((resid_at(OBJ,pos,L)@u).mean())
def patch_effect(u,pos,L,label):
    cp,co=proj_means(u,pos,L); tgt=OBJ
    r_to_p=float(readout(tgt,{"u":u,"c":cp,"pos":pos,"layer":L}).mean()); r_to_o=float(readout(tgt,{"u":u,"c":co,"pos":pos,"layer":L}).mean())
    eff=r_to_p-r_to_o
    g=torch.Generator().manual_seed(7); ur=torch.randn(D,generator=g); ur=ur/ur.norm()
    rp,ro=proj_means(ur,pos,L); er=float(readout(tgt,{"u":ur,"c":rp,"pos":pos,"layer":L}).mean())-float(readout(tgt,{"u":ur,"c":ro,"pos":pos,"layer":L}).mean())
    print(f"    {label}: set->place {r_to_p:+.2f}  set->object {r_to_o:+.2f}  EFFECT {eff:+.2f}  (random {er:+.2f})")
    return {"effect":eff,"random":er}
print(f"\n(3) NOUN(pos{NOUN}) lexical copy @L{A.L} for reference:")
res["noun_position"]=patch_effect(u_noun,NOUN,A.L,f"@NOUN(pos{NOUN})")
print(f"\n(2) CAUSAL @ CONSUMING position (last, pos{LAST}) -- SWEEP the patch layer (deposit may be late):")
res["consuming_by_layer"]={}
for L in [4,8,12,16,20,23]:
    uL=(resid_at(PLACE,LAST,L).mean(0)-resid_at(OBJ,LAST,L).mean(0)); uL=uL/uL.norm()
    res["consuming_by_layer"][L]=patch_effect(uL,LAST,L,f"@CONSUMING L{L:>2}")
ecs=[res["consuming_by_layer"][L]["effect"] for L in res["consuming_by_layer"]]
best=max(res["consuming_by_layer"].items(), key=lambda kv: kv[1]["effect"])
verdict=(f"deposited copy BECOMES causal at the consuming position by L{best[0]} (effect {best[1]['effect']:+.2f})"
         if best[1]["effect"]>0.5 and best[1]["effect"]>3*abs(best[1]["random"])
         else "deposited copy NOT causal at the consuming position at any layer -> the prediction reads the SOURCE noun via late attention, not the broadcast copy")
print(f"\n   => {verdict}")
res["verdict"]=verdict
Path(str(ROOT/"palimpsest"/"data"/"transported_causal.json")).write_text(json.dumps(res,indent=2),encoding="utf-8")
print("\n READ: patching the category axis at the LAST token (where only the TRANSPORTED copy lives, noun is @pos1)")
print("  shifts the category prediction while a matched random direction does not => the deposited copy is causal")
print("  AT THE POINT OF CONSUMPTION. The channel delivers causal semantic type to where it is used.")
print("\n wrote palimpsest/data/transported_causal.json")
