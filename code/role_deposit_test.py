#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/role_deposit_test.py
DO OTHER GRAMMATICAL ROLES GET A DEPOSIT CHANNEL?

The subject->verb deposit channel is established (feature_deposit_map.py): shared low-rank
map R re-encodes subject type signature at the verb (held-out cos 0.54 vs identity 0.10).
This script tests whether similar channels exist for:

  (A) Subject -> verb  (baseline replication)
  (B) Object -> pronoun  (object features re-encoded at a referring pronoun?)
  (C) Object -> clause-end  (deposited generally, not just at a consumer?)
  (D) Cross-role: apply R_subj to object lexical -> does it predict object transport?

If R_obj generalises to held-out words -> object channel exists.
If R_subj predicts object transport -> one shared deposit op across roles.
If R_obj does NOT generalise -> deposit only where the training signal demands it.

Templates (all single-token positions verified):
  Subject lex:  "The {w} near the table"            -> w @ pos 1
  Subject tr:   "The {w} near the {N2} {V}"         -> V @ pos 5
  Object lex:   "The {S} saw the {w} by the wall"   -> w @ pos 4
  Object->pro:  "The {S} saw the {w} near the {N2} and it" -> "it" @ pos 9
  Object->end:  "The {S} saw the {w} by the wall"   -> "wall" @ pos 7

Usage: .venv/Scripts/python.exe palimpsest/code/role_deposit_test.py
Output: palimpsest/data/role_deposit_test.json
"""
import sys, json, argparse, random
from pathlib import Path
import numpy as np, torch
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass
ROOT = Path(__file__).resolve().parent.parent.parent

P = argparse.ArgumentParser()
P.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--llex", type=int, default=8)
P.add_argument("--ltr", type=int, default=12)
P.add_argument("--ns", type=int, default=12)   # samples per word
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"ROLE DEPOSIT TEST | {A.model} | lex L{A.llex} -> tr L{A.ltr} | {DEV}")
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
for p in model.parameters(): p.requires_grad_(False)
D = model.config.hidden_size

def tid(s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None

def enc(s, first=False):
    return tok(s if first else " " + s, add_special_tokens=False)["input_ids"]

def has(w): return tid(" " + w) is not None

rng = random.Random(A.seed)

def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))

# ---------- word inventories ----------
NUMP = [("key","keys"),("book","books"),("door","doors"),("car","cars"),("box","boxes"),
        ("dog","dogs"),("cat","cats"),("boy","boys"),("girl","girls"),("table","tables"),
        ("bird","birds"),("tree","trees"),("wall","walls"),("road","roads"),("house","houses"),
        ("chair","chairs"),("lamp","lamps"),("cup","cups"),("hand","hands"),("king","kings")]
MASC = [w for w in ["man","king","boy","father","son","brother","uncle","prince","actor",
        "priest","duke","lord","waiter","nephew"] if has(w)]
FEM  = [w for w in ["woman","queen","girl","mother","daughter","sister","aunt","princess",
        "actress","lady","nun","duchess","widow","niece"] if has(w)]
ANIM = [w for w in ["man","woman","dog","cat","boy","girl","king","queen","bird","horse",
        "lion","cow","fish","child"] if has(w)]
INAN = [w for w in ["car","box","table","chair","door","wall","book","lamp","rock","cup",
        "road","window","stone","desk","clock","knife"] if has(w)]

def mk_words():
    W = {}
    for s, p in NUMP:
        if tid(" " + s): W.setdefault(s, {})["number"] = "sg"
        if tid(" " + p): W.setdefault(p, {})["number"] = "pl"
    for w in MASC: W.setdefault(w, {})["gender"] = "m"
    for w in FEM:  W.setdefault(w, {})["gender"] = "f"
    for w in ANIM: W.setdefault(w, {})["anim"] = "a"
    for w in INAN: W.setdefault(w, {})["anim"] = "i"
    return {w: t for w, t in W.items() if tid(" " + w)}

WORDS = mk_words()
WL = sorted(WORDS)
print(f"  {len(WL)} unique noun tokens")

# context pools
N2POOL = [w for w in INAN if has(w)]
VBR = [w for w in ["hurt","blamed","saw","watched","washed","cleaned","defended",
       "scratched","followed","pushed"] if has(w)]
SUBJ_POOL = [w for w in ["boy","girl","man","woman","king","queen","lord","lady"] if has(w)]
OBJ_VERB = [w for w in ["saw","found","watched","hit","caught","held","liked","helped",
            "chased","loved"] if has(w)]
TAILS = ["near the table", "by the window", "in the room", "on the shelf", "beside the wall"]

# ---------- residual collection ----------
@torch.no_grad()
def resid_batch(seqs, pos, L):
    out = []
    for i in range(0, len(seqs), 64):
        b = seqs[i:i+64]
        hs = model(torch.tensor(b, device=DEV), output_hidden_states=True).hidden_states[L]
        for j in range(len(b)):
            out.append(hs[j, pos[i+j]].cpu())
    return torch.stack(out)

# ---------- template builders ----------
def subj_lex_seq(w):
    """'The {w} near the table' -> w @ pos 1"""
    return enc("The", True) + [tid(" " + w)] + enc(rng.choice(TAILS))

def subj_tr_seq(w):
    """'The {w} near the {N2} {V}' -> V @ pos 5"""
    return (enc("The", True) + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(N2POOL))] + [tid(" " + rng.choice(VBR))])

def obj_lex_seq(w):
    """'The {S} saw the {w} by the wall' -> w @ pos 4"""
    return (enc("The", True) + [tid(" " + rng.choice(SUBJ_POOL))]
            + [tid(" " + rng.choice(OBJ_VERB))] + enc("the") + [tid(" " + w)]
            + enc("by") + enc("the") + enc("wall"))

def obj_pro_seq(w):
    """'The {S} saw the {w} near the {N2} and it' -> 'it' @ pos 9"""
    return (enc("The", True) + [tid(" " + rng.choice(SUBJ_POOL))]
            + [tid(" " + rng.choice(OBJ_VERB))] + enc("the") + [tid(" " + w)]
            + enc("near") + enc("the") + [tid(" " + rng.choice(N2POOL))]
            + enc("and") + enc("it"))

def obj_end_seq(w):
    """'The {S} saw the {w} by the wall' -> 'wall' @ pos 7"""
    return obj_lex_seq(w)  # same template, different read position

# ---------- per-word signatures ----------
def collect_sigs(mk_seq, pos, L, ns):
    """Collect mean residual at (pos, L) for each word in WL."""
    sigs = []
    for w in WL:
        seqs = [mk_seq(w) for _ in range(ns)]
        sigs.append(resid_batch(seqs, [pos] * ns, L).mean(0))
    return torch.stack(sigs)

# ---------- fit & eval ridge map ----------
def fit_R(XL, XT, rows):
    Xc = XL[rows]; Yc = XT[rows]
    lam = 1e-1 * Xc.pow(2).sum(0).mean()
    G = Xc.T @ Xc + lam * torch.eye(D)
    return torch.linalg.solve(G, Xc.T @ Yc)

def eval_R(R, XL, XT, rows):
    pred = XL[rows] @ R
    cs = [cos(pred[i], XT[rows][i]) for i in range(len(rows))]
    r2 = float(1 - (XT[rows] - pred).pow(2).sum() / XT[rows].pow(2).sum())
    return float(np.mean(cs)), r2

# ---------- feature direction helper ----------
def fdir(pool_a, pool_b, mk_seq, pos, L, n=220):
    sa = [mk_seq(rng.choice(pool_a)) for _ in range(n)]
    sb = [mk_seq(rng.choice(pool_b)) for _ in range(n)]
    ma = resid_batch(sa, [pos] * n, L).mean(0)
    mb = resid_batch(sb, [pos] * n, L).mean(0)
    w = ma - mb
    return w / (w.norm() + 1e-9)

# ---------- eta-squared (word identity) ----------
def eta2(mk_seq, pos, L, nper=12):
    words = rng.sample(WL, min(16, len(WL)))
    samp = {w: resid_batch([mk_seq(w) for _ in range(nper)], [pos] * nper, L) for w in words}
    allX = torch.cat([samp[w] for w in words])
    gm = allX.mean(0)
    sst = (allX - gm).pow(2).sum()
    ssb = sum(len(samp[w]) * ((samp[w].mean(0) - gm).pow(2).sum()) for w in words)
    return float(ssb / sst)

# ========== COLLECT ==========
print("\n--- Collecting signatures ---")
print("  subject lexical (pos 1, L{}) ...".format(A.llex))
XL_subj = collect_sigs(subj_lex_seq, 1, A.llex, A.ns)
print("  subject transport (pos 5, L{}) ...".format(A.ltr))
XT_subj = collect_sigs(subj_tr_seq, 5, A.ltr, A.ns)
print("  object lexical (pos 4, L{}) ...".format(A.llex))
XL_obj  = collect_sigs(obj_lex_seq, 4, A.llex, A.ns)
print("  object -> pronoun (pos 9, L{}) ...".format(A.ltr))
XT_obj_pro = collect_sigs(obj_pro_seq, 9, A.ltr, A.ns)
print("  object -> clause-end (pos 7, L{}) ...".format(A.ltr))
XT_obj_end = collect_sigs(obj_end_seq, 7, A.ltr, A.ns)

# center (noun-dependent component)
def center(X):
    mu = X.mean(0); return X - mu, mu
XL_subj_c, mu_ls = center(XL_subj)
XT_subj_c, mu_ts = center(XT_subj)
XL_obj_c,  mu_lo = center(XL_obj)
XT_op_c,   mu_tp = center(XT_obj_pro)
XT_oe_c,   mu_te = center(XT_obj_end)

# train/test split (same for all, by word index)
idx = list(range(len(WL))); rng.shuffle(idx)
cut = int(0.75 * len(idx))
tr, te = idx[:cut], idx[cut:]
print(f"  train {len(tr)} words, test {len(te)} words")

# ========== (A) SUBJECT -> VERB ==========
print("\n=== (A) SUBJECT -> VERB (baseline) ===")
R_subj = fit_R(XL_subj_c, XT_subj_c, tr)
cs_tr, r2_tr = eval_R(R_subj, XL_subj_c, XT_subj_c, tr)
cs_te, r2_te = eval_R(R_subj, XL_subj_c, XT_subj_c, te)
# null
perm = te[:]; rng.shuffle(perm)
pred_n = XL_subj_c[te] @ R_subj
cs_null = float(np.mean([cos(pred_n[i], XT_subj_c[perm][i]) for i in range(len(te))]))
# identity
cs_id = float(np.mean([cos(XL_subj_c[te][i], XT_subj_c[te][i]) for i in range(len(te))]))
print(f"  R_subj: train cos {cs_tr:.3f} | HELD-OUT cos {cs_te:.3f} R² {r2_te:.3f}")
print(f"  baselines: identity {cs_id:.3f} | null {cs_null:.3f}")
res_subj = {"train_cos": cs_tr, "heldout_cos": cs_te, "heldout_r2": r2_te,
            "identity_cos": cs_id, "null_cos": cs_null}

# ========== (B) OBJECT -> PRONOUN ==========
print("\n=== (B) OBJECT -> PRONOUN ===")
R_obj_pro = fit_R(XL_obj_c, XT_op_c, tr)
cs_tr_o, r2_tr_o = eval_R(R_obj_pro, XL_obj_c, XT_op_c, tr)
cs_te_o, r2_te_o = eval_R(R_obj_pro, XL_obj_c, XT_op_c, te)
pred_n2 = XL_obj_c[te] @ R_obj_pro
cs_null_o = float(np.mean([cos(pred_n2[i], XT_op_c[perm][i]) for i in range(len(te))]))
cs_id_o = float(np.mean([cos(XL_obj_c[te][i], XT_op_c[te][i]) for i in range(len(te))]))
print(f"  R_obj_pro: train cos {cs_tr_o:.3f} | HELD-OUT cos {cs_te_o:.3f} R² {r2_te_o:.3f}")
print(f"  baselines: identity {cs_id_o:.3f} | null {cs_null_o:.3f}")
res_obj_pro = {"train_cos": cs_tr_o, "heldout_cos": cs_te_o, "heldout_r2": r2_te_o,
               "identity_cos": cs_id_o, "null_cos": cs_null_o}

# ========== (C) OBJECT -> CLAUSE-END ==========
print("\n=== (C) OBJECT -> CLAUSE-END (no explicit consumer) ===")
R_obj_end = fit_R(XL_obj_c, XT_oe_c, tr)
cs_tr_e, r2_tr_e = eval_R(R_obj_end, XL_obj_c, XT_oe_c, tr)
cs_te_e, r2_te_e = eval_R(R_obj_end, XL_obj_c, XT_oe_c, te)
pred_n3 = XL_obj_c[te] @ R_obj_end
cs_null_e = float(np.mean([cos(pred_n3[i], XT_oe_c[perm][i]) for i in range(len(te))]))
cs_id_e = float(np.mean([cos(XL_obj_c[te][i], XT_oe_c[te][i]) for i in range(len(te))]))
print(f"  R_obj_end: train cos {cs_tr_e:.3f} | HELD-OUT cos {cs_te_e:.3f} R² {r2_te_e:.3f}")
print(f"  baselines: identity {cs_id_e:.3f} | null {cs_null_e:.3f}")
res_obj_end = {"train_cos": cs_tr_e, "heldout_cos": cs_te_e, "heldout_r2": r2_te_e,
               "identity_cos": cs_id_e, "null_cos": cs_null_e}

# ========== (D) CROSS-ROLE: R_subj applied to object lexical ==========
print("\n=== (D) CROSS-ROLE: R_subj -> object transport ===")
# apply R_subj to object lexical, see if it predicts object->pronoun transport
pred_cross = XL_obj_c[te] @ R_subj
cs_cross_pro = float(np.mean([cos(pred_cross[i], XT_op_c[te][i]) for i in range(len(te))]))
cs_cross_end = float(np.mean([cos(pred_cross[i], XT_oe_c[te][i]) for i in range(len(te))]))
print(f"  R_subj applied to object -> pronoun: cos {cs_cross_pro:.3f}")
print(f"  R_subj applied to object -> clause-end: cos {cs_cross_end:.3f}")
print(f"  (compare: R_subj on subject -> verb held-out cos {cs_te:.3f})")
res_cross = {"subj_R_to_obj_pronoun": cs_cross_pro, "subj_R_to_obj_end": cs_cross_end}

# ========== (E) FEATURE BATTERY ==========
print("\n=== FEATURE BATTERY ===")
SG = [s for s, _ in NUMP if has(s)]; PL = [p for _, p in NUMP if has(p)]
FEATS = {
    "number": (SG, PL),
    "gender": (MASC, FEM),
    "animacy": (ANIM, INAN),
}

def battery(name, mk_lex, pos_lex, mk_tr, pos_tr, R_map):
    U, S, Vh = torch.linalg.svd(R_map)
    Uk = U[:, :30]
    null_e = 30.0 / D  # random expected energy
    print(f"\n  {name}:")
    res = {}
    for fname, (pa, pb) in FEATS.items():
        pa2 = [w for w in pa if has(w)]; pb2 = [w for w in pb if has(w)]
        if len(pa2) < 3 or len(pb2) < 3:
            print(f"    {fname:>10}: insufficient words"); continue
        lF = fdir(pa2, pb2, mk_lex, pos_lex, A.llex)
        tF = fdir(pa2, pb2, mk_tr, pos_tr, A.ltr)
        e = float((Uk.T @ lF).pow(2).sum() / (lF @ lF + 1e-9))
        mapped = lF @ R_map; cmap = cos(mapped / (mapped.norm() + 1e-9), tF)
        v = ("CARRIED" if e > 4 * null_e and cmap > 0.5 else
             ("filtered" if e < 2 * null_e else "partial"))
        res[fname] = {"chan_energy": round(e, 3), "transport_cos": round(cmap, 3), "verdict": v}
        print(f"    {fname:>10}: energy {e:.3f} (null ~{null_e:.3f})  transport cos {cmap:.3f}  -> {v}")
    return res

bat_subj = battery("SUBJECT->VERB", subj_lex_seq, 1, subj_tr_seq, 5, R_subj)
bat_obj_pro = battery("OBJECT->PRONOUN", obj_lex_seq, 4, obj_pro_seq, 9, R_obj_pro)
bat_obj_end = battery("OBJECT->CLAUSE-END", obj_lex_seq, 4, obj_end_seq, 7, R_obj_end)

# ========== (F) WORD IDENTITY (eta^2) ==========
print("\n=== WORD IDENTITY (eta^2) ===")
e_subj_lex = eta2(subj_lex_seq, 1, A.llex)
e_subj_tr  = eta2(subj_tr_seq, 5, A.ltr)
e_obj_lex  = eta2(obj_lex_seq, 4, A.llex)
e_obj_pro  = eta2(obj_pro_seq, 9, A.ltr)
e_obj_end  = eta2(obj_end_seq, 7, A.ltr)
print(f"  subject: lexical {e_subj_lex:.3f} -> verb {e_subj_tr:.3f}")
print(f"  object:  lexical {e_obj_lex:.3f} -> pronoun {e_obj_pro:.3f} | clause-end {e_obj_end:.3f}")
eta_res = {"subj_lex": e_subj_lex, "subj_verb": e_subj_tr,
           "obj_lex": e_obj_lex, "obj_pronoun": e_obj_pro, "obj_end": e_obj_end}

# ========== (G) EFFECTIVE RANK comparison ==========
sv_subj = torch.linalg.svdvals(R_subj)
sv_obj_pro = torch.linalg.svdvals(R_obj_pro)
sv_obj_end = torch.linalg.svdvals(R_obj_end)
er = lambda s: float((s.sum()**2) / (s.pow(2).sum()))
print(f"\n=== EFFECTIVE RANK ===")
print(f"  R_subj: {er(sv_subj):.0f}  R_obj_pro: {er(sv_obj_pro):.0f}  R_obj_end: {er(sv_obj_end):.0f}")

# ========== VERDICT ==========
print("\n" + "=" * 60)
subj_has = cs_te > cs_id + 0.15 and cs_te > cs_null + 0.2
obj_pro_has = cs_te_o > cs_id_o + 0.15 and cs_te_o > cs_null_o + 0.2
obj_end_has = cs_te_e > cs_id_e + 0.15 and cs_te_e > cs_null_e + 0.2
cross_works = cs_cross_pro > cs_null + 0.15

print(f"SUBJECT->VERB channel:   {'YES' if subj_has else 'NO'}  (held-out {cs_te:.3f} vs id {cs_id:.3f} / null {cs_null:.3f})")
print(f"OBJECT->PRONOUN channel: {'YES' if obj_pro_has else 'NO'}  (held-out {cs_te_o:.3f} vs id {cs_id_o:.3f} / null {cs_null_o:.3f})")
print(f"OBJECT->CLAUSE-END:      {'YES' if obj_end_has else 'NO'}  (held-out {cs_te_e:.3f} vs id {cs_id_e:.3f} / null {cs_null_e:.3f})")
print(f"CROSS-ROLE (R_subj->obj):{'YES' if cross_works else 'NO'}  (cos {cs_cross_pro:.3f})")

if subj_has and not obj_pro_has:
    print("\n=> DEPOSIT IS ROLE-SPECIFIC: exists for subject->verb but NOT object->pronoun.")
    print("   Consistent with: deposit channel emerges where agreement/prediction DEMANDS it.")
    print("   English has subject-verb agreement but no object agreement -> no object channel.")
elif subj_has and obj_pro_has and cross_works:
    print("\n=> SHARED GENERIC DEPOSIT OP: one map works across roles.")
elif subj_has and obj_pro_has and not cross_works:
    print("\n=> ROLE-SPECIFIC DEPOSIT MAPS: both roles have channels, but different maps.")

res = {
    "model": A.model, "llex": A.llex, "ltr": A.ltr, "nwords": len(WL),
    "subject_verb": res_subj,
    "object_pronoun": res_obj_pro,
    "object_clause_end": res_obj_end,
    "cross_role": res_cross,
    "battery_subject": bat_subj,
    "battery_obj_pronoun": bat_obj_pro,
    "battery_obj_clause_end": bat_obj_end,
    "eta2": eta_res,
    "eff_rank": {"subj": er(sv_subj), "obj_pro": er(sv_obj_pro), "obj_end": er(sv_obj_end)},
}
out = ROOT / "palimpsest" / "data" / "role_deposit_test.json"
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"\nwrote {out}")
