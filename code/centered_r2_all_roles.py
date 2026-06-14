#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/centered_r2_all_roles.py
Centered R² for all four grammatical roles + causal subspace filtering.

For each role:
  1. Centered R² (word-specific transport quality)
  2. Cross-category R² (wrong-word control)
  3. ΔR² (word-specific increment)
  4. R² restricted to the CAUSAL SUBSPACE: project reps onto the
     feature directions (number, gender, animacy) before fitting R.
     This measures transport in the subspace we know is causally used.
  5. R² restricted to the CHANNEL subspace (top-30 SVs of R).

Usage: .venv/Scripts/python.exe palimpsest/code/centered_r2_all_roles.py
"""
import sys, json, random
from pathlib import Path
import torch
try: sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except AttributeError: pass

ROOT = Path(__file__).resolve().parent.parent.parent

import argparse
P = argparse.ArgumentParser()
P.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
P.add_argument("--seed", type=int, default=42)
P.add_argument("--llex", type=int, default=8)
P.add_argument("--ltr", type=int, default=12)
P.add_argument("--ns", type=int, default=20)
P.add_argument("--n_splits", type=int, default=10)
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"CENTERED R² ALL ROLES | {A.model} | L{A.llex}->L{A.ltr} | {DEV}")
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32,
                                              low_cpu_mem_usage=True).to(DEV).eval()
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

# --- word inventory with features ---
MASC = [w for w in ["man","king","boy","father","son","brother","uncle","prince","actor",
        "priest","duke","lord","waiter","nephew"] if has(w)]
FEM  = [w for w in ["woman","queen","girl","mother","daughter","sister","aunt","princess",
        "actress","lady","nun","duchess","widow","niece"] if has(w)]
ANIM = [w for w in ["man","woman","dog","cat","boy","girl","king","queen","bird","horse",
        "lion","cow","child","bear","deer","wolf","frog","mouse"] if has(w)]
INAN_W = [w for w in ["car","box","table","chair","door","wall","book","lamp","rock","cup",
        "road","window","stone","desk","clock","knife","plate","ship","bottle"] if has(w)]
NUMP = [("key","keys"),("book","books"),("door","doors"),("car","cars"),("box","boxes"),
        ("dog","dogs"),("cat","cats"),("boy","boys"),("girl","girls"),("table","tables"),
        ("bird","birds"),("tree","trees"),("wall","walls"),("road","roads"),("house","houses"),
        ("chair","chairs"),("lamp","lamps"),("cup","cups"),("hand","hands"),("king","kings")]

WORDS = {}
for s, p in NUMP:
    if tid(" " + s): WORDS.setdefault(s, {})["number"] = "sg"
    if tid(" " + p): WORDS.setdefault(p, {})["number"] = "pl"
for w in MASC: WORDS.setdefault(w, {})["gender"] = "m"
for w in FEM:  WORDS.setdefault(w, {})["gender"] = "f"
for w in ANIM: WORDS.setdefault(w, {})["anim"] = "a"
for w in INAN_W: WORDS.setdefault(w, {})["anim"] = "i"
# Add more words from diverse categories
for w in ["faith","truth","peace","rage","fear","hope","pride","shame","grief","joy",
          "bread","rice","apple","soup","cake","heart","bone","head","foot","arm",
          "country","forest","village","harbor","bridge","island","mountain","valley",
          "lake","river"]:
    if has(w): WORDS.setdefault(w, {})["anim"] = "i"

WL = sorted([w for w in WORDS if tid(" " + w)])
print(f"  {len(WL)} words")

# context pools
INAN_POOL = [w for w in INAN_W if has(w)]
VERBS = [w for w in ["hurt","blamed","saw","watched","cleaned","defended",
         "followed","pushed","lifted","dropped"] if has(w)]
SUBJ_POOL = [w for w in ["boy","girl","man","woman","king","queen","lord","lady"] if has(w)]
OBJ_VERB = [w for w in ["saw","found","watched","hit","caught","held","liked","helped"] if has(w)]
TAILS = ["near the table", "by the window", "in the room", "on the shelf", "beside the wall"]

@torch.no_grad()
def get_resid(input_ids, pos, L):
    hs = model(torch.tensor([input_ids], device=DEV),
               output_hidden_states=True).hidden_states[L]
    return hs[0, pos, :].cpu()

def cos(a, b): return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))

# === Role templates ===
def subj_lex_seq(w):
    return enc("The", True) + [tid(" " + w)] + enc(rng.choice(TAILS))

def subj_tr_seq(w):
    return (enc("The", True) + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN_POOL))] + [tid(" " + rng.choice(VERBS))])

def obj_lex_seq(w):
    return (enc("The", True) + [tid(" " + rng.choice(SUBJ_POOL))]
            + [tid(" " + rng.choice(OBJ_VERB))] + enc("the") + [tid(" " + w)]
            + enc("by") + enc("the") + enc("wall"))

def obj_pro_seq(w):
    return (enc("The", True) + [tid(" " + rng.choice(SUBJ_POOL))]
            + [tid(" " + rng.choice(OBJ_VERB))] + enc("the") + [tid(" " + w)]
            + enc("near") + enc("the") + [tid(" " + rng.choice(INAN_POOL))]
            + enc("and") + enc("it"))

def obj_end_seq(w):
    return obj_lex_seq(w)

ROLES = {
    "subj_verb": {
        "lex_fn": subj_lex_seq, "lex_pos": 1,
        "tr_fn": subj_tr_seq, "tr_pos": 5,
    },
    "obj_pronoun": {
        "lex_fn": obj_lex_seq, "lex_pos": 4,
        "tr_fn": obj_pro_seq, "tr_pos": 9,
    },
    "obj_clause_end": {
        "lex_fn": obj_lex_seq, "lex_pos": 4,
        "tr_fn": obj_end_seq, "tr_pos": 7,
    },
}

# === Collect and measure for each role ===
def ridge_r2(X, Y, train, test, lam_scale=0.1):
    Xtr, Ytr = X[train], Y[train]
    Xte, Yte = X[test], Y[test]
    if Xtr.shape[1] == 0: return 0.0
    lam = lam_scale * Xtr.pow(2).sum(0).mean()
    G = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1])
    R = torch.linalg.solve(G, Xtr.T @ Ytr)
    Ypred = Xte @ R
    ss_res = (Yte - Ypred).pow(2).sum()
    ss_tot = Yte.pow(2).sum()
    if ss_tot < 1e-12: return 0.0
    return float(1 - ss_res / ss_tot)

def compute_feature_dirs(X_lex, words):
    """Compute feature directions from centered lexical reps."""
    dirs = {}
    # Number
    sg_idx = [i for i, w in enumerate(words) if WORDS.get(w, {}).get("number") == "sg"]
    pl_idx = [i for i, w in enumerate(words) if WORDS.get(w, {}).get("number") == "pl"]
    if len(sg_idx) > 2 and len(pl_idx) > 2:
        d = X_lex[sg_idx].mean(0) - X_lex[pl_idx].mean(0)
        dirs["number"] = d / (d.norm() + 1e-9)
    # Gender
    m_idx = [i for i, w in enumerate(words) if WORDS.get(w, {}).get("gender") == "m"]
    f_idx = [i for i, w in enumerate(words) if WORDS.get(w, {}).get("gender") == "f"]
    if len(m_idx) > 2 and len(f_idx) > 2:
        d = X_lex[m_idx].mean(0) - X_lex[f_idx].mean(0)
        dirs["gender"] = d / (d.norm() + 1e-9)
    # Animacy
    a_idx = [i for i, w in enumerate(words) if WORDS.get(w, {}).get("anim") == "a"]
    i_idx = [i for i, w in enumerate(words) if WORDS.get(w, {}).get("anim") == "i"]
    if len(a_idx) > 2 and len(i_idx) > 2:
        d = X_lex[a_idx].mean(0) - X_lex[i_idx].mean(0)
        dirs["animacy"] = d / (d.norm() + 1e-9)
    return dirs

def project_onto_subspace(X, basis_vecs):
    """Project X onto the subspace spanned by basis_vecs (list of unit vectors)."""
    if len(basis_vecs) == 0:
        return torch.zeros(X.shape[0], 0)
    B = torch.stack(basis_vecs)  # [k, D]
    # Orthogonalize via QR
    Q, _ = torch.linalg.qr(B.T)  # [D, k]
    return X @ Q  # [N, k]

all_results = {}

for role_name, role_cfg in ROLES.items():
    print(f"\n{'='*60}")
    print(f"  ROLE: {role_name}")
    print(f"{'='*60}")

    # Collect lexical and transport reps
    X_lex, Y_tr = [], []
    for w in WL:
        lex_reps, tr_reps = [], []
        for _ in range(A.ns):
            seq_lex = role_cfg["lex_fn"](w)
            seq_tr = role_cfg["tr_fn"](w)
            lex_reps.append(get_resid(seq_lex, role_cfg["lex_pos"], A.llex))
            tr_reps.append(get_resid(seq_tr, role_cfg["tr_pos"], A.ltr))
        X_lex.append(torch.stack(lex_reps).mean(0))
        Y_tr.append(torch.stack(tr_reps).mean(0))

    X_lex = torch.stack(X_lex)
    Y_tr = torch.stack(Y_tr)
    N = len(WL)

    # Center
    Xc = X_lex - X_lex.mean(0, keepdim=True)
    Yc = Y_tr - Y_tr.mean(0, keepdim=True)

    # Cross-category control
    cats_of = {w: WORDS.get(w, {}).get("anim", "?") for w in WL}
    Y_wrong = []
    for w in WL:
        own = cats_of[w]
        others = [ww for ww in WL if cats_of[ww] != own and cats_of[ww] != "?"]
        if not others: others = [ww for ww in WL if ww != w]
        wrong_w = rng.choice(others)
        tr_reps = []
        for _ in range(A.ns):
            seq = role_cfg["tr_fn"](wrong_w)
            tr_reps.append(get_resid(seq, role_cfg["tr_pos"], A.ltr))
        Y_wrong.append(torch.stack(tr_reps).mean(0))
    Y_wrong = torch.stack(Y_wrong)
    Ywc = Y_wrong - Y_wrong.mean(0, keepdim=True)

    # Feature directions (from lexical reps)
    feat_dirs_lex = compute_feature_dirs(Xc, WL)
    # Also from transport reps
    feat_dirs_tr = compute_feature_dirs(Yc, WL)

    print(f"  Feature directions found: {list(feat_dirs_lex.keys())}")

    # Project onto causal subspace (feature directions in lex and tr space)
    lex_basis = list(feat_dirs_lex.values())
    tr_basis = list(feat_dirs_tr.values())
    Xc_feat = project_onto_subspace(Xc, lex_basis)  # [N, k]
    Yc_feat = project_onto_subspace(Yc, tr_basis)    # [N, k]

    # Fit R on full space (top-30 channel via SVD of the full R)
    # First fit full R to get SVD
    idxs_all = list(range(N))
    lam = 0.1 * Xc.pow(2).sum(0).mean()
    G = Xc.T @ Xc + lam * torch.eye(D)
    R_full = torch.linalg.solve(G, Xc.T @ Yc)
    U, S, Vt = torch.linalg.svd(R_full, full_matrices=False)
    # Channel subspace: top-30 left singular vectors
    k_chan = 30
    U30 = U[:, :k_chan]  # [D, 30]
    Xc_chan = Xc @ U30   # [N, 30]
    # We need to project Y onto the corresponding output subspace
    V30 = Vt[:k_chan, :].T  # [D, 30]
    Yc_chan = Yc @ V30  # [N, 30]

    # Run splits
    matched_r2, cross_r2, feat_r2, chan_r2 = [], [], [], []
    for s in range(A.n_splits):
        idx = list(range(N))
        random.Random(A.seed + s).shuffle(idx)
        n_train = int(0.75 * N)
        train, test = idx[:n_train], idx[n_train:]

        matched_r2.append(ridge_r2(Xc, Yc, train, test))
        cross_r2.append(ridge_r2(Xc, Ywc, train, test))
        if Xc_feat.shape[1] > 0 and Yc_feat.shape[1] > 0:
            feat_r2.append(ridge_r2(Xc_feat, Yc_feat, train, test))
        chan_r2.append(ridge_r2(Xc_chan, Yc_chan, train, test))

    def ms(vals):
        m = sum(vals)/len(vals)
        s = (sum((v-m)**2 for v in vals)/len(vals))**0.5
        return m, s

    mm, sm = ms(matched_r2)
    mc, sc = ms(cross_r2)
    mf, sf = ms(feat_r2) if feat_r2 else (0, 0)
    mch, sch = ms(chan_r2)

    print(f"  Full space:      matched R² = {mm:.4f}±{sm:.4f}  cross-cat = {mc:.4f}±{sc:.4f}  ΔR² = {mm-mc:.4f}")
    print(f"  Feature subspace ({len(lex_basis)}d): R² = {mf:.4f}±{sf:.4f}")
    print(f"  Channel subspace ({k_chan}d):  R² = {mch:.4f}±{sch:.4f}")

    all_results[role_name] = {
        "n_words": N,
        "full_matched_r2": mm, "full_matched_std": sm,
        "full_cross_r2": mc, "full_cross_std": sc,
        "full_delta_r2": mm - mc,
        "feature_subspace_dim": len(lex_basis),
        "feature_r2": mf, "feature_std": sf,
        "channel_dim": k_chan,
        "channel_r2": mch, "channel_std": sch,
    }

print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")
print(f"{'Role':<20} {'Full R²':>10} {'Cross R²':>10} {'ΔR²':>8} {'Feat R²':>10} {'Chan R²':>10}")
print("-" * 70)
for rn, r in all_results.items():
    print(f"  {rn:<18} {r['full_matched_r2']:>10.4f} {r['full_cross_r2']:>10.4f} "
          f"{r['full_delta_r2']:>8.4f} {r['feature_r2']:>10.4f} {r['channel_r2']:>10.4f}")

out = ROOT / "palimpsest" / "data" / "centered_r2_all_roles.json"
out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
print(f"\nSaved to {out}")
