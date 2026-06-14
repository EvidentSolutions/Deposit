#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/global_structure_control_v2.py
CONFOUND TEST v2: word-specific transport vs global structure.

v1 swapped within the same word list (dog↔cat — nearly identical features).
v2 uses structured swaps across maximally different categories:

  (A) Matched: real word pairings (baseline)
  (B) Same-category swap: animal↔animal, object↔object
  (C) Cross-category swap: animal↔abstract, person↔object, etc.
  (D) Shuffled: random permutation of all words
  (E) Fixed-word: ALL sentences use the same noun ("dog"), so the
      consuming position sees identical word content every time.
      R fit on the (diverse source reps) → (fixed consumer reps).
      This is pure global structure — zero word-specific signal.

Also: 10 random train/test splits for bootstrap CIs.

Usage: .venv/Scripts/python.exe palimpsest/code/global_structure_control_v2.py
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
P.add_argument("--ns", type=int, default=16)
P.add_argument("--n_splits", type=int, default=10)
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"GLOBAL STRUCTURE CONTROL v2 | {A.model} | L{A.llex}->L{A.ltr} | {DEV}")
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
def cos(a, b): return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))

rng = random.Random(A.seed)

# --- Diverse word categories (maximally different) ---
CATEGORIES = {
    "animal":   ["dog", "cat", "bird", "horse", "lion", "cow", "fish", "deer",
                 "wolf", "frog", "mouse", "bear"],
    "person":   ["king", "queen", "boy", "girl", "man", "woman", "actor", "priest",
                 "teacher", "lord", "mother", "father"],
    "object":   ["car", "box", "chair", "lamp", "cup", "desk", "clock", "knife",
                 "plate", "ship", "bottle", "door"],
    "place":    ["country", "forest", "village", "harbor", "bridge", "island",
                 "mountain", "valley", "lake", "river", "field", "road"],
    "abstract": ["faith", "truth", "peace", "rage", "fear", "hope", "pride",
                 "shame", "grief", "joy", "love", "hate"],
    "food":     ["bread", "rice", "apple", "soup", "cake", "milk", "cheese",
                 "butter", "sugar", "salt", "meat", "fish"],
    "body":     ["heart", "bone", "hand", "head", "foot", "arm", "leg", "eye",
                 "tooth", "lung", "brain", "skin"],
}

# filter to single-token words and deduplicate
all_words = {}
for cat, words in CATEGORIES.items():
    for w in words:
        if has(w) and w not in all_words:  # first category wins on duplicates
            all_words[w] = cat

WL = sorted(all_words.keys())
CAT_OF = {w: all_words[w] for w in WL}
print(f"  {len(WL)} words across {len(set(CAT_OF.values()))} categories")
for cat in sorted(set(CAT_OF.values())):
    members = [w for w in WL if CAT_OF[w] == cat]
    print(f"    {cat}: {len(members)} — {members[:5]}...")

# --- context pools ---
INAN = [w for w in ["table", "window", "stone", "shelf", "bench", "fence",
        "pillar", "barrel", "crate", "bucket"] if has(w)]
VERBS = [w for w in ["hurt", "blamed", "saw", "watched", "cleaned", "defended",
         "followed", "pushed", "lifted", "dropped"] if has(w)]
TAILS = ["near the table", "by the window", "in the room", "on the shelf", "beside the wall"]

# --- residual collection ---
@torch.no_grad()
def resid_batch(seqs, pos, L):
    out = []
    for i in range(0, len(seqs), 64):
        b = seqs[i:i+64]
        hs = model(torch.tensor(b, device=DEV), output_hidden_states=True).hidden_states[L]
        for j in range(len(b)):
            out.append(hs[j, pos[i+j]].cpu())
    return torch.stack(out)

def subj_lex_seq(w):
    return enc("The", True) + [tid(" " + w)] + enc(rng.choice(TAILS))

def subj_tr_seq(w):
    return (enc("The", True) + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN))] + [tid(" " + rng.choice(VERBS))])

# --- collect all reps ---
print("\nCollecting lexical reps (noun @ pos 1, L8)...")
X_lex = []
for w in WL:
    seqs = [subj_lex_seq(w) for _ in range(A.ns)]
    X_lex.append(resid_batch(seqs, [1]*A.ns, A.llex).mean(0))
X_lex = torch.stack(X_lex)

print("Collecting transport reps (verb @ pos 5, L12) — matched...")
Y_matched = []
for w in WL:
    seqs = [subj_tr_seq(w) for _ in range(A.ns)]
    Y_matched.append(resid_batch(seqs, [5]*A.ns, A.ltr).mean(0))
Y_matched = torch.stack(Y_matched)

# --- build swap assignments ---
def same_cat_swap(wl, cat_of):
    """Swap each word with another word from the SAME category."""
    by_cat = {}
    for w in wl:
        by_cat.setdefault(cat_of[w], []).append(w)
    mapping = {}
    for cat, members in by_cat.items():
        shuffled = members.copy()
        # ensure no word maps to itself
        for _ in range(100):
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(members, shuffled)):
                break
        for a, b in zip(members, shuffled):
            mapping[a] = b
    return mapping

def cross_cat_swap(wl, cat_of):
    """Swap each word with a word from a DIFFERENT category."""
    mapping = {}
    cats = sorted(set(cat_of.values()))
    by_cat = {c: [w for w in wl if cat_of[w] == c] for c in cats}
    for w in wl:
        own_cat = cat_of[w]
        other_cats = [c for c in cats if c != own_cat]
        chosen_cat = rng.choice(other_cats)
        mapping[w] = rng.choice(by_cat[chosen_cat])
    return mapping

print("Collecting transport reps for swap conditions...")

# Same-category swap
same_swap = same_cat_swap(WL, CAT_OF)
Y_same_cat = []
for w in WL:
    sw = same_swap[w]
    seqs = [subj_tr_seq(sw) for _ in range(A.ns)]
    Y_same_cat.append(resid_batch(seqs, [5]*A.ns, A.ltr).mean(0))
Y_same_cat = torch.stack(Y_same_cat)

# Cross-category swap
cross_swap = cross_cat_swap(WL, CAT_OF)
Y_cross_cat = []
for w in WL:
    sw = cross_swap[w]
    seqs = [subj_tr_seq(sw) for _ in range(A.ns)]
    Y_cross_cat.append(resid_batch(seqs, [5]*A.ns, A.ltr).mean(0))
Y_cross_cat = torch.stack(Y_cross_cat)

# Fixed-word: all sentences use "dog"
FIXED = "dog"
print(f"Collecting transport reps — FIXED word ('{FIXED}' in every sentence)...")
Y_fixed = []
for _ in WL:
    seqs = [subj_tr_seq(FIXED) for _ in range(A.ns)]
    Y_fixed.append(resid_batch(seqs, [5]*A.ns, A.ltr).mean(0))
Y_fixed = torch.stack(Y_fixed)

# --- fit and evaluate with bootstrap over splits ---
N = len(WL)

def fit_R(X, Y, rows):
    Xc = X[rows]; Yc = Y[rows]
    lam = 1e-1 * Xc.pow(2).sum(0).mean()
    G = Xc.T @ Xc + lam * torch.eye(D)
    return torch.linalg.solve(G, Xc.T @ Yc)

def eval_R(R, X, Y, rows):
    cs = [cos(R @ X[i], Y[i]) for i in rows]
    return sum(cs) / len(cs)

conditions = {
    "matched":    Y_matched,
    "same_cat":   Y_same_cat,
    "cross_cat":  Y_cross_cat,
    "fixed_word": Y_fixed,
}

print(f"\nRunning {A.n_splits} random train/test splits...\n")
all_results = {k: [] for k in conditions}
# also: shuffled (permute the matched targets)
all_results["shuffled"] = []

for s in range(A.n_splits):
    idxs = list(range(N))
    random.Random(A.seed + s).shuffle(idxs)
    n_train = int(0.75 * N)
    train, test = idxs[:n_train], idxs[n_train:]

    for name, Y in conditions.items():
        R = fit_R(X_lex, Y, train)
        c = eval_R(R, X_lex, Y, test)
        all_results[name].append(c)

    # shuffled
    perm = list(range(N))
    random.Random(A.seed + s + 1000).shuffle(perm)
    Y_shuf = Y_matched[perm]
    R_shuf = fit_R(X_lex, Y_shuf, train)
    c_shuf = eval_R(R_shuf, X_lex, Y_shuf, test)
    all_results["shuffled"].append(c_shuf)

print("=== RESULTS (mean ± std over 10 splits) ===\n")
print(f"{'Condition':<18} {'Mean cos':>10} {'Std':>8}  Description")
print("-" * 75)
desc = {
    "matched":    "correct word pairing (real transport)",
    "same_cat":   "dog→cat, king→queen (same semantic category)",
    "cross_cat":  "dog→chair, king→river (different category)",
    "shuffled":   "random permutation of matched targets",
    "fixed_word": f"all sentences contain '{FIXED}' (pure positional)",
}
summary = {}
for name in ["matched", "same_cat", "cross_cat", "shuffled", "fixed_word"]:
    vals = all_results[name]
    m = sum(vals) / len(vals)
    s = (sum((v - m)**2 for v in vals) / len(vals)) ** 0.5
    print(f"  {name:<16} {m:>10.4f} {s:>8.4f}  {desc[name]}")
    summary[name] = {"mean": m, "std": s, "values": vals}

word_specific = summary["matched"]["mean"] - summary["cross_cat"]["mean"]
global_floor = summary["fixed_word"]["mean"]
print(f"\n  Word-specific increment (matched - cross_cat): {word_specific:.4f}")
print(f"  Global positional floor (fixed_word):          {global_floor:.4f}")
print(f"  Category-specific increment (same_cat - cross_cat): "
      f"{summary['same_cat']['mean'] - summary['cross_cat']['mean']:.4f}")

# Show swap examples
print("\n  Same-category swap examples:")
for w in WL[:8]:
    print(f"    {w} ({CAT_OF[w]}) → {same_swap[w]} ({CAT_OF[same_swap[w]]})")
print("  Cross-category swap examples:")
for w in WL[:8]:
    print(f"    {w} ({CAT_OF[w]}) → {cross_swap[w]} ({CAT_OF[cross_swap[w]]})")

out = ROOT / "palimpsest" / "data" / "global_structure_control_v2.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nSaved to {out}")
