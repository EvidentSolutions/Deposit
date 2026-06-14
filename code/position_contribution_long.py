#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/position_contribution_long.py
POSITION CONTRIBUTION with longer contexts.

Short template (6 tokens): "The {w} near the {N2} {V}"
Long template (~25 tokens): "{filler}. The {w} near the {N2} {V}"

The filler is 3-4 sentences of neutral prose (fixed across words).
Does the noun's unique contribution shrink further when there's
more context for the consuming position to draw on?

Also test a naturalistic template (~40 tokens) with varied fillers.

Usage: .venv/Scripts/python.exe palimpsest/code/position_contribution_long.py
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
print(f"POSITION CONTRIBUTION (long) | {A.model} | L{A.llex}->L{A.ltr} | {DEV}")
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

# --- words ---
CATEGORIES = {
    "animal":   ["dog", "cat", "bird", "horse", "lion", "cow", "deer", "wolf", "frog", "mouse", "bear"],
    "person":   ["king", "queen", "boy", "girl", "man", "woman", "actor", "priest", "teacher", "lord"],
    "object":   ["car", "box", "chair", "lamp", "cup", "desk", "clock", "knife", "plate", "ship", "bottle", "door"],
    "place":    ["country", "forest", "village", "harbor", "bridge", "island", "mountain", "valley", "lake", "river"],
    "abstract": ["faith", "truth", "peace", "rage", "fear", "hope", "pride", "shame", "grief", "joy"],
    "food":     ["bread", "rice", "apple", "soup", "cake", "milk", "cheese", "butter", "sugar", "salt"],
    "body":     ["heart", "bone", "hand", "head", "foot", "arm", "leg", "eye", "tooth", "brain", "skin"],
}
all_words = {}
for cat, words in CATEGORIES.items():
    for w in words:
        if has(w) and w not in all_words:
            all_words[w] = cat
WL = sorted(all_words.keys())
print(f"  {len(WL)} words")

INAN = [w for w in ["table", "window", "stone", "shelf", "bench", "fence",
        "pillar", "barrel", "crate", "bucket"] if has(w)]
VERBS = [w for w in ["hurt", "blamed", "saw", "watched", "cleaned", "defended",
         "followed", "pushed", "lifted", "dropped"] if has(w)]

# --- filler prefixes ---
FILLERS = [
    "The weather was clear and the road stretched ahead for miles.",
    "It was a quiet morning in the village and nothing seemed unusual.",
    "The old building stood at the corner of the street near the park.",
    "Several people gathered in the square to watch the parade go by.",
    "A light breeze carried the smell of fresh bread from the bakery.",
    "The river ran slowly through the valley below the stone bridge.",
    "Clouds gathered on the horizon but the sun still shone brightly.",
    "The train arrived on time and passengers stepped onto the platform.",
]

@torch.no_grad()
def get_resid(input_ids, pos, L):
    """Get residual at (pos, L) for a single sequence."""
    hs = model(torch.tensor([input_ids], device=DEV),
               output_hidden_states=True).hidden_states[L]
    return hs[0, pos, :].cpu()

@torch.no_grad()
def get_resid_batch(seqs, positions, L):
    """Get residuals, handling variable-length by processing one at a time."""
    out = []
    for seq, pos in zip(seqs, positions):
        out.append(get_resid(seq, pos, L))
    return torch.stack(out)

def cos(a, b): return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))

# =============================================
# CONDITION 1: SHORT (6 tokens) — baseline
# =============================================
def make_short(w):
    """'The {w} near the {N2} {V}' — 6 tokens"""
    return (enc("The", True) + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN))] + [tid(" " + rng.choice(VERBS))])

SHORT_LEN = 6
SHORT_NOUN_POS = 1
SHORT_VERB_POS = 5

# =============================================
# CONDITION 2: LONG PREFIX (~25-30 tokens)
# =============================================
def make_long(w):
    """'{filler}. The {w} near the {N2} {V}'"""
    filler = rng.choice(FILLERS)
    prefix_ids = tok(filler, add_special_tokens=False)["input_ids"]
    core = (enc("The") + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN))] + [tid(" " + rng.choice(VERBS))])
    full = prefix_ids + core
    noun_pos = len(prefix_ids) + 1  # +1 because "The" is one token after prefix
    verb_pos = len(prefix_ids) + 5
    return full, noun_pos, verb_pos

# =============================================
# CONDITION 3: VERY LONG (2 filler sentences)
# =============================================
def make_very_long(w):
    """'{filler1}. {filler2}. The {w} near the {N2} {V}'"""
    f1 = rng.choice(FILLERS)
    f2 = rng.choice([f for f in FILLERS if f != f1])
    prefix_ids = tok(f1 + " " + f2, add_special_tokens=False)["input_ids"]
    core = (enc("The") + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN))] + [tid(" " + rng.choice(VERBS))])
    full = prefix_ids + core
    noun_pos = len(prefix_ids) + 1
    verb_pos = len(prefix_ids) + 5
    return full, noun_pos, verb_pos

# --- collect for each condition ---
def collect_condition(name, make_fn, is_short=False):
    print(f"\n--- {name} ---")
    X_noun = []  # noun residual at L_lex
    Y_verb = []  # verb residual at L_tr

    for wi, w in enumerate(WL):
        noun_reps = []
        verb_reps = []
        for _ in range(A.ns):
            if is_short:
                seq = make_fn(w)
                npos, vpos = SHORT_NOUN_POS, SHORT_VERB_POS
            else:
                seq, npos, vpos = make_fn(w)
            noun_reps.append(get_resid(seq, npos, A.llex))
            verb_reps.append(get_resid(seq, vpos, A.ltr))
        X_noun.append(torch.stack(noun_reps).mean(0))
        Y_verb.append(torch.stack(verb_reps).mean(0))

    X_noun = torch.stack(X_noun)
    Y_verb = torch.stack(Y_verb)

    # center
    Xc = X_noun - X_noun.mean(0, keepdim=True)
    Yc = Y_verb - Y_verb.mean(0, keepdim=True)

    # Also: collect verb reps from WRONG-WORD sentences (cross-category)
    cats = sorted(set(all_words.values()))
    by_cat = {c: [w for w in WL if all_words[w] == c] for c in cats}
    Y_wrong = []
    for w in WL:
        own_cat = all_words[w]
        other_cats = [c for c in cats if c != own_cat]
        wrong_w = rng.choice(by_cat[rng.choice(other_cats)])
        wrong_reps = []
        for _ in range(A.ns):
            if is_short:
                seq = make_fn(wrong_w)
                vpos = SHORT_VERB_POS
            else:
                seq, _, vpos = make_fn(wrong_w)
            wrong_reps.append(get_resid(seq, vpos, A.ltr))
        Y_wrong.append(torch.stack(wrong_reps).mean(0))
    Y_wrong = torch.stack(Y_wrong)
    Y_wrong_c = Y_wrong - Y_wrong.mean(0, keepdim=True)

    # Fixed-word control
    Y_fixed = []
    for _ in WL:
        fixed_reps = []
        for _ in range(A.ns):
            if is_short:
                seq = make_fn("dog")
                vpos = SHORT_VERB_POS
            else:
                seq, _, vpos = make_fn("dog")
            fixed_reps.append(get_resid(seq, vpos, A.ltr))
        Y_fixed.append(torch.stack(fixed_reps).mean(0))
    Y_fixed = torch.stack(Y_fixed)
    Y_fixed_c = Y_fixed - Y_fixed.mean(0, keepdim=True)

    N = len(WL)

    def ridge_r2(X, Y, train, test):
        Xtr, Ytr = X[train], Y[train]
        Xte, Yte = X[test], Y[test]
        lam = 0.1 * Xtr.pow(2).sum(0).mean()
        G = Xtr.T @ Xtr + lam * torch.eye(D)
        R = torch.linalg.solve(G, Xtr.T @ Ytr)
        Ypred = Xte @ R
        ss_res = (Yte - Ypred).pow(2).sum()
        ss_tot = Yte.pow(2).sum()
        return float(1 - ss_res / ss_tot)

    def ridge_cos(X, Y, train, test):
        Xtr, Ytr = X[train], Y[train]
        Xte, Yte = X[test], Y[test]
        lam = 0.1 * Xtr.pow(2).sum(0).mean()
        G = Xtr.T @ Xtr + lam * torch.eye(D)
        R = torch.linalg.solve(G, Xtr.T @ Ytr)
        Ypred = Xte @ R
        cs = [float(torch.dot(Ypred[i], Yte[i]) / (Ypred[i].norm() * Yte[i].norm() + 1e-9))
              for i in range(len(test))]
        return sum(cs) / len(cs)

    matched_r2s, matched_cos = [], []
    wrong_r2s, fixed_r2s = [], []

    for s in range(A.n_splits):
        idxs = list(range(N))
        random.Random(A.seed + s).shuffle(idxs)
        n_train = int(0.75 * N)
        train, test = idxs[:n_train], idxs[n_train:]

        matched_r2s.append(ridge_r2(Xc, Yc, train, test))
        matched_cos.append(ridge_cos(X_noun, Y_verb, train, test))
        wrong_r2s.append(ridge_r2(Xc, Y_wrong_c, train, test))
        fixed_r2s.append(ridge_r2(Xc, Y_fixed_c, train, test))

    mr2 = sum(matched_r2s) / len(matched_r2s)
    mc = sum(matched_cos) / len(matched_cos)
    wr2 = sum(wrong_r2s) / len(wrong_r2s)
    fr2 = sum(fixed_r2s) / len(fixed_r2s)

    print(f"  Matched R² (centered):     {mr2:.4f}")
    print(f"  Matched cos (uncentered):   {mc:.4f}")
    print(f"  Cross-category R² (centered): {wr2:.4f}")
    print(f"  Fixed-word R² (centered):  {fr2:.4f}")
    print(f"  Word-specific ΔR²:         {mr2 - wr2:.4f}")

    return {
        "matched_r2": mr2, "matched_cos": mc,
        "cross_cat_r2": wr2, "fixed_r2": fr2,
        "word_specific_delta_r2": mr2 - wr2,
    }

results = {}
results["short_6tok"] = collect_condition("SHORT (6 tokens)", make_short, is_short=True)
results["long_25tok"] = collect_condition("LONG (~25 tokens, 1 filler sentence)", make_long)
results["very_long_45tok"] = collect_condition("VERY LONG (~45 tokens, 2 filler sentences)", make_very_long)

print("\n\n=== SUMMARY ===\n")
print(f"{'Condition':<25} {'Matched R²':>12} {'Cross-cat R²':>14} {'ΔR²':>8} {'Matched cos':>13}")
print("-" * 75)
for name, r in results.items():
    print(f"  {name:<23} {r['matched_r2']:>12.4f} {r['cross_cat_r2']:>14.4f} "
          f"{r['word_specific_delta_r2']:>8.4f} {r['matched_cos']:>13.4f}")

out = ROOT / "palimpsest" / "data" / "position_contribution_long.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved to {out}")
