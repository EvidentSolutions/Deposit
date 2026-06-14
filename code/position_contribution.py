#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/position_contribution.py
POSITION CONTRIBUTION ANALYSIS: which source positions contribute
word-specific information to the consuming position?

For template "The {w} near the {N2} {V}" (verb @ pos 5):
  pos 0 = "The"    (fixed)
  pos 1 = {w}      (the noun — varies)
  pos 2 = "near"   (fixed)
  pos 3 = "the"    (fixed)
  pos 4 = {N2}     (distractor noun — varies randomly)
  pos 5 = {V}      (verb — varies randomly, this is the consuming position)

Method: for each source position p, collect L8 residuals across all
words. Center across words (remove global/positional mean). Fit ridge
regression from centered position-p reps to centered consuming-position
L12 reps. Held-out R² measures how much of the word-varying component
at the consumer is predicted by position p.

If the noun (pos 1) has the highest R², the word-specific signal at the
consuming position comes primarily from the source noun. If adjacent
positions dominate, it's local context leakage.

Also: partial R² — fit from pos 1 AFTER regressing out all other positions.

Usage: .venv/Scripts/python.exe palimpsest/code/position_contribution.py
"""
import sys, json, random
from pathlib import Path
import numpy as np
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
P.add_argument("--ns", type=int, default=20)  # samples per word
P.add_argument("--n_splits", type=int, default=10)
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"POSITION CONTRIBUTION | {A.model} | L{A.llex}->L{A.ltr} | {DEV}")
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

# --- collect residuals at ALL positions ---
@torch.no_grad()
def resid_all_pos(seqs, L, seq_len):
    """Return [n_seqs, seq_len, D] tensor of residuals at layer L."""
    out = []
    for i in range(0, len(seqs), 32):
        b = seqs[i:i+32]
        hs = model(torch.tensor(b, device=DEV), output_hidden_states=True).hidden_states[L]
        out.append(hs[:, :seq_len, :].cpu())
    return torch.cat(out, dim=0)

def make_seq(w):
    """'The {w} near the {N2} {V}' — 6 tokens, verb @ pos 5"""
    return (enc("The", True) + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN))] + [tid(" " + rng.choice(VERBS))])

SEQ_LEN = 6  # The w near the N2 V
CONSUME_POS = 5  # verb position
NOUN_POS = 1

print(f"\nCollecting residuals at all {SEQ_LEN} positions for {len(WL)} words...")
print(f"  Source (all positions): L{A.llex}")
print(f"  Consumer (pos {CONSUME_POS}): L{A.ltr}")

# Per word: collect ns samples, average, get [seq_len, D] at L_lex
# and [D] at L_tr for the consuming position
X_all = []  # [n_words, seq_len, D] at L_lex
Y_consume = []  # [n_words, D] at L_tr

for wi, w in enumerate(WL):
    seqs = [make_seq(w) for _ in range(A.ns)]
    # L_lex residuals at all positions
    h_lex = resid_all_pos(seqs, A.llex, SEQ_LEN)  # [ns, seq_len, D]
    X_all.append(h_lex.mean(0))  # [seq_len, D]
    # L_tr residual at consuming position
    h_tr = resid_all_pos(seqs, A.ltr, SEQ_LEN)  # [ns, seq_len, D]
    Y_consume.append(h_tr[:, CONSUME_POS, :].mean(0))  # [D]
    if (wi + 1) % 20 == 0:
        print(f"    {wi+1}/{len(WL)} words done")

X_all = torch.stack(X_all)  # [n_words, seq_len, D]
Y_consume = torch.stack(Y_consume)  # [n_words, D]
N = len(WL)

print(f"\n  X_all shape: {X_all.shape}")
print(f"  Y_consume shape: {Y_consume.shape}")

# --- center across words (remove global mean per position) ---
X_centered = X_all - X_all.mean(0, keepdim=True)  # [N, seq_len, D]
Y_centered = Y_consume - Y_consume.mean(0, keepdim=True)  # [N, D]

# --- per-position R² ---
def ridge_r2(X, Y, train, test, lam_scale=0.1):
    """Fit ridge X->Y on train, return R² on test."""
    Xtr, Ytr = X[train], Y[train]
    Xte, Yte = X[test], Y[test]
    lam = lam_scale * Xtr.pow(2).sum(0).mean()
    G = Xtr.T @ Xtr + lam * torch.eye(X.shape[1])
    R = torch.linalg.solve(G, Xtr.T @ Ytr)
    Ypred = Xte @ R
    ss_res = (Yte - Ypred).pow(2).sum()
    ss_tot = Yte.pow(2).sum()  # already centered
    return float(1 - ss_res / ss_tot)

def ridge_cos(X, Y, train, test, lam_scale=0.1):
    """Fit ridge X->Y on train, return mean cos on test."""
    Xtr, Ytr = X[train], Y[train]
    Xte, Yte = X[test], Y[test]
    lam = lam_scale * Xtr.pow(2).sum(0).mean()
    G = Xtr.T @ Xtr + lam * torch.eye(X.shape[1])
    R = torch.linalg.solve(G, Xtr.T @ Ytr)
    Ypred = Xte @ R
    cos_vals = []
    for i in range(len(test)):
        c = float(torch.dot(Ypred[i], Yte[i]) / (Ypred[i].norm() * Yte[i].norm() + 1e-9))
        cos_vals.append(c)
    return sum(cos_vals) / len(cos_vals)

print(f"\nRunning position contribution analysis ({A.n_splits} splits)...\n")

pos_labels = ["The", "{noun}", "near", "the", "{N2}", "{verb}"]
results_r2 = {p: [] for p in range(SEQ_LEN)}
results_cos = {p: [] for p in range(SEQ_LEN)}

# Also: multi-position regression (all positions except the consuming one)
results_all_pos = []
results_noun_only = []
results_no_noun = []  # all positions except noun and consumer

for s in range(A.n_splits):
    idxs = list(range(N))
    random.Random(A.seed + s).shuffle(idxs)
    n_train = int(0.75 * N)
    train, test = idxs[:n_train], idxs[n_train:]

    for p in range(SEQ_LEN):
        Xp = X_centered[:, p, :]  # [N, D]
        r2 = ridge_r2(Xp, Y_centered, train, test)
        c = ridge_cos(Xp, Y_centered, train, test)
        results_r2[p].append(r2)
        results_cos[p].append(c)

    # All source positions concatenated (except consumer itself)
    src_positions = [p for p in range(SEQ_LEN) if p != CONSUME_POS]
    X_concat = X_centered[:, src_positions, :].reshape(N, -1)  # [N, 5*D]
    r2_all = ridge_r2(X_concat, Y_centered, train, test)
    results_all_pos.append(r2_all)

    # Noun only (already done above, but collect separately)
    results_noun_only.append(results_r2[NOUN_POS][-1])

    # All except noun
    no_noun_pos = [p for p in range(SEQ_LEN) if p != CONSUME_POS and p != NOUN_POS]
    X_no_noun = X_centered[:, no_noun_pos, :].reshape(N, -1)
    r2_no_noun = ridge_r2(X_no_noun, Y_centered, train, test)
    results_no_noun.append(r2_no_noun)

print("=== PER-POSITION R² (word-specific, centered) ===\n")
print(f"{'Pos':>4}  {'Label':>8}  {'Mean R²':>10}  {'Std':>8}  {'Mean cos':>10}  {'Std':>8}")
print("-" * 62)
summary = {}
for p in range(SEQ_LEN):
    r2_vals = results_r2[p]
    cos_vals = results_cos[p]
    mr2 = sum(r2_vals) / len(r2_vals)
    sr2 = (sum((v - mr2)**2 for v in r2_vals) / len(r2_vals)) ** 0.5
    mc = sum(cos_vals) / len(cos_vals)
    sc = (sum((v - mc)**2 for v in cos_vals) / len(cos_vals)) ** 0.5
    marker = " ← NOUN" if p == NOUN_POS else (" ← CONSUMER" if p == CONSUME_POS else "")
    print(f"  {p:>2}  {pos_labels[p]:>8}  {mr2:>10.4f}  {sr2:>8.4f}  {mc:>10.4f}  {sc:>8.4f}{marker}")
    summary[f"pos_{p}_{pos_labels[p]}"] = {"r2_mean": mr2, "r2_std": sr2, "cos_mean": mc, "cos_std": sc}

mr2_all = sum(results_all_pos) / len(results_all_pos)
mr2_noun = sum(results_noun_only) / len(results_noun_only)
mr2_no_noun = sum(results_no_noun) / len(results_no_noun)

print(f"\n  All source positions (concat): R² = {mr2_all:.4f}")
print(f"  Noun only:                     R² = {mr2_noun:.4f}")
print(f"  All except noun:               R² = {mr2_no_noun:.4f}")
print(f"  Noun's unique contribution:    ΔR² ≈ {mr2_all - mr2_no_noun:.4f}")
print(f"    (= all-positions R² minus all-except-noun R²)")

summary["all_source_r2"] = mr2_all
summary["noun_only_r2"] = mr2_noun
summary["no_noun_r2"] = mr2_no_noun
summary["noun_unique_delta_r2"] = mr2_all - mr2_no_noun

out = ROOT / "palimpsest" / "data" / "position_contribution.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nSaved to {out}")
