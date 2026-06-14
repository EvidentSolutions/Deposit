#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/layer_pair_sweep.py
Grid search over source/consumer layer pairs for the deposit map.

For subject→verb, sweep L_lex ∈ {2,4,6,8,10,12,14,16} and
L_tr ∈ {4,6,8,10,12,14,16,18,20,22} (L_tr > L_lex).
Report centered R² at each pair.

Usage: .venv/Scripts/python.exe palimpsest/code/layer_pair_sweep.py
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
P.add_argument("--ns", type=int, default=16)
P.add_argument("--n_splits", type=int, default=5)
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"LAYER PAIR SWEEP | {A.model} | {DEV}")
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32,
                                              low_cpu_mem_usage=True).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
for p in model.parameters(): p.requires_grad_(False)
D = model.config.hidden_size
NL = len(model.gpt_neox.layers)

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
TAILS = ["near the table", "by the window", "in the room", "on the shelf", "beside the wall"]

NOUN_POS = 1  # in "The {w} ..."
VERB_POS = 5  # in "The {w} near the {N2} {V}"

def subj_lex_seq(w):
    return enc("The", True) + [tid(" " + w)] + enc(rng.choice(TAILS))

def subj_tr_seq(w):
    return (enc("The", True) + [tid(" " + w)] + enc("near") + enc("the")
            + [tid(" " + rng.choice(INAN))] + [tid(" " + rng.choice(VERBS))])

# --- collect ALL layers at once ---
print("Collecting residuals at all layers (this takes a minute)...")

@torch.no_grad()
def collect_all_layers(make_seq, pos, ns):
    """For each word, collect mean residual at `pos` at EVERY layer."""
    # Returns dict: layer_idx -> [N, D] tensor
    per_layer = {L: [] for L in range(NL + 1)}  # +1 for embedding layer
    for wi, w in enumerate(WL):
        seqs = [make_seq(w) for _ in range(ns)]
        # Run one batch per word, get all hidden states
        all_hs = []
        for i in range(0, len(seqs), 32):
            batch = seqs[i:i+32]
            out = model(torch.tensor(batch, device=DEV), output_hidden_states=True)
            # out.hidden_states is tuple of (NL+1) tensors, each [batch, seq_len, D]
            for L in range(NL + 1):
                if len(all_hs) <= L:
                    all_hs.append([])
                all_hs[L].append(out.hidden_states[L][:, pos, :].cpu())
        for L in range(NL + 1):
            h = torch.cat(all_hs[L], dim=0).mean(0)  # [D]
            per_layer[L].append(h)
        if (wi + 1) % 20 == 0:
            print(f"    {wi+1}/{len(WL)}")
    for L in per_layer:
        per_layer[L] = torch.stack(per_layer[L])  # [N, D]
    return per_layer

print("  Collecting noun position (pos 1) residuals...")
noun_layers = collect_all_layers(subj_lex_seq, NOUN_POS, A.ns)

print("  Collecting verb position (pos 5) residuals...")
verb_layers = collect_all_layers(subj_tr_seq, VERB_POS, A.ns)

# --- sweep ---
L_LEX_RANGE = [2, 4, 6, 8, 10, 12, 14, 16]
L_TR_RANGE = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

N = len(WL)

def ridge_r2_centered(X_raw, Y_raw, train, test):
    X = X_raw - X_raw.mean(0, keepdim=True)
    Y = Y_raw - Y_raw.mean(0, keepdim=True)
    Xtr, Ytr = X[train], Y[train]
    Xte, Yte = X[test], Y[test]
    lam = 0.1 * Xtr.pow(2).sum(0).mean()
    G = Xtr.T @ Xtr + lam * torch.eye(D)
    R = torch.linalg.solve(G, Xtr.T @ Ytr)
    Ypred = Xte @ R
    ss_res = (Yte - Ypred).pow(2).sum()
    ss_tot = Yte.pow(2).sum()
    if ss_tot < 1e-12: return 0.0
    return float(1 - ss_res / ss_tot)

print(f"\nSweeping {len(L_LEX_RANGE)} x {len(L_TR_RANGE)} layer pairs ({A.n_splits} splits each)...\n")

results = {}
for l_lex in L_LEX_RANGE:
    for l_tr in L_TR_RANGE:
        if l_tr <= l_lex:
            continue
        X = noun_layers[l_lex]
        Y = verb_layers[l_tr]
        r2s = []
        for s in range(A.n_splits):
            idxs = list(range(N))
            random.Random(A.seed + s).shuffle(idxs)
            n_train = int(0.75 * N)
            train, test = idxs[:n_train], idxs[n_train:]
            r2s.append(ridge_r2_centered(X, Y, train, test))
        m = sum(r2s) / len(r2s)
        results[f"{l_lex}->{l_tr}"] = m
        print(f"  L{l_lex:>2} -> L{l_tr:>2}: centered R² = {m:.4f}")

# Print as a grid
print(f"\n{'':>6}", end="")
for l_tr in L_TR_RANGE:
    print(f"  L{l_tr:>2}", end="")
print()
for l_lex in L_LEX_RANGE:
    print(f"L{l_lex:>2}  ", end="")
    for l_tr in L_TR_RANGE:
        key = f"{l_lex}->{l_tr}"
        if key in results:
            v = results[key]
            print(f" {v:5.2f}", end="")
        else:
            print(f"   --", end="")
    print()

out = ROOT / "palimpsest" / "data" / "layer_pair_sweep.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved to {out}")
