#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
palimpsest/code/forced_alternative_test.py
FORCE THE SECOND-CHOICE TOKEN AND TRACE THE CASCADE

At the prediction position, the model has a bundle for the top-1 prediction.
Does it also have one for the second-most-likely token? Force it and see:

  1. Get top-1 and top-2 predictions at the prompt end
  2. Force each as the first generated token, continue greedily for 7 more
  3. Compare: at the prompt end, what bundle did each have?
  4. How quickly do the two cascades diverge?
  5. Does the number constraint hold for the forced alternative?

Template: "The {noun} near the wall" (sg/pl nouns)

Usage: .venv/Scripts/python.exe palimpsest/code/forced_alternative_test.py
Output: palimpsest/data/forced_alternative_test.json
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
P.add_argument("--n_continue", type=int, default=7)
P.add_argument("--topk", type=int, default=5)  # check top-k alternatives
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"FORCED ALTERNATIVE TEST | {A.model} | top-{A.topk} | {DEV}")
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
for p in model.parameters(): p.requires_grad_(False)
NL = len(model.gpt_neox.layers)
D = model.config.hidden_size
W_U = model.embed_out.weight.detach()

def tid(s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None

def enc(s, first=False):
    return tok(s if first else " " + s, add_special_tokens=False)["input_ids"]

def has(w): return tid(" " + w) is not None
def decode(t): return tok.decode([t])

rng = random.Random(A.seed)

# ---------- nouns ----------
NUMP = [(s, p) for s, p in [
    ("dog","dogs"),("cat","cats"),("boy","boys"),("girl","girls"),("bird","birds"),
    ("tree","trees"),("car","cars"),("house","houses"),("wall","walls"),("road","roads"),
    ("king","kings"),("door","doors"),("chair","chairs"),("lamp","lamps"),("cup","cups"),
    ("book","books"),("hand","hands"),("key","keys"),("box","boxes"),("ship","ships"),
] if has(s) and has(p)]
SG_NOUNS = [s for s, _ in NUMP]
PL_NOUNS = [p for _, p in NUMP]

PROMPT_LEN = 5  # "The {noun} near the wall"

def make_prompt(noun):
    return enc("The", True) + [tid(" " + noun)] + enc("near") + enc("the") + enc("wall")

# ---------- get top-k at prompt end, then force each and continue ----------
print("\n--- Phase 1: get top-k predictions at prompt end ---")

all_nouns = SG_NOUNS + PL_NOUNS
is_plural = [False] * len(SG_NOUNS) + [True] * len(PL_NOUNS)
prompts = [make_prompt(n) for n in all_nouns]
prompt_tensor = torch.tensor(prompts, device=DEV)  # (N, 5)

with torch.no_grad():
    out = model(prompt_tensor, output_hidden_states=True)
    logits_end = out.logits[:, -1, :]  # (N, vocab) at pos 4

# top-k tokens per prompt
topk_probs, topk_ids = torch.topk(torch.softmax(logits_end, dim=-1), A.topk, dim=-1)
# topk_ids: (N, topk), topk_probs: (N, topk)

# Show a few
print("\n  Top-5 predictions at prompt end:")
for i in [0, len(SG_NOUNS)-1, len(SG_NOUNS), len(all_nouns)-1]:
    noun = all_nouns[i]
    tops = [(decode(topk_ids[i, k].item()), f"{topk_probs[i, k].item():.3f}") for k in range(A.topk)]
    print(f"    '{noun}': {tops}")

# ---------- Phase 2: force each top-k token, continue greedily ----------
print(f"\n--- Phase 2: force top-1 through top-{A.topk}, continue {A.n_continue} tokens ---")

@torch.no_grad()
def generate_from_forced(prompt_ids, forced_token_id, n_continue):
    """Force one token after prompt, then continue greedily."""
    seq = list(prompt_ids) + [forced_token_id]
    for _ in range(n_continue):
        inp = torch.tensor([seq], device=DEV)
        logits = model(inp).logits[0, -1, :]
        seq.append(int(logits.argmax()))
    return seq

# For each prompt, for each top-k alternative, generate a continuation
results_by_rank = {k: [] for k in range(A.topk)}

for i in range(len(all_nouns)):
    for k in range(A.topk):
        forced_id = topk_ids[i, k].item()
        seq = generate_from_forced(prompts[i], forced_id, A.n_continue)
        results_by_rank[k].append(seq)

# Show examples
print("\n  Example continuations (first sg noun, first pl noun):")
for i in [0, len(SG_NOUNS)]:
    noun = all_nouns[i]
    print(f"\n  '{noun}' ({'pl' if is_plural[i] else 'sg'}):")
    for k in range(min(A.topk, 4)):
        text = tok.decode(results_by_rank[k][i])
        p = topk_probs[i, k].item()
        print(f"    rank-{k+1} (p={p:.3f}): {text}")

# ---------- Phase 3: run each forced sequence through model, analyze bundles ----------
print(f"\n--- Phase 3: analyzing bundles for each alternative ---")

# For each rank k, run all sequences through the model and measure:
# (a) At pos 4 (prompt end): logit-lens P of the forced token and each subsequent token
# (b) Number agreement of the forced token
# (c) How quickly the cascade diverges from rank-1

W_U_cpu = W_U.cpu()
L_FINAL = NL
LAYERS = [0, 8, 12, 16, 20, NL]

print(f"\n{'='*70}")
print(f"  BUNDLE ANALYSIS FOR EACH ALTERNATIVE PREDICTION")
print(f"{'='*70}")

sg_idx = [i for i in range(len(all_nouns)) if not is_plural[i]]
pl_idx = [i for i in range(len(all_nouns)) if is_plural[i]]

summary = {}
for k in range(A.topk):
    seqs = results_by_rank[k]
    seq_len = len(seqs[0])
    seq_tensor = torch.tensor(seqs, device=DEV)

    # Get hidden states at prompt end (pos 4)
    with torch.no_grad():
        out_k = model(seq_tensor, output_hidden_states=True)

    # (a) Logit-lens P of each future token, FROM the prompt end
    hs_prompt_end = out_k.hidden_states[L_FINAL][:, PROMPT_LEN - 1, :].cpu()
    future_probs = []
    for offset in range(1, min(A.n_continue + 2, seq_len - PROMPT_LEN + 1)):
        target = seq_tensor[:, PROMPT_LEN - 1 + offset].cpu()
        logits = hs_prompt_end @ W_U_cpu.T
        probs = torch.softmax(logits, dim=-1)
        p = torch.gather(probs, 1, target.unsqueeze(1)).squeeze(1)
        future_probs.append(float(p.mean()))

    # (b) Forced token analysis
    forced_tokens = [seqs[i][PROMPT_LEN] for i in range(len(seqs))]
    forced_words = [decode(t) for t in forced_tokens]

    # Number agreement of forced token
    def check_number(word):
        w = word.strip().lower()
        if w in ("is", "was", "has"): return "sg"
        if w in ("are", "were", "have"): return "pl"
        if w.endswith("s") and w not in ("is","was","has","us","this","its"): return "sg"  # 3sg present
        return None

    num_agree = 0; num_total = 0
    for i in range(len(seqs)):
        vn = check_number(forced_words[i])
        if vn:
            expected = "pl" if is_plural[i] else "sg"
            if vn == expected: num_agree += 1
            num_total += 1

    # (c) Token overlap with rank-1 at each future position
    rank1_seqs = results_by_rank[0]
    overlaps = []
    for offset in range(seq_len - PROMPT_LEN):
        pos = PROMPT_LEN + offset
        match = sum(1 for i in range(len(seqs)) if seqs[i][pos] == rank1_seqs[i][pos])
        overlaps.append(match / len(seqs))

    # (d) sg/pl token divergence at each position
    frac_differ_by_pos = []
    for pos in range(PROMPT_LEN, seq_len):
        tok_sg = [seqs[i][pos] for i in sg_idx]
        tok_pl = [seqs[i][pos] for i in pl_idx]
        n = min(len(tok_sg), len(tok_pl))
        diff = sum(1 for j in range(n) if tok_sg[j] != tok_pl[j]) / n if n > 0 else 0
        frac_differ_by_pos.append(diff)

    # (e) At the prompt end, what's the number-direction score for this alternative?
    hs_mid = out_k.hidden_states[NL // 2][:, PROMPT_LEN - 1, :].cpu()  # L12
    d_num = hs_mid[pl_idx].mean(0) - hs_mid[sg_idx].mean(0)
    d_num = d_num / (d_num.norm() + 1e-9)

    # Mean forced-token probability
    mean_p = float(topk_probs[:, k].mean())
    from collections import Counter
    common_forced = Counter(forced_words).most_common(5)

    print(f"\n  RANK {k+1} (mean P = {mean_p:.3f}):")
    print(f"    Forced token: {common_forced}")
    print(f"    Number agree: {num_agree}/{num_total}"
          f" ({num_agree/num_total*100:.0f}%)" if num_total > 0 else "    Number agree: N/A")
    print(f"    Logit-lens P from prompt end: ", end="")
    for j, p in enumerate(future_probs[:6]):
        print(f"t+{j+1}={p:.4f} ", end="")
    print()
    print(f"    Overlap with rank-1: ", end="")
    for j, o in enumerate(overlaps[:6]):
        print(f"t+{j+1}={o:.0%} ", end="")
    print()
    print(f"    sg/pl differ: ", end="")
    for j, d in enumerate(frac_differ_by_pos[:6]):
        print(f"t+{j+1}={d:.0%} ", end="")
    print()

    summary[k] = {
        "mean_prob": mean_p,
        "common_forced": [(w, c) for w, c in common_forced],
        "num_agree": num_agree, "num_total": num_total,
        "future_probs_from_prompt": future_probs,
        "overlap_with_rank1": overlaps,
        "sg_pl_differ": frac_differ_by_pos,
    }

# ---------- Phase 4: the key comparison ----------
print(f"\n{'='*70}")
print(f"  KEY COMPARISON: does the prompt-end bundle serve multiple alternatives?")
print(f"{'='*70}")

# At pos 4, logit-lens P for rank-1 vs rank-2 forced tokens
print(f"\n  From the prompt end (L{L_FINAL} logit lens):")
print(f"    Rank-1 forced token: P = {summary[0]['future_probs_from_prompt'][0]:.4f}")
print(f"    Rank-2 forced token: P = {summary[1]['future_probs_from_prompt'][0]:.4f}")
if A.topk >= 3:
    print(f"    Rank-3 forced token: P = {summary[2]['future_probs_from_prompt'][0]:.4f}")

# Number agreement across ranks
print(f"\n  Number agreement of forced token:")
for k in range(min(A.topk, 5)):
    s = summary[k]
    pct = f"{s['num_agree']/s['num_total']*100:.0f}%" if s['num_total'] > 0 else "N/A"
    print(f"    Rank-{k+1}: {s['num_agree']}/{s['num_total']} ({pct}) | "
          f"mean P = {s['mean_prob']:.3f}")

# Cascade divergence
print(f"\n  Cascade divergence (overlap of rank-k with rank-1):")
print(f"    {'':>8}", end="")
for j in range(6):
    print(f"  {'t+'+str(j+1):>6}", end="")
print()
for k in range(1, min(A.topk, 5)):
    print(f"    rank-{k+1}:", end="")
    for j in range(6):
        print(f"  {summary[k]['overlap_with_rank1'][j]:>6.0%}", end="")
    print()

# Does the number constraint hold for ALL alternatives?
print(f"\n  Number constraint across alternatives:")
all_agree = all(summary[k]['num_agree'] == summary[k]['num_total']
                for k in range(min(A.topk, 5)) if summary[k]['num_total'] > 0)
if all_agree:
    print("  => ALL top-k alternatives respect the number constraint!")
    print("     The deposit bundle serves MULTIPLE competing predictions, not just the argmax.")
else:
    for k in range(min(A.topk, 5)):
        s = summary[k]
        if s['num_total'] > 0 and s['num_agree'] < s['num_total']:
            print(f"  => Rank-{k+1} BREAKS number agreement "
                  f"({s['num_agree']}/{s['num_total']})")

out = ROOT / "palimpsest" / "data" / "forced_alternative_test.json"
out.write_text(json.dumps({"model": A.model, "topk": A.topk, "summary": {str(k): v for k, v in summary.items()}},
               indent=2), encoding="utf-8")
print(f"\nwrote {out}")
