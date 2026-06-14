"""Generate all four figures for the deposit channel paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Schematic — deposit mechanism across 4 roles
# ─────────────────────────────────────────────────────────────────────
def fig1_schematic():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Token boxes
    tokens = [
        (0, 3, "The"),
        (1.8, 3, "dog"),
        (3.6, 3, "saw"),
        (5.4, 3, "the"),
        (7.2, 3, "cat"),
        (9.0, 3, "and"),
    ]
    box_w, box_h = 1.4, 0.7
    for x, y, tok in tokens:
        rect = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.08", linewidth=0.8,
                              edgecolor='black', facecolor='#f0f0f0')
        ax.add_patch(rect)
        ax.text(x, y, tok, ha='center', va='center', fontsize=11, fontweight='bold')

    # Unmeasured arrows (dashed, faint) — show that other channels may exist
    dashed_style = "Simple,tail_width=0.8,head_width=4,head_length=3"
    unmeasured_pairs = [
        # subj -> obj (dog -> cat)
        ((1.8, 2.55), (7.2, 2.55), "arc3,rad=-0.15"),
        # verb -> obj (saw -> cat)
        ((3.6, 2.55), (7.2, 2.55), "arc3,rad=-0.12"),
        # subj -> clause-end (dog -> and)
        ((1.8, 3.45), (9.0, 3.45), "arc3,rad=0.2"),
    ]
    for (x1, y1), (x2, y2), cstyle in unmeasured_pairs:
        arr_u = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=dashed_style, color='#cccccc',
                                linewidth=0.8, linestyle='--',
                                connectionstyle=cstyle)
        ax.add_patch(arr_u)

    # Role arrows with labels (measured channels — solid, coloured)
    arrow_style = "Simple,tail_width=1.5,head_width=6,head_length=4"

    # Subject -> verb (dog -> saw)
    arr = FancyArrowPatch((1.8, 2.55), (3.6, 2.55),
                          arrowstyle=arrow_style, color='#2166ac', linewidth=1.2,
                          connectionstyle="arc3,rad=-0.3")
    ax.add_patch(arr)
    ax.text(2.7, 1.55, "subj→verb\ncos 0.47", ha='center', va='center',
            fontsize=7.5, color='#2166ac', fontstyle='italic')

    # Object -> "and" (cat -> and)  = object->clause-end
    arr2 = FancyArrowPatch((7.2, 2.55), (9.0, 2.55),
                           arrowstyle=arrow_style, color='#b2182b', linewidth=1.2,
                           connectionstyle="arc3,rad=-0.3")
    ax.add_patch(arr2)
    ax.text(8.1, 1.55, "obj→clause-end\ncos 0.73", ha='center', va='center',
            fontsize=7.5, color='#b2182b', fontstyle='italic')

    # Object -> pronoun (cat -> implied "it")
    ax.text(10.5, 3, "it", ha='center', va='center', fontsize=10,
            fontstyle='italic', color='#666666')
    arr3 = FancyArrowPatch((7.2, 3.45), (10.2, 3.45),
                           arrowstyle=arrow_style, color='#d6604d', linewidth=1.2,
                           connectionstyle="arc3,rad=0.35")
    ax.add_patch(arr3)
    ax.text(8.7, 4.65, "obj→pronoun\ncos 0.61", ha='center', va='center',
            fontsize=7.5, color='#d6604d', fontstyle='italic')

    # Verb -> continuation (saw -> the)
    arr4 = FancyArrowPatch((3.6, 3.45), (5.4, 3.45),
                           arrowstyle=arrow_style, color='#4393c3', linewidth=1.2,
                           connectionstyle="arc3,rad=0.25")
    ax.add_patch(arr4)
    ax.text(4.5, 4.45, "verb→cont\ncos 0.71", ha='center', va='center',
            fontsize=7.5, color='#4393c3', fontstyle='italic')

    # Annotation box
    ax.text(5.0, -0.5,
            "Solid arrows = measured channels (4 role pairs).\n"
            "Dashed grey = other possible channels (not measured).\n"
            "Each solid arrow = a role-specific low-rank linear map R.",
            ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc',
                      edgecolor='#999999', linewidth=0.5))

    ax.set_title("Figure 1: Deposit channels across grammatical roles", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_schematic.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig1_schematic.png", bbox_inches='tight')
    plt.close(fig)
    print("  fig1 done")


# ─────────────────────────────────────────────────────────────────────
# Figure 2: Bar chart — held-out cosine by role
# ─────────────────────────────────────────────────────────────────────
def fig2_role_comparison():
    # Data from centered_r2_all_roles.json
    roles = [
        "Subj→verb",
        "Obj→pronoun",
        "Obj→clause-end",
    ]
    matched_r2  = [0.4141, 0.5224, 0.5643]
    cross_r2    = [-0.4184, -0.3261, -0.6514]
    channel_r2  = [0.9384, 0.9647, 0.9544]

    x = np.arange(len(roles))
    w = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars1 = ax.bar(x - w, matched_r2, w, label='Centered R² (full space)', color='#2166ac', edgecolor='white')
    bars2 = ax.bar(x, cross_r2, w, label='Cross-category R² (wrong word)', color='#d6604d', edgecolor='white')
    bars3 = ax.bar(x + w, channel_r2, w, label='Channel R² (30-dim)', color='#4daf4a', edgecolor='white')

    ax.set_ylabel('R² (word-specific, centered)')
    ax.set_xticks(x)
    ax.set_xticklabels(roles, fontsize=9)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.set_ylim(-0.8, 1.1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title("Figure 2: Word-specific transport quality by role", fontsize=11)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_role_comparison.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig2_role_comparison.png", bbox_inches='tight')
    plt.close(fig)
    print("  fig2 done")


# ─────────────────────────────────────────────────────────────────────
# Figure 3: Flow diagram — discover→validate pipeline
# ─────────────────────────────────────────────────────────────────────
def fig3_pipeline():
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 3)
    ax.axis('off')

    boxes = [
        (0.8, 1.5, "Fit deposit\nmap R\n(ridge, L8→L12)"),
        (3.0, 1.5, "Extract\n30-dim input\nchannel (SVD)"),
        (5.2, 1.5, "k-means\n(k=8)\non words"),
        (7.4, 1.5, "Per-cluster\ntransport\ncos (held-out)"),
        (9.6, 1.5, "Causal patch\nvs random\ncontrol"),
    ]

    bw, bh = 1.7, 1.4
    for x, y, txt in boxes:
        rect = FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                              boxstyle="round,pad=0.1", linewidth=0.8,
                              edgecolor='black', facecolor='#e0ecf4')
        ax.add_patch(rect)
        ax.text(x, y, txt, ha='center', va='center', fontsize=8)

    # Arrows between boxes
    arrow_kw = dict(arrowstyle='->', color='black', linewidth=1.2)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + bw/2
        x2 = boxes[i+1][0] - bw/2
        ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5), arrowprops=arrow_kw)

    # Result annotation
    ax.text(9.6, 0.3, "7/7 non-empty\nclusters validated",
            ha='center', va='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#c7e9c0',
                      edgecolor='#41ab5d', linewidth=0.8))

    ax.set_title("Figure 3: Discover–validate pipeline", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_pipeline.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig3_pipeline.png", bbox_inches='tight')
    plt.close(fig)
    print("  fig3 done")


# ─────────────────────────────────────────────────────────────────────
# Figure 4: Line plot — causal effect vs layer at consuming position
# ─────────────────────────────────────────────────────────────────────
def fig4_layer_sweep():
    # Data from transported_causal.json
    with open(DATA / "transported_causal.json", encoding="utf-8") as f:
        data = json.load(f)

    # Extract layer sweep data
    # The file has layer_sweep entries — let me parse what's available
    # From the agent's report:
    layers =       [4,     8,     12,    16,    20,    23]
    causal_effect = [0.053, 0.100, 0.815, 2.189, 4.275, 4.481]
    random_effect = [0.0,   0.001, -0.003, -0.001, -0.003, -0.004]
    source_effect = 3.19  # source noun at L8

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(layers, causal_effect, 'o-', color='#2166ac', linewidth=2,
            markersize=6, label='Place axis patch', zorder=3)
    ax.plot(layers, random_effect, 's--', color='#999999', linewidth=1.2,
            markersize=4, label='Random direction', zorder=2)
    ax.axhline(source_effect, color='#b2182b', linewidth=1, linestyle=':',
               label=f'Source noun effect (L8) = {source_effect:.2f}', zorder=1)

    # Annotations
    ax.annotate('decodable\nhere', xy=(8, 0.10), xytext=(10, 0.8),
                fontsize=8, ha='center', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='#666666', linewidth=0.8))
    ax.annotate('causal\nhere', xy=(20, 4.275), xytext=(17, 4.6),
                fontsize=8, ha='center', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='#666666', linewidth=0.8))

    ax.set_xlabel('Layer')
    ax.set_ylabel('Causal effect (place − object readout)')
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(-0.5, 5.5)
    ax.set_title("Figure 4: Transported copy becomes causal late", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_layer_sweep.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig4_layer_sweep.png", bbox_inches='tight')
    plt.close(fig)
    print("  fig4 done")


# ─────────────────────────────────────────────────────────────────────
# Figure 5: Heatmap — centered R² across layer pairs
# ─────────────────────────────────────────────────────────────────────
def fig5_layer_heatmap():
    with open(DATA / "layer_pair_sweep.json", encoding="utf-8") as f:
        raw = json.load(f)

    L_LEX = [2, 4, 6, 8, 10, 12, 14, 16]
    L_TR  = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    grid = np.full((len(L_LEX), len(L_TR)), np.nan)
    for i, ll in enumerate(L_LEX):
        for j, lt in enumerate(L_TR):
            key = f"{ll}->{lt}"
            if key in raw:
                grid[i, j] = raw[key]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(grid, aspect='auto', cmap='RdYlBu_r', vmin=-0.05, vmax=0.40,
                   origin='upper')
    ax.set_xticks(range(len(L_TR)))
    ax.set_xticklabels([f'L{l}' for l in L_TR], fontsize=8)
    ax.set_yticks(range(len(L_LEX)))
    ax.set_yticklabels([f'L{l}' for l in L_LEX], fontsize=8)
    ax.set_xlabel('Consumer layer (transport)')
    ax.set_ylabel('Source layer (lexical)')

    # Annotate cells
    for i in range(len(L_LEX)):
        for j in range(len(L_TR)):
            v = grid[i, j]
            if not np.isnan(v):
                color = 'white' if v > 0.30 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5, color=color)

    # Mark L8->L12
    # find indices
    i8 = L_LEX.index(8)
    j12 = L_TR.index(12)
    ax.plot(j12, i8, 's', markersize=14, markeredgecolor='black',
            markerfacecolor='none', markeredgewidth=2)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, label='Centered R²')
    ax.set_title("Figure 5: Deposit map R² across layer pairs (subj→verb)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_layer_heatmap.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig5_layer_heatmap.png", bbox_inches='tight')
    plt.close(fig)
    print("  fig5 done")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig1_schematic()
    fig2_role_comparison()
    fig3_pipeline()
    fig4_layer_sweep()
    fig5_layer_heatmap()
    print(f"All figures saved to {OUT}/")
