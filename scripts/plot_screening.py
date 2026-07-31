"""
Visualize + summarize the Stage-1 screening (per-matrix, one matrix pruned at a time).

Outputs to experiments/screen/plots/:
  - screening_redundancy_map.png : Layer x Matrix ΔPPL heatmaps (methods x sparsities)
  - screening_divergence_map.png : Layer x Matrix output-divergence (KL) heatmaps
  - screening_damage_vs_sparsity.png : mean ΔPPL vs sparsity per method (with best point marked)

Runs offline from the saved JSON (no model needed).
"""

import json
import glob
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREEN_DIR = "experiments/screen"
OUT_DIR = os.path.join(SCREEN_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DENSE_PPL = 13.559  # dense model, WikiText-2 test, 20% subset (same subset as screening)

# Data-aware methods (used for the redundancy/divergence maps and trajectories).
METHODS = ["wanda", "sparsegpt", "sparsegpt_recon"]
# Full ladder incl. the baseline controls (used for the method-comparison line plot).
LADDER = ["random", "magnitude", "wanda", "sparsegpt", "sparsegpt_recon"]
RATIOS = ["0.10", "0.20", "0.40"]

# Every method must be compared on IDENTICAL cells. Layers 32/33/34 were later measured for
# sparsegpt_recon only (to fix the policy's sensitivity labels), so including them here would
# silently compute that method's median over 8 layers and the baselines' over 5 -- and the extra
# cells are mostly robust, which would flatter it. Aggregate figures use the canonical set.
# The 32-34 data lives in FINDINGS.md; completing that grid for all methods is a backlog item.
CANON_LAYERS = [0, 9, 18, 27, 35]
MATRICES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Colorblind-safe categorical palette (Okabe-Ito), assigned in fixed order.
METHOD_COLOR = {
    "random": "#999999", "magnitude": "#D55E00",
    "wanda": "#0072B2", "sparsegpt": "#E69F00", "sparsegpt_recon": "#009E73",
}
METHOD_LABEL = {
    "random": "Random (floor, 5 seeds)", "magnitude": "Magnitude (weight-only)",
    "wanda": "Wanda", "sparsegpt": "SparseGPT (mask)", "sparsegpt_recon": "SparseGPT (reconstruct)",
}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load():
    """Load every method. Random has one file per seed, so values are averaged over seeds."""
    from collections import defaultdict
    acc, kacc = defaultdict(list), defaultdict(list)
    layers = set()
    for method in LADDER:
        for ratio in RATIOS:
            for f in glob.glob(os.path.join(SCREEN_DIR, f"{method}_p{ratio}", "layer*.json")):
                s = json.load(open(f))
                if s["layer"] not in CANON_LAYERS:
                    continue          # keep every method on identical cells (see CANON_LAYERS)
                layers.add(s["layer"])
                for r in s["results"]:
                    div = r.get("divergence") or {}
                    key = (method, ratio, s["layer"], r["matrix"])
                    acc[key].append(r["perplexity"] - DENSE_PPL)
                    if div.get("kl_dense_pruned") is not None:
                        kacc[key].append(div["kl_dense_pruned"])

    data = {}
    for (method, ratio, layer, mat), v in acc.items():
        k = kacc.get((method, ratio, layer, mat))
        data.setdefault((method, ratio), {}).setdefault(layer, {})[mat] = {
            "dppl": float(np.mean(v)),
            "kl": float(np.mean(k)) if k else None,
        }
    return data, sorted(layers)


def grid(data, layers, key):
    out = {}
    for method in METHODS:
        for ratio in RATIOS:
            M = np.full((len(MATRICES), len(layers)), np.nan)
            cell = data.get((method, ratio), {})
            for li, L in enumerate(layers):
                for mi, mat in enumerate(MATRICES):
                    v = cell.get(L, {}).get(mat, {}).get(key)
                    if v is not None:
                        M[mi, li] = v
            out[(method, ratio)] = M
    return out


def heatmaps(mats, layers, main, sub, cbar_label, fname, cmap, vmax):
    # Rows = pruning level (one row each), columns = method, values written in every cell.
    fig, axes = plt.subplots(len(RATIOS), len(METHODS), figsize=(17, 13))
    norm = plt.Normalize(0, vmax)
    im = None
    for i, ratio in enumerate(RATIOS):
        for j, method in enumerate(METHODS):
            ax = axes[i][j]
            M = mats[(method, ratio)]
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
            for mi in range(M.shape[0]):
                for li in range(M.shape[1]):
                    v = M[mi, li]
                    if not np.isnan(v):
                        tc = "white" if norm(v) < 0.55 else "black"
                        ax.text(li, mi, f"{v:.2f}", ha="center", va="center", fontsize=8, color=tc)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, fontsize=9)
            ax.set_yticks(range(len(MATRICES)))
            ax.set_yticklabels(MATRICES if j == 0 else [], fontsize=8.5)
            if i == 0:
                ax.set_title(METHOD_LABEL[method], fontsize=12.5, fontweight="bold", color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(f"{int(float(ratio) * 100)}% of the matrix pruned",
                              fontsize=12, fontweight="bold", color=INK)
            if i == len(RATIOS) - 1:
                ax.set_xlabel("layer", fontsize=9)
    fig.suptitle(main, fontsize=16, fontweight="bold", y=0.995, color=INK)
    fig.text(0.5, 0.965, sub, ha="center", fontsize=11, color=MUTED)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", fname)


def mark_min(ax, xs, ys, color):
    """Circle the minimum point, drop a dashed line to the y-axis, label its value."""
    i = int(np.argmin(ys))
    xmin, ymin = xs[i], ys[i]
    x0 = ax.get_xlim()[0]
    ax.plot([x0, xmin], [ymin, ymin], linestyle="--", color=color, linewidth=1.1, alpha=0.8, zorder=2)
    ax.scatter([xmin], [ymin], s=140, facecolors="none", edgecolors=color, linewidths=1.8, zorder=6)
    ax.annotate(f"best {ymin:.3f}", xy=(x0, ymin), xytext=(2, 4), textcoords="offset points",
                fontsize=9, color=color, fontweight="bold", ha="left", va="bottom")


def damage_vs_sparsity(data, layers):
    xs = [int(float(r) * 100) for r in RATIOS]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    all_x, all_y, all_c = [], [], []
    for method in LADDER:
        med, q1, q3 = [], [], []
        for ratio in RATIOS:
            cell = data.get((method, ratio), {})
            vals = np.array([cell[L][m]["dppl"] for L in layers for m in MATRICES
                             if L in cell and m in cell[L]])
            med.append(np.median(vals)); q1.append(np.percentile(vals, 25)); q3.append(np.percentile(vals, 75))
        ax.fill_between(xs, q1, q3, color=METHOD_COLOR[method], alpha=0.13, linewidth=0)
        ax.plot(xs, med, marker="o", linewidth=2, markersize=7,
                color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        all_x += xs; all_y += med; all_c.append(METHOD_COLOR[method])
    # mark the single best (lowest) median across all methods
    gi = int(np.argmin(all_y))
    mark_min(ax, all_x, all_y, all_c[gi // len(xs)])

    ax.set_xlabel("tile sparsity (%)", fontsize=11)
    ax.set_ylabel("ΔPPL per matrix — median (line), middle 50% (band)", fontsize=11)
    fig.suptitle("Method ladder: typical pruning damage vs sparsity",
                 fontsize=14, fontweight="bold", color=INK, y=0.99)
    ax.set_title("one matrix pruned at a time · median over 35 layer×matrix combos · band = middle 50% · baselines included as the floor",
                 fontsize=9, color=MUTED, pad=8)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, "screening_damage_vs_sparsity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved screening_damage_vs_sparsity.png")


def individual_plots(data, layers, method):
    """Small multiples: one panel per matrix type, one line per layer, ΔPPL vs sparsity.

    Shows the per-combination trajectory shape that the heatmap grid doesn't draw.
    """
    xs = [int(float(r) * 100) for r in RATIOS]
    layer_colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(layers)))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for mi, mat in enumerate(MATRICES):
        ax = axes[mi]
        for li, L in enumerate(layers):
            ys = [data[(method, r)][L][mat]["dppl"] for r in RATIOS]
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.9,
                    color=layer_colors[li], label=f"layer {L}")
        ax.set_title(mat, fontsize=12, fontweight="bold", color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.7, alpha=0.5)
        ax.grid(alpha=0.22)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if mi % 4 == 0:
            ax.set_ylabel("ΔPPL vs dense", fontsize=9.5)
        ax.set_xlabel("sparsity (%)", fontsize=9)

    axes[7].axis("off")
    handles, lbls = axes[0].get_legend_handles_labels()
    axes[7].legend(handles, lbls, title="layer (depth)", loc="center", frameon=False, fontsize=12)

    fig.suptitle(f"Per-matrix pruning damage vs sparsity — {METHOD_LABEL[method]}",
                 fontsize=15, fontweight="bold", color=INK, y=1.0)
    fig.text(0.5, 0.955, "each matrix pruned alone · one panel per matrix type · one line per layer · y-scales differ per panel",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUT_DIR, f"screening_individual_{method}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved screening_individual_{method}.png")


def main():
    data, layers = load()
    dppl = grid(data, layers, "dppl")
    kl = grid(data, layers, "kl")
    dppl_vmax = np.nanpercentile(np.concatenate([m.ravel() for m in dppl.values()]), 95)
    kl_vmax = np.nanpercentile(np.concatenate([m.ravel() for m in kl.values()]), 95)

    heatmaps(
        dppl, layers,
        "Redundancy map — perplexity damage per matrix",
        f"each matrix pruned alone · ΔPPL vs dense ({DENSE_PPL}) · brighter = more sensitive, dark = redundant",
        "ΔPPL vs dense", "screening_redundancy_map.png", "viridis", dppl_vmax,
    )
    heatmaps(
        kl, layers,
        "Behavioural-drift map — output divergence per matrix",
        "each matrix pruned alone · KL(dense ‖ pruned) · brighter = the pruned model behaves more differently",
        "KL divergence", "screening_divergence_map.png", "magma", kl_vmax,
    )
    damage_vs_sparsity(data, layers)
    for method in METHODS:
        individual_plots(data, layers, method)


if __name__ == "__main__":
    main()
