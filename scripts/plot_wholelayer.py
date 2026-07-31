"""
Visualize + summarize the whole-layer (Stage 2) results (all 7 matrices of a layer
pruned together, evaluated once).

Outputs to experiments/screen_wholelayer/plots/:
  - wholelayer_damage_vs_depth.png : cumulative ΔPPL vs layer, per method, per sparsity
  - wholelayer_accumulation.png     : whole-layer ΔPPL vs sum of individual-matrix ΔPPL

Runs offline from the saved JSON.
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

DENSE = 13.559
METHODS = ["wanda", "sparsegpt", "sparsegpt_recon"]
RATIOS = ["0.10", "0.20", "0.40"]
OUT = "experiments/screen_wholelayer/plots"
os.makedirs(OUT, exist_ok=True)

METHOD_COLOR = {"wanda": "#0072B2", "sparsegpt": "#E69F00", "sparsegpt_recon": "#009E73"}
METHOD_LABEL = {"wanda": "Wanda", "sparsegpt": "SparseGPT (mask)", "sparsegpt_recon": "SparseGPT (reconstruct)"}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load_wholelayer():
    wl = {}
    for m in METHODS:
        for r in RATIOS:
            for f in glob.glob(f"experiments/screen_wholelayer/{m}_p{r}/layer*.json"):
                s = json.load(open(f))
                wl.setdefault((m, r), {})[s["layer"]] = s["perplexity"]
    return wl


def load_screening_sums():
    sc = {}
    for m in METHODS:
        for r in RATIOS:
            for f in glob.glob(f"experiments/screen/{m}_p{r}/layer*.json"):
                s = json.load(open(f))
                sc.setdefault((m, r), {})[s["layer"]] = sum(rr["perplexity"] - DENSE for rr in s["results"])
    return sc


def mark_min_pt(ax, x, y, color):
    x0 = ax.get_xlim()[0]
    ax.plot([x0, x], [y, y], linestyle="--", color=color, linewidth=1.0, alpha=0.8, zorder=2)
    ax.scatter([x], [y], s=130, facecolors="none", edgecolors=color, linewidths=1.7, zorder=6)
    ax.annotate(f"{y:.2f}", xy=(x0, y), xytext=(2, 3), textcoords="offset points",
                fontsize=8.5, color=color, fontweight="bold", ha="left", va="bottom")


def main():
    wl = load_wholelayer()
    sc = load_screening_sums()
    layers = sorted(next(iter(wl.values())).keys())

    # Plot 1: cumulative ΔPPL vs layer, one panel per sparsity
    fig, axes = plt.subplots(1, len(RATIOS), figsize=(16, 5.4), sharey=True)
    for j, r in enumerate(RATIOS):
        ax = axes[j]
        best = (None, None, 1e9)
        for m in METHODS:
            ys = [wl[(m, r)][L] - DENSE for L in layers]
            ax.plot(layers, ys, marker="o", linewidth=2, markersize=6,
                    color=METHOD_COLOR[m], label=METHOD_LABEL[m])
            k = int(np.argmin(ys))
            if ys[k] < best[2]:
                best = (layers[k], ys[k], ys[k], METHOD_COLOR[m])
        ax.set_xlim(min(layers) - 2, max(layers) + 1)
        mark_min_pt(ax, best[0], best[1], best[3])
        ax.set_title(f"{int(float(r) * 100)}% of every matrix pruned", fontsize=11, color=INK)
        ax.set_xlabel("layer", fontsize=10)
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.set_ylabel("perplexity increase vs dense  (ΔPPL)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9.5)
    fig.suptitle("Whole-layer pruning: damage across model depth",
                 fontsize=15, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, 0.965, "all 7 matrices of ONE layer pruned together, each layer tested independently · lower = more robust · dashed line marks the best point",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "wholelayer_damage_vs_depth.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved wholelayer_damage_vs_depth.png")

    # Plot 2: accumulation @20%
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    mx = 0.0
    for m in METHODS:
        wl_vals = [wl[(m, "0.20")][L] - DENSE for L in layers]
        sum_vals = [sc[(m, "0.20")][L] for L in layers]
        ax.scatter(sum_vals, wl_vals, s=70, color=METHOD_COLOR[m], label=METHOD_LABEL[m],
                   edgecolors="white", linewidths=0.6, zorder=5)
        mx = max(mx, max(wl_vals), max(sum_vals))
    ax.plot([0, mx * 1.05], [0, mx * 1.05], "--", color=MUTED, alpha=0.8,
            label="y = x  (damage adds up linearly)")
    ax.set_xlabel("sum of individual-matrix ΔPPL  (pruned one at a time)", fontsize=10.5)
    ax.set_ylabel("whole-layer ΔPPL  (all 7 pruned together)", fontsize=10.5)
    ax.set_title("Is layer damage additive?", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.5, 1.015, "@20% · points BELOW the line → sub-additive (per-matrix damages overlap)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "wholelayer_accumulation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved wholelayer_accumulation.png")


if __name__ == "__main__":
    main()
