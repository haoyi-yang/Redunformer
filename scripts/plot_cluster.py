"""
Cluster & depth analysis plots -- Rathore's S4 specified output.

  "Produce line plots containing: layer number; mask-only perplexity change; reconstructed
   perplexity change; reconstruction improvement. This shows both depth-wise sensitivity and
   the value of compensation."

Rows = matrix type, columns = sparsity. Each panel draws mask-only and reconstructed against
layer; the shaded band between them IS the reconstruction improvement, so all three of his
requested quantities are in one readable panel rather than three overlapping lines.

Two clusters are scanned with a gap between them (robust 15-21, sensitive 29-35). The segments
are drawn SEPARATELY -- joining them would draw a line across layers 22-28 that were never
measured, inventing a trend.

Outputs experiments/screen_cluster_s4/plots/cluster_depth_reconstruction.png
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

S4_DIR = "experiments/archive/screen_cluster_s4"  # moved to archive/ (tile-64 shape-evidence)
OUT_DIR = os.path.join(S4_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DENSE = 13.559
ROBUST = [15, 16, 17, 18, 19, 20, 21]
SENSITIVE = [29, 30, 31, 32, 33, 34, 35]
RATIOS = ["0.10", "0.20", "0.40"]
MATRICES = ["k_proj", "up_proj"]        # easiest / hardest to reconstruct (measured)
MASK_C, RECON_C = "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load(method):
    """{(ratio, layer, matrix): delta_ppl}"""
    out = {}
    for f in glob.glob(os.path.join(S4_DIR, f"{method}_p*", "layer*.json")):
        d = json.load(open(f))
        r = f"{float(d['prune_ratio']):.2f}"
        for res in d["results"]:
            out[(r, d["layer"], res["matrix"])] = res["perplexity"] - DENSE
    return out


def segments(cell, ratio, matrix, layers):
    xs = [L for L in layers if (ratio, L, matrix) in cell]
    ys = [cell[(ratio, L, matrix)] for L in xs]
    return xs, ys


def main():
    mask = load("sparsegpt")
    recon = load("sparsegpt_recon")
    if not mask:
        print("no mask-only cluster data yet")
        return
    have = sorted({r for (r, _, _) in mask})
    print("mask ratios:", have, "| recon ratios:", sorted({r for (r, _, _) in recon}) or "none yet")

    # sharey per ROW: within a matrix type the panels must be comparable across sparsity, or the
    # per-panel autoscale hides the growth (up_proj runs to 1.4 at 10% and 3.9 at 40%; separate
    # scales make those look identical). Rows keep independent scales because k_proj and up_proj
    # differ by ~5x and sharing across rows would flatten k_proj into a line.
    fig, axes = plt.subplots(len(MATRICES), len(RATIOS), figsize=(16.5, 8.4),
                             sharex=True, sharey="row")
    for i, matrix in enumerate(MATRICES):
        for j, ratio in enumerate(RATIOS):
            ax = axes[i][j]
            for cluster, label in ((ROBUST, "robust 15-21"), (SENSITIVE, "sensitive 29-35")):
                mx, my = segments(mask, ratio, matrix, cluster)
                rx, ry = segments(recon, ratio, matrix, cluster)
                if mx:
                    ax.plot(mx, my, "-o", color=MASK_C, markersize=4.5, linewidth=1.9, zorder=3)
                if rx:
                    ax.plot(rx, ry, "-s", color=RECON_C, markersize=4.5, linewidth=1.9, zorder=3)
                # the band between the two curves IS the reconstruction improvement
                if mx and rx and mx == rx:
                    ax.fill_between(mx, my, ry, where=[a >= b for a, b in zip(my, ry)],
                                    color=RECON_C, alpha=0.16, linewidth=0)
                    ax.fill_between(mx, my, ry, where=[a < b for a, b in zip(my, ry)],
                                    color=MASK_C, alpha=0.16, linewidth=0)
            ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.6)
            ax.grid(alpha=0.2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(f"{int(float(ratio) * 100)}% tile sparsity", fontsize=11.5,
                             fontweight="bold", color=INK)
            if j == 0:
                ax.set_ylabel(f"{matrix}\nΔPPL vs dense", fontsize=11, fontweight="bold", color=INK)
            if i == len(MATRICES) - 1:
                ax.set_xlabel("layer", fontsize=10)

    h = [plt.Line2D([], [], color=MASK_C, marker="o", label="mask-only (no repair)"),
         plt.Line2D([], [], color=RECON_C, marker="s", label="reconstructed (repaired)"),
         plt.Rectangle((0, 0), 1, 1, color=RECON_C, alpha=0.16, label="repair helped"),
         plt.Rectangle((0, 0), 1, 1, color=MASK_C, alpha=0.16, label="repair made it WORSE")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cluster & depth analysis — does reconstructability follow a depth pattern?",
                 fontsize=15, fontweight="bold", color=INK, y=1.0)
    fig.text(0.5, 0.955, "matrix types chosen by RECONSTRUCTABILITY: k_proj easiest to repair "
                         "(72.5%), up_proj hardest (−3.4%) · gap at 22-28 is unmeasured, so the "
                         "clusters are drawn as separate segments",
             ha="center", fontsize=9.5, color=MUTED)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = os.path.join(OUT_DIR, "cluster_depth_reconstruction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
