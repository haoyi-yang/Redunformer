"""
Headline plot: whole-model uniform pruning, perplexity + KL vs sparsity.

Reads experiments/wholemodel/wholemodel_{method}_p{ratio}.json (one whole-model
perplexity per method/sparsity) and writes experiments/wholemodel/plots/.
Perplexity uses a log y-axis (the range spans 13 -> tens of millions).
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

DENSE = 13.22  # full-eval dense perplexity (whole-model sweep used eval_frac=1.0)
METHODS = ["wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon"]
COLOR = {"wanda": "#0072B2", "sparsegpt": "#E69F00", "sparsegpt_recon": "#009E73",
         "wanda_recon": "#CC79A7"}
LABEL = {"wanda": "Wanda", "sparsegpt": "SparseGPT (mask)",
         "sparsegpt_recon": "SparseGPT (select + repair)",
         "wanda_recon": "Wanda select + SparseGPT repair"}
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = "experiments/wholemodel/plots"
os.makedirs(OUT, exist_ok=True)


def main():
    data = {}
    for f in glob.glob("experiments/wholemodel/wholemodel_*.json"):
        j = json.load(open(f))
        data.setdefault(j["method"], {})[j["prune_ratio"]] = (
            j["perplexity"], (j.get("divergence") or {}).get("kl_dense_pruned"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for m in METHODS:
        rs = sorted(data.get(m, {}))
        xs = [int(r * 100) for r in rs]
        ppl = [data[m][r][0] for r in rs]
        kl = [data[m][r][1] for r in rs]
        ax1.plot(xs, ppl, marker="o", lw=2, ms=7, color=COLOR[m], label=LABEL[m])
        ax2.plot(xs, kl, marker="o", lw=2, ms=7, color=COLOR[m], label=LABEL[m])

    ax1.axhline(DENSE, ls="--", color=MUTED, lw=1.2, alpha=0.9, label=f"dense = {DENSE}")
    ax1.set_yscale("log")
    ax1.set_xlabel("global tile sparsity (%)", fontsize=11)
    ax1.set_ylabel("perplexity  (log scale, lower = better)", fontsize=11)
    ax1.set_title("Perplexity vs sparsity", fontsize=13, fontweight="bold", color=INK)

    ax2.set_xlabel("global tile sparsity (%)", fontsize=11)
    ax2.set_ylabel("KL(dense ‖ pruned)  (lower = better)", fontsize=11)
    ax2.set_title("Behavioural divergence vs sparsity", fontsize=13, fontweight="bold", color=INK)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5)

    fig.suptitle("Whole-model uniform pruning — the repair is what matters, not the selection",
                 fontsize=15, fontweight="bold", color=INK, y=1.04)
    fig.text(0.5, 0.975,
             "every matrix in all 36 layers pruned at one sparsity, evaluated once (full WikiText-2) · "
             "note the two repaired methods sit on top of each other",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT, "wholemodel_headline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("saved wholemodel_headline.png")


if __name__ == "__main__":
    main()
