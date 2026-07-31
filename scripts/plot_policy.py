"""
Policy A (uniform) vs Policy B (sensitivity-aware) head-to-head -- Rathore Strategy 5.

Both policies remove the SAME total number of tiles; only the distribution differs.
So any gap is attributable to *where* the tiles were taken from, not how many.

Outputs to experiments/wholemodel/plots/:
  - policy_a_vs_b.png : perplexity curves (left) + budget-matched gain ratio (right)

Honesty guard: once a model's perplexity passes DESTROYED_PPL it is gibberish, and the
A/B ratio between two gibberish models is noise, not a result. Those points are drawn
faded and excluded from the "B wins" reading.

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

WM_DIR = "experiments/wholemodel"
OUT_DIR = os.path.join(WM_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DENSE_PPL = 13.22          # dense Qwen3-4B, WikiText-2 test, full eval
DESTROYED_PPL = 1000.0     # beyond this the model is gibberish; ratios stop meaning anything

METHODS = ["wanda", "sparsegpt", "sparsegpt_recon"]
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

METHOD_COLOR = {"wanda": "#0072B2", "sparsegpt": "#E69F00", "sparsegpt_recon": "#009E73"}
METHOD_LABEL = {"wanda": "Wanda", "sparsegpt": "SparseGPT (mask)", "sparsegpt_recon": "SparseGPT (reconstruct)"}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load():
    """{(method, policy, ratio): perplexity} from the whole-model JSONs."""
    rows = {}
    for f in glob.glob(os.path.join(WM_DIR, "*.json")):
        d = json.load(open(f))
        m, p = d.get("method"), d.get("prune_ratio")
        if m is None or p is None:
            continue
        ppl = d.get("perplexity")
        if ppl is None and "results" in d and d["results"]:
            ppl = d["results"][0].get("perplexity")
        if ppl is not None:
            rows[(m, d.get("policy", "uniform"), round(float(p), 2))] = ppl
    return rows


def curves(ax, rows):
    """Left panel: absolute perplexity, A solid vs B dashed, log y."""
    ax.axhline(DENSE_PPL, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate(f"dense = {DENSE_PPL}", xy=(5, DENSE_PPL), xytext=(0, -12),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.axhspan(DESTROYED_PPL, 1e9, color="#d55e00", alpha=0.055, linewidth=0)
    ax.annotate("model is gibberish above here", xy=(52, DESTROYED_PPL * 2.2),
                fontsize=8.5, color="#d55e00", alpha=0.85)

    for m in METHODS:
        for pol, style, lab in (("uniform", "-", "A uniform"), ("sensitivity", "--", "B sensitivity-aware")):
            xs = [r * 100 for r in RATIOS if (m, pol, r) in rows]
            ys = [rows[(m, pol, r)] for r in RATIOS if (m, pol, r) in rows]
            if not xs:
                continue
            ax.plot(xs, ys, style, marker="o" if pol == "uniform" else "s",
                    markersize=5, linewidth=2, color=METHOD_COLOR[m], alpha=0.95,
                    label=f"{METHOD_LABEL[m]} — {lab}")

    ax.set_yscale("log")
    ax.set_xlabel("tile sparsity (%)", fontsize=10.5)
    ax.set_ylabel("WikiText-2 perplexity (log scale)", fontsize=10.5)
    ax.set_title("Absolute damage — same tile budget, different distribution",
                 fontsize=11, color=MUTED, pad=8)
    ax.grid(True, alpha=0.22, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)


def gains(ax, rows):
    """Right panel: gain = ppl_A / ppl_B. >1 means B wins at the same budget.

    Only points where at least one model is still intelligible are drawn. A ratio
    between two gibberish models is noise, and drawing it -- even faded -- hands the
    eye a huge excursion that means nothing. Each line simply stops where its method
    dies, which is itself the honest reading.
    """
    ax.axhline(1.0, color=INK, linewidth=1.2)
    ax.annotate("parity — no benefit from the policy", xy=(3.5, 1.02), fontsize=8.5, color=INK)
    ax.axhspan(0, 1, color="#d55e00", alpha=0.05, linewidth=0)
    ax.axhspan(1, 3, color="#009e73", alpha=0.05, linewidth=0)
    ax.annotate("B better", xy=(3.5, 1.60), fontsize=9.5, color="#009e73", fontweight="bold")
    ax.annotate("B worse", xy=(3.5, 0.55), fontsize=9.5, color="#d55e00", fontweight="bold")

    for m in METHODS:
        trust, dead_from = [], None
        for r in RATIOS:
            a, b = rows.get((m, "uniform", r)), rows.get((m, "sensitivity", r))
            if not (a and b):
                continue
            if a > DESTROYED_PPL and b > DESTROYED_PPL:
                dead_from = dead_from if dead_from is not None else r * 100
                continue
            trust.append((r * 100, a / b))
        if not trust:
            continue

        xs = [p[0] for p in trust]
        ys = [p[1] for p in trust]
        ax.plot(xs, ys, "-", marker="o", markersize=7, linewidth=2.4,
                color=METHOD_COLOR[m], label=METHOD_LABEL[m])

        # Mark where the method stops being measurable at all.
        if dead_from is not None:
            ax.plot([xs[-1]], [ys[-1]], marker="x", markersize=11, markeredgewidth=2.4,
                    color=METHOD_COLOR[m])
            ax.annotate(f"both models destroyed\nbeyond {int(xs[-1])}%",
                        xy=(xs[-1], ys[-1]), xytext=(6, 10), textcoords="offset points",
                        fontsize=8, color=METHOD_COLOR[m], style="italic")

        # Crossover: where a measurable gain first drops through parity.
        for (x0, y0), (x1, y1) in zip(trust, trust[1:]):
            if y0 >= 1 > y1:
                xc = x0 + (x1 - x0) * (y0 - 1) / (y0 - y1)
                ax.plot([xc, xc], [1, 0.47], "--", color=METHOD_COLOR[m], linewidth=1.2, alpha=0.85)
                ax.annotate(f"crossover\n{xc:.0f}%", xy=(xc, 0.455), ha="center", va="top",
                            fontsize=8.5, color=METHOD_COLOR[m], fontweight="bold")
                break

    ax.text(0.5, -0.155, "omitted: points where both policies exceed perplexity "
                         f"{DESTROYED_PPL:.0f} — the ratio between two destroyed models is noise, not a result",
            transform=ax.transAxes, ha="center", fontsize=8.2, color=MUTED, style="italic")
    ax.set_ylim(0.42, 1.85)
    ax.set_xlim(0, 75)
    ax.set_xlabel("tile sparsity (%)", fontsize=10.5)
    ax.set_ylabel("gain = perplexity(A) / perplexity(B)", fontsize=10.5)
    ax.set_title("Does spending the budget unevenly pay off?  (dashed = crossover)",
                 fontsize=11, color=MUTED, pad=8)
    ax.grid(True, alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def main():
    rows = load()
    have = sorted({k[1] for k in rows})
    print("policies found:", have)
    missing = [(m, r) for m in METHODS for r in RATIOS if (m, "sensitivity", r) not in rows]
    if missing:
        print(f"note: {len(missing)} Policy B cells still missing (sweep unfinished): {missing}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    curves(ax1, rows)
    gains(ax2, rows)

    fig.suptitle("Policy A (uniform) vs Policy B (sensitivity-aware) — budget-matched whole-model pruning",
                 fontsize=15, fontweight="bold", color=INK, y=1.0)
    fig.text(0.5, 0.945,
             "identical total tiles removed · B protects sensitive regions and over-prunes robust ones · "
             "sensitivity measured at 20% sparsity",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "policy_a_vs_b.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
