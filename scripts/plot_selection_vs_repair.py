"""
Selection vs repair at whole-model scale (findings 4c / 4d / 4e).

The single figure for the presentation's strongest result. Whole-model perplexity vs sparsity,
against the random control floor:

  - the grey band = the random floor (min..max over 5 seeds). "No better than chance" lives here.
  - REPAIR methods (solid, filled markers) dive BELOW the band -- they beat the coin flip.
  - SELECTION-ONLY methods (dashed) sit INSIDE the band at 5% -- no better than random.
  - random_recon (random selection + repair) lands ON the other repair methods, which is the
    whole point: once you repair, which tiles you picked stops mattering.
  - magnitude is off-scale (4,942 -> 30M); annotated, not plotted.

Encoding is composite by design (colour + line style + marker), so identity never rests on colour
alone -- the node palette validator was unavailable in this environment, so secondary encoding is
the safety net (Okabe-Ito is CVD-safe by construction; this makes it robust regardless).

Runs offline from the saved JSON.
"""

import json
import glob
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WM = "experiments/wholemodel"
OUT = "experiments/wholemodel/plots/selection_vs_repair.png"
DENSE = 13.22
RATIOS = [0.05, 0.10, 0.20]
INK, MUTED = "#1a1a1a", "#6b6b6b"

# method -> (colour [Okabe-Ito], style, marker, label, group)
#   group: repair methods solid+filled; selection-only dashed; drawn so the two groups read at a
#   glance even in greyscale / for CVD readers.
METHODS = {
    "sparsegpt_recon": ("#009E73", "-",  "o", "SparseGPT (select + repair)", "repair"),
    "wanda_recon":     ("#CC79A7", "-",  "s", "Wanda select + repair",       "repair"),
    "random_recon":    ("#56B4E9", "-",  "D", "RANDOM select + repair",      "repair"),
    "wanda":           ("#0072B2", "--", "^", "Wanda (select only)",         "select"),
    "sparsegpt":       ("#E69F00", "--", "v", "SparseGPT (select only)",     "select"),
}


def load():
    rnd = defaultdict(list); rr = defaultdict(list); det = defaultdict(dict)
    for fn in glob.glob(f"{WM}/wholemodel_*.json"):
        d = json.load(open(fn)); m = d.get("method")
        if d.get("policy", "uniform") != "uniform" or m is None:
            continue
        r = d.get("results") or []
        ppl = d.get("perplexity") or (r[0].get("perplexity") if r else None)
        if ppl is None:
            continue
        p = round(float(d["prune_ratio"]), 2)
        if m == "random":
            rnd[p].append(ppl)
        elif m == "random_recon":
            rr[p].append(ppl)
        else:
            det[m][p] = ppl
    return rnd, rr, det


def main():
    rnd, rr, det = load()
    xs = [p * 100 for p in RATIOS]
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(15.5, 7), gridspec_kw={"width_ratios": [1.7, 1]})

    ax.axhline(DENSE, color=MUTED, linestyle=":", linewidth=1.3)
    ax.annotate(f"dense = {DENSE}", xy=(4.6, DENSE * 0.86), fontsize=9, color=MUTED)

    # random floor as a band (min..max over seeds) + median line
    lo = [min(rnd[p]) for p in RATIOS]; hi = [max(rnd[p]) for p in RATIOS]
    med = [np.median(rnd[p]) for p in RATIOS]
    ax.fill_between(xs, lo, hi, color="#999999", alpha=0.25, linewidth=0, zorder=1)
    ax.plot(xs, med, color="#666666", linewidth=1.6, linestyle=(0, (1, 1)), zorder=2,
            marker="x", markersize=7, label="Random floor (median; band = 5-seed range)")

    # random_recon: median + its own seed range, so it reads as a distribution too
    rr_med = [np.median(rr[p]) for p in RATIOS if rr[p]]
    rr_lo = [min(rr[p]) for p in RATIOS if rr[p]]; rr_hi = [max(rr[p]) for p in RATIOS if rr[p]]

    for m, (c, ls, mk, lab, grp) in METHODS.items():
        if m == "random_recon":
            ys = rr_med
            ax.fill_between(xs[:len(ys)], rr_lo, rr_hi, color=c, alpha=0.13, linewidth=0, zorder=2)
        else:
            ys = [det[m].get(p) for p in RATIOS]
        lw = 2.6 if grp == "repair" else 1.9
        ax.plot(xs[:len(ys)], ys, color=c, linestyle=ls, linewidth=lw, marker=mk,
                markersize=8.5 if grp == "repair" else 7, markerfacecolor=c if grp == "repair" else "white",
                markeredgecolor=c, markeredgewidth=1.6, label=lab, zorder=5 if grp == "repair" else 4)

    # magnitude is off-scale -- state it rather than plot it
    ax.annotate("magnitude (naive control):\n4,942 → 601,384 → off-scale",
                xy=(0.97, 0.35), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.5, color="#D55E00", style="italic")

    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{int(x)}%" for x in xs])
    ax.set_xlim(4, 21)
    ax.set_xlabel("global tile sparsity", fontsize=11.5)
    ax.set_ylabel("WikiText-2 perplexity (log scale, lower = better)", fontsize=11.5)
    ax.set_title("Full sparsity range", fontsize=11, color=MUTED, pad=6)
    ax.grid(True, alpha=0.22, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    # ---- zoom panel: 5% only, LINEAR, so "repair below the band, selection inside it" is unmissable
    p = 0.05
    band_lo, band_hi = min(rnd[p]), max(rnd[p])
    axz.axhspan(band_lo, band_hi, color="#999999", alpha=0.22, linewidth=0)
    axz.axhline(np.median(rnd[p]), color="#666666", linestyle=(0, (1, 1)), linewidth=1.4)
    axz.annotate("random floor\n(5 seeds, ×)", xy=(0.5, 60), ha="center", fontsize=8.5,
                 color="#555555", style="italic")
    for sd in rnd[p]:
        axz.scatter([0.5], [sd], marker="x", s=45, color="#888888", zorder=3)
    axz.axhline(DENSE, color=MUTED, linestyle=":", linewidth=1.2)
    axz.annotate(f"dense {DENSE}", xy=(0.05, DENSE + 0.6), fontsize=8.5, color=MUTED, va="bottom")

    order = [("sparsegpt_recon", 1.3), ("wanda_recon", 1.65), ("random_recon", 2.0),
             ("wanda", 2.55), ("sparsegpt", 2.9)]
    for m, x in order:
        c, ls, mk, lab, grp = METHODS[m]
        y = np.median(rr[p]) if m == "random_recon" else det[m][p]
        axz.scatter([x], [y], marker=mk, s=190, facecolors=c if grp == "repair" else "white",
                    edgecolors=c, linewidths=2, zorder=5)
        axz.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 12 if grp == "repair" else -17),
                     textcoords="offset points", ha="center", fontsize=8.5, color=c, fontweight="bold")
    axz.annotate("REPAIR\nbelow the floor\n→ beats chance 5/5", xy=(0.55, 16.5), ha="center",
                 va="center", fontsize=9, color="#009E73", fontweight="bold")
    axz.annotate("SELECT only\ninside floor → = chance", xy=(2.72, 35), ha="center",
                 fontsize=9, color="#0072B2", fontweight="bold")
    axz.set_xlim(0, 3.5); axz.set_ylim(9, 82)
    axz.set_xticks([])
    axz.set_ylabel("perplexity at 5% (linear)", fontsize=10.5)
    axz.set_title("Zoom: 5% — the usable operating point", fontsize=11, color=MUTED, pad=14)
    for s in ("top", "right"):
        axz.spines[s].set_visible(False)

    fig.suptitle("At usable sparsity, repair is the whole method — selection is worthless",
                 fontsize=15.5, fontweight="bold", color=INK, y=1.0)
    fig.text(0.5, 0.945, "whole model · random floor = 5 seeds · random+repair lands on the "
                         "data-aware repaired methods, so which tiles you pick stops mattering once you repair",
             ha="center", fontsize=9.5, color=MUTED)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", OUT)


if __name__ == "__main__":
    main()
