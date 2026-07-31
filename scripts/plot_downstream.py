"""
Downstream capability vs sparsity, and the perplexity-vs-capability divergence.

Two questions:
  1. How much of the model's LEARNED ability survives pruning, and where does it die?
     Raw accuracy flatters a pruned model, because a broken model still scores chance
     (25% on HellaSwag/ARC, 50% on PIQA) by guessing. So we report RETAINED ability:

         retained = (acc_pruned - chance) / (acc_dense - chance)

     0% = no better than guessing. 100% = dense.

  2. Is perplexity a trustworthy proxy for that? (Finding 7 says no.) The right panel
     pairs each config's perplexity against its retained ability, so the gap is visible
     rather than asserted.

Outputs to experiments/downstream/plots/:
  - downstream_capability.png

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

DS_DIR = "experiments/downstream"
WM_DIR = "experiments/wholemodel"
OUT_DIR = os.path.join(DS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DENSE_PPL = 13.22
CHANCE = {"hellaswag": 0.25, "piqa": 0.50, "arc_easy": 0.25}
TASK_COLOR = {"hellaswag": "#0072B2", "piqa": "#E69F00", "arc_easy": "#009E73"}
TASK_LABEL = {"hellaswag": "HellaSwag", "piqa": "PIQA", "arc_easy": "ARC-Easy"}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def acc_of(d, task):
    v = d.get("results", {}).get(task, {})
    return v.get("acc_norm,none", v.get("acc,none"))


def load():
    """[{sparsity, policy, method, accs{}, ppl}] plus the dense reference accuracies."""
    dense, rows = None, []
    for f in glob.glob(os.path.join(DS_DIR, "*.json")):
        d = json.load(open(f))
        if d.get("dense"):
            dense = {t: acc_of(d, t) for t in CHANCE}
            continue
        rows.append({
            "method": d.get("method"),
            "sparsity": float(d.get("prune_ratio")),
            "policy": d.get("policy", "uniform"),
            "accs": {t: acc_of(d, t) for t in CHANCE},
            "ppl": lookup_ppl(d.get("method"), d.get("prune_ratio"), d.get("policy", "uniform")),
        })
    return dense, sorted(rows, key=lambda r: r["sparsity"])


def lookup_ppl(method, ratio, policy):
    """Pair each downstream config with the perplexity of the same config, if we ran it."""
    short = str(int(round(float(ratio) * 100)))
    pol = "" if policy == "uniform" else f"_{policy}"
    p = os.path.join(WM_DIR, f"wholemodel_{method}_p{short}{pol}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ppl = d.get("perplexity")
    if ppl is None and d.get("results"):
        ppl = d["results"][0].get("perplexity")
    return ppl


def retained(acc, dense_acc, task):
    """Share of ABOVE-CHANCE ability kept. Guarding against a dense-==-chance divide."""
    c = CHANCE[task]
    if acc is None or dense_acc is None or abs(dense_acc - c) < 1e-9:
        return None
    return (acc - c) / (dense_acc - c) * 100


def capability_panel(ax, dense, rows):
    uni = [r for r in rows if r["policy"] == "uniform" and r["method"] == "sparsegpt_recon"]
    for task in CHANCE:
        xs = [r["sparsity"] * 100 for r in uni if r["accs"][task] is not None]
        ys = [retained(r["accs"][task], dense[task], task) for r in uni if r["accs"][task] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, "-o", color=TASK_COLOR[task], linewidth=2.3, markersize=7,
                label=TASK_LABEL[task])

    ax.axhline(100, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate("dense", xy=(0.4, 101), fontsize=8.5, color=MUTED)
    ax.axhline(0, color="#d55e00", linewidth=1.4)
    ax.annotate("no better than guessing", xy=(0.4, 3), fontsize=9, color="#d55e00", fontweight="bold")
    ax.axhspan(-12, 0, color="#d55e00", alpha=0.06, linewidth=0)

    ax.set_xlabel("tile sparsity (%)", fontsize=10.5)
    ax.set_ylabel("ability retained vs dense (%)", fontsize=10.5)
    ax.set_title("How much of the LEARNED ability survives?  (reconstruct, uniform)",
                 fontsize=11, color=MUTED, pad=8)
    ax.set_ylim(-12, 112)
    ax.grid(alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5)


def divergence_panel(ax, dense, rows):
    """The money panel: perplexity says one thing, capability says another.

    The reference curve is recon+uniform. Other configs (a different POLICY, and a different
    METHOD entirely) are overlaid as separate markers rather than joined into the same line --
    they are not one trajectory. Whether they land ON the curve is itself the test: if they do,
    perplexity predicts capability regardless of how the damage was inflicted, and its only
    defect is the nonlinearity. If they scattered off it, perplexity would be method-dependent
    and useless as a cross-method proxy.
    """
    base = [r for r in rows if r["ppl"] and r["method"] == "sparsegpt_recon" and r["policy"] == "uniform"]
    other = [r for r in rows if r["ppl"] and r not in base]

    for task in CHANCE:
        xs = [r["ppl"] for r in base if r["accs"][task] is not None]
        ys = [retained(r["accs"][task], dense[task], task) for r in base if r["accs"][task] is not None]
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "-o", color=TASK_COLOR[task],
                    linewidth=2, markersize=7, label=TASK_LABEL[task], zorder=4)

    # Overlay the off-curve configs: square = different policy, triangle = different method.
    for r in other:
        mk = "s" if r["method"] == "sparsegpt_recon" else "^"
        for task in CHANCE:
            if r["accs"][task] is None:
                continue
            ax.scatter([r["ppl"]], [retained(r["accs"][task], dense[task], task)],
                       marker=mk, s=70, facecolors="none", edgecolors=TASK_COLOR[task],
                       linewidths=1.8, zorder=5)
    ax.scatter([], [], marker="s", s=70, facecolors="none", edgecolors=MUTED,
               linewidths=1.8, label="sensitivity policy (below curve)")
    ax.scatter([], [], marker="^", s=70, facecolors="none", edgecolors=MUTED,
               linewidths=1.8, label="wanda (beyond curve range)")
    ax.annotate("the sensitivity policy sits 10-16pp BELOW the curve: at equal perplexity it\n"
                "delivers LESS capability. Its perplexity gain overstates its real gain ~3x —\n"
                "unsurprising, since its sensitivity map was itself built FROM perplexity.",
                xy=(0.035, 0.15), xycoords="axes fraction", fontsize=8.2, color="#D55E00",
                style="italic")

    ax.axvline(DENSE_PPL, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate(f"dense ppl {DENSE_PPL}", xy=(DENSE_PPL * 1.04, 6), fontsize=8.5, color=MUTED, rotation=90)
    ax.axhline(0, color="#d55e00", linewidth=1.4)
    ax.axhline(50, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("half the model's ability gone", xy=(DENSE_PPL * 1.15, 52), fontsize=8.5, color=MUTED)

    ax.set_xscale("log")
    ax.set_xlabel("WikiText-2 perplexity (log scale)  —  'only 2x worse' is not benign", fontsize=10.5)
    ax.set_ylabel("ability retained vs dense (%)", fontsize=10.5)
    ax.set_title("Is perplexity a trustworthy proxy for capability?", fontsize=11, color=MUTED, pad=8)
    ax.set_ylim(-12, 112)
    ax.grid(alpha=0.22, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5)


def main():
    dense, rows = load()
    if dense is None:
        print("no dense baseline found -- run: python scripts/run_downstream.py --dense")
        return
    if not rows:
        print("no pruned downstream results yet")
        return
    print(f"dense: " + ", ".join(f"{t}={dense[t]:.4f}" for t in CHANCE))
    for r in rows:
        ret = {t: retained(r["accs"][t], dense[t], t) for t in CHANCE}
        s = ", ".join(f"{t}={ret[t]:.0f}%" for t in CHANCE if ret[t] is not None)
        print(f"  {r['method']:16} p{r['sparsity']:.2f} {r['policy']:12} ppl={r['ppl']} | retained {s}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    capability_panel(ax1, dense, rows)
    divergence_panel(ax2, dense, rows)
    fig.suptitle("Downstream capability under tile pruning — what perplexity hides",
                 fontsize=15, fontweight="bold", color=INK, y=1.0)
    fig.text(0.5, 0.945, "retained = (acc − chance) / (acc_dense − chance) · a broken model still scores "
                         "chance by guessing, so raw accuracy flatters it",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "downstream_capability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
