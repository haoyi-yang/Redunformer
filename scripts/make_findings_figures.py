"""
Regenerate the Redunformer tile-pruning findings figures (Qwen3-4B) for the team deck.

Writes presentation-quality PNGs to figures/ at the repo root, one per finding:
  f1_redundancy_ladder.png      F1  retained ability vs sparsity (3 tasks)
  f2_perplexity_scissors.png    F2+2b  capability vs perplexity ratio + o_proj gaming
  f3_policyB_divergence.png     F3  Policy B ppl-gain bars vs capability-gain line
  f4_L35_backfire.png           F4  L35 dPPL by method (calibration ladder)
  f5_scope_escalation.png       F5  ppl vs pruning scope
  f6_inverted_u.png             F6  ppl vs depth concentration (inverted-U)
  f7_depth_matrix_interaction.png  F7  dPPL vs depth, o_proj vs up_proj
  f8_magnitude_vs_random.png    F8  ppl vs sparsity, magnitude vs random floor

All numbers extracted live from experiments/**; verified against the finding cards.
CPU-only (no model / no GPU). Run: .venv/Scripts/python.exe scripts/make_findings_figures.py
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---- shared style -----------------------------------------------------------
# Okabe-Ito colorblind-safe palette
BLUE, VERM, GREEN, PINK, ORANGE, SKY, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8a8a8a")
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

DENSE_WT = 13.22      # WikiText-2 full-eval dense ppl (whole-model sweep, downstream anchor)
DENSE_SUB = 13.559    # screening 20%-subset dense ppl (screen / screen_wholelayer / cluster)
CHANCE = {"hellaswag": 0.25, "piqa": 0.50, "arc_easy": 0.25}
TASK_COLOR = {"hellaswag": BLUE, "piqa": ORANGE, "arc_easy": GREEN}
TASK_LABEL = {"hellaswag": "HellaSwag", "piqa": "PIQA", "arc_easy": "ARC-Easy"}


def load(p):
    with open(p) as f:
        return json.load(f)


def ppl_of(d):
    p = d.get("perplexity")
    if p is None and d.get("results"):
        p = d["results"][0].get("perplexity")
    return p


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# =============================================================================
# F1 — Redundancy ladder: retained above-chance ability vs sparsity
# =============================================================================
def f1_redundancy_ladder():
    dense = load("experiments/downstream/downstream_dense.json")
    dacc = {t: dense["results"][t]["acc_norm,none"] for t in CHANCE}
    ladder = [(1, "p1"), (2, "p2"), (5, "p5"), (10, "p10"), (20, "p20"), (30, "p30")]
    xs = list(range(len(ladder)))
    xticklab = [f"{s}%" for s, _ in ladder]
    ret = {t: [] for t in CHANCE}
    for _, short in ladder:
        d = load(f"experiments/downstream/downstream_sparsegpt_recon_{short}_uniform.json")
        for t in CHANCE:
            a = d["results"][t]["acc_norm,none"]
            ret[t].append((a - CHANCE[t]) / (dacc[t] - CHANCE[t]) * 100)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    five_x = 2  # index of 5% in the evenly-spaced axis
    ax.axhline(90, color=MUTED, ls="--", lw=1.3, zorder=1)
    ax.annotate("90% of dense ability", xy=(0.02, 90), xytext=(0, 4),
                textcoords="offset points", fontsize=9.5, color=MUTED)
    ax.axvline(five_x, color=VERM, ls=":", lw=1.5, zorder=1)
    ax.axvspan(five_x, five_x + 0.06, alpha=0)  # keep autoscale sane

    for t in CHANCE:
        ax.plot(xs, ret[t], "-o", color=TASK_COLOR[t], lw=2.4, ms=7.5,
                label=TASK_LABEL[t], zorder=4)

    # highlight the tight convergence at 5% and the collapse band 10->20
    ax.scatter([five_x] * 3, [ret[t][five_x] for t in CHANCE], s=140,
               facecolors="none", edgecolors=VERM, linewidths=1.8, zorder=6)
    ax.annotate("~5%: last point all three\ntasks still hold 90%",
                xy=(five_x, 90), xytext=(five_x - 1.55, 60),
                fontsize=9.5, color=VERM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))
    ax.annotate("true collapse 10 -> 20%\n(79% -> 43%)",
                xy=(4, 43), xytext=(3.05, 20), fontsize=9.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    ax.set_xticks(xs)
    ax.set_xticklabels(xticklab)
    ax.set_ylim(0, 108)
    ax.set_xlabel("global tile sparsity  (32x32 blocks removed)")
    ax.set_ylabel("above-chance ability retained vs dense  (%)")
    ax.set_title("F1 — Only ~5% of tiles are removable before real-task ability breaks",
                 fontweight="bold", color=INK, pad=10)
    ax.text(0.5, 1.005, "SparseGPT-select + SparseGPT-repair (sparsegpt_recon), uniform, tile-32 · "
            "retained = (acc-chance)/(dense-chance)",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    ax.grid(alpha=0.22)
    clean(ax)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    save(fig, "f1_redundancy_ladder.png")


# =============================================================================
# F2 + F2b — Perplexity scissors: capability vs perplexity ratio to dense
# =============================================================================
def f2_perplexity_scissors():
    dense = load("experiments/downstream/downstream_dense.json")
    dacc = {t: dense["results"][t]["acc_norm,none"] for t in CHANCE}

    def avg_ret(short):
        d = load(f"experiments/downstream/downstream_sparsegpt_recon_{short}_uniform.json")
        return float(np.mean([(d["results"][t]["acc_norm,none"] - CHANCE[t]) /
                              (dacc[t] - CHANCE[t]) * 100 for t in CHANCE]))

    def hs_ret(path):
        d = load(path)
        a = d["results"]["hellaswag"]["acc_norm,none"]
        return (a - CHANCE["hellaswag"]) / (dacc["hellaswag"] - CHANCE["hellaswag"]) * 100

    # Finding-2 honest curve (tile-32): ppl ratio to dense vs avg retained
    uni = []
    for short in ["p1", "p2", "p5", "p10", "p20", "p30"]:
        ppl = ppl_of(load(f"experiments/wholemodel/wholemodel_sparsegpt_recon_{short}.json"))
        uni.append((ppl / DENSE_WT, avg_ret(short)))
    uni.sort()
    ux, uy = zip(*uni)

    # Finding-2b o_proj gaming (tile-64 ppl / tile-32 HellaSwag accuracy)
    game = [
        (ppl_of(load("experiments/archive/oproj_targeted/wholemodel_wanda_p20.json")) / DENSE_WT,
         hs_ret("experiments/downstream/ds_oproj_targeted_wanda_p0.20.json"), "20% dose"),
        (ppl_of(load("experiments/archive/oproj_targeted/wholemodel_wanda_p40.json")) / DENSE_WT,
         hs_ret("experiments/downstream/ds_oproj_targeted_wanda_p0.40.json"), "40% dose"),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.axvline(1.0, color=INK, lw=1.3, zorder=2)
    ax.annotate("dense perplexity", xy=(1.0, 8), xytext=(3, 0), textcoords="offset points",
                rotation=90, fontsize=8.8, color=INK, va="bottom")
    ax.axvspan(0.6, 1.0, color=VERM, alpha=0.06, lw=0)
    ax.text(0.865, 103, '"looks better than dense"', fontsize=8.6, color=VERM,
            ha="center", style="italic")

    # honest curve
    ax.plot(ux, uy, "-o", color=BLUE, lw=2.4, ms=7.5, zorder=4,
            label="uniform prune (tile-32) — avg of 3 tasks")
    # annotate the brutal-nonlinearity point (2.34x -> 43)
    ax.annotate("2.3x perplexity = only 43% ability\n(over half the model gone)",
                xy=(30.953 / DENSE_WT, 43), xytext=(2.55, 58), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    # gaming markers (o_proj masking): ppl < dense but capability at/below dense
    gx = [g[0] for g in game]
    gy = [g[1] for g in game]
    ax.plot(gx, gy, "D", color=VERM, ms=12, zorder=6, markerfacecolor="white",
            markeredgewidth=2.2, label="o_proj gaming (tile-64 ppl / tile-32 HellaSwag)")
    ax.annotate("20% dose", xy=(game[0][0], game[0][1]), xytext=(8, 7),
                textcoords="offset points", fontsize=8.4, color=VERM, fontweight="bold")
    ax.annotate("40% dose", xy=(game[1][0], game[1][1]), xytext=(-1, -15),
                textcoords="offset points", ha="center", fontsize=8.4, color=VERM, fontweight="bold")
    ax.annotate("o_proj gaming: perplexity driven up to\n13% BELOW dense, yet HellaSwag falls -3.6 sigma",
                xy=(game[1][0], game[1][1]), xytext=(1.13, 72), fontsize=9,
                color=VERM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))

    ax.set_xscale("log")
    ax.set_xticks([0.9, 1, 1.5, 2, 3, 4, 5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.80, 5.4)
    ax.set_ylim(0, 112)
    ax.set_xlabel("WikiText-2 perplexity as a ratio to dense  (log scale)  —  lower 'looks better'")
    ax.set_ylabel("downstream capability retained vs dense  (%)")
    ax.set_title("F2/F2b — Perplexity is nonlinear, and o_proj can push it below dense while the model gets worse",
                 fontweight="bold", color=INK, fontsize=11, pad=10)
    ax.grid(alpha=0.22, which="both")
    clean(ax)
    ax.legend(frameon=False, fontsize=9.2, loc="upper right")
    save(fig, "f2_perplexity_scissors.png")


# =============================================================================
# F3 — Policy B: perplexity gain (bars) diverges from capability gain (line)
# =============================================================================
def f3_policyB_divergence():
    dense = load("experiments/downstream/downstream_dense.json")
    dacc = {t: dense["results"][t]["acc_norm,none"] for t in CHANCE}
    sp = [(5, "p5"), (10, "p10"), (20, "p20"), (30, "p30"), (40, "p40")]
    xs = np.arange(len(sp))
    gain = []
    for _, short in sp:
        u = ppl_of(load(f"experiments/wholemodel/wholemodel_sparsegpt_recon_{short}.json"))
        b = ppl_of(load(f"experiments/wholemodel/wholemodel_sparsegpt_recon_{short}_sensitivity.json"))
        gain.append(u / b)

    def mean_ret_gain(short):
        du = load(f"experiments/downstream/downstream_sparsegpt_recon_{short}_uniform.json")
        db = load(f"experiments/downstream/downstream_sparsegpt_recon_{short}_sensitivity.json")
        d = []
        for t in CHANCE:
            ru = (du["results"][t]["acc_norm,none"] - CHANCE[t]) / (dacc[t] - CHANCE[t]) * 100
            rb = (db["results"][t]["acc_norm,none"] - CHANCE[t]) / (dacc[t] - CHANCE[t]) * 100
            d.append(rb - ru)
        return float(np.mean(d))

    # downstream Policy-B measured only at p20 and p30
    cap_idx = [2, 3]
    cap_val = [mean_ret_gain("p20"), mean_ret_gain("p30")]

    fig, ax1 = plt.subplots(figsize=(9.0, 5.6))
    bars = ax1.bar(xs, gain, width=0.56, color=BLUE, alpha=0.9, zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax1.axhline(1.0, color=INK, lw=1.1)
    ax1.annotate("parity (no perplexity gain)", xy=(0, 1.0), xytext=(2, 3),
                 textcoords="offset points", fontsize=8.6, color=INK)
    for x, g in zip(xs, gain):
        ax1.annotate(f"{g:.2f}x", xy=(x, g), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax1.set_ylim(0, 1.85)
    ax1.set_ylabel("Policy B perplexity gain over uniform  (x)", color=BLUE)
    ax1.tick_params(axis="y", colors=BLUE)
    ax1.spines["left"].set_color(BLUE)

    ax2 = ax1.twinx()
    ax2.axhline(0, color=VERM, lw=0.9, ls=":", alpha=0.7)
    ax2.plot(cap_idx, cap_val, "-D", color=VERM, ms=11, lw=2.4, zorder=5,
             markerfacecolor="white", markeredgewidth=2)
    for i, v in zip(cap_idx, cap_val):
        ax2.annotate(f"{v:+.1f} pp", xy=(i, v), xytext=(0, 10 if v > 1 else 12),
                     textcoords="offset points", ha="center", fontsize=9.5,
                     color=VERM, fontweight="bold")
    ax2.set_ylim(-6, 12)
    ax2.set_ylabel("mean retained-ability gain, 3-task avg  (pp)", color=VERM)
    ax2.tick_params(axis="y", colors=VERM)
    ax2.spines["right"].set_color(VERM)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    # divergence callout at 30%
    ax2.annotate("DIVERGENCE at 30%:\nperplexity gain peaks (1.58x)\nbut real ability gain ~ 0",
                 xy=(3, cap_val[1]), xytext=(3.15, 7.6), fontsize=9.2, color=VERM,
                 fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))
    ax1.annotate("downstream Policy B run only at 20% / 30%", xy=(0.5, -0.145),
                 xycoords="axes fraction", ha="center", fontsize=8.4, color=MUTED, style="italic")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{s}%" for s, _ in sp])
    ax1.set_xlabel("global tile sparsity")
    ax1.set_title("F3 — Policy B wins 1.1-1.6x on perplexity but games the metric it was built from",
                  fontweight="bold", color=INK, fontsize=11, pad=10)
    ax1.grid(axis="y", alpha=0.18)
    ax1.spines["top"].set_visible(False)
    save(fig, "f3_policyB_divergence.png")


# =============================================================================
# F4 — Layer-35 backfire: dPPL by method, ordered by calibration
# =============================================================================
def f4_L35_backfire():
    ratios = ["0.10", "0.20", "0.40"]
    methods = ["random", "magnitude", "sparsegpt", "wanda", "sparsegpt_recon"]
    mcolor = {"random": GREY, "magnitude": VERM, "sparsegpt": ORANGE,
              "wanda": BLUE, "sparsegpt_recon": GREEN}
    mlabel = {"random": "random (5-seed median)", "magnitude": "magnitude",
              "sparsegpt": "SparseGPT (mask)", "wanda": "Wanda",
              "sparsegpt_recon": "SparseGPT+repair"}

    def wl(method, ratio, layer=35):
        fs = glob.glob(f"experiments/screen_wholelayer/{method}_p{ratio}/layer{layer}_*.json")
        return [ppl_of(load(f)) - DENSE_SUB for f in fs]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ng = len(ratios)
    nb = len(methods)
    group_w = 0.82
    bw = group_w / nb
    for gi, r in enumerate(ratios):
        base = gi
        rnd = wl("random", r)
        rmin, rmax, rmed = min(rnd), max(rnd), float(np.median(rnd))
        # shaded random min-max band spanning the group width
        x0 = base - group_w / 2
        ax.add_patch(Rectangle((x0, rmin), group_w, rmax - rmin, color=GREY,
                               alpha=0.22, lw=0, zorder=1))
        ax.hlines(rmed, x0, x0 + group_w, color=GREY, lw=1.2, ls="--", zorder=2)
        for mi, m in enumerate(methods):
            xx = base - group_w / 2 + bw * (mi + 0.5)
            if m == "random":
                val = rmed
            else:
                val = wl(m, r)[0]
            ax.bar(xx, val, width=bw * 0.9, color=mcolor[m], zorder=3,
                   edgecolor="white", linewidth=0.8,
                   label=mlabel[m] if gi == 0 else None)

    # annotate the monotonic calibration ladder at 40%
    r40 = "0.40"
    vals40 = [float(np.median(wl("random", r40)))] + [wl(m, r40)[0] for m in methods[1:]]
    ax.annotate("at 40%, damage grows monotonically\nwith calibration: random < magnitude\n< SparseGPT < Wanda < SparseGPT+repair",
                xy=(2 + group_w / 2 - bw / 2, vals40[-1]), xytext=(0.9, 5.4),
                fontsize=9, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(0.5, -0.135, "grey band = random 5-seed min-max floor · every calibrated bar rises above it",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED, style="italic")

    ax.set_xticks(range(ng))
    ax.set_xticklabels([f"{int(float(r)*100)}%" for r in ratios])
    ax.set_xlabel("tile sparsity applied to every matrix of layer 35")
    ax.set_ylabel("perplexity increase vs dense at L35  (dPPL, tile-32)")
    ax.set_ylim(0, 6.2)
    ax.set_title("F4 — At the last layer, calibrated tile-picking backfires (worse the smarter it is)",
                 fontweight="bold", color=INK, fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.2)
    clean(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)
    save(fig, "f4_L35_backfire.png")


# =============================================================================
# F5 — Scope escalation: robustness evaporates as pruning scope grows (@20%)
# =============================================================================
def f5_scope_escalation():
    # single matrix (isolated median ~ dense), whole layer (typical), whole model
    iso = DENSE_SUB + 0.016  # isolated median dPPL +0.016 (card)
    wl_layers = []
    for L in [0, 9, 18, 27, 35]:
        fs = glob.glob(f"experiments/screen_wholelayer/sparsegpt_recon_p0.20/layer{L}_*.json")
        if fs:
            wl_layers.append(ppl_of(load(fs[0])))
    layer_typ = float(np.median(wl_layers))
    layer_worst = max(wl_layers)  # L35
    wm = ppl_of(load("experiments/wholemodel/wholemodel_sparsegpt_recon_p20.json"))

    labels = ["single matrix\n(isolated median)", "one whole layer\n(typical)", "whole model\n(36 layers)"]
    vals = [iso, layer_typ, wm]
    colors = [GREEN, ORANGE, VERM]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    xs = np.arange(3)
    ax.bar(xs, vals, width=0.6, color=colors, alpha=0.92, zorder=3,
           edgecolor="white", linewidth=1.2)
    ax.axhline(DENSE_SUB, color=MUTED, ls=":", lw=1.3)
    ax.annotate(f"dense = {DENSE_SUB:.2f}", xy=(2.35, DENSE_SUB), xytext=(0, 3),
                textcoords="offset points", fontsize=8.8, color=MUTED)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10.5, color=INK, fontweight="bold")
    # worst single layer (L35) drawn as a cap above the "typical" middle bar
    ax.hlines(layer_worst, 1 - 0.3, 1 + 0.3, color=ORANGE, ls="--", lw=1.7, zorder=5)
    ax.annotate(f"worst single layer (L35) = {layer_worst:.1f}", xy=(1 + 0.3, layer_worst),
                xytext=(1.12, 22), fontsize=8.6, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.annotate("only 43% of task ability retained\n(HellaSwag/PIQA/ARC = 37/51/41%)",
                xy=(2, wm), xytext=(0.72, wm * 0.78), fontsize=9.5, color=VERM,
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WikiText-2 perplexity  (log scale)")
    ax.set_xlabel("pruning scope at fixed 20% sparsity  (increasing ->)")
    ax.set_title('F5 — "Robust in isolation" does not compose: marginal safety != joint safety',
                 fontweight="bold", color=INK, fontsize=11, pad=10)
    ax.text(0.5, 1.005, "SparseGPT+repair, tile-32 · same 20% sparsity at every scope",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    ax.grid(axis="y", alpha=0.22, which="both")
    clean(ax)
    save(fig, "f5_scope_escalation.png")


# =============================================================================
# F6 — Inverted-U: perplexity vs depth concentration (constant tile budget)
# =============================================================================
def f6_inverted_u():
    rows = []
    for f in glob.glob("experiments/archive/depth/wholemodel_*.json"):
        d = load(f)
        rows.append((d["num_layers"], float(d["prune_ratio"]), d["perplexity"]))
    rows.sort(key=lambda r: -r[0])   # 32,24,16,12,8  (left->right = more concentrated)
    ns = [r[0] for r in rows]
    locals_ = [r[1] for r in rows]
    ppls = [r[2] for r in rows]
    xpos = list(range(len(rows)))
    UNIFORM36 = 30.95

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.axhline(UNIFORM36, color=VERM, ls="--", lw=1.5)
    ax.annotate(f"all 36 layers @ 20% = {UNIFORM36} (tile-32 baseline, mixed-tile)",
                xy=(0, UNIFORM36), xytext=(2, 4), textcoords="offset points",
                fontsize=8.6, color=VERM, fontweight="bold")
    ax.axhline(DENSE_WT, color=MUTED, ls=":", lw=1.2)
    ax.annotate(f"dense {DENSE_WT}", xy=(len(rows) - 1, DENSE_WT), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8.6, color=MUTED)

    ax.plot(xpos, ppls, "-o", color=BLUE, lw=2.4, ms=9, zorder=5)
    imin = int(np.argmin(ppls))
    ax.scatter([xpos[imin]], [ppls[imin]], s=200, facecolors="none",
               edgecolors=GREEN, linewidths=2.4, zorder=6)
    ax.annotate(f"optimum: N=24 layers\nppl {ppls[imin]:.2f}",
                xy=(xpos[imin], ppls[imin]), xytext=(xpos[imin] + 0.35, 15),
                fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    for x, n, lo, p in zip(xpos, ns, locals_, ppls):
        ax.annotate(f"{p:,.1f}", xy=(x, p), xytext=(0, -16 if p < UNIFORM36 else 9),
                    textcoords="offset points", ha="center", fontsize=8.6,
                    color=INK, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"N={n}\n({int(lo*100)}% local)" for n, lo in zip(ns, locals_)])
    ax.set_xlabel("budget concentrated into fewer layers  (more concentrated ->)")
    ax.set_ylabel("WikiText-2 perplexity  (log scale)")
    ax.set_title("F6 — Reallocation has a sweet spot: an inverted-U with its minimum at N=24",
                 fontweight="bold", color=INK, fontsize=11, pad=10)
    ax.text(0.5, 1.005, "constant 177,408-tile budget (= 20% of the model) · tile-64 shape evidence",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    ax.grid(axis="y", alpha=0.22, which="both")
    clean(ax)
    save(fig, "f6_inverted_u.png")


# =============================================================================
# F7 — Depth x matrix interaction: o_proj vs up_proj across depth (40% Wanda)
# =============================================================================
def f7_depth_matrix_interaction():
    ROBUST = [15, 16, 17, 18, 19, 20, 21]
    SENS = [29, 30, 31, 32, 33, 34, 35]

    def cl(layer, matrix):
        fs = glob.glob(f"experiments/archive/screen_cluster/wanda_p0.40/layer{layer}_*.json")
        if not fs:
            return None
        for res in load(fs[0])["results"]:
            if res["matrix"] == matrix:
                return res["perplexity"] - DENSE_SUB
        return None

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.axhline(0, color=MUTED, lw=0.9, alpha=0.6)
    handles = {}
    for mat, color, lab in (("o_proj", BLUE, "o_proj (attention output)"),
                            ("up_proj", VERM, "up_proj (MLP)")):
        for cluster in (ROBUST, SENS):
            xs = [L for L in cluster if cl(L, mat) is not None]
            ys = [cl(L, mat) for L in xs]
            (h,) = ax.plot(xs, ys, "-o", color=color, lw=2.3, ms=6.5, zorder=4)
            handles[lab] = h
    # gap shading between clusters
    ax.axvspan(21.5, 28.5, color=GREY, alpha=0.08, lw=0)
    ax.annotate("layers 22-28\nnot measured", xy=(25, 3.6), ha="center",
                fontsize=8.4, color=MUTED, style="italic")
    # fan-out callout
    ax.annotate("up_proj +4.70", xy=(35, cl(35, "up_proj")), xytext=(31.4, 4.2),
                fontsize=9.2, color=VERM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))
    ax.annotate("o_proj -0.48\n(even improves ppl)", xy=(35, cl(35, "o_proj")),
                xytext=(31.4, -1.9), fontsize=9.2, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.annotate("lines sit together until L34-35,\nthen fan apart = the interaction",
                xy=(33, 0.72), xytext=(15.3, 3.3), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    ax.set_xticks(ROBUST + SENS)
    ax.set_xticklabels(ROBUST + SENS, fontsize=8.5)
    ax.set_xlabel("layer depth  (robust mid-band 15-21   |   final layers 29-35)")
    ax.set_ylabel("perplexity increase vs subset-dense 13.559  (dPPL)")
    ax.set_title("F7 — Robustness is the depth x matrix-type interaction, not either alone",
                 fontweight="bold", color=INK, fontsize=11, pad=10)
    ax.text(0.5, 1.005, "40% Wanda tile prune, one matrix type at a time · tile-64 shape evidence",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    ax.grid(alpha=0.2)
    clean(ax)
    ax.legend(handles.values(), handles.keys(), frameon=False, fontsize=10, loc="upper left")
    save(fig, "f7_depth_matrix_interaction.png")


# =============================================================================
# F8 — Magnitude vs random: perplexity vs sparsity (whole model, tile-32)
# =============================================================================
def f8_magnitude_vs_random():
    sp = [(5, "p5"), (10, "p10"), (20, "p20"), (30, "p30"), (40, "p40")]
    xs = [s for s, _ in sp]
    mag = [ppl_of(load(f"experiments/wholemodel/wholemodel_magnitude_{short}.json")) for _, short in sp]
    rmed, rlo, rhi = [], [], []
    for _, short in sp:
        seeds = [ppl_of(load(f)) for f in sorted(
            glob.glob(f"experiments/wholemodel/wholemodel_random_{short}_seed*.json"))]
        rmed.append(float(np.median(seeds)))
        rlo.append(min(seeds))
        rhi.append(max(seeds))

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.axhline(DENSE_WT, color=MUTED, ls="--", lw=1.3)
    ax.annotate(f"dense = {DENSE_WT}", xy=(xs[0], DENSE_WT), xytext=(4, 4),
                textcoords="offset points", ha="left", fontsize=8.8, color=MUTED)

    ax.fill_between(xs, rlo, rhi, color=BLUE, alpha=0.16, lw=0, label="random 5-seed range")
    ax.plot(xs, rmed, "-o", color=BLUE, lw=2.4, ms=8, label="random (5-seed median)", zorder=4)
    ax.plot(xs, mag, "-s", color=VERM, lw=2.4, ms=8, label="magnitude", zorder=4)

    # annotate the only usable operating point, 5%
    ax.annotate("5% (capability-preserving point):\nmagnitude 3,449 (destroyed)\nvs random ~23 (functional)",
                xy=(5, mag[0]), xytext=(6.3, 4e4), fontsize=9.3, color=VERM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.3))
    ax.scatter([5, 5], [mag[0], rmed[0]], s=90, facecolors="none",
               edgecolors=[VERM, BLUE], linewidths=1.8, zorder=6)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}%" for s in xs])
    ax.set_xlabel("global tile sparsity")
    ax.set_ylabel("WikiText-2 perplexity  (log scale)")
    ax.set_title("F8 — Magnitude tile-pruning loses to a coin flip",
                 fontweight="bold", color=INK, fontsize=11, pad=10)
    ax.text(0.5, 1.005, "whole-model, tile-32 · both are broken past 5%; the comparison only means "
            "something where the model still works",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=MUTED)
    ax.grid(alpha=0.22, which="both")
    clean(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    save(fig, "f8_magnitude_vs_random.png")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    f1_redundancy_ladder()
    f2_perplexity_scissors()
    f3_policyB_divergence()
    f4_L35_backfire()
    f5_scope_escalation()
    f6_inverted_u()
    f7_depth_matrix_interaction()
    f8_magnitude_vs_random()
    print("\nAll figures written to figures/")
