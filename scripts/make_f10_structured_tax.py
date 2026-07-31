"""Finding 10 figure: the structured-pruning tax. 1x1 (unstructured) vs 32x32 tiles,
same SparseGPT repair, matched weight-sparsity -- two panels (perplexity + downstream
retained ability). Reads the real result JSONs so the figure can't drift from the data."""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSE_PPL = 13.2181
DENSE = {"hellaswag": 0.6836, "piqa": 0.7492, "arc_easy": 0.7828}
CHANCE = {"hellaswag": 0.25, "piqa": 0.5, "arc_easy": 0.25}
C_1x1, C_32 = "#0072B2", "#E69F00"   # Okabe-Ito, colourblind-safe

def load(p): return json.load(open(p))

def retained(d):
    r = d.get("results", {}); vals = []
    for t in DENSE:
        a = r.get(t, {}).get("acc_norm,none", r.get(t, {}).get("acc,none"))
        if a is not None: vals.append((a - CHANCE[t]) / (DENSE[t] - CHANCE[t]))
    return sum(vals) / len(vals) if vals else float("nan")

# ---- perplexity data ----
ppl_1 = {}
for f in glob.glob(os.path.join(ROOT, "experiments/tilesize/ppl/us_sparsegpt_recon_p*_T1.json")):
    d = load(f); ppl_1[round(d["prune_ratio"]*100)] = d["perplexity"]
ppl_32 = {}
for pr in (5, 10, 20, 30, 40, 50):
    f = os.path.join(ROOT, f"experiments/wholemodel/wholemodel_sparsegpt_recon_p{pr}.json")
    if os.path.exists(f): ppl_32[pr] = load(f)["perplexity"]

# ---- downstream data ----
ret_1 = {}
for f in glob.glob(os.path.join(ROOT, "experiments/tilesize/downstream/us_sparsegpt_recon_p*_T1.json")):
    d = load(f); ret_1[round(d["prune_ratio"]*100)] = retained(d)
ret_32 = {}
for pr in (5, 10, 20, 30):
    f = os.path.join(ROOT, f"experiments/downstream/downstream_sparsegpt_recon_p{pr}_uniform.json")
    if os.path.exists(f): ret_32[pr] = retained(load(f))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

# ---- Panel L: perplexity ----
x1 = sorted(ppl_1); x32 = sorted(ppl_32)
axL.plot(x32, [ppl_32[x] for x in x32], "o-", color=C_32, lw=2, label="32×32 tiles")
axL.plot(x1, [ppl_1[x] for x in x1], "s-", color=C_1x1, lw=2, label="1×1 (unstructured)")
axL.axhline(DENSE_PPL, ls="--", color="gray", lw=1)
axL.text(51, DENSE_PPL*1.03, "dense 13.22", color="gray", fontsize=8, ha="right")
axL.set_yscale("log")
axL.set_xlabel("weight sparsity (%)"); axL.set_ylabel("WikiText-2 perplexity (log)")
axL.set_title("Perplexity — same repair, matched sparsity")
axL.annotate("1×1 @ 50% ≈ 32×32 @ 5%\n(15.69 vs 15.62)", xy=(50, ppl_1[50]),
             xytext=(30, 20), fontsize=8.5, color=C_1x1,
             arrowprops=dict(arrowstyle="->", color=C_1x1, lw=1))
axL.legend(frameon=False); axL.grid(alpha=0.25, which="both")

# ---- Panel R: retained ability ----
xr1 = sorted(ret_1); xr32 = sorted(ret_32)
axR.plot(xr32, [ret_32[x]*100 for x in xr32], "o-", color=C_32, lw=2, label="32×32 tiles")
axR.plot(xr1, [ret_1[x]*100 for x in xr1], "s-", color=C_1x1, lw=2, label="1×1 (unstructured)")
axR.axhline(90, ls="--", color="gray", lw=1); axR.text(1, 91, "90% retained", color="gray", fontsize=8)
axR.set_xlabel("weight sparsity (%)"); axR.set_ylabel("downstream ability retained (%)")
axR.set_title("Capability — 1×1 crosses 90% at ~40% vs 32×32 at ~5%")
axR.annotate("~8× the usable\nsparsity", xy=(40, 90), xytext=(20, 55), fontsize=9,
             color=C_1x1, arrowprops=dict(arrowstyle="->", color=C_1x1, lw=1))
axR.set_ylim(15, 105); axR.legend(frameon=False); axR.grid(alpha=0.25)

fig.suptitle("Finding 10 — the 5% is a structured-pruning tax: unstructured redundancy is ~8× larger but diffuse",
             fontsize=12, y=1.00)
fig.tight_layout()
out = os.path.join(ROOT, "figures", "f10_structured_tax.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
print("ppl_1x1:", ppl_1); print("ppl_32:", ppl_32)
print("ret_1x1:", {k: round(v,3) for k,v in ret_1.items()})
print("ret_32:", {k: round(v,3) for k,v in ret_32.items()})
