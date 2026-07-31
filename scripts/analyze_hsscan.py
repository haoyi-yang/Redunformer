"""
Metric-specificity test (Seb's HellaSwag-selection experiment).

Compares the layers where pruning o_proj *improves HellaSwag accuracy* against the layers where it
*improves WikiText perplexity*. If the two beneficial sets differ, the "free lunch" is
metric-specific -- each metric manufactures its own beneficial layers -- which is the constructive
proof that the o_proj improvement is overfitting to whatever you measured, not real redundancy.

  WikiText-improving @40% (from the cluster scan, delta_ppl < 0): {17,18,19,20,21,32,33,34,35}
  HellaSwag-improving @40%: computed here from experiments/hsscan/hs_L*.json (acc > dense)

Reads the per-layer HellaSwag ablations. Prints the two sets, their overlap, and the verdict.
No GPU.
"""

import json
import glob
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DENSE_HS = 0.6836          # dense HellaSwag acc_norm (measured)
DENSE_HS_SE = 0.0046       # its stderr
CLUSTER = [15, 16, 17, 18, 19, 20, 21, 29, 30, 31, 32, 33, 34, 35]
WIKITEXT_IMPROVING_40 = {17, 18, 19, 20, 21, 32, 33, 34, 35}


def load_hs():
    """{layer: hellaswag acc_norm} from the per-layer o_proj@40% ablations."""
    out = {}
    for f in glob.glob("experiments/hsscan/hs_L*.json"):
        d = json.load(open(f))
        v = d.get("results", {}).get("hellaswag", {})
        acc = v.get("acc_norm,none", v.get("acc,none"))
        L = int(os.path.basename(f).split("_L")[1].split(".")[0])
        if acc is not None:
            out[L] = acc
    return out


def main():
    hs = load_hs()
    if not hs:
        print("no hsscan results yet")
        return
    print(f"dense HellaSwag acc_norm = {DENSE_HS:.4f} (±{DENSE_HS_SE:.4f})\n")
    print(f"{'layer':>6} {'HS acc':>8} {'delta':>9} {'sigma':>7}   {'improves HS?':>13}   WikiText@40 improves?")
    hs_improving = set()
    for L in CLUSTER:
        if L not in hs:
            print(f"{L:>6}   (missing)")
            continue
        a = hs[L]; delta = a - DENSE_HS; z = delta / DENSE_HS_SE
        improves = delta > 0
        if improves:
            hs_improving.add(L)
        wt = "yes" if L in WIKITEXT_IMPROVING_40 else "no"
        mark = "  IMPROVES" if z > 1 else ("  (up, ns)" if improves else "  down")
        print(f"{L:>6} {a:>8.4f} {delta:>+9.4f} {z:>+7.1f}   {mark:>13}   {wt}")

    print()
    print(f"HellaSwag-improving set: {sorted(hs_improving)}")
    print(f"WikiText-improving set : {sorted(WIKITEXT_IMPROVING_40)}")
    both = hs_improving & WIKITEXT_IMPROVING_40
    only_hs = hs_improving - WIKITEXT_IMPROVING_40
    only_wt = WIKITEXT_IMPROVING_40 - hs_improving
    print(f"  improve BOTH metrics : {sorted(both)}")
    print(f"  only HellaSwag       : {sorted(only_hs)}")
    print(f"  only WikiText        : {sorted(only_wt)}")
    print()
    n = len(CLUSTER)
    agree = sum(1 for L in CLUSTER if L in hs
                and ((L in hs_improving) == (L in WIKITEXT_IMPROVING_40)))
    print(f"the two metrics agree on {agree}/{n} layers")
    if len(only_hs) + len(only_wt) >= 3:
        print("VERDICT: the beneficial sets DIFFER -> the o_proj 'improvement' is metric-specific")
        print("         (overfitting to whatever you measure, not robust redundancy) -- finding 2b confirmed.")
    else:
        print("VERDICT: the sets largely AGREE -> the improvement may be a genuine (metric-general) property")
        print("         -- would warrant a clean hold-out follow-up.")


if __name__ == "__main__":
    main()
