"""
Sensitivity-aware pruning policy (Rathore Strategy 5, Policy B).

Policy A (uniform) prunes every matrix at the same sparsity. Policy B instead spends
the SAME global tile budget unevenly: less on sensitive regions, more on redundant
ones. The budget matching is what makes the comparison fair -- both policies remove
approximately the same total number of tiles.

Pipeline:
  1. load_sensitivity()  - per (layer, matrix) damage from the screening runs
  2. classify()          - sensitive / moderate / robust
  3. expand_to_model()   - our screening measured only a few layers; map every layer
                           to its nearest measured layer's class
  4. allocate()          - solve for the scale that makes the tile-weighted mean
                           sparsity equal the target (bisection, respects a cap)
"""

import json
import glob
import os

SENSITIVE, MODERATE, ROBUST = "sensitive", "moderate", "robust"


def load_sensitivity(screen_dir, method="sparsegpt_recon", ref_ratio="0.20", dense_ppl=13.559):
    """{(layer, matrix): delta_ppl} from the screening JSONs of one method/sparsity."""
    sens = {}
    for f in glob.glob(os.path.join(screen_dir, f"{method}_p{ref_ratio}", "layer*.json")):
        d = json.load(open(f))
        for r in d["results"]:
            sens[(d["layer"], r["matrix"])] = r["perplexity"] - dense_ppl
    if not sens:
        raise ValueError(f"No screening data found for {method} p{ref_ratio} in {screen_dir}")
    return sens


def classify(sens, sensitive_thr=0.30, moderate_thr=0.05):
    """Bucket each measured (layer, matrix) by how much pruning it hurt."""
    out = {}
    for k, v in sens.items():
        if v >= sensitive_thr:
            out[k] = SENSITIVE
        elif v >= moderate_thr:
            out[k] = MODERATE
        else:
            out[k] = ROBUST
    return out


def nearest_measured(layer, measured):
    return min(measured, key=lambda m: abs(m - layer))


def expand_to_model(classes, all_layers, matrices, measured):
    """Assign a class to every (layer, matrix) in the model.

    The screening only measured a few representative layers, so each unmeasured layer
    inherits the class of the nearest measured layer (per matrix type).
    """
    full = {}
    for layer in all_layers:
        src = nearest_measured(layer, measured)
        for m in matrices:
            full[(layer, m)] = classes.get((src, m), ROBUST)
    return full


def allocate(classes_full, tiles, target_sparsity,
             mults=None, max_ratio=0.95, tol=1e-6):
    """Per-(layer,matrix) ratios whose TILE-WEIGHTED MEAN equals target_sparsity.

    tiles: {(layer, matrix): num_full_tiles}
    Returns ({(layer,matrix): ratio}, info dict).
    """
    if mults is None:
        mults = {SENSITIVE: 0.30, MODERATE: 1.00, ROBUST: 1.60}

    total_tiles = sum(tiles.values())
    budget = target_sparsity * total_tiles

    def removed(scale):
        return sum(min(mults[classes_full[k]] * scale, max_ratio) * n for k, n in tiles.items())

    # Bisect on the scale factor until the removed-tile count matches the budget.
    lo, hi = 0.0, max_ratio / min(mults.values()) + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if removed(mid) < budget:
            lo = mid
        else:
            hi = mid
        if abs(removed(mid) - budget) <= tol * total_tiles:
            break
    scale = (lo + hi) / 2

    ratios = {k: min(mults[classes_full[k]] * scale, max_ratio) for k in tiles}
    achieved = removed(scale) / total_tiles
    per_class = {}
    for c in (SENSITIVE, MODERATE, ROBUST):
        ks = [k for k in tiles if classes_full[k] == c]
        if ks:
            per_class[c] = {
                "ratio": min(mults[c] * scale, max_ratio),
                "matrices": len(ks),
                "tiles": sum(tiles[k] for k in ks),
            }
    return ratios, {"target": target_sparsity, "achieved": achieved, "scale": scale, "per_class": per_class}
