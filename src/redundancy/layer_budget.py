"""
Budget-matched layer-wide allocation -- Rathore's Wanda Strategy 3.

Uniform pruning forces every matrix in a layer to the same sparsity. W3 instead hands the whole
LAYER one tile budget and lets the allocation between its seven matrices fall out of the scores:

  1. Count the total tiles across all seven matrices.
  2. Define a total layer budget (e.g. 20% of those tiles).
  3. Rank or normalise the Wanda tile scores WITHIN their matrices.
  4. Select that many lowest-ranked candidates across the whole layer.
  5. Robust matrices then lose more tiles, sensitive ones fewer.

The allocation is an OUTPUT, not an input -- "the actual percentages must be determined by the
measured Wanda rankings and must not be assumed in advance".

Why normalisation is not optional, and why it cannot be rank-based
------------------------------------------------------------------
Raw Wanda scores are not comparable across matrices: |W_ij|*||X_:,j|| inherits each matrix's
weight and activation scale, so pooling raw scores mostly ranks matrices by their scale rather
than by how expendable their tiles are.

But pure RANK normalisation is degenerate. If each matrix's tiles are mapped to uniform
percentiles, then taking the globally-lowest N% returns exactly N% from every matrix -- i.e.
uniform pruning, the very baseline W3 exists to beat. It cannot produce a non-uniform allocation
even in principle.

So we scale-normalise: divide each matrix's tile scores by that matrix's mean tile score. Scores
then read as "how expendable is this tile relative to its own matrix", which IS comparable across
matrices, and a matrix with a heavier left tail rightly surrenders more tiles. `mode="raw"` is
kept for contrast.
"""

import numpy as np


def pooled_allocation(scores_by_matrix, budget_ratio, mode="mean"):
    """Pick the layer's lowest-ranked tiles across all matrices under one shared budget.

    scores_by_matrix : {matrix_name: list of (score, r, c)} -- Wanda tile scores per matrix
    budget_ratio     : fraction of the LAYER's total tiles to remove
    mode             : "mean" -> divide each matrix's scores by its own mean (default)
                       "raw"  -> pool untransformed scores (kept for contrast; scale-dominated)

    Returns (pruned_by_matrix, info) where pruned_by_matrix is {matrix: [(r, c), ...]} and info
    reports the allocation that emerged, per matrix.
    """
    pooled = []
    for m, scored in scores_by_matrix.items():
        vals = np.array([s for s, _, _ in scored], dtype=np.float64)
        if mode == "mean":
            denom = vals.mean()
            norm = vals / denom if denom > 0 else vals
        elif mode == "raw":
            norm = vals
        else:
            raise ValueError(f"unknown mode {mode!r}")
        for (nv, (_, r, c)) in zip(norm, scored):
            pooled.append((float(nv), m, r, c))

    total = len(pooled)
    n_prune = int(total * budget_ratio)
    pooled.sort(key=lambda x: x[0])
    chosen = pooled[:n_prune]

    pruned_by_matrix = {m: [] for m in scores_by_matrix}
    for _, m, r, c in chosen:
        pruned_by_matrix[m].append((r, c))

    info = {
        "mode": mode,
        "budget_ratio": budget_ratio,
        "layer_total_tiles": total,
        "layer_pruned_tiles": n_prune,
        "achieved_ratio": n_prune / total if total else 0.0,
        "per_matrix": {
            m: {
                "tiles": len(scores_by_matrix[m]),
                "pruned": len(pruned_by_matrix[m]),
                # the emergent sparsity -- uniform would make every one of these == budget_ratio
                "ratio": len(pruned_by_matrix[m]) / len(scores_by_matrix[m])
                if scores_by_matrix[m] else 0.0,
            }
            for m in scores_by_matrix
        },
    }
    return pruned_by_matrix, info
