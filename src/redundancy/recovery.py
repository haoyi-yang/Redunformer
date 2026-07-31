"""
Reconstruction utilities for tile pruning (SparseGPT, report eq. 23).

Removing a tile T zeros the weight block W[rows_of_T, cols_of_T]. The remaining
weights of the affected output rows can then be updated to best reconstruct the
original layer output XW^T, using the inverse input-Gram matrix Hinv.

The reconstruction objective ||XW^T - X W_hat^T||_F^2 decomposes over output
rows, so each row is reconstructed independently and exactly. Output rows in the
same tile-row-block share the same set of pruned columns P, so they are solved
together. For that block, with S = (Hinv[P, P])^-1 and update operator
A = Hinv[:, P] @ S, the optimal reconstructed weights are

    W_hat[block, :] = W[block, :] - W[block, P] @ A^T

which sets W_hat[block, P] = 0 (all the block's pruned tiles are removed) and
adjusts the surviving columns to compensate. Because every row-block is solved
once against its full pruned-column set, previously pruned tiles are never
written back into -- the result is the exact least-squares reconstruction for
the chosen pruning mask.

The only remaining "one-shot" aspect is that Hinv comes from the dense model:
it is not re-derived after other matrices/layers are pruned (report sec. 4.8,
"sequential dependency"), which is standard for one-shot post-training pruning.
"""

import torch


def _damped_inverse(H, damp):
    cols = H.shape[0]
    diag_mean = torch.diag(H).mean()
    Hd = H.clone()
    idx = torch.arange(cols, device=H.device)
    Hd[idx, idx] += damp * diag_mean
    return torch.linalg.inv(Hd)


def reconstruct_prune_tiles(weight, hessian, tile_size, prune_ratio, damp=1e-2, eps=1e-8):
    """Rank tiles by SparseGPT reconstructed error (eq. 23), prune the
    lowest-error fraction, and apply the exact compensating weight updates in
    place (jointly per output-row-block).

    weight   : the target nn.Linear weight tensor (modified in place)
    hessian  : the calibration Gram matrix H = X^T X for this matrix

    Returns (num_tiles, num_pruned).
    """
    Wf = weight.detach().float()
    H = hessian.detach().float().to(Wf.device)
    rows, cols = Wf.shape

    Hinv = _damped_inverse(H, damp)
    denom = (Wf @ H * Wf).sum().item() + eps

    # 1. Score every tile independently by its eq. 23 reconstructed error.
    #    S depends only on the tile's column block, so compute it once per column.
    schur_by_col = {}
    for c in range(0, cols, tile_size):
        if c + tile_size <= cols:
            schur_by_col[c] = torch.linalg.inv(Hinv[c:c + tile_size, c:c + tile_size])

    scored = []
    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            if r + tile_size <= rows and c + tile_size <= cols:
                Wt = Wf[r:r + tile_size, c:c + tile_size]
                num = (Wt @ schur_by_col[c] * Wt).sum().item()
                scored.append((max(num, 0.0) / denom, r, c))

    scored.sort(key=lambda x: x[0])
    num_prune = int(len(scored) * prune_ratio)
    pruned = scored[:num_prune]

    # 2. Group pruned tiles by row-block and reconstruct each block jointly over
    #    the union of its pruned columns (exact for the chosen mask).
    pruned_cols_by_row = {}
    for _, r, c in pruned:
        pruned_cols_by_row.setdefault(r, []).extend(range(c, c + tile_size))

    with torch.no_grad():
        for r, pcols in pruned_cols_by_row.items():
            idx = torch.tensor(sorted(pcols), device=Wf.device)
            S = torch.linalg.inv(Hinv[idx][:, idx])           # (|P|, |P|)
            A = Hinv[:, idx] @ S                               # (in, |P|)
            block = weight[r:r + tile_size, idx].float()       # (tile_size, |P|)
            weight[r:r + tile_size, :] -= (block @ A.t()).to(weight.dtype)

    return len(scored), num_prune
