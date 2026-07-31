"""
Tile-level importance metrics adapted from Wanda and SparseGPT.

Each function takes a weight matrix W (shape [out_features, in_features])
partitioned into tile_size x tile_size tiles, plus a calibration statistic
collected on the dense model, and returns a list of (score, row, col) for
every full tile.

Convention: a LOWER score means the tile is a better pruning candidate
(it contributes less / removing it changes the layer output least). This
matches the magnitude pruner in run_pruning.py, so tiles are pruned by
sorting ascending and removing the first N.
"""

import torch


def _full_tile_positions(rows, cols, tile_size):
    positions = []
    for r in range(0, rows, tile_size):
        for c in range(0, cols, tile_size):
            if r + tile_size <= rows and c + tile_size <= cols:
                positions.append((r, c))
    return positions


def wanda_tile_scores(weight, col_norms, tile_size):
    """Wanda tile score (report eq. 12):

        S_T = mean_{(i,j) in T} |W_ij| * ||X_:,j||_2

    col_norms is the vector of per-input-column activation norms
    (= sqrt(diag(H)) from the calibration Gram matrix).
    """
    W = weight.detach().float()
    cn = col_norms.detach().float().to(W.device)
    rows, cols = W.shape

    scored = []
    for r, c in _full_tile_positions(rows, cols, tile_size):
        tile = W[r:r + tile_size, c:c + tile_size].abs()
        weighted = tile * cn[c:c + tile_size].unsqueeze(0)   # broadcast norm over rows
        scored.append((weighted.mean().item(), r, c))

    return scored


def sparsegpt_tile_errors(weight, hessian, tile_size, eps=1e-8):
    """SparseGPT masked-tile error (report eq. 22):

        E_T = ||X W^T - X W_{-T}^T||_F^2 / (||X W^T||_F^2 + eps)

    Removing tile T only changes the output columns it feeds, and with
    H = X^T X the squared error reduces to a quadratic form:

        numerator(T)  = trace(W_T   H_cc W_T^T)   # H_cc = H[cols_of_T, cols_of_T]
        denominator   = trace(W     H    W^T)      # constant for the matrix

    A low error means the tile can be removed with little change to the
    layer output.
    """
    W = weight.detach().float()
    H = hessian.detach().float().to(W.device)
    rows, cols = W.shape

    # trace(W H W^T) = sum((W @ H) * W)
    denom = (W @ H * W).sum().item() + eps

    scored = []
    for r, c in _full_tile_positions(rows, cols, tile_size):
        Wt = W[r:r + tile_size, c:c + tile_size]
        Hcc = H[c:c + tile_size, c:c + tile_size]
        num = (Wt @ Hcc * Wt).sum().item()
        scored.append((num / denom, r, c))

    return scored


def sparsegpt_reconstructed_errors(weight, hessian, tile_size, eps=1e-8, damp=1e-2):
    """SparseGPT reconstructed-tile error (report eq. 23).

    After removing a tile, the surviving weights of the affected output rows may
    be re-optimised to reconstruct the original output. With the damped inverse
    Gram matrix Hinv = (H + damp*mean(diag(H))*I)^-1, the minimum achievable
    error for a tile is a quadratic form over its input columns P:

        trace(W_T S W_T^T) / (trace(W H W^T) + eps),   S = ( Hinv[P, P] )^-1

    S is the Schur complement of the surviving columns, so this error is always
    <= the eq. 22 masked error (reconstruction can only help). Low error means
    the tile's function can be absorbed by the remaining weights.

    This is the pure ranking/validation metric; applying the compensating weight
    updates when actually pruning lives in redundancy.recovery.
    """
    W = weight.detach().float()
    H = hessian.detach().float().to(W.device)
    rows, cols = W.shape

    # Damping stabilises the inverse when input features are weakly excited
    # (near-zero diagonal entries in H). Standard SparseGPT practice.
    diag_mean = torch.diag(H).mean()
    Hdamp = H.clone()
    idx = torch.arange(cols, device=W.device)
    Hdamp[idx, idx] += damp * diag_mean
    Hinv = torch.linalg.inv(Hdamp)

    denom = (W @ H * W).sum().item() + eps

    # S depends only on the tile's column block, so compute it once per column.
    schur_by_col = {}
    for c in range(0, cols, tile_size):
        if c + tile_size <= cols:
            schur_by_col[c] = torch.linalg.inv(Hinv[c:c + tile_size, c:c + tile_size])

    scored = []
    for r, c in _full_tile_positions(rows, cols, tile_size):
        Wt = W[r:r + tile_size, c:c + tile_size]
        num = (Wt @ schur_by_col[c] * Wt).sum().item()
        scored.append((max(num, 0.0) / denom, r, c))

    return scored
