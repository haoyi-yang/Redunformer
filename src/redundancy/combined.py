"""
Ablation: does sparsegpt_recon win by its SELECTION or by its RECONSTRUCTION?

The current ladder confounds the two. sparsegpt_recon both ranks tiles by the eq-23
reconstructed error AND applies the compensating weight update, while wanda ranks by a
cheap per-weight proxy and never repairs. So "recon is best" could mean either "eq-23
picks better tiles" or "repairing at all is what matters" -- we cannot tell which.

This module supplies the missing cell of the 2x2: Wanda's selection with SparseGPT's
repair.

                    | no repair   | repair
    ----------------|-------------|----------------------
    Wanda select    | wanda       | wanda_recon  (here)
    eq-23 select    | sparsegpt   | sparsegpt_recon

  wanda_recon vs wanda            -> what the repair alone is worth
  wanda_recon vs sparsegpt_recon  -> what the selection alone is worth

Only the Gram matrix H is needed: Wanda's column norms are sqrt(diag(H)), so this reuses
the statistic the sparsegpt path already collects -- no extra calibration pass.
"""

import random as _random

import torch

from .recovery import _damped_inverse
from .scoring import wanda_tile_scores, _full_tile_positions


def col_norms_from_hessian(hessian):
    """Wanda's per-column activation norms: ||X_:,j||_2 = sqrt(diag(X^T X))."""
    return torch.sqrt(torch.diag(hessian.detach().float()).clamp_min(0))


def reconstruct_given_tiles(weight, hessian, pruned_tiles, tile_size, damp=1e-2):
    """Apply the exact SparseGPT reconstruction for an ALREADY-CHOSEN tile set.

    pruned_tiles: iterable of (row, col) tile top-left offsets.

    Same math as recovery.reconstruct_prune_tiles step 2: solved jointly per output
    row-block over the union of that block's pruned columns, so the update zeros every
    pruned tile exactly and never writes back into an earlier-pruned one. Splitting this
    out is what lets any selection rule be paired with the reconstruction.
    """
    H = hessian.detach().float().to(weight.device)
    Hinv = _damped_inverse(H, damp)

    pruned_cols_by_row = {}
    for r, c in pruned_tiles:
        pruned_cols_by_row.setdefault(r, []).extend(range(c, c + tile_size))

    with torch.no_grad():
        for r, pcols in pruned_cols_by_row.items():
            idx = torch.tensor(sorted(set(pcols)), device=weight.device)
            S = torch.linalg.inv(Hinv[idx][:, idx])          # (|P|, |P|)
            A = Hinv[:, idx] @ S                              # (in, |P|)
            block = weight[r:r + tile_size, idx].float()      # (tile_size, |P|)
            weight[r:r + tile_size, :] -= (block @ A.t()).to(weight.dtype)


def prune_random_recon(weight, tile_size, prune_ratio, hessian, seed, damp=1e-2):
    """RANDOM selection + SparseGPT repair -- the decisive test of whether selection matters.

    Selects the EXACT same tiles as run_pruning.prune_random at the same seed: identical tile
    ordering (row-major, full tiles only) and identical `random.Random(seed).shuffle`. So

        random  vs  random_recon   isolates REPAIR   (same tiles, repair on/off)
        random_recon vs wanda_recon isolates SELECTION (repair on both, tiles differ)

    If random_recon matches wanda_recon, selection is irrelevant *given* repair -- you may pick
    tiles by coin flip. If it does not, selection is doing real work that our whole-model control
    (where plain random already beats plain wanda) would otherwise have hidden.

    Returns (num_tiles, num_pruned) -- same contract as the other prune_* functions.
    """
    rows, cols = weight.shape
    tiles = _full_tile_positions(rows, cols, tile_size)
    rng = _random.Random(seed)
    rng.shuffle(tiles)
    num_prune = int(len(tiles) * prune_ratio)
    pruned = tiles[:num_prune]

    reconstruct_given_tiles(weight, hessian, pruned, tile_size, damp)

    return len(tiles), num_prune


def prune_wanda_recon(weight, tile_size, prune_ratio, hessian, damp=1e-2):
    """Select tiles by Wanda score, remove them with SparseGPT's compensating update.

    Returns (num_tiles, num_pruned) -- same contract as the other prune_* functions.
    """
    scored = wanda_tile_scores(weight, col_norms_from_hessian(hessian), tile_size)
    scored.sort(key=lambda x: x[0])
    num_prune = int(len(scored) * prune_ratio)
    pruned = [(r, c) for _, r, c in scored[:num_prune]]

    reconstruct_given_tiles(weight, hessian, pruned, tile_size, damp)

    return len(scored), num_prune
