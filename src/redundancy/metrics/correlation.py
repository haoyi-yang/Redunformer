"""Pairwise neuron correlation within an FFN layer.

We never materialize a full ``width x width`` correlation matrix for wide layers.
Instead we (optionally) restrict to the highest-variance neurons, then scan the
upper triangle in row-blocks to keep only the top-``k`` most correlated pairs.
This implements the "duplicate neuron" probe from the proposal (|rho| > ~0.9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CorrelatedPair:
    neuron_i: int
    neuron_j: int
    correlation: float

    def as_tuple(self) -> tuple[int, int, float]:
        return self.neuron_i, self.neuron_j, self.correlation


def _zscore_columns(matrix: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mu = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    valid = sd > eps
    z = np.zeros_like(matrix, dtype=np.float64)
    z[:, valid] = (matrix[:, valid] - mu[valid]) / sd[valid]
    return z, valid


def top_k_correlated_pairs(
    activations: np.ndarray,
    *,
    k: int = 50,
    max_neurons: int = 2048,
    abs_threshold: float | None = None,
    block_size: int = 256,
) -> list[CorrelatedPair]:
    """Return the top-``k`` most correlated neuron pairs in one layer.

    Args:
        activations: ``[tokens, width]`` activation subsample (e.g. a reservoir).
        k: number of pairs to return, ranked by ``|correlation|``.
        max_neurons: cap on neurons considered; the highest-variance neurons are
            kept when the layer is wider (bounds the cost to ``max_neurons^2``).
        abs_threshold: if set, only pairs with ``|rho| >= abs_threshold`` are kept.
        block_size: row-block size for the streaming upper-triangle scan.
    """
    if activations.ndim != 2:
        raise ValueError("activations must be 2D [tokens, width]")
    tokens, width = activations.shape
    if tokens < 3 or width < 2:
        return []

    if width > max_neurons:
        keep = np.argsort(activations.var(axis=0))[-max_neurons:]
        keep.sort()
    else:
        keep = np.arange(width)

    z, valid = _zscore_columns(activations[:, keep])
    n = z.shape[1]

    best: list[tuple[float, int, int, float]] = []  # (|rho|, i, j, rho) min-heap

    import heapq

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # Correlation of block rows [start:end] against all columns >= start.
        block = z[:, start:end].T @ z[:, start:]  # [block, n-start]
        block /= tokens
        for local_i in range(end - start):
            i = start + local_i
            if not valid[i]:
                continue
            # Only j > i (strict upper triangle).
            row = block[local_i, (i - start) + 1 :]
            cols = np.arange(i + 1, n)
            if row.size == 0:
                continue
            mag = np.abs(row)
            if abs_threshold is not None:
                sel = mag >= abs_threshold
                if not sel.any():
                    continue
                row, cols, mag = row[sel], cols[sel], mag[sel]
            for c, rho, a in zip(cols, row, mag):
                if not valid[c]:
                    continue
                if len(best) < k:
                    heapq.heappush(best, (a, int(keep[i]), int(keep[c]), float(rho)))
                elif a > best[0][0]:
                    heapq.heapreplace(best, (a, int(keep[i]), int(keep[c]), float(rho)))

    best.sort(key=lambda t: t[0], reverse=True)
    return [CorrelatedPair(i, j, rho) for _, i, j, rho in best]


def dense_correlation_matrix(activations: np.ndarray, neuron_ids: Sequence[int]) -> np.ndarray:
    """Dense Pearson correlation matrix for a small set of neurons."""
    sub = activations[:, list(neuron_ids)]
    z, _ = _zscore_columns(sub)
    return (z.T @ z) / sub.shape[0]


def build_correlation_neighborhood(
    activations: np.ndarray,
    pairs: Sequence[CorrelatedPair],
    *,
    size: int = 40,
) -> list[int]:
    """Pick neurons for a readable correlation heatmap.

    Starts from the neurons in the strongest pairs, then pads with the
    highest-variance neurons so the heatmap is a real block (not 2x2) even when
    few pairs clear the correlation threshold.
    """
    ids: list[int] = []
    for p in pairs:
        for nid in (p.neuron_i, p.neuron_j):
            if nid not in ids:
                ids.append(nid)
        if len(ids) >= size:
            break
    if len(ids) < size:
        order = np.argsort(activations.var(axis=0))[::-1]
        for nid in order:
            nid = int(nid)
            if nid not in ids:
                ids.append(nid)
            if len(ids) >= size:
                break
    return sorted(ids[:size])


def duplication_score(
    activations: np.ndarray,
    *,
    max_neurons: int = 512,
    top_frac: float = 0.02,
    block_size: int = 256,
) -> float:
    """Layer-level duplication signal: mean of the top per-neuron max |correlation|.

    High when a layer contains many neurons with a near-duplicate partner. Used
    to rank layers for the correlation drill-down (replaces ranking by the
    saturated lazy-fraction).
    """
    _, best = max_abs_correlation_per_neuron(
        activations, max_neurons=max_neurons, block_size=block_size
    )
    if best.size == 0:
        return 0.0
    k = max(1, int(round(top_frac * best.size)))
    return float(np.sort(best)[-k:].mean())


def max_abs_correlation_per_neuron(
    activations: np.ndarray,
    *,
    max_neurons: int = 2048,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """For each kept neuron, its strongest off-diagonal ``|correlation|``.

    Returns ``(neuron_indices, max_abs_corr)`` over the (variance-capped) neuron
    set. Useful as a per-neuron "duplication" score complementing frequency.
    """
    tokens, width = activations.shape
    if width > max_neurons:
        keep = np.argsort(activations.var(axis=0))[-max_neurons:]
        keep.sort()
    else:
        keep = np.arange(width)

    z, valid = _zscore_columns(activations[:, keep])
    n = z.shape[1]
    best = np.zeros(n, dtype=np.float64)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        corr = (z[:, start:end].T @ z) / tokens  # [block, n]
        for local_i in range(end - start):
            i = start + local_i
            corr[local_i, i] = 0.0  # ignore self
        best[start:end] = np.max(np.abs(corr), axis=1)
    best[~valid] = 0.0
    return keep, best
