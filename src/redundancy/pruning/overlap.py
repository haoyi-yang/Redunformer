"""Cross-task neuron comparison (hypothesis H3, Weeks 11–12).

H3 says neurons that look redundant under WikiText-2 are *not* redundant under
PIQA. Testing it needs two measurement artifacts for the same model, one per
calibration corpus, and then two kinds of comparison:

* **Set agreement** — do the two corpora nominate the same neurons for removal?
  The pruning-relevant quantity is the *bottom* of the ranking (what would be
  masked), so both tails are reported; the top tail is the "most active neurons"
  overlap the proposal originally described.
* **Rank agreement** — Spearman correlation of the full importance ranking,
  which catches the case where the tails agree by chance while the middle of the
  distribution does not.

Both are computed per layer, because Weeks 5–8 found redundancy to be strongly
depth-dependent; a single global number would average away the layers where the
two corpora actually disagree.

A low overlap is only *evidence for* H3. The behavioural test — masking the
WikiText-redundant set and measuring PIQA accuracy against masking the
PIQA-redundant set — is run by ``scripts/run_pruning.py`` on each artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from redundancy.pruning.selection import MeasurementArtifact, neurons_to_mask


@dataclass
class LayerOverlap:
    """Agreement between two calibration corpora for one layer."""

    layer_name: str
    layer_index: int
    width: int
    count: int
    jaccard_bottom: float
    jaccard_top: float
    spearman: float

    def summary(self) -> dict:
        return {
            "layer": self.layer_name,
            "layer_index": self.layer_index,
            "width": self.width,
            "count": self.count,
            "jaccard_bottom": self.jaccard_bottom,
            "jaccard_top": self.jaccard_top,
            "spearman": self.spearman,
        }


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Intersection over union of two neuron index sets."""
    set_a, set_b = set(int(x) for x in a), set(int(x) for x in b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def expected_random_jaccard(width: int, count: int) -> float:
    """Jaccard two *independent* random sets of this size would score.

    Overlap has to be judged against this floor: picking the lowest 10% of 1024
    neurons twice at random already agrees on roughly 5% of the union, so a
    small positive Jaccard is not evidence of shared structure.
    """
    if width <= 0 or count <= 0:
        return 0.0
    expected_intersection = count * count / width
    union = 2 * count - expected_intersection
    return expected_intersection / union if union > 0 else 0.0


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values`` with tied entries sharing their mean rank.

    Tie handling is what makes a constant vector rank as constant; naive
    ``argsort(argsort(x))`` would hand it an arbitrary permutation and report a
    spurious correlation.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    _, first, inverse, counts = np.unique(
        sorted_values, return_index=True, return_inverse=True, return_counts=True
    )
    group_rank = first.astype(np.float64) + (counts - 1) / 2.0
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = group_rank[inverse]
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two score vectors (NaN if either is flat)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    rank_a = average_ranks(a)
    rank_b = average_ranks(b)
    std_a, std_b = rank_a.std(), rank_b.std()
    if std_a <= 0 or std_b <= 0:
        return float("nan")
    return float(((rank_a - rank_a.mean()) * (rank_b - rank_b.mean())).mean() / (std_a * std_b))


def compare_artifacts(
    reference: MeasurementArtifact,
    other: MeasurementArtifact,
    *,
    ratio: float = 0.1,
    strategy: str = "importance",
) -> list[LayerOverlap]:
    """Per-layer agreement between two calibration corpora at one removal ratio.

    Layers are matched by *index* rather than name so the comparison survives
    two runs that measured different layer subsets.
    """
    ref_scores = reference.scores(strategy)
    other_scores = other.scores(strategy)
    results: list[LayerOverlap] = []
    for name, layer_index in zip(reference.layer_names, reference.layer_indices):
        other_name = other.name_for_layer(layer_index)
        if other_name is None:
            continue
        ref_vals = ref_scores.get(name)
        other_vals = other_scores.get(other_name)
        if ref_vals is None or other_vals is None or ref_vals.size != other_vals.size:
            continue
        width = int(ref_vals.size)
        count = neurons_to_mask(width, ratio)
        ref_order = np.argsort(ref_vals, kind="stable")
        other_order = np.argsort(other_vals, kind="stable")
        results.append(
            LayerOverlap(
                layer_name=name,
                layer_index=layer_index,
                width=width,
                count=count,
                jaccard_bottom=jaccard(ref_order[:count], other_order[:count]),
                jaccard_top=jaccard(ref_order[-count:], other_order[-count:]),
                spearman=spearman(ref_vals, other_vals),
            )
        )
    return results


def overlap_summary(overlaps: Sequence[LayerOverlap]) -> dict:
    """Aggregate per-layer overlaps, including the random-agreement floor."""
    if not overlaps:
        return {"num_layers": 0}
    bottom = np.array([o.jaccard_bottom for o in overlaps], dtype=np.float64)
    top = np.array([o.jaccard_top for o in overlaps], dtype=np.float64)
    rho = np.array([o.spearman for o in overlaps], dtype=np.float64)
    chance = np.array(
        [expected_random_jaccard(o.width, o.count) for o in overlaps], dtype=np.float64
    )
    return {
        "num_layers": len(overlaps),
        "mean_jaccard_bottom": float(bottom.mean()),
        "min_jaccard_bottom": float(bottom.min()),
        "max_jaccard_bottom": float(bottom.max()),
        "mean_jaccard_top": float(top.mean()),
        "mean_spearman": float(np.nanmean(rho)),
        "mean_chance_jaccard": float(chance.mean()),
        "bottom_over_chance": float(bottom.mean() / chance.mean()) if chance.mean() > 0 else 0.0,
    }
