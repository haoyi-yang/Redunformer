"""Duplicate-neuron ablation conditions (hypothesis H5).

The measurement stage finds pairs of neurons whose activations are nearly
collinear (|rho| >= 0.9). Correlation alone does not prove the network only
needs one of them, so this module builds the ablation conditions that test it:

* ``duplicate_one``   — mask one neuron of each pair (the lower-importance twin)
* ``duplicate_both``  — mask both, to see how much of the damage the surviving
  twin was actually absorbing
* ``importance_matched`` — mask *uncorrelated* neurons with matched importance,
  the control that isolates "is a duplicate" from "is unimportant"
* ``random_matched``  — mask random uncorrelated neurons at the same count
* ``lowest_importance`` — mask the least-important neurons at the same count,
  i.e. the guided-pruning reference at this (tiny) budget

H5 predicts ``duplicate_one`` degrades noticeably less than
``importance_matched`` at the same neuron count, and close to ``duplicate_both``
minus the surviving twin's own contribution.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np

from redundancy.pruning.selection import MeasurementArtifact


@dataclass
class DuplicatePair:
    """One near-duplicate pair, oriented so ``drop`` is the cheaper twin."""

    layer_name: str
    drop: int
    keep: int
    correlation: float


@dataclass
class AblationCondition:
    """A named set of neurons to mask, plus how it was constructed."""

    label: str
    masked: dict[str, np.ndarray]
    description: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def num_masked(self) -> int:
        return sum(int(idx.size) for idx in self.masked.values())


def select_disjoint_pairs(
    artifact: MeasurementArtifact,
    *,
    min_abs_corr: float = 0.9,
    max_pairs_per_layer: int | None = None,
    layer_filter: list[str] | None = None,
) -> list[DuplicatePair]:
    """Pick non-overlapping duplicate pairs, strongest correlation first.

    Pairs are made disjoint because the reported top-k often share a neuron
    (a small clique of near-identical units); masking one neuron from each of
    two overlapping pairs would silently change the removal budget.
    """
    names = artifact.filter_layers(layer_filter)
    importance = artifact.metrics.get("importance", {})
    pairs: list[DuplicatePair] = []
    for name in names:
        entries = artifact.correlation_pairs.get(name, [])
        ranked = sorted(entries, key=lambda e: abs(e[2]), reverse=True)
        used: set[int] = set()
        taken = 0
        for i, j, rho in ranked:
            if abs(rho) < min_abs_corr or i in used or j in used:
                continue
            imp = importance.get(name)
            if imp is not None and imp[j] < imp[i]:
                drop, keep = j, i
            else:
                drop, keep = i, j
            pairs.append(DuplicatePair(layer_name=name, drop=drop, keep=keep, correlation=rho))
            used.update((i, j))
            taken += 1
            if max_pairs_per_layer is not None and taken >= max_pairs_per_layer:
                break
    return pairs


def _group_by_layer(pairs: list[DuplicatePair]) -> dict[str, list[DuplicatePair]]:
    grouped: dict[str, list[DuplicatePair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.layer_name, []).append(pair)
    return grouped


def _paired_neurons(pairs: list[DuplicatePair]) -> dict[str, set[int]]:
    """All neurons involved in any duplicate pair, per layer (excluded from controls)."""
    involved: dict[str, set[int]] = {}
    for pair in pairs:
        involved.setdefault(pair.layer_name, set()).update((pair.drop, pair.keep))
    return involved


def _match_by_importance(
    importance: np.ndarray, targets: np.ndarray, excluded: set[int]
) -> np.ndarray:
    """Pick one distinct neuron per target with the closest importance value.

    Candidates are kept in an importance-sorted list and removed as they are
    consumed, so no neuron is selected twice and the matched set spans the same
    importance range as the duplicates it controls for.
    """
    candidates = [i for i in range(importance.size) if i not in excluded]
    candidates.sort(key=lambda i: importance[i])
    values = [importance[i] for i in candidates]
    chosen: list[int] = []
    for target in targets:
        if not candidates:
            break
        pos = bisect_left(values, target)
        best = pos
        if pos >= len(values):
            best = len(values) - 1
        elif pos > 0 and abs(values[pos - 1] - target) <= abs(values[pos] - target):
            best = pos - 1
        chosen.append(candidates.pop(best))
        values.pop(best)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def build_conditions(
    artifact: MeasurementArtifact,
    pairs: list[DuplicatePair],
    *,
    seed: int = 42,
) -> list[AblationCondition]:
    """Build the H5 ablation conditions for a set of duplicate pairs."""
    if not pairs:
        return []

    grouped = _group_by_layer(pairs)
    excluded = _paired_neurons(pairs)
    importance = artifact.metrics.get("importance", {})
    rng = np.random.default_rng(seed)

    drop_only: dict[str, np.ndarray] = {}
    both: dict[str, np.ndarray] = {}
    matched: dict[str, np.ndarray] = {}
    random_matched: dict[str, np.ndarray] = {}
    lowest: dict[str, np.ndarray] = {}

    for name, layer_pairs in grouped.items():
        drops = np.array([p.drop for p in layer_pairs], dtype=np.int64)
        keeps = np.array([p.keep for p in layer_pairs], dtype=np.int64)
        drop_only[name] = np.sort(drops)
        both[name] = np.sort(np.concatenate([drops, keeps]))

        imp = importance.get(name)
        width = artifact.widths[name]
        blocked = excluded.get(name, set())
        pool = np.array([i for i in range(width) if i not in blocked], dtype=np.int64)
        count = min(drops.size, pool.size)
        random_matched[name] = np.sort(rng.choice(pool, size=count, replace=False))

        if imp is not None:
            matched[name] = _match_by_importance(imp, imp[drops], blocked)
            order = np.argsort(imp, kind="stable")
            lowest[name] = np.sort(order[: drops.size]).astype(np.int64)
        else:
            matched[name] = random_matched[name]
            lowest[name] = random_matched[name]

    def _mean_importance(sel: dict[str, np.ndarray]) -> float:
        vals = [
            importance[name][idx]
            for name, idx in sel.items()
            if name in importance and idx.size
        ]
        return float(np.concatenate(vals).mean()) if vals else 0.0

    conditions = [
        AblationCondition(
            label="duplicate_one",
            masked=drop_only,
            description="one neuron of each near-duplicate pair (lower-importance twin)",
        ),
        AblationCondition(
            label="duplicate_both",
            masked=both,
            description="both neurons of each near-duplicate pair",
        ),
        AblationCondition(
            label="importance_matched",
            masked=matched,
            description="uncorrelated neurons with importance matched to duplicate_one",
        ),
        AblationCondition(
            label="random_matched",
            masked=random_matched,
            description="random uncorrelated neurons, same count as duplicate_one",
        ),
        AblationCondition(
            label="lowest_importance",
            masked=lowest,
            description="least-important neurons, same count as duplicate_one",
        ),
    ]
    for cond in conditions:
        cond.meta["mean_importance"] = _mean_importance(cond.masked)
        cond.meta["layers"] = {name: int(idx.size) for name, idx in cond.masked.items()}
    return conditions


def pairs_summary(pairs: list[DuplicatePair]) -> list[dict]:
    return [
        {
            "layer": p.layer_name,
            "drop": p.drop,
            "keep": p.keep,
            "correlation": p.correlation,
        }
        for p in pairs
    ]
