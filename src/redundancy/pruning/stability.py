"""How much does the random baseline depend on its seed? (#18)

Every `random` column in Weeks 9–14 is **one** seeded draw. That is a real gap:
the guided-vs-random gap is the headline result of the whole project, and it is
reported against a baseline with no error bar. Where the gap is large (0.6B at
25%: 5.889× vs 1.566×) a single draw is obviously enough; where it is small and
occasionally *negative* (gpt2-large at 5–10%, −0.02× baseline) it is not obvious
at all, and that cell is currently unfalsifiable.

Re-running the sweep with more seeds costs GPU hours we do not have. This module
closes most of the gap for free: the seed only enters through
:func:`select_neurons`, so the *selection* half of the variance can be resampled
offline from the measurement ``.npz`` alone — no model, no forward pass, no
tokenizer. What varies across seeds is **which** neurons get masked, and the
summary statistic for that is how much of each layer's importance mass the draw
happens to remove.

**What this bounds and what it does not.** It bounds the spread of the *inputs*
to the evaluation, not the spread of perplexity itself: mapping removed
importance to ΔPPL still needs forward passes. So a tight distribution here means
"another seed would have masked a near-identical amount of importance", which
makes a large reported gap safe to trust. It does **not** license a confidence
interval on any perplexity number. Read the output as a sanity bound on the
baseline, and quote it as such.

Two statistics are produced per ratio:

``importance_removed_fraction``
    Share of total layer importance the draw removes, per seed — mean, standard
    deviation, min/max, and a percentile bootstrap CI of the mean. Compared
    against the *guided* selection at the same sparsity, which is what the
    random arm is a control for.

``mean pairwise Jaccard``
    Agreement between two independent draws, averaged over layers and seed
    pairs, against the analytic floor from :func:`expected_random_jaccard`.
    This one is a self-check rather than a result: draws that agree far more
    than chance would mean the seeding is not doing what we think it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from redundancy.pruning.overlap import expected_random_jaccard, jaccard
from redundancy.pruning.selection import (
    RANDOM_STRATEGY,
    MeasurementArtifact,
    neurons_to_mask,
    select_neurons,
    selection_stats,
)

DEFAULT_NUM_SEEDS = 20
DEFAULT_BASE_SEED = 42
DEFAULT_BOOTSTRAP = 2000


@dataclass
class RatioStability:
    """Seed-to-seed spread of the random baseline at one removal ratio."""

    ratio: float
    seeds: list[int]
    num_masked: int
    importance_removed: np.ndarray
    mean_pairwise_jaccard: float
    chance_jaccard: float
    guided_importance_removed: float
    bootstrap_ci: tuple[float, float]

    @property
    def mean(self) -> float:
        return float(self.importance_removed.mean())

    @property
    def std(self) -> float:
        # Sample standard deviation: these seeds are a sample of the draws we
        # could have taken, not the population of them.
        return float(self.importance_removed.std(ddof=1)) if self.seeds[1:] else 0.0

    @property
    def spread(self) -> float:
        """Full observed range, which is the honest thing to quote for 20 draws."""
        return float(self.importance_removed.max() - self.importance_removed.min())

    @property
    def guided_gap_in_sds(self) -> float:
        """Distance from the guided arm to the random mean, in seed standard deviations.

        Large values mean the ranking is doing something no random draw would
        stumble into, so the single-seed baseline was never the weak link.
        """
        if self.std <= 0:
            return float("inf") if self.guided_importance_removed != self.mean else 0.0
        return float((self.mean - self.guided_importance_removed) / self.std)

    def summary(self) -> dict:
        return {
            "ratio": self.ratio,
            "num_seeds": len(self.seeds),
            "seeds": list(self.seeds),
            "neurons_masked": self.num_masked,
            "importance_removed_mean": self.mean,
            "importance_removed_std": self.std,
            "importance_removed_min": float(self.importance_removed.min()),
            "importance_removed_max": float(self.importance_removed.max()),
            "importance_removed_range": self.spread,
            "bootstrap_ci_95": list(self.bootstrap_ci),
            "mean_pairwise_jaccard": self.mean_pairwise_jaccard,
            "chance_jaccard": self.chance_jaccard,
            "guided_importance_removed": self.guided_importance_removed,
            "guided_gap_in_seed_sds": self.guided_gap_in_sds,
        }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    num_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Deliberately non-parametric: with 20 draws there is no reason to assume the
    removed-importance distribution is normal, and the quantity is a bounded
    fraction.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(num_resamples, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def mean_pairwise_jaccard(selections: Sequence[dict[str, np.ndarray]]) -> float:
    """Average agreement between every pair of draws, over layers and pairs."""
    if len(selections) < 2:
        return float("nan")
    per_pair = [
        float(np.mean([jaccard(a[name], b[name]) for name in a if name in b]))
        for a, b in combinations(selections, 2)
    ]
    return float(np.mean(per_pair)) if per_pair else float("nan")


def chance_jaccard(
    artifact: MeasurementArtifact, ratio: float, layer_names: Sequence[str]
) -> float:
    """Analytic Jaccard floor for two independent draws, averaged over layers."""
    floors = [
        expected_random_jaccard(
            artifact.widths[name], neurons_to_mask(artifact.widths[name], ratio)
        )
        for name in layer_names
    ]
    return float(np.mean(floors)) if floors else 0.0


def random_selection_stability(
    artifact: MeasurementArtifact,
    ratio: float,
    *,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
    layer_filter: Sequence[str] | None = None,
    guided_strategy: str = "importance",
    num_resamples: int = DEFAULT_BOOTSTRAP,
) -> RatioStability:
    """Resample the random baseline's *selection* at one ratio, seeds only.

    ``base_seed`` is the seed the production runs used, and it is included as the
    first draw so the committed numbers can be located inside the distribution
    rather than compared to it from outside.
    """
    seeds = [base_seed + i for i in range(max(1, num_seeds))]
    names = artifact.filter_layers(layer_filter)

    selections = [
        select_neurons(artifact, RANDOM_STRATEGY, ratio, seed=seed, layer_filter=layer_filter)
        for seed in seeds
    ]
    removed = np.array(
        [selection_stats(artifact, sel)["importance_removed_fraction"] for sel in selections],
        dtype=np.float64,
    )
    guided = select_neurons(artifact, guided_strategy, ratio, layer_filter=layer_filter)

    return RatioStability(
        ratio=ratio,
        seeds=seeds,
        num_masked=int(sum(int(idx.size) for idx in selections[0].values())),
        importance_removed=removed,
        mean_pairwise_jaccard=mean_pairwise_jaccard(selections),
        chance_jaccard=chance_jaccard(artifact, ratio, names),
        guided_importance_removed=float(
            selection_stats(artifact, guided)["importance_removed_fraction"]
        ),
        bootstrap_ci=bootstrap_mean_ci(removed, num_resamples=num_resamples, seed=base_seed),
    )


def stability_summary(results: Sequence[RatioStability]) -> dict:
    """Aggregate verdict across ratios, in the form the write-up needs."""
    if not results:
        return {"num_ratios": 0}
    widest = max(results, key=lambda r: r.spread)
    closest = min(results, key=lambda r: abs(r.guided_gap_in_sds))
    return {
        "num_ratios": len(results),
        "max_importance_removed_range": widest.spread,
        "max_range_at_ratio": widest.ratio,
        "min_guided_gap_in_seed_sds": closest.guided_gap_in_sds,
        "min_gap_at_ratio": closest.ratio,
        # The guided arm has to sit clear of the seed noise at every ratio for
        # the single-seed baselines in the committed reports to be safe.
        "guided_outside_seed_noise": all(r.guided_gap_in_sds > 3.0 for r in results),
    }
