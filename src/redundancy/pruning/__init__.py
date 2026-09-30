"""Neuron masking, replacement, and cross-task interventions (Weeks 9–12)."""

from __future__ import annotations

from redundancy.pruning.masks import NeuronMasker, layer_width, resolve_layer_names
from redundancy.pruning.merge import (
    MergePlan,
    NeuronMerger,
    PairActivationCollector,
    fit_linear_map,
    fit_merge_plans,
    merge_stats,
    plans_to_masked,
)
from redundancy.pruning.overlap import (
    LayerOverlap,
    compare_artifacts,
    expected_random_jaccard,
    jaccard,
    overlap_summary,
    spearman,
)
from redundancy.pruning.pairs import (
    AblationCondition,
    DuplicatePair,
    build_conditions,
    pairs_summary,
    select_disjoint_pairs,
)
from redundancy.pruning.selection import (
    MeasurementArtifact,
    RANDOM_STRATEGY,
    STRATEGIES,
    STRATEGY_METRIC,
    find_latest_measurement,
    load_measurement,
    neurons_to_mask,
    resolve_layer_filter,
    select_neurons,
    selection_stats,
)
from redundancy.pruning.stability import (
    RatioStability,
    bootstrap_mean_ci,
    chance_jaccard,
    mean_pairwise_jaccard,
    random_selection_stability,
    stability_summary,
)

__all__ = [
    "NeuronMasker",
    "layer_width",
    "resolve_layer_names",
    "MeasurementArtifact",
    "load_measurement",
    "find_latest_measurement",
    "select_neurons",
    "selection_stats",
    "neurons_to_mask",
    "resolve_layer_filter",
    "STRATEGIES",
    "STRATEGY_METRIC",
    "RANDOM_STRATEGY",
    "DuplicatePair",
    "AblationCondition",
    "select_disjoint_pairs",
    "build_conditions",
    "pairs_summary",
    "MergePlan",
    "NeuronMerger",
    "PairActivationCollector",
    "fit_linear_map",
    "fit_merge_plans",
    "merge_stats",
    "plans_to_masked",
    "LayerOverlap",
    "compare_artifacts",
    "overlap_summary",
    "jaccard",
    "expected_random_jaccard",
    "spearman",
    "RatioStability",
    "random_selection_stability",
    "stability_summary",
    "bootstrap_mean_ci",
    "mean_pairwise_jaccard",
    "chance_jaccard",
]
