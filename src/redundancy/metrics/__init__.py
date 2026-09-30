"""Redundancy metrics (Weeks 5–8: neuron / FFN-unit measurement)."""

from __future__ import annotations

from redundancy.metrics.activation_stats import (
    LayerActivationResult,
    NeuronActivationStats,
)
from redundancy.metrics.correlation import (
    CorrelatedPair,
    build_correlation_neighborhood,
    dense_correlation_matrix,
    duplication_score,
    max_abs_correlation_per_neuron,
    top_k_correlated_pairs,
)

__all__ = [
    "LayerActivationResult",
    "NeuronActivationStats",
    "CorrelatedPair",
    "top_k_correlated_pairs",
    "max_abs_correlation_per_neuron",
    "dense_correlation_matrix",
    "build_correlation_neighborhood",
    "duplication_score",
]
