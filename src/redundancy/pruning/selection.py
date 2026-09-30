"""Rank neurons from a Weeks 5–8 measurement artifact and pick masking sets.

The measurement stage writes a JSON summary plus a per-neuron ``.npz``. This
module turns those arrays into ``{layer_name: neuron_indices}`` selections at a
given removal ratio, one selection per ranking strategy.

Ratios are applied **per layer** (matched layer-wise sparsity), because that is
the apples-to-apples comparison against the random baseline: a global ranking
would concentrate removals in whichever layer happens to have the smallest
activation scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Ranking strategy -> per-neuron array in the measurement .npz. Neurons with the
# *lowest* score are masked first.
STRATEGY_METRIC = {
    "importance": "importance",
    "frequency": "freq_rms",
    "magnitude": "mean_abs",
    "rms": "rms",
}
RANDOM_STRATEGY = "random"
STRATEGIES = (*STRATEGY_METRIC, RANDOM_STRATEGY)


@dataclass
class MeasurementArtifact:
    """Per-neuron measurement arrays plus the metadata needed to replay them."""

    json_path: Path
    npz_path: Path
    model_id: str
    layer_names: list[str]
    layer_indices: list[int]
    widths: dict[str, int]
    metrics: dict[str, dict[str, np.ndarray]]
    correlation_pairs: dict[str, list[tuple[int, int, float]]] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def scores(self, strategy: str) -> dict[str, np.ndarray]:
        metric = STRATEGY_METRIC.get(strategy)
        if metric is None:
            raise KeyError(f"Unknown ranking strategy: {strategy!r}")
        arrays = self.metrics.get(metric)
        if arrays is None:
            raise KeyError(f"Measurement artifact has no '{metric}' arrays.")
        return arrays

    def name_for_layer(self, layer_index: int) -> str | None:
        for name, idx in zip(self.layer_names, self.layer_indices):
            if idx == layer_index:
                return name
        return None

    def total_neurons(self, layer_filter: Sequence[str] | None = None) -> int:
        names = self.filter_layers(layer_filter)
        return sum(self.widths[name] for name in names)

    def calibration_label(self) -> str:
        """Short description of the corpus this ranking was measured on.

        Worth printing before any run: a ranking is only valid for the corpus it
        was calibrated on, and the intervention scripts accept any artifact.
        """
        dataset = self.summary.get("dataset") or {}
        parts = [str(dataset.get(key)) for key in ("dataset_id", "split") if dataset.get(key)]
        builder = dataset.get("text_builder")
        if builder:
            parts.append(f"prompts={builder}")
        return " / ".join(parts) if parts else "unknown corpus"

    def filter_layers(self, layer_filter: Sequence[str] | None = None) -> list[str]:
        if layer_filter is None:
            return list(self.layer_names)
        allowed = set(layer_filter)
        return [name for name in self.layer_names if name in allowed]


def load_measurement(json_path: str | Path) -> MeasurementArtifact:
    """Load a ``measurement_*.json`` summary and its sibling ``.npz`` arrays."""
    json_path = Path(json_path)
    with json_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    npz_path = json_path.with_suffix(".npz")
    if not npz_path.exists():
        recorded = summary.get("artifacts", {}).get("npz")
        if recorded and Path(recorded).exists():
            npz_path = Path(recorded)
        else:
            raise FileNotFoundError(f"No per-neuron .npz next to {json_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        names = [str(n) for n in data["names"]]
        layer_indices = [int(i) for i in data["layer_index"]]
        metrics: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            if key in ("names", "layer_index"):
                continue
            stack = data[key]
            metrics[key] = {
                name: np.asarray(stack[row], dtype=np.float64) for row, name in enumerate(names)
            }

    widths = {name: int(metrics["rms"][name].size) for name in names}
    pairs = {}
    for layer_key, entries in summary.get("correlation", {}).get("top_pairs", {}).items():
        layer_index = int(layer_key)
        name = names[layer_indices.index(layer_index)] if layer_index in layer_indices else None
        if name is None:
            continue
        pairs[name] = [
            (int(e["neuron_i"]), int(e["neuron_j"]), float(e["correlation"])) for e in entries
        ]

    return MeasurementArtifact(
        json_path=json_path,
        npz_path=npz_path,
        model_id=str(summary.get("model", {}).get("model_id", "unknown")),
        layer_names=names,
        layer_indices=layer_indices,
        widths=widths,
        metrics=metrics,
        correlation_pairs=pairs,
        summary=summary,
    )


def find_latest_measurement(results_dir: str | Path, model_id: str | None = None) -> Path:
    """Newest ``measurement_*.json`` in ``results_dir``, optionally for one model."""
    results_dir = Path(results_dir)
    pattern = "measurement_*.json"
    if model_id:
        pattern = f"measurement_{model_id.replace('/', '__')}_*.json"
    candidates = sorted(results_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {pattern} found in {results_dir}")
    return candidates[-1]


def neurons_to_mask(width: int, ratio: float) -> int:
    """Per-layer masking budget at a removal ratio (0 stays exactly 0)."""
    if ratio <= 0:
        return 0
    return min(width, max(1, int(round(ratio * width))))


def select_neurons(
    artifact: MeasurementArtifact,
    strategy: str,
    ratio: float,
    *,
    seed: int = 42,
    layer_filter: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Neurons to mask per layer at ``ratio`` under one ranking strategy.

    Score-based strategies mask the lowest-scoring neurons; ``random`` samples
    uniformly at the same per-layer count, which is the matched-sparsity
    baseline the intervention is judged against.
    """
    names = artifact.filter_layers(layer_filter)
    if ratio <= 0:
        return {name: np.empty(0, dtype=np.int64) for name in names}

    if strategy == RANDOM_STRATEGY:
        rng = np.random.default_rng(seed)
        return {
            name: np.sort(
                rng.choice(
                    artifact.widths[name],
                    size=neurons_to_mask(artifact.widths[name], ratio),
                    replace=False,
                )
            ).astype(np.int64)
            for name in names
        }

    scores = artifact.scores(strategy)
    selected: dict[str, np.ndarray] = {}
    for name in names:
        vals = scores[name]
        count = neurons_to_mask(vals.size, ratio)
        # Stable sort so ties resolve by neuron index and runs stay reproducible.
        order = np.argsort(vals, kind="stable")
        selected[name] = np.sort(order[:count]).astype(np.int64)
    return selected


def selection_stats(
    artifact: MeasurementArtifact, selected: dict[str, np.ndarray]
) -> dict[str, float]:
    """Summarize a selection: how many neurons and how much importance it removes."""
    importance = artifact.metrics.get("importance", {})
    total_masked = sum(int(idx.size) for idx in selected.values())
    total_neurons = sum(artifact.widths[name] for name in selected)
    imp_masked = 0.0
    imp_total = 0.0
    for name, idx in selected.items():
        imp = importance.get(name)
        if imp is None:
            continue
        imp_total += float(imp.sum())
        if idx.size:
            imp_masked += float(imp[idx].sum())
    return {
        "num_masked": float(total_masked),
        "masked_fraction": total_masked / total_neurons if total_neurons else 0.0,
        "importance_removed_fraction": imp_masked / imp_total if imp_total > 0 else 0.0,
    }


def resolve_layer_filter(
    artifact: MeasurementArtifact, scope: str | Iterable[int] | None
) -> list[str] | None:
    """Turn a config ``layers`` scope into explicit layer names.

    Accepts ``"all"``/``None`` or an explicit list of layer indices.
    """
    if scope is None or scope == "all":
        return None
    names = [artifact.name_for_layer(int(i)) for i in scope]
    resolved = [n for n in names if n is not None]
    if not resolved:
        raise ValueError(f"No measured layers matched the requested scope: {scope}")
    return resolved
