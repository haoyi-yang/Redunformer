"""Offline unit tests for the Weeks 9-10 masking intervention.

Uses a toy SwiGLU model and a synthetic measurement artifact, so nothing is
downloaded and no GPU is required.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.pruning import (  # noqa: E402
    NeuronMasker,
    build_conditions,
    find_latest_measurement,
    load_measurement,
    neurons_to_mask,
    resolve_layer_filter,
    select_disjoint_pairs,
    select_neurons,
    selection_stats,
)


class _ToyMLP(torch.nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _ToyLayer(torch.nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.mlp = _ToyMLP(hidden, inter)

    def forward(self, x):
        return self.mlp(x)


class _ToyModel(torch.nn.Module):
    def __init__(self, hidden=8, inter=16, n_layers=2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer(hidden, inter) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def _write_measurement(tmp_path: Path, *, width: int = 16, n_layers: int = 2) -> Path:
    """Synthetic measurement artifact matching what run_measurement.py writes."""
    names = [f"layers.{i}.mlp.down_proj" for i in range(n_layers)]
    rng = np.random.default_rng(0)
    importance = np.stack([np.arange(width, dtype=float) + 1.0 for _ in range(n_layers)])
    freq = np.stack([rng.random(width) for _ in range(n_layers)])
    npz_path = tmp_path / "measurement_toy_20260101T000000Z.npz"
    np.savez_compressed(
        npz_path,
        layer_index=np.arange(n_layers),
        names=np.array(names),
        importance=importance,
        freq_rms=freq,
        freq_fixed=freq,
        mean_abs=importance / 2.0,
        rms=importance / 3.0,
        col_norm=np.ones((n_layers, width)),
    )
    payload = {
        "model": {"model_id": "toy/model"},
        "correlation": {
            "top_pairs": {
                "0": [
                    {"neuron_i": 2, "neuron_j": 5, "correlation": 0.99},
                    {"neuron_i": 2, "neuron_j": 9, "correlation": 0.98},  # overlaps, skipped
                    {"neuron_i": 7, "neuron_j": 11, "correlation": 0.95},
                    {"neuron_i": 1, "neuron_j": 3, "correlation": 0.5},  # below threshold
                ]
            }
        },
        "artifacts": {"npz": str(npz_path)},
    }
    json_path = npz_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def test_masker_zeroes_selected_neurons_and_restores():
    torch.manual_seed(0)
    model = _ToyModel()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        clean = model(x).clone()

    captured: list[torch.Tensor] = []
    with NeuronMasker(model) as masker:
        # Registered after the masker so this pre-hook observes the masked input.
        handle = model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inputs: captured.append(inputs[0].detach().clone()) and None
        )
        assert len(masker.layer_names) == 2
        assert masker.widths[masker.layer_names[0]] == 16
        masked = masker.apply({masker.layer_names[0]: [0, 3, 7]})
        assert masked == 3
        assert masker.masked_fraction() == pytest.approx(3 / 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(out, clean)
        # The masked channels really are zero on the way into down_proj.
        np.testing.assert_allclose(captured[-1][..., [0, 3, 7]].numpy(), 0.0, atol=0)
        assert captured[-1].abs().sum() > 0  # other channels survive

        masker.clear()
        with torch.no_grad():
            restored = model(x)
        assert torch.allclose(restored, clean, atol=1e-6)
        handle.remove()

    # Hooks are removed on exit, so the model is untouched afterwards.
    with torch.no_grad():
        assert torch.allclose(model(x), clean, atol=1e-6)


def test_masker_rejects_out_of_range_index():
    model = _ToyModel()
    with NeuronMasker(model) as masker:
        with pytest.raises(IndexError):
            masker.apply({masker.layer_names[0]: [99]})


def test_masker_ignores_unknown_layers():
    model = _ToyModel()
    with NeuronMasker(model) as masker:
        assert masker.apply({"vision_tower.mlp.down_proj": [0, 1]}) == 0


def test_neurons_to_mask_budget():
    assert neurons_to_mask(100, 0.0) == 0
    assert neurons_to_mask(100, 0.10) == 10
    assert neurons_to_mask(16, 0.01) == 1  # never silently rounds down to nothing
    assert neurons_to_mask(16, 2.0) == 16


def test_selection_picks_lowest_scores(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    assert artifact.model_id == "toy/model"
    assert artifact.widths[artifact.layer_names[0]] == 16

    selected = select_neurons(artifact, "importance", 0.25, seed=0)
    for indices in selected.values():
        # importance is 1..16 by index, so the cheapest quarter is indices 0-3.
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])

    stats = selection_stats(artifact, selected)
    assert stats["num_masked"] == 8
    assert stats["masked_fraction"] == pytest.approx(0.25)
    # Bottom 4 of 1..16 hold 10/136 of the layer importance.
    assert stats["importance_removed_fraction"] == pytest.approx(10 / 136)


def test_random_selection_matches_count_and_is_seeded(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    a = select_neurons(artifact, "random", 0.25, seed=7)
    b = select_neurons(artifact, "random", 0.25, seed=7)
    c = select_neurons(artifact, "random", 0.25, seed=8)
    for name in a:
        assert a[name].size == 4
        np.testing.assert_array_equal(a[name], b[name])
        assert len(np.unique(a[name])) == 4
    assert any((a[name] != c[name]).any() for name in a)


def test_zero_ratio_masks_nothing(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    for strategy in ("importance", "frequency", "random"):
        selected = select_neurons(artifact, strategy, 0.0)
        assert all(idx.size == 0 for idx in selected.values())


def test_layer_filter_scopes_selection(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    assert resolve_layer_filter(artifact, "all") is None
    only_second = resolve_layer_filter(artifact, [1])
    assert only_second == ["layers.1.mlp.down_proj"]
    selected = select_neurons(artifact, "importance", 0.5, layer_filter=only_second)
    assert set(selected) == set(only_second)
    with pytest.raises(ValueError):
        resolve_layer_filter(artifact, [99])


def test_find_latest_measurement(tmp_path):
    json_path = _write_measurement(tmp_path)
    assert find_latest_measurement(tmp_path) == json_path
    with pytest.raises(FileNotFoundError):
        find_latest_measurement(tmp_path, model_id="other/model")


def test_duplicate_pairs_are_disjoint_and_oriented(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    pairs = select_disjoint_pairs(artifact, min_abs_corr=0.9)
    # (2,9) overlaps with the stronger (2,5) and (1,3) is below threshold.
    assert len(pairs) == 2
    assert {(p.drop, p.keep) for p in pairs} == {(2, 5), (7, 11)}
    # importance grows with index, so the lower-index twin is the one dropped.
    for p in pairs:
        assert p.drop < p.keep
        assert abs(p.correlation) >= 0.9
    neurons = [n for p in pairs for n in (p.drop, p.keep)]
    assert len(neurons) == len(set(neurons))


def test_pair_conditions_are_count_matched(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    pairs = select_disjoint_pairs(artifact, min_abs_corr=0.9)
    conditions = {c.label: c for c in build_conditions(artifact, pairs, seed=0)}
    assert set(conditions) == {
        "duplicate_one",
        "duplicate_both",
        "importance_matched",
        "random_matched",
        "lowest_importance",
    }
    n = conditions["duplicate_one"].num_masked
    assert n == len(pairs)
    assert conditions["duplicate_both"].num_masked == 2 * n
    for label in ("importance_matched", "random_matched", "lowest_importance"):
        assert conditions[label].num_masked == n

    paired = {2, 5, 7, 11}
    layer = "layers.0.mlp.down_proj"
    # Controls must avoid the duplicate pairs, or they would not be controls.
    assert not paired & set(conditions["importance_matched"].masked[layer].tolist())
    assert not paired & set(conditions["random_matched"].masked[layer].tolist())
    # Matched control tracks the importance of the neurons it stands in for.
    assert conditions["importance_matched"].meta["mean_importance"] == pytest.approx(
        conditions["duplicate_one"].meta["mean_importance"], rel=0.35
    )
    # ... and the lowest-importance reference is strictly cheaper.
    assert (
        conditions["lowest_importance"].meta["mean_importance"]
        < conditions["duplicate_one"].meta["mean_importance"]
    )


def test_pair_conditions_are_maskable_on_a_real_module(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    pairs = select_disjoint_pairs(artifact, min_abs_corr=0.9)
    conditions = build_conditions(artifact, pairs, seed=0)
    model = _ToyModel()
    # The toy model names layers "layers.N.mlp.down_proj", matching the artifact.
    with NeuronMasker(model, layer_names=artifact.layer_names) as masker:
        for cond in conditions:
            assert masker.apply(cond.masked) == cond.num_masked


def test_intervention_plots_write_files(tmp_path):
    from redundancy import plotting

    ratios = [0.0, 0.05, 0.1, 0.25]
    series = {"importance": [10, 11, 13, 20], "random": [10, 12, 16, 30]}
    p1 = plotting.plot_removal_curves(ratios, series, tmp_path / "curve.png", baseline=10.0)
    p2 = plotting.plot_condition_bars(
        ["duplicate_one", "importance_matched"],
        [0.01, 0.08],
        tmp_path / "bars.png",
        title="t",
        ylabel="y",
        highlight=("duplicate_one",),
    )
    for p in (p1, p2):
        assert Path(p).exists() and Path(p).stat().st_size > 0
