"""Offline unit tests for the Weeks 5-8 neuron measurement metrics.

These use synthetic tensors and a tiny toy module, so they run without any model
or dataset download.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.hooks import (  # noqa: E402
    NeuronActivationCollector,
    input_column_norms,
    iter_ffn_down_projections,
)
from redundancy.metrics import (  # noqa: E402
    build_correlation_neighborhood,
    dense_correlation_matrix,
    duplication_score,
    NeuronActivationStats,
    top_k_correlated_pairs,
)


def test_activation_stats_basic_aggregates():
    stats = NeuronActivationStats(fixed_eps=1e-3, rms_alpha=1e-3, max_reservoir_tokens=1000, seed=0)
    stats.update("layer", 0, torch.tensor([[1.0, 0.0, 5.0], [-1.0, 0.0, 5.0]]))
    stats.update("layer", 0, torch.tensor([[1.0, 0.0, -5.0]]))

    (res,) = stats.finalize()
    assert res.width == 3
    assert res.num_tokens == 3
    np.testing.assert_allclose(res.mean_abs, [1.0, 0.0, 5.0], rtol=1e-6)
    np.testing.assert_allclose(res.rms, [1.0, 0.0, 5.0], rtol=1e-6)
    # neuron 1 never fires; neurons 0 and 2 always fire.
    np.testing.assert_allclose(res.freq_fixed, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(res.freq_rms, [1.0, 0.0, 1.0])
    assert res.lazy_fraction(0.0) == pytest.approx(1 / 3)


def test_lazy_neuron_detected():
    rng = np.random.default_rng(0)
    stats = NeuronActivationStats(fixed_eps=1e-3, rms_alpha=1e-2, max_reservoir_tokens=4096, seed=0)
    # 4 active neurons + 1 dead neuron.
    for _ in range(50):
        block = torch.from_numpy(rng.normal(size=(64, 5)).astype(np.float32))
        block[:, 4] = 0.0
        stats.update("L", 3, block)
    (res,) = stats.finalize()
    assert res.freq_rms[4] == pytest.approx(0.0)
    assert (res.freq_rms[:4] > 0.5).all()
    assert res.lazy_fraction(0.01) == pytest.approx(0.2, abs=1e-6)


def test_multi_alpha_frequency_sweep_is_monotone():
    rng = np.random.default_rng(2)
    stats = NeuronActivationStats(
        rms_alpha=1e-3,
        rms_alphas=[1e-3, 1e-1, 1.0, 5.0],
        max_reservoir_tokens=4096,
        seed=0,
    )
    for _ in range(40):
        stats.update("L", 0, torch.from_numpy(rng.normal(size=(64, 6)).astype(np.float32)))
    (res,) = stats.finalize()
    # Higher alpha => higher threshold => lower firing frequency (per neuron).
    means = [res.freq_by_alpha[a].mean() for a in [1e-3, 1e-1, 1.0, 5.0]]
    assert means == sorted(means, reverse=True)
    # As alpha grows the lazy fraction must not decrease.
    lazy = [res.lazy_fraction_at_alpha(a, 0.01) for a in [1e-3, 1e-1, 1.0, 5.0]]
    assert lazy == sorted(lazy)


def test_importance_from_weight_col_norm():
    stats = NeuronActivationStats(max_reservoir_tokens=64, seed=0)
    stats.update("L", 0, torch.tensor([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]]))
    (res,) = stats.finalize()
    # RMS = [2, 0, 1]; col norms [1, 1, 10] -> importance [2, 0, 10].
    res.set_weight_col_norm(np.array([1.0, 1.0, 10.0]))
    np.testing.assert_allclose(res.importance, [2.0, 0.0, 10.0], rtol=1e-6)
    # Bottom-2/3 neurons (0 and 1) hold 2/12 of total importance.
    assert res.importance_share_bottom(2 / 3) == pytest.approx(2.0 / 12.0, rel=1e-6)
    # Fallback when no weights: importance == rms.
    res.set_weight_col_norm(None)
    np.testing.assert_allclose(res.importance, res.rms)


def test_input_column_norms_linear():
    layer = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 0.0, 1.0], [4.0, 0.0, 0.0]]))
    norms = input_column_norms(layer)
    # Columns (input neurons): ||[3,4]||=5, ||[0,0]||=0, ||[1,0]||=1.
    np.testing.assert_allclose(norms, [5.0, 0.0, 1.0], rtol=1e-6)


def test_dense_corr_neighborhood_and_duplication():
    rng = np.random.default_rng(3)
    base = rng.normal(size=300)
    cols = [base, base + 0.01 * rng.normal(size=300)] + [rng.normal(size=300) for _ in range(6)]
    X = np.stack(cols, axis=1).astype(np.float32)
    pairs = top_k_correlated_pairs(X, k=5, max_neurons=20, abs_threshold=0.9)
    nbhd = build_correlation_neighborhood(X, pairs, size=5)
    assert len(nbhd) == 5
    assert 0 in nbhd and 1 in nbhd  # the duplicate pair is included
    mat = dense_correlation_matrix(X, nbhd)
    assert mat.shape == (5, 5)
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-6)
    # A layer with a near-duplicate scores higher than pure noise.
    noise = rng.normal(size=(300, 8)).astype(np.float32)
    assert duplication_score(X, max_neurons=20) > duplication_score(noise, max_neurons=20)


def test_top_k_correlated_pairs_finds_duplicate():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(300, 1))
    cols = [
        base[:, 0],
        base[:, 0] + 0.01 * rng.normal(size=300),  # near-duplicate of neuron 0
        rng.normal(size=300),
        rng.normal(size=300),
    ]
    X = np.stack(cols, axis=1).astype(np.float32)
    pairs = top_k_correlated_pairs(X, k=3, max_neurons=10, abs_threshold=0.9)
    assert pairs, "expected at least one highly-correlated pair"
    top = pairs[0]
    assert {top.neuron_i, top.neuron_j} == {0, 1}
    assert top.correlation > 0.9


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


def test_collector_discovers_and_captures_ffn():
    model = _ToyModel(hidden=8, inter=16, n_layers=2)
    found = list(iter_ffn_down_projections(model))
    assert {layer_idx for layer_idx, _, _ in found} == {0, 1}
    assert all(name.endswith("down_proj") for _, name, _ in found)

    stats = NeuronActivationStats(max_reservoir_tokens=256, seed=0)
    with NeuronActivationCollector(model, stats) as collector:
        model(torch.randn(3, 5, 8))  # [batch, seq, hidden]
        col_norms = collector.input_weight_norms()

    results = stats.finalize()
    assert len(results) == 2
    for res in results:
        assert res.width == 16
        assert res.num_tokens == 15  # 3 * 5 token positions
        norm = col_norms[res.name]
        assert norm is not None and norm.shape == (16,)
        res.set_weight_col_norm(norm)
        assert res.col_norm is not None
        assert res.importance.shape == (16,)
        assert 0.0 <= res.importance_share_bottom(0.1) <= 1.0


def test_plotting_helpers_write_files(tmp_path):
    from redundancy import plotting

    rng = np.random.default_rng(0)
    freqs = {i: rng.random(64) for i in range(6)}
    p1 = plotting.plot_frequency_histograms(freqs, tmp_path / "hist.png")
    p2 = plotting.plot_freq_distribution_heatmap(freqs, tmp_path / "depth.png")
    p3 = plotting.plot_lazy_fraction_by_depth(
        list(freqs), [float(v.mean()) for v in freqs.values()], tmp_path / "lazy.png"
    )
    corr = np.corrcoef(rng.normal(size=(200, 8)).T)
    p4 = plotting.plot_correlation_heatmap(corr, tmp_path / "corr.png")
    imp = {i: rng.random(64) * (i + 1) for i in range(6)}
    p5 = plotting.plot_importance_histograms(imp, tmp_path / "imp_hist.png")
    p6 = plotting.plot_metric_by_depth(
        list(imp), [float(v.mean()) for v in imp.values()], tmp_path / "imp_depth.png",
        title="t", ylabel="y",
    )
    sweep = {a: rng.random(6) for a in (1e-3, 1e-1, 1.0)}
    p7 = plotting.plot_lazy_fraction_sweep(list(range(6)), sweep, tmp_path / "sweep.png")
    for p in (p1, p2, p3, p4, p5, p6, p7):
        assert Path(p).exists() and Path(p).stat().st_size > 0


def test_calibration_block_builder():
    from redundancy.data import build_calibration_blocks

    class _Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
            n = len(text.split())
            ids = torch.arange(n).unsqueeze(0)
            return {"input_ids": ids}

    texts = [" ".join(str(i) for i in range(1000))]
    blocks = build_calibration_blocks(texts, _Tok(), seq_len=128, max_blocks=5)
    assert len(blocks) == 5
    assert all(b.shape == (1, 128) for b in blocks)
