"""Offline unit tests for the random-baseline seed stability analysis (#18).

Pure numpy against a synthetic measurement artifact: no model, no GPU, no
downloads, and no forward passes - which is the whole point of the module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.pruning import (  # noqa: E402
    bootstrap_mean_ci,
    load_measurement,
    mean_pairwise_jaccard,
    random_selection_stability,
    select_neurons,
    selection_stats,
    stability_summary,
)


def _write_measurement(tmp_path: Path, *, width: int = 64, n_layers: int = 3) -> Path:
    """Synthetic measurement artifact matching what run_measurement.py writes."""
    names = [f"layers.{i}.mlp.down_proj" for i in range(n_layers)]
    rng = np.random.default_rng(0)
    # Heavy-tailed importance, like the real thing: a few neurons hold most of
    # the mass, which is what makes a random draw's removed-importance vary.
    importance = np.stack([np.exp(np.linspace(0.0, 6.0, width)) for _ in range(n_layers)])
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
        "dataset": {"dataset_id": "wikitext", "split": "train"},
        "artifacts": {"npz": str(npz_path)},
    }
    json_path = npz_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def test_bootstrap_ci_brackets_the_mean():
    values = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.10, 0.12]
    lo, hi = bootstrap_mean_ci(values, num_resamples=500, seed=1)
    assert lo < float(np.mean(values)) < hi
    assert lo >= min(values) and hi <= max(values)
    # Degenerate inputs must not raise: a one-seed run is a legal (useless) run.
    assert bootstrap_mean_ci([0.3]) == (0.3, 0.3)
    assert all(np.isnan(x) for x in bootstrap_mean_ci([]))


def test_draws_vary_with_seed_and_are_reproducible(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    a = random_selection_stability(artifact, 0.25, num_seeds=8, base_seed=42)
    b = random_selection_stability(artifact, 0.25, num_seeds=8, base_seed=42)

    np.testing.assert_allclose(a.importance_removed, b.importance_removed)
    assert a.std > 0, "seeds must actually change which neurons are drawn"
    assert len(set(a.importance_removed.tolist())) > 1
    assert a.spread == pytest.approx(
        a.importance_removed.max() - a.importance_removed.min()
    )


def test_production_seed_is_the_first_draw(tmp_path):
    """The committed runs used seed 42; it has to be inside the distribution."""
    artifact = load_measurement(_write_measurement(tmp_path))
    result = random_selection_stability(artifact, 0.10, num_seeds=6, base_seed=42)

    assert result.seeds[0] == 42
    committed = selection_stats(artifact, select_neurons(artifact, "random", 0.10, seed=42))
    assert result.importance_removed[0] == pytest.approx(
        committed["importance_removed_fraction"]
    )
    assert result.num_masked == int(committed["num_masked"])


def test_pairwise_agreement_sits_at_the_chance_floor(tmp_path):
    """Independent draws must agree at chance; more would mean broken seeding."""
    artifact = load_measurement(_write_measurement(tmp_path))
    result = random_selection_stability(artifact, 0.25, num_seeds=12, base_seed=42)

    assert result.chance_jaccard > 0
    assert result.mean_pairwise_jaccard == pytest.approx(result.chance_jaccard, rel=0.35)


def test_guided_arm_sits_far_outside_the_seed_noise(tmp_path):
    """The point of the analysis: is the single-seed baseline good enough?"""
    artifact = load_measurement(_write_measurement(tmp_path))
    results = [
        random_selection_stability(artifact, ratio, num_seeds=12, base_seed=42)
        for ratio in (0.10, 0.25, 0.50)
    ]
    for r in results:
        assert r.guided_importance_removed < r.importance_removed.min()
        assert r.guided_gap_in_sds > 3.0

    summary = stability_summary(results)
    assert summary["num_ratios"] == 3
    assert summary["guided_outside_seed_noise"] is True
    assert summary["max_range_at_ratio"] in (0.10, 0.25, 0.50)
    assert summary["min_guided_gap_in_seed_sds"] > 3.0


def test_layer_filter_scopes_the_analysis(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    one_layer = ["layers.1.mlp.down_proj"]
    scoped = random_selection_stability(
        artifact, 0.25, num_seeds=4, base_seed=42, layer_filter=one_layer
    )
    everything = random_selection_stability(artifact, 0.25, num_seeds=4, base_seed=42)
    assert scoped.num_masked * 3 == everything.num_masked


def test_zero_ratio_is_degenerate_but_does_not_raise(tmp_path):
    artifact = load_measurement(_write_measurement(tmp_path))
    result = random_selection_stability(artifact, 0.0, num_seeds=4, base_seed=42)

    assert result.num_masked == 0
    assert result.std == 0.0
    assert result.spread == 0.0
    assert result.guided_importance_removed == 0.0
    assert result.guided_gap_in_sds == 0.0  # no spread and no difference
    assert result.mean_pairwise_jaccard == 1.0  # two empty sets agree completely


def test_mean_pairwise_jaccard_edge_cases():
    one = [{"a": np.array([1, 2, 3])}]
    assert np.isnan(mean_pairwise_jaccard(one))
    identical = [{"a": np.array([1, 2, 3])}, {"a": np.array([1, 2, 3])}]
    assert mean_pairwise_jaccard(identical) == pytest.approx(1.0)
    disjoint = [{"a": np.array([1, 2])}, {"a": np.array([3, 4])}]
    assert mean_pairwise_jaccard(disjoint) == pytest.approx(0.0)
