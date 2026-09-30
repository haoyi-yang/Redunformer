"""Offline unit tests for the Weeks 11-12 stage.

Covers the three deliverables - neuron replacement (merge), LoRA recovery, and
cross-task overlap (H3) - on toy models and synthetic artifacts, so nothing is
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

from redundancy import plotting  # noqa: E402
from redundancy.data import TEXT_BUILDERS  # noqa: E402
from redundancy.hooks import input_columns  # noqa: E402
from redundancy.pruning.overlap import average_ranks  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    DuplicatePair,
    MergePlan,
    NeuronMasker,
    NeuronMerger,
    PairActivationCollector,
    compare_artifacts,
    expected_random_jaccard,
    fit_linear_map,
    fit_merge_plans,
    jaccard,
    load_measurement,
    merge_stats,
    overlap_summary,
    plans_to_masked,
    resolve_layer_names,
    spearman,
)
from redundancy.recovery import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    RecoveryConfig,
    _lr_scale,
    build_lora_model,
    recovery_fraction,
    resolve_target_modules,
    train_lora,
    trainable_parameter_count,
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


def _duplicate_neuron(model: _ToyModel, layer: int, source: int, target: int) -> None:
    """Make ``target``'s post-activation identical to ``source``'s in one layer.

    Copying both projection rows makes the two neurons exact twins, which is the
    limiting case a merge should reproduce with zero error.
    """
    mlp = model.layers[layer].mlp
    with torch.no_grad():
        mlp.gate_proj.weight[target] = mlp.gate_proj.weight[source]
        mlp.up_proj.weight[target] = mlp.up_proj.weight[source]


# --------------------------------------------------------------------------
# merge: fitting the substitution
# --------------------------------------------------------------------------


def test_fit_linear_map_recovers_known_relation():
    rng = np.random.default_rng(0)
    source = rng.normal(size=500)
    target = 2.5 * source - 0.75
    alpha, beta, r2 = fit_linear_map(target, source)
    assert alpha == pytest.approx(2.5, abs=1e-9)
    assert beta == pytest.approx(-0.75, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_linear_map_reports_low_r2_for_unrelated_signals():
    rng = np.random.default_rng(1)
    alpha, beta, r2 = fit_linear_map(rng.normal(size=800), rng.normal(size=800))
    assert abs(alpha) < 0.2
    assert r2 < 0.1


def test_fit_linear_map_handles_degenerate_input():
    # A constant survivor carries no information: fall back to the mean, r2 = 0.
    alpha, beta, r2 = fit_linear_map(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    assert alpha == 0.0
    assert beta == pytest.approx(2.0)
    assert r2 == 0.0
    assert fit_linear_map(np.array([1.0]), np.array([1.0])) == (0.0, 0.0, 0.0)


def test_input_columns_matches_weight_slice():
    layer = torch.nn.Linear(6, 4, bias=False)
    cols = input_columns(layer, [1, 4])
    assert cols.shape == (2, 4)
    np.testing.assert_allclose(cols[0], layer.weight.detach().numpy()[:, 1], atol=1e-6)
    np.testing.assert_allclose(cols[1], layer.weight.detach().numpy()[:, 4], atol=1e-6)
    assert input_columns(layer, [99]) is None


def test_pair_activation_collector_captures_requested_columns():
    torch.manual_seed(0)
    model = _ToyModel()
    x = torch.randn(3, 5, 8)
    name = "layers.0.mlp.down_proj"
    with PairActivationCollector(model, {name: [2, 5]}) as collector:
        with torch.no_grad():
            model(x)
            model(x)
    indices, matrix = collector.activations()[name]
    np.testing.assert_array_equal(indices, [2, 5])
    # Two forward passes over 3x5 positions.
    assert matrix.shape == (2 * 3 * 5, 2)


# --------------------------------------------------------------------------
# merge: applying the substitution
# --------------------------------------------------------------------------


def test_merger_with_identity_plan_is_lossless():
    """drop == keep with alpha=1 removes a neuron and adds its exact contribution.

    This pins the whole hook path - column extraction, stashing, and the output
    correction - against the unmasked forward pass.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        clean = model(x).clone()

    plan = MergePlan("layers.0.mlp.down_proj", drop=3, keep=3, alpha=1.0, beta=0.0, r2=1.0)
    with NeuronMerger(model, [plan]):
        with torch.no_grad():
            merged = model(x)
    assert torch.allclose(merged, clean, atol=1e-5)

    # Hooks are gone afterwards.
    with torch.no_grad():
        assert torch.allclose(model(x), clean, atol=1e-6)


def test_merge_recovers_exact_duplicate_where_masking_does_not():
    """End-to-end: collect, fit, merge on a genuinely duplicated neuron."""
    torch.manual_seed(0)
    model = _ToyModel()
    _duplicate_neuron(model, layer=0, source=2, target=5)
    x = torch.randn(4, 6, 8)
    with torch.no_grad():
        clean = model(x).clone()

    name = "layers.0.mlp.down_proj"
    with PairActivationCollector(model, {name: [2, 5]}) as collector:
        with torch.no_grad():
            model(x)
    plans = fit_merge_plans(
        collector.activations(),
        [DuplicatePair(layer_name=name, drop=5, keep=2, correlation=1.0)],
    )
    assert len(plans) == 1
    assert plans[0].alpha == pytest.approx(1.0, abs=1e-4)
    assert plans[0].r2 == pytest.approx(1.0, abs=1e-6)

    with NeuronMasker(model) as masker:
        masker.apply(plans_to_masked(plans))
        with torch.no_grad():
            masked = model(x)
    mask_error = float((masked - clean).abs().max())
    assert mask_error > 1e-4  # masking a live neuron really does change the output

    with NeuronMerger(model, plans):
        with torch.no_grad():
            merged = model(x)
    merge_error = float((merged - clean).abs().max())
    assert merge_error < 1e-4
    assert merge_error < mask_error


def test_merger_bias_term_is_optional():
    torch.manual_seed(0)
    model = _ToyModel()
    x = torch.randn(2, 3, 8)
    plan = MergePlan("layers.0.mlp.down_proj", drop=1, keep=4, alpha=0.5, beta=2.0, r2=0.9)
    with NeuronMerger(model, [plan], include_bias=True):
        with torch.no_grad():
            with_bias = model(x).clone()
    with NeuronMerger(model, [plan], include_bias=False):
        with torch.no_grad():
            without_bias = model(x).clone()
    assert not torch.allclose(with_bias, without_bias)


def test_merger_skips_layers_absent_from_the_model():
    model = _ToyModel()
    good = MergePlan("layers.0.mlp.down_proj", drop=1, keep=2, alpha=1.0, beta=0.0, r2=1.0)
    missing = MergePlan("layers.99.mlp.down_proj", drop=1, keep=2, alpha=1.0, beta=0.0, r2=1.0)
    with NeuronMerger(model, [good, missing]) as merger:
        assert merger.num_merged() == 1
        assert [p.layer_name for p in merger.skipped] == ["layers.99.mlp.down_proj"]
    with pytest.raises(RuntimeError):
        NeuronMerger(model, [missing])


def test_plans_to_masked_and_merge_stats():
    plans = [
        MergePlan("a", drop=5, keep=1, alpha=1.0, beta=0.0, r2=0.9),
        MergePlan("a", drop=2, keep=3, alpha=-2.0, beta=0.0, r2=0.5),
        MergePlan("b", drop=7, keep=8, alpha=1.0, beta=0.0, r2=0.7),
    ]
    masked = plans_to_masked(plans)
    np.testing.assert_array_equal(masked["a"], [2, 5])
    np.testing.assert_array_equal(masked["b"], [7])
    stats = merge_stats(plans)
    assert stats["num_plans"] == 3
    assert stats["mean_r2"] == pytest.approx(0.7)
    assert stats["min_r2"] == pytest.approx(0.5)
    assert stats["mean_abs_alpha"] == pytest.approx(4.0 / 3.0)
    assert merge_stats([])["num_plans"] == 0


# --------------------------------------------------------------------------
# layer-name resolution across wrappers
# --------------------------------------------------------------------------


def test_resolve_layer_names_handles_wrapper_prefixes():
    discovered = [
        "base_model.model.model.layers.0.mlp.down_proj",
        "base_model.model.model.layers.1.mlp.down_proj",
    ]
    mapping = resolve_layer_names(discovered, ["model.layers.1.mlp.down_proj"])
    assert mapping == {
        "model.layers.1.mlp.down_proj": "base_model.model.model.layers.1.mlp.down_proj"
    }
    # Exact matches win, unknown names are simply dropped.
    assert resolve_layer_names(discovered, [discovered[0]]) == {discovered[0]: discovered[0]}
    assert resolve_layer_names(discovered, ["nope"]) == {}


# --------------------------------------------------------------------------
# H3: cross-task overlap
# --------------------------------------------------------------------------


def test_jaccard_and_chance_floor():
    assert jaccard([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert jaccard([1, 2], [3, 4]) == pytest.approx(0.0)
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 4)
    assert jaccard([], []) == pytest.approx(1.0)
    # Two independent 10%-sized draws from 1000 neurons agree on ~5% of the union.
    assert expected_random_jaccard(1000, 100) == pytest.approx(100 / 19 / 100, rel=1e-6)
    assert expected_random_jaccard(100, 100) == pytest.approx(1.0)
    assert expected_random_jaccard(0, 10) == 0.0


def test_average_ranks_shares_rank_within_ties():
    np.testing.assert_allclose(average_ranks(np.array([10.0, 20.0, 30.0])), [0.0, 1.0, 2.0])
    # Two-way tie at the bottom takes the mean of ranks 0 and 1.
    np.testing.assert_allclose(average_ranks(np.array([5.0, 5.0, 9.0])), [0.5, 0.5, 2.0])
    np.testing.assert_allclose(average_ranks(np.ones(4)), [1.5, 1.5, 1.5, 1.5])


def test_spearman_endpoints():
    values = np.array([3.0, 1.0, 2.0, 5.0])
    assert spearman(values, values) == pytest.approx(1.0)
    assert spearman(values, -values) == pytest.approx(-1.0)
    # A flat vector has no ranking, so the correlation is undefined rather than 0.
    assert np.isnan(spearman(np.ones(4), values))
    assert np.isnan(spearman(np.array([1.0]), np.array([1.0])))
    assert np.isnan(spearman(np.arange(4.0), np.arange(5.0)))


def _write_artifact(path: Path, importance: np.ndarray, *, model_id="toy/model") -> Path:
    n_layers, width = importance.shape
    names = [f"model.layers.{i}.mlp.down_proj" for i in range(n_layers)]
    npz_path = path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        layer_index=np.arange(n_layers),
        names=np.array(names),
        importance=importance,
        freq_rms=importance,
        mean_abs=importance,
        rms=importance,
    )
    path.write_text(
        json.dumps({"model": {"model_id": model_id}, "artifacts": {"npz": str(npz_path)}}),
        encoding="utf-8",
    )
    return path


def test_compare_artifacts_identical_and_reversed(tmp_path):
    importance = np.stack([np.arange(20, dtype=float) + 1 for _ in range(3)])
    same = load_measurement(
        _write_artifact(tmp_path / "measurement_toy_a.json", importance.copy())
    )
    reversed_ = load_measurement(
        _write_artifact(tmp_path / "measurement_toy_b.json", importance[:, ::-1].copy())
    )

    identical = compare_artifacts(same, same, ratio=0.25)
    assert len(identical) == 3
    for entry in identical:
        assert entry.count == 5
        assert entry.jaccard_bottom == pytest.approx(1.0)
        assert entry.jaccard_top == pytest.approx(1.0)
        assert entry.spearman == pytest.approx(1.0)
    assert overlap_summary(identical)["mean_jaccard_bottom"] == pytest.approx(1.0)

    opposed = compare_artifacts(same, reversed_, ratio=0.25)
    for entry in opposed:
        # The cheapest neurons under one corpus are the dearest under the other.
        assert entry.jaccard_bottom == pytest.approx(0.0)
        assert entry.spearman == pytest.approx(-1.0)
    summary = overlap_summary(opposed)
    assert summary["mean_spearman"] == pytest.approx(-1.0)
    assert summary["mean_chance_jaccard"] > 0
    assert overlap_summary([])["num_layers"] == 0


def test_piqa_prompt_builder_uses_the_labelled_solution():
    builder = TEXT_BUILDERS["piqa"]
    row = {"goal": "boil water", "sol1": "use a pot", "sol2": "use a sock", "label": 0}
    assert builder(row) == "Question: boil water\nAnswer: use a pot"
    assert builder({**row, "label": 1}).endswith("use a sock")


def test_calibration_label_reports_the_corpus(tmp_path):
    """Guards against replaying a ranking on the corpus it was not measured on."""
    path = _write_artifact(tmp_path / "measurement_toy_c.json", np.ones((1, 4)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset"] = {
        "dataset_id": "baber/piqa",
        "split": "train",
        "text_builder": "piqa",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_measurement(path).calibration_label() == "baber/piqa / train / prompts=piqa"

    bare = _write_artifact(tmp_path / "measurement_toy_d.json", np.ones((1, 4)))
    assert load_measurement(bare).calibration_label() == "unknown corpus"


def test_plot_overlap_by_depth_writes_a_figure(tmp_path):
    out = plotting.plot_overlap_by_depth(
        [0, 1, 2],
        {"masked set (bottom)": [0.3, 0.4, 0.5], "most active (top)": [0.2, 0.3, 0.4]},
        tmp_path / "figures" / "overlap.png",
        chance=0.05,
    )
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------
# recovery: LoRA fine-tuning with masks live
# --------------------------------------------------------------------------


def _tiny_causal_lm():
    """A 2-layer Qwen2 built from config - no download, same module names."""
    from transformers import AutoModelForCausalLM, Qwen2Config

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config)


def test_resolve_target_modules_keeps_only_existing_names():
    model = _tiny_causal_lm()
    assert resolve_target_modules(model, DEFAULT_TARGET_MODULES) == list(DEFAULT_TARGET_MODULES)
    assert resolve_target_modules(model, ["q_proj", "not_a_module"]) == ["q_proj"]
    assert resolve_target_modules(model, ["not_a_module"]) == []


def test_lr_scale_warms_up_then_decays():
    assert _lr_scale(0, 100, 10) == pytest.approx(0.1)
    assert _lr_scale(9, 100, 10) == pytest.approx(1.0)
    assert _lr_scale(10, 100, 10) == pytest.approx(1.0)
    assert _lr_scale(99, 100, 10) < 0.05
    # No warmup requested: start at full rate.
    assert _lr_scale(0, 10, 0) == pytest.approx(1.0)


def test_lora_leaves_down_proj_unwrapped_so_masks_survive():
    """The mask hooks must still fire after PEFT rewrites the module tree."""
    model = _tiny_causal_lm()
    cfg = RecoveryConfig(rank=4, steps=1, grad_accum=1)
    peft_model = build_lora_model(model, cfg)

    trainable, total = trainable_parameter_count(peft_model)
    assert 0 < trainable < total

    down = peft_model.get_submodule("base_model.model.model.layers.0.mlp.down_proj")
    assert isinstance(down, torch.nn.Linear)  # untouched by LoRA

    ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        unmasked = peft_model(ids).logits.clone()

    # Artifact-style names, as written by run_measurement.py on the bare model.
    artifact_names = [f"model.layers.{i}.mlp.down_proj" for i in range(2)]
    with NeuronMasker(peft_model, layer_names=artifact_names) as masker:
        assert len(masker.layer_names) == 2
        assert masker.apply({artifact_names[0]: [0, 1, 2, 3]}) == 4
        with torch.no_grad():
            masked = peft_model(ids).logits
        assert not torch.allclose(masked, unmasked)


def test_train_lora_updates_only_adapters_with_masks_active():
    model = _tiny_causal_lm()
    cfg = RecoveryConfig(rank=4, steps=3, grad_accum=2, batch_size=2, warmup_steps=1)
    peft_model = build_lora_model(model, cfg)

    base_weight = peft_model.get_submodule(
        "base_model.model.model.layers.0.mlp.down_proj"
    ).weight.detach().clone()
    base_q = peft_model.get_submodule(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer"
    ).weight.detach().clone()

    blocks = [torch.randint(0, 64, (1, 16)) for _ in range(5)]
    artifact_names = [f"model.layers.{i}.mlp.down_proj" for i in range(2)]
    with NeuronMasker(peft_model, layer_names=artifact_names) as masker:
        masker.apply({artifact_names[1]: [0, 1]})
        history = train_lora(peft_model, blocks, cfg)

    assert history.steps == 3
    assert len(history.losses) == 3
    assert all(np.isfinite(history.losses))
    assert history.summary()["final_loss"] == history.losses[-1]

    # Frozen base weights really are frozen; the adapters carry the update.
    after_down = peft_model.get_submodule(
        "base_model.model.model.layers.0.mlp.down_proj"
    ).weight.detach()
    after_q = peft_model.get_submodule(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer"
    ).weight.detach()
    assert torch.allclose(after_down, base_weight)
    assert torch.allclose(after_q, base_q)
    lora_b = peft_model.get_submodule(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default"
    ).weight.detach()
    assert lora_b.abs().sum() > 0  # started at zero, so training moved it
    assert not peft_model.training


def test_train_lora_requires_blocks_and_trainable_parameters():
    cfg = RecoveryConfig(steps=1, grad_accum=1)
    model = _tiny_causal_lm()
    with pytest.raises(ValueError):
        train_lora(build_lora_model(model, cfg), [], cfg)
    frozen = _tiny_causal_lm()
    for param in frozen.parameters():
        param.requires_grad_(False)
    with pytest.raises(RuntimeError):
        train_lora(frozen, [torch.randint(0, 64, (1, 8))], cfg)


def test_recovery_fraction_semantics():
    assert recovery_fraction(10.0, 15.0, 10.0) == pytest.approx(1.0)  # fully restored
    assert recovery_fraction(10.0, 15.0, 15.0) == pytest.approx(0.0)  # no change
    assert recovery_fraction(10.0, 15.0, 12.5) == pytest.approx(0.5)
    assert recovery_fraction(10.0, 15.0, 16.0) == pytest.approx(-0.2)  # made it worse
    assert recovery_fraction(10.0, 10.0, 9.0) is None  # nothing to recover
