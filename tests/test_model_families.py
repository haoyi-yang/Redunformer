"""Offline unit tests for non-Qwen model families.

Everything the measurement and masking code does to an FFN assumes the
``nn.Linear`` weight layout ``[out_features, in_features]``. GPT-2 breaks that
assumption: its MLP is built from ``transformers.pytorch_utils.Conv1D``, which
computes ``x @ W`` and therefore stores ``[in_features, out_features]``. These
tests pin the orientation handling on a toy GPT-2-shaped model, so nothing is
downloaded and no GPU is required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch
from transformers.pytorch_utils import Conv1D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.hooks import (  # noqa: E402
    NeuronActivationCollector,
    input_column_norms,
    input_columns,
    is_transposed_linear,
    iter_ffn_down_projections,
)
from redundancy.pruning.masks import NeuronMasker, layer_width  # noqa: E402
from redundancy.pruning.merge import MergePlan, NeuronMerger  # noqa: E402

HIDDEN = 8
INTER = 16


class _ToyGPT2Attention(torch.nn.Module):
    """Only here so the tests can prove ``attn.c_proj`` is not mistaken for FFN."""

    def __init__(self) -> None:
        super().__init__()
        self.c_proj = Conv1D(HIDDEN, HIDDEN)

    def forward(self, x):
        return self.c_proj(x)


class _ToyGPT2MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = Conv1D(INTER, HIDDEN)
        self.c_proj = Conv1D(HIDDEN, INTER)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class _ToyGPT2Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _ToyGPT2Attention()
        self.mlp = _ToyGPT2MLP()

    def forward(self, x):
        return x + self.mlp(x + self.attn(x))


class _ToyGPT2(torch.nn.Module):
    """Mirrors GPT-2's ``transformer.h.<i>.mlp.c_proj`` naming."""

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([_ToyGPT2Block() for _ in range(n_layers)])

    def forward(self, x):
        for block in self.h:
            x = block(x)
        return x


@pytest.fixture()
def model() -> _ToyGPT2:
    torch.manual_seed(0)
    return _ToyGPT2().eval()


def test_conv1d_is_detected_as_transposed(model: _ToyGPT2) -> None:
    assert is_transposed_linear(model.h[0].mlp.c_proj) is True
    assert is_transposed_linear(torch.nn.Linear(4, 6)) is False


def test_discovery_finds_mlp_c_proj_and_skips_attention(model: _ToyGPT2) -> None:
    found = list(iter_ffn_down_projections(model))
    assert [name for _, name, _ in found] == ["h.0.mlp.c_proj", "h.1.mlp.c_proj"]
    assert [idx for idx, _, _ in found] == [0, 1]


def test_layer_width_is_the_intermediate_size_not_the_output(model: _ToyGPT2) -> None:
    """The regression this guards: ``weight.shape[1]`` on a Conv1D gives HIDDEN."""
    module = model.h[0].mlp.c_proj
    assert tuple(module.weight.shape) == (INTER, HIDDEN)
    assert layer_width(module) == INTER


def test_column_norms_reduce_over_the_output_axis(model: _ToyGPT2) -> None:
    module = model.h[0].mlp.c_proj
    norms = input_column_norms(module)
    expected = module.weight.data.norm(dim=1).numpy()
    assert norms.shape == (INTER,)
    np.testing.assert_allclose(norms, expected, rtol=1e-6)


def test_input_columns_returns_per_neuron_output_directions(model: _ToyGPT2) -> None:
    module = model.h[0].mlp.c_proj
    cols = input_columns(module, [2, 7])
    assert cols.shape == (2, HIDDEN)
    np.testing.assert_allclose(cols[0], module.weight.data[2].numpy(), rtol=1e-6)
    np.testing.assert_allclose(cols[1], module.weight.data[7].numpy(), rtol=1e-6)


def test_collector_captures_intermediate_width_activations(model: _ToyGPT2) -> None:
    seen: dict[str, tuple[int, ...]] = {}

    def record(name, _layer_index, act):
        seen[name] = tuple(act.shape)

    with NeuronActivationCollector(model, record) as collector:
        assert collector.layers == {"h.0.mlp.c_proj": 0, "h.1.mlp.c_proj": 1}
        model(torch.randn(2, 5, HIDDEN))

    assert seen["h.0.mlp.c_proj"] == (10, INTER)
    norms = collector.input_weight_norms()
    assert norms["h.0.mlp.c_proj"].shape == (INTER,)


def test_masking_matches_manually_zeroing_the_activation(model: _ToyGPT2) -> None:
    """A masked neuron must be indistinguishable from one that never fired."""
    x = torch.randn(1, 4, HIDDEN)
    dropped = [1, 5, 11]

    with torch.no_grad():
        baseline = model(x)

    mlp = model.h[0].mlp
    original_forward = mlp.forward

    def zeroed_forward(inp):
        hidden = mlp.act(mlp.c_fc(inp))
        hidden = hidden.clone()
        hidden[..., dropped] = 0.0
        return mlp.c_proj(hidden)

    mlp.forward = zeroed_forward
    with torch.no_grad():
        manual = model(x)
    mlp.forward = original_forward

    with NeuronMasker(model) as masker:
        assert masker.widths == {"h.0.mlp.c_proj": INTER, "h.1.mlp.c_proj": INTER}
        assert masker.apply({"h.0.mlp.c_proj": dropped}) == len(dropped)
        with torch.no_grad():
            masked = model(x)

    torch.testing.assert_close(masked, manual)
    assert (masked - baseline).abs().max() > 1e-6

    with torch.no_grad():
        torch.testing.assert_close(model(x), baseline)


def test_mask_index_beyond_intermediate_width_is_rejected(model: _ToyGPT2) -> None:
    with NeuronMasker(model) as masker:
        with pytest.raises(IndexError):
            masker.apply({"h.0.mlp.c_proj": [INTER]})


def test_merge_restores_an_exactly_duplicated_neuron(model: _ToyGPT2) -> None:
    """With ``h_drop == h_keep`` the merge is exact, so any transpose slip shows up."""
    keep, drop = 3, 9
    with torch.no_grad():
        model.h[0].mlp.c_fc.weight[:, drop] = model.h[0].mlp.c_fc.weight[:, keep]
        model.h[0].mlp.c_fc.bias[drop] = model.h[0].mlp.c_fc.bias[keep]

    x = torch.randn(1, 6, HIDDEN)
    with torch.no_grad():
        baseline = model(x)

    plan = MergePlan("h.0.mlp.c_proj", drop=drop, keep=keep, alpha=1.0, beta=0.0, r2=1.0)
    with NeuronMerger(model, [plan]) as merger:
        assert merger.num_merged() == 1
        with torch.no_grad():
            merged = model(x)

    torch.testing.assert_close(merged, baseline, rtol=1e-4, atol=1e-5)
