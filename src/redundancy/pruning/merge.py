"""Neuron replacement / merge for near-duplicate pairs (Weeks 11–12).

Weeks 9–10 found that masking *one* twin of a near-duplicate pair is far cheaper
than masking both: the survivor absorbs much of the function. That is the
signature of a *replaceable* neuron, and this module tests it directly — instead
of simply deleting the redundant twin, we hand its contribution to its partner.

## The arithmetic

A neuron reaches the residual stream only through its ``down_proj`` column, so
an FFN output is ``sum_k h_k * W[:, k]``. For a near-duplicate pair (drop ``i``,
keep ``j``) we fit a scalar map on calibration activations:

    h_i  ~=  alpha * h_j + beta

Masking ``i`` removes ``h_i * W[:, i]``. Substituting the fit gives back

    (alpha * h_j + beta) * W[:, i]

so the merged layer behaves like ``W[:, j] <- W[:, j] + alpha * W[:, i]`` plus a
constant ``beta * W[:, i]``. We apply that as a **forward-hook correction rather
than a weight edit**, for two reasons: it keeps the "masking, not surgery"
contract from the proposal (parameter shapes and stored weights are untouched),
and it works on 4-bit models where writing back a merged column would mean
re-quantizing.

``alpha`` is fitted by least squares on real activations rather than derived
from the reported correlation, because the correlation alone fixes only the sign
and shape of the relationship, not the scale ratio between the two neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from redundancy.hooks import input_columns, iter_ffn_down_projections
from redundancy.pruning.masks import layer_width, resolve_layer_names


@dataclass
class MergePlan:
    """One replacement: mask ``drop``, route its contribution through ``keep``."""

    layer_name: str
    drop: int
    keep: int
    alpha: float
    beta: float
    r2: float
    correlation: float = 0.0

    def summary(self) -> dict:
        return {
            "layer": self.layer_name,
            "drop": self.drop,
            "keep": self.keep,
            "alpha": self.alpha,
            "beta": self.beta,
            "r2": self.r2,
            "correlation": self.correlation,
        }


class PairActivationCollector:
    """Capture only the neuron columns needed to fit merge plans.

    Hooks just the layers that contain pairs and keeps only the requested
    columns, so the calibration pass costs a few floats per token instead of a
    full activation matrix.
    """

    def __init__(self, model: torch.nn.Module, neurons_by_layer: Mapping[str, Sequence[int]]):
        found = {name: module for _, name, module in iter_ffn_down_projections(model)}
        alias = resolve_layer_names(found, neurons_by_layer.keys())
        self._columns: dict[str, np.ndarray] = {}
        self._buffers: dict[str, list[torch.Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for requested, indices in neurons_by_layer.items():
            actual = alias.get(requested)
            if actual is None:
                continue
            idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
            if idx.size == 0:
                continue
            self._columns[requested] = idx
            self._buffers[requested] = []
            self._handles.append(
                found[actual].register_forward_pre_hook(self._make_hook(requested, idx))
            )

    def _make_hook(self, key: str, idx: np.ndarray):
        cols = torch.from_numpy(idx)

        def hook(_module, inputs):
            tensor = inputs[0]
            flat = tensor.detach().reshape(-1, tensor.shape[-1])
            self._buffers[key].append(flat[:, cols.to(flat.device)].to(torch.float32).cpu())

        return hook

    def activations(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per layer: ``(neuron_indices, [tokens, n_neurons])`` activations."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, chunks in self._buffers.items():
            if not chunks:
                continue
            out[key] = (self._columns[key], torch.cat(chunks, dim=0).numpy())
        return out

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "PairActivationCollector":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()


def fit_linear_map(target: np.ndarray, source: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit ``target ~= alpha * source + beta``; returns (a, b, r2).

    ``r2`` says how much of the redundant neuron a merge can actually recover:
    at ``r2 = 1`` the survivor determines it exactly, and a low value means the
    pair is correlated but not substitutable.
    """
    target = np.asarray(target, dtype=np.float64).ravel()
    source = np.asarray(source, dtype=np.float64).ravel()
    if target.size < 2 or source.size != target.size:
        return 0.0, 0.0, 0.0
    var = float(source.var())
    if var <= 1e-12:
        return 0.0, float(target.mean()), 0.0
    alpha = float(np.cov(target, source, bias=True)[0, 1] / var)
    beta = float(target.mean() - alpha * source.mean())
    residual = target - (alpha * source + beta)
    sst = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / sst if sst > 0 else 0.0
    return alpha, beta, r2


def fit_merge_plans(
    activations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    pairs: Iterable,
) -> list[MergePlan]:
    """Fit one :class:`MergePlan` per duplicate pair from collected activations.

    ``pairs`` are :class:`~redundancy.pruning.pairs.DuplicatePair` instances.
    """
    plans: list[MergePlan] = []
    for pair in pairs:
        entry = activations.get(pair.layer_name)
        if entry is None:
            continue
        indices, matrix = entry
        lookup = {int(n): col for col, n in enumerate(indices)}
        if pair.drop not in lookup or pair.keep not in lookup:
            continue
        alpha, beta, r2 = fit_linear_map(
            matrix[:, lookup[pair.drop]], matrix[:, lookup[pair.keep]]
        )
        plans.append(
            MergePlan(
                layer_name=pair.layer_name,
                drop=pair.drop,
                keep=pair.keep,
                alpha=alpha,
                beta=beta,
                r2=r2,
                correlation=getattr(pair, "correlation", 0.0),
            )
        )
    return plans


class NeuronMerger:
    """Mask the redundant twin and route its contribution through its partner.

    Each hooked layer gets a pre-hook (zero the dropped channels, remember the
    surviving partners' values) and a forward hook that adds
    ``(alpha * h_keep + beta) x W[:, drop]`` back to the output.

    Set ``include_bias=False`` to drop the constant ``beta`` term, which
    isolates how much of the recovery comes from rescaling alone.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        plans: Sequence[MergePlan],
        *,
        include_bias: bool = True,
    ) -> None:
        found: dict[str, torch.nn.Module] = {}
        for _, name, module in iter_ffn_down_projections(model):
            if layer_width(module) is not None:
                found[name] = module
        alias = resolve_layer_names(found, {p.layer_name for p in plans})

        self.include_bias = include_bias
        self._plans: dict[str, list[MergePlan]] = {}
        self._modules: dict[str, torch.nn.Module] = {}
        self._drop_idx: dict[str, torch.Tensor] = {}
        self._keep_idx: dict[str, torch.Tensor] = {}
        self._alpha: dict[str, torch.Tensor] = {}
        self._beta: dict[str, torch.Tensor] = {}
        self._cols: dict[str, torch.Tensor] = {}
        self.skipped: list[MergePlan] = []

        for plan in plans:
            actual = alias.get(plan.layer_name)
            if actual is None:
                self.skipped.append(plan)
                continue
            self._plans.setdefault(actual, []).append(plan)
            self._modules[actual] = found[actual]

        for name, layer_plans in list(self._plans.items()):
            cols = input_columns(self._modules[name], [p.drop for p in layer_plans])
            if cols is None:
                # Without the dropped columns there is nothing to redistribute.
                self.skipped.extend(layer_plans)
                del self._plans[name]
                del self._modules[name]
                continue
            self._drop_idx[name] = torch.tensor([p.drop for p in layer_plans], dtype=torch.long)
            self._keep_idx[name] = torch.tensor([p.keep for p in layer_plans], dtype=torch.long)
            self._alpha[name] = torch.tensor([p.alpha for p in layer_plans], dtype=torch.float32)
            self._beta[name] = torch.tensor(
                [p.beta if include_bias else 0.0 for p in layer_plans], dtype=torch.float32
            )
            self._cols[name] = torch.from_numpy(cols)  # [n_plans, out_features]

        if not self._plans:
            raise RuntimeError("No merge plans could be attached to the model.")

        self._stash: dict[str, torch.Tensor] = {}
        self._cast: dict[tuple[str, torch.device, torch.dtype], tuple[torch.Tensor, ...]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in self._modules.items():
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._make_post_hook(name)))

    @property
    def plans(self) -> list[MergePlan]:
        return [p for group in self._plans.values() for p in group]

    def num_merged(self) -> int:
        return len(self.plans)

    def _tensors(self, name: str, device: torch.device, dtype: torch.dtype):
        key = (name, device, dtype)
        cached = self._cast.get(key)
        if cached is None:
            cached = (
                self._drop_idx[name].to(device),
                self._keep_idx[name].to(device),
                self._alpha[name].to(device=device, dtype=dtype),
                self._beta[name].to(device=device, dtype=dtype),
                self._cols[name].to(device=device, dtype=dtype),
            )
            self._cast[key] = cached
        return cached

    def _make_pre_hook(self, name: str):
        def hook(_module, inputs):
            tensor = inputs[0]
            drop_idx, keep_idx, _, _, _ = self._tensors(name, tensor.device, tensor.dtype)
            self._stash[name] = tensor.index_select(-1, keep_idx)
            mask = torch.ones(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
            mask[drop_idx] = 0
            return (tensor * mask,) + tuple(inputs[1:])

        return hook

    def _make_post_hook(self, name: str):
        def hook(_module, _inputs, output):
            keep_vals = self._stash.pop(name, None)
            if keep_vals is None:
                return None
            _, _, alpha, beta, cols = self._tensors(name, output.device, output.dtype)
            coefficients = keep_vals * alpha + beta  # [..., n_plans]
            return output + coefficients @ cols

        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._stash.clear()
        self._cast.clear()

    def __enter__(self) -> "NeuronMerger":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()


def plans_to_masked(plans: Sequence[MergePlan]) -> dict[str, np.ndarray]:
    """The mask-only selection matching a merge plan set, for the control arm."""
    grouped: dict[str, list[int]] = {}
    for plan in plans:
        grouped.setdefault(plan.layer_name, []).append(plan.drop)
    return {name: np.sort(np.asarray(idx, dtype=np.int64)) for name, idx in grouped.items()}


def merge_stats(plans: Sequence[MergePlan]) -> dict[str, float]:
    """Aggregate fit quality, which bounds how well a merge can possibly work."""
    if not plans:
        return {"num_plans": 0.0}
    r2 = np.array([p.r2 for p in plans], dtype=np.float64)
    alpha = np.array([p.alpha for p in plans], dtype=np.float64)
    return {
        "num_plans": float(len(plans)),
        "mean_r2": float(r2.mean()),
        "min_r2": float(r2.min()),
        "mean_abs_alpha": float(np.abs(alpha).mean()),
    }
