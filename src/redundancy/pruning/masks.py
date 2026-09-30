"""Neuron masking for the Weeks 9–10 intervention.

We *mask*, we do not structurally remove: parameter shapes never change, so no
runtime or FLOP claim is made (proposal Q1/Q8). A neuron is one post-SwiGLU
activation channel, which enters the residual stream only through ``down_proj``.
Zeroing that channel on the way into ``down_proj`` therefore removes the
neuron's entire contribution, which is exactly the ablation the measurement
stage ranks neurons for.

The masker registers one forward pre-hook per FFN layer and keeps them for its
whole lifetime, so a removal-ratio sweep can re-point the masks between
evaluations without re-registering hooks:

    with NeuronMasker(model) as masker:
        for ratio in ratios:
            masker.apply(select_neurons(artifact, "importance", ratio))
            evaluate(...)
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from redundancy.hooks import is_transposed_linear, iter_ffn_down_projections


def resolve_layer_names(
    discovered: Iterable[str], requested: Iterable[str]
) -> dict[str, str]:
    """Map requested layer names onto the model's actual module names.

    A measurement artifact records names as they appear on the bare model
    (``model.layers.0.mlp.down_proj``), but wrappers re-prefix them — PEFT, for
    instance, exposes ``base_model.model.model.layers.0.mlp.down_proj``. Exact
    matches win; otherwise a unique suffix match is accepted. Ambiguous or
    missing names are dropped, matching the masker's tolerance for artifacts
    recorded on a superset of layers.
    """
    available = set(discovered)
    mapping: dict[str, str] = {}
    for name in requested:
        if name in available:
            mapping[name] = name
            continue
        candidates = [
            other
            for other in available
            if other.endswith("." + name) or name.endswith("." + other)
        ]
        if len(candidates) == 1:
            mapping[name] = candidates[0]
    return mapping


def layer_width(module: torch.nn.Module) -> int | None:
    """Number of input neurons of a down-projection.

    The declared width is preferred because bitsandbytes 4-bit weights are
    stored packed, so ``weight.shape`` does not give the logical width.
    ``nn.Linear`` calls it ``in_features``; GPT-2's ``Conv1D`` calls it ``nx``
    and stores its weight transposed, so the shape fallback picks the axis that
    matches the module's layout.
    """
    for attr in ("in_features", "nx"):
        width = getattr(module, attr, None)
        if isinstance(width, int) and width > 0:
            return width
    weight = getattr(module, "weight", None)
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[0] if is_transposed_linear(module) else weight.shape[1])
    return None


class NeuronMasker:
    """Zero selected FFN neurons via forward pre-hooks on the down projections."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_names: Sequence[str] | None = None,
    ) -> None:
        found: dict[str, torch.nn.Module] = {}
        widths: dict[str, int] = {}
        for _, name, module in iter_ffn_down_projections(model):
            width = layer_width(module)
            if width is None:
                continue
            found[name] = module
            widths[name] = width

        self._alias: dict[str, str] = {}
        if layer_names is not None:
            self._alias = resolve_layer_names(found, layer_names)
            keep = set(self._alias.values())
            found = {name: mod for name, mod in found.items() if name in keep}
            widths = {name: w for name, w in widths.items() if name in keep}

        self._modules = found
        self._widths = widths
        if not self._modules:
            raise RuntimeError("No maskable FFN down-projections found in the model.")

        self._masks: dict[str, torch.Tensor] = {}
        self._cache: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}
        self._handles = [
            module.register_forward_pre_hook(self._make_hook(name))
            for name, module in self._modules.items()
        ]

    @property
    def layer_names(self) -> list[str]:
        return list(self._modules)

    @property
    def widths(self) -> dict[str, int]:
        return dict(self._widths)

    def _make_hook(self, name: str):
        def hook(_module, inputs):
            mask = self._masks.get(name)
            if mask is None:
                return None
            tensor = inputs[0]
            key = (name, tensor.device, tensor.dtype)
            cached = self._cache.get(key)
            if cached is None:
                cached = mask.to(device=tensor.device, dtype=tensor.dtype)
                self._cache[key] = cached
            return (tensor * cached,) + tuple(inputs[1:])

        return hook

    def apply(self, masked: Mapping[str, Sequence[int]]) -> int:
        """Mask the given neuron indices per layer, replacing any previous masks.

        Unknown layer names are ignored so a measurement artifact recorded on a
        superset of layers (e.g. a hybrid model whose vision tower never fires)
        can be replayed unchanged. Returns the total number of masked neurons.
        """
        self.clear()
        total = 0
        for requested, indices in masked.items():
            name = self._alias.get(requested, requested)
            module_width = self._widths.get(name)
            if module_width is None:
                continue
            idx = np.asarray(indices, dtype=np.int64).ravel()
            if idx.size == 0:
                continue
            if idx.min() < 0 or idx.max() >= module_width:
                raise IndexError(
                    f"Neuron index out of range for {name} (width {module_width}): "
                    f"[{idx.min()}, {idx.max()}]"
                )
            mask = torch.ones(module_width, dtype=torch.float32)
            mask[torch.from_numpy(idx)] = 0.0
            self._masks[name] = mask
            total += int(idx.size)
        return total

    def clear(self) -> None:
        self._masks.clear()
        self._cache.clear()

    def masked_counts(self) -> dict[str, int]:
        return {name: int((mask == 0).sum()) for name, mask in self._masks.items()}

    def masked_fraction(self) -> float:
        """Masked neurons as a fraction of all maskable neurons."""
        total = sum(self._widths.values())
        if total == 0:
            return 0.0
        return sum(self.masked_counts().values()) / total

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self) -> "NeuronMasker":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()
