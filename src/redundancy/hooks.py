"""Forward hooks for FFN neuron activation capture (Weeks 5–8 measurements).

The "neuron" unit for Group 5 is one post-SwiGLU activation channel inside an
FFN block, i.e. one entry of ``act_fn(gate_proj(x)) * up_proj(x)``. That tensor
is exactly the *input* to the ``down_proj`` linear layer, so we capture it with a
forward **pre**-hook on the down projection. For GPT-2 style MLPs (single GELU,
no gate) the equivalent unit is the input to ``mlp.c_proj`` / ``mlp.fc2``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterator

import numpy as np
import torch

# Module-name suffixes that correspond to an FFN "down" projection. The input to
# these modules is the per-neuron (intermediate-width) activation we measure.
_DOWN_PROJ_SUFFIXES = ("down_proj", "c_proj", "fc2", "wo")
_LAYER_INDEX_RE = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")


class ActivationStore:
    """Accumulates module outputs registered via hooks.

    Kept for ad-hoc inspection; the measurement pipeline uses
    :class:`NeuronActivationCollector` instead so it never stores the full
    activation history.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def clear(self) -> None:
        for tensors in self._buffers.values():
            tensors.clear()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def register(self, module: torch.nn.Module, name: str) -> None:
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self._buffers[name].append(tensor.detach().cpu())

        self._handles.append(module.register_forward_hook(hook))

    def get(self, name: str) -> list[torch.Tensor]:
        return self._buffers[name]


def iter_mlp_modules(model: torch.nn.Module, name_filter: Callable[[str], bool] | None = None):
    """Yield (name, module) pairs likely corresponding to FFN / MLP blocks."""
    for name, module in model.named_modules():
        lname = name.lower()
        if "mlp" in lname or "ffn" in lname or "feed_forward" in lname:
            if name_filter is None or name_filter(name):
                yield name, module


def hook_gate_proj(model: torch.nn.Module, store: ActivationStore) -> None:
    """Capture gate-projection outputs into ``store`` (ad-hoc inspection)."""
    for name, module in model.named_modules():
        if name.lower().endswith("gate_proj"):
            store.register(module, name)


def hook_up_proj(model: torch.nn.Module, store: ActivationStore) -> None:
    """Capture up-projection outputs into ``store`` (ad-hoc inspection)."""
    for name, module in model.named_modules():
        if name.lower().endswith("up_proj"):
            store.register(module, name)


def hook_gate_up_proj(model: torch.nn.Module, store: ActivationStore) -> None:
    """Capture both gate- and up-projection outputs into ``store``."""
    for name, module in model.named_modules():
        if name.lower().endswith(("gate_proj", "up_proj")):
            store.register(module, name)


def _layer_index_from_name(name: str) -> int:
    """Best-effort extraction of the transformer layer index from a module name."""
    match = _LAYER_INDEX_RE.search("." + name + ".")
    return int(match.group(1)) if match else -1


def iter_ffn_down_projections(
    model: torch.nn.Module,
) -> Iterator[tuple[int, str, torch.nn.Module]]:
    """Yield ``(layer_index, name, module)`` for each FFN down-projection.

    The input to each yielded module is the post-activation (intermediate-width)
    neuron tensor. We restrict ``c_proj`` to modules living under an ``mlp`` so we
    do not accidentally pick up the attention output projection in GPT-2.
    """
    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, torch.nn.Linear) and not hasattr(module, "weight"):
            continue
        if not lname.endswith(_DOWN_PROJ_SUFFIXES):
            continue
        if lname.endswith(("c_proj", "fc2", "wo")) and "mlp" not in lname and "ffn" not in lname:
            continue
        yield _layer_index_from_name(name), name, module


def is_transposed_linear(module: torch.nn.Module) -> bool:
    """True for GPT-2 style ``Conv1D``, whose weight is ``[in_features, out_features]``.

    ``transformers.pytorch_utils.Conv1D`` computes ``x @ W + b`` instead of
    ``x @ W.T + b``, so its weight is the transpose of an ``nn.Linear``'s. Every
    caller here indexes neurons as *columns* of an ``[out, in]`` matrix, so the
    orientation has to be normalized before any reduction. Detected structurally
    (``nf`` output width and no ``in_features``) so re-exports and subclasses
    still match.
    """
    if isinstance(module, torch.nn.Linear):
        return False
    return hasattr(module, "nf") and not hasattr(module, "in_features")


def input_column_norms(module: torch.nn.Module) -> np.ndarray | None:
    """Return per-input-neuron L2 norms of a (down-projection) weight.

    For a weight oriented ``[out_features, in_features]`` the input neuron ``i``
    is column ``i``; its norm is ``||W[:, i]||_2``. Handles bitsandbytes 4-bit
    (``Params4bit``) weights by dequantizing first. Returns ``None`` if the
    weight is unavailable or cannot be dequantized (importance then falls back to
    activation RMS alone).
    """
    mat = _dense_weight(module)
    if mat is None:
        return None
    return mat.norm(dim=0).detach().cpu().numpy()


def _dense_weight(module: torch.nn.Module) -> torch.Tensor | None:
    """Materialize a weight as float32 ``[out_features, in_features]``.

    Dequantizes 4-bit weights and transposes GPT-2 ``Conv1D`` layers so callers
    can assume the ``nn.Linear`` layout.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    try:
        if weight.__class__.__name__ == "Params4bit":
            import bitsandbytes.functional as bnb_f

            dequant = bnb_f.dequantize_4bit(weight.data, weight.quant_state)
            mat = dequant.to(torch.float32)
        else:
            mat = weight.data.to(torch.float32)
    except Exception:
        return None
    if mat.ndim != 2:
        return None
    return mat.T.contiguous() if is_transposed_linear(module) else mat


def input_columns(module: torch.nn.Module, indices) -> np.ndarray | None:
    """Return ``W[:, indices]`` of a (down-projection) Linear as ``[n, out_features]``.

    Column ``i`` is the output direction that input neuron ``i`` writes into, so
    these are the vectors a neuron-merge has to redistribute. Handles
    bitsandbytes 4-bit weights by dequantizing first; returns ``None`` when the
    weight is unavailable.
    """
    mat = _dense_weight(module)
    if mat is None:
        return None
    idx = np.asarray(indices, dtype=np.int64).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= mat.shape[1]):
        return None
    return mat[:, torch.from_numpy(idx)].T.detach().cpu().numpy()


class NeuronActivationCollector:
    """Register pre-hooks on FFN down-projections and stream activations to stats.

    Activations are never accumulated in full: each captured batch is folded into
    a running :class:`~redundancy.metrics.activation_stats.NeuronActivationStats`
    instance via the provided callback, keeping memory bounded by the reservoir
    size rather than the calibration-set size.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        on_activation: Callable[[str, int, torch.Tensor], None],
    ) -> None:
        self._on_activation = on_activation
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layers: dict[str, int] = {}
        self._modules: dict[str, torch.nn.Module] = {}
        for layer_index, name, module in iter_ffn_down_projections(model):
            self.layers[name] = layer_index
            self._modules[name] = module
            self._handles.append(module.register_forward_pre_hook(self._make_hook(name, layer_index)))
        if not self.layers:
            raise RuntimeError(
                "No FFN down-projection modules found. Checked suffixes: "
                f"{_DOWN_PROJ_SUFFIXES}."
            )

    def _make_hook(self, name: str, layer_index: int):
        def hook(_module, inputs):
            tensor = inputs[0] if isinstance(inputs, tuple) else inputs
            # [batch, seq, width] -> [tokens, width], float32 on CPU.
            act = tensor.detach().reshape(-1, tensor.shape[-1]).to(torch.float32).cpu()
            self._on_activation(name, layer_index, act)

        return hook

    def input_weight_norms(self) -> dict[str, np.ndarray | None]:
        """Per-layer down-projection input-column norms (for neuron importance)."""
        return {name: input_column_norms(module) for name, module in self._modules.items()}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "NeuronActivationCollector":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()
