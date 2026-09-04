"""
Per-row magnitude pruning (DSnoT-compatible mask layout).

Matches the unstructured initial mask used by the official DSnoT repo when
``initial_method=magnitude``: for each output row, zero the
``⌊sparsity · C_in⌋`` smallest-|W| entries (equal zero-count per row).

This differs from ``magnitude.py``, which uses one global threshold over the
flattened weight matrix and therefore produces uneven per-row sparsity —
masks that DSnoT ``base=existing`` cannot refine.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import transformers
except ImportError:  # pragma: no cover
    transformers = None

ALGORITHM_NAME = "magnitude_dsnot"

PARAMETERS = {
    "sparsity": {
        "type": float,
        "prompt": "Sparsity (0-1)",
        "default": 0.5,
    },
}


def _weight_as_out_in(module: nn.Module) -> torch.Tensor:
    W = module.weight.data
    if isinstance(module, nn.Conv2d):
        return W.flatten(1)
    if transformers is not None and isinstance(module, transformers.Conv1D):
        return W.t()
    return W


def _write_weight(module: nn.Module, W_out: torch.Tensor) -> None:
    if transformers is not None and isinstance(module, transformers.Conv1D):
        module.weight.data = W_out.t().to(dtype=module.weight.dtype)
    else:
        module.weight.data = W_out.reshape(module.weight.shape).to(
            dtype=module.weight.dtype
        )


@torch.no_grad()
def apply_magnitude_dsnot_pruning(model: nn.Module, sparsity: float) -> None:
    """Zero the smallest-|w| fraction of weights in each output row."""
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")

    model.eval()
    total_pruned = 0
    total_weights = 0

    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = transformers is not None and isinstance(
            module, transformers.Conv1D
        )
        is_conv2d = isinstance(module, nn.Conv2d)
        if not (is_linear or is_conv1d or is_conv2d):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        W = _weight_as_out_in(module).float()
        cols = W.shape[1]
        k = int(cols * sparsity)
        total_weights += W.numel()
        if k <= 0:
            continue

        abs_w = torch.abs(W)
        indices = torch.sort(abs_w, dim=-1, stable=True).indices[:, :k]
        mask = torch.ones_like(W, dtype=torch.bool)
        mask.scatter_(1, indices, False)
        total_pruned += int((~mask).sum().item())
        W_out = W.clone()
        W_out[~mask] = 0
        _write_weight(module, W_out)

    actual = total_pruned / total_weights if total_weights else 0.0
    print(
        f"magnitude_dsnot: pruned {total_pruned:,} / {total_weights:,} "
        f"weights ({actual:.2%}, target sparsity={sparsity})"
    )


def magnitude_dsnot_prune_model(
    model: nn.Module, sparsity: float, **_kwargs
) -> nn.Module:
    apply_magnitude_dsnot_pruning(model, sparsity)
    return model


def prune(model: nn.Module, **kwargs) -> nn.Module:
    return magnitude_dsnot_prune_model(model, **kwargs)
