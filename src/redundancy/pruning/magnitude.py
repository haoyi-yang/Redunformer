from __future__ import annotations

import torch
import torch.nn as nn

ALGORITHM_NAME = "magnitude"

PARAMETERS = {
    "sparsity": {
        "type": float,
        "prompt": "Sparsity (0-1)",
        "default": 0.5,
    },
}


def apply_magnitude_pruning(model: nn.Module, pruning_ratio: float) -> None:
    model.eval()

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                if hasattr(module, "weight") and module.weight is not None:
                    weights = module.weight.data

                    abs_weights = torch.abs(weights)
                    threshold = torch.quantile(abs_weights, pruning_ratio)
                    mask = (abs_weights >= threshold).float()

                    module.weight.data.mul_(mask)


def magnitude_prune_model(model: nn.Module, sparsity: float, **_kwargs) -> nn.Module:
    apply_magnitude_pruning(model, sparsity)
    return model


def prune(model: nn.Module, **kwargs) -> nn.Module:
    return magnitude_prune_model(model, **kwargs)
