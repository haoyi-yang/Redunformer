from __future__ import annotations

import torch

ALGORITHM_NAME = "random_pruning"

PARAMETERS = {
    "sparsity": {
        "type": float,
        "prompt": "Sparsity (0-1)",
        "default": 0.2,
    },
    "seed": {
        "type": int,
        "prompt": "Random seed",
        "default": 42,
    },
}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def random_prune_model(model, sparsity: float, seed: int = 42) -> None:
    """
    Randomly sets a fraction of weights to zero.

    sparsity=0.2 -> 20% of weights become zero.
    """

    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be between 0 and 1")

    generator = torch.Generator()
    generator.manual_seed(seed)

    total_pruned = 0
    total_weights = 0

    with torch.no_grad():
        for param in model.parameters():

            if param.ndim == 0:
                continue

            total_weights += param.numel()

            mask = (
                torch.rand(
                    param.shape,
                    generator=generator,
                    device=param.device,
                )
                > sparsity
            )

            total_pruned += (~mask).sum().item()

            param.mul_(mask)

    actual = total_pruned / total_weights

    print(
        f"Pruned {total_pruned:,} / {total_weights:,} "
        f"weights ({actual:.2%})"
    )

def prune_model(model, sparsity: float, seed: int = 42):
    random_prune_model(
        model=model,
        sparsity=sparsity,
        seed=seed,
    )


def prune(model, **kwargs):
    random_prune_model(model=model, **kwargs)
    return model
