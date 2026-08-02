"""Model loading utilities."""

from copy import deepcopy

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


def get_device(device: str | None = None) -> torch.device:
    """Auto-select device: user-specified > CUDA > MPS > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name: str = "gpt2",
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[GPT2LMHeadModel, AutoTokenizer]:
    dev = get_device(device)
    if dtype is None:
        dtype = torch.float32 if dev.type == "cpu" else torch.float16

    print(f"Loading '{model_name}' on {dev} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, dtype=dtype)
    model.to(dev).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = getattr(model.config, "n_layer", "?")
    print(f"  {n_params / 1e6:.1f}M params | {n_layers} layers | vocab {model.config.vocab_size}")

    return model, tokenizer


def prune_model_blocks(model: GPT2LMHeadModel, blocks_to_remove: list[int]) -> GPT2LMHeadModel:
    unique_blocks = sorted(set(blocks_to_remove))
    if not unique_blocks:
        return model

    total_blocks = len(model.transformer.h)
    invalid = [idx for idx in unique_blocks if idx < 0 or idx >= total_blocks]
    if invalid:
        raise IndexError(f"Block indices out of range for pruning: {invalid}")

    keep_blocks = []
    new_layer_idx = 0
    for old_idx, block in enumerate(model.transformer.h):
        if old_idx in unique_blocks:
            continue
        if hasattr(block, "attn") and hasattr(block.attn, "layer_idx"):
            block.attn.layer_idx = new_layer_idx
        if hasattr(block, "crossattention") and hasattr(block.crossattention, "layer_idx"):
            block.crossattention.layer_idx = new_layer_idx
        keep_blocks.append(block)
        new_layer_idx += 1

    model.transformer.h = torch.nn.ModuleList(keep_blocks)
    model.config.n_layer = len(keep_blocks)
    return model


def clone_and_prune_model(model: GPT2LMHeadModel, blocks_to_remove: list[int]) -> GPT2LMHeadModel:
    pruned_model = deepcopy(model)
    return prune_model_blocks(pruned_model, blocks_to_remove)
