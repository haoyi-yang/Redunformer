"""Model loading utilities."""

from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device(device: str | None = None) -> torch.device:
    """Auto-select device: user-specified > CUDA > MPS > CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transformer_blocks(model):
    """Return the decoder transformer blocks for supported architectures."""

    # GPT-2 family
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # Qwen / Llama / Mistral-style architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    raise ValueError(
        f"Unsupported model architecture: {type(model).__name__}"
    )


def set_transformer_blocks(model, blocks):
    """Replace the decoder transformer blocks for supported architectures."""

    # GPT-2 family
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = blocks
        return

    # Qwen / Llama / Mistral-style architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = blocks
        return

    raise ValueError(
        f"Unsupported model architecture: {type(model).__name__}"
    )


def load_model(
    model_name: str = "gpt2-medium",
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    dev = get_device(device)

    if dtype is None:
        dtype = torch.float32 if dev.type == "cpu" else torch.float16

    print(f"Loading '{model_name}' on {dev} ({dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )

    model.to(dev).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = getattr(
        model.config,
        "num_hidden_layers",
        getattr(model.config, "n_layer", "?"),
    )

    print(
        f"  {n_params / 1e6:.1f}M params | {n_layers} layers | "
        f"vocab {model.config.vocab_size}"
    )

    return model, tokenizer


def prune_model_blocks(
    model: AutoModelForCausalLM,
    blocks_to_remove: list[int],
) -> AutoModelForCausalLM:
    unique_blocks = sorted(set(blocks_to_remove))

    if not unique_blocks:
        return model

    total_blocks = len(get_transformer_blocks(model))

    invalid = [
        idx
        for idx in unique_blocks
        if idx < 0 or idx >= total_blocks
    ]

    if invalid:
        raise IndexError(
            f"Block indices out of range for pruning: {invalid}"
        )

    keep_blocks = []
    new_layer_idx = 0

    for old_idx, block in enumerate(get_transformer_blocks(model)):
        if old_idx in unique_blocks:
            continue

        # GPT-2-style attention
        if hasattr(block, "attn") and hasattr(block.attn, "layer_idx"):
            block.attn.layer_idx = new_layer_idx

        # Qwen / Llama / Mistral-style attention
        if (
            hasattr(block, "self_attn")
            and hasattr(block.self_attn, "layer_idx")
        ):
            block.self_attn.layer_idx = new_layer_idx

        # Models with cross-attention
        if (
            hasattr(block, "crossattention")
            and hasattr(block.crossattention, "layer_idx")
        ):
            block.crossattention.layer_idx = new_layer_idx

        keep_blocks.append(block)
        new_layer_idx += 1

    set_transformer_blocks(
        model,
        torch.nn.ModuleList(keep_blocks),
    )

    if hasattr(model.config, "n_layer"):
        model.config.n_layer = len(keep_blocks)

    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(keep_blocks)

    return model


def clone_and_prune_model(
    model: AutoModelForCausalLM,
    blocks_to_remove: list[int],
) -> AutoModelForCausalLM:
    pruned_model = deepcopy(model)
    return prune_model_blocks(pruned_model, blocks_to_remove)
