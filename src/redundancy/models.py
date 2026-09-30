"""Load Hugging Face causal language models (Qwen3.5 family)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from redundancy.config import REPO_ROOT
from redundancy.safetensors_pread import enable_pread_safetensors
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

enable_pread_safetensors()


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    load_in_4bit: bool


def _resolve_dtype(name: str) -> str | torch.dtype:
    key = (name or "auto").lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown torch_dtype: {name}")
    return _DTYPE_MAP[key]


def load_causal_lm(cfg: dict[str, Any]) -> LoadedModel:
    """Load a causal LM from a model config dict (see configs/models/*.yaml)."""
    use_fallback = bool(cfg.get("_use_fallback", False))
    model_id = cfg["fallback_model_id"] if use_fallback else cfg["model_id"]
    load_in_4bit = bool(cfg.get("fallback_load_in_4bit", False)) if use_fallback else bool(
        cfg.get("load_in_4bit", False)
    )
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    device_map = cfg.get("device_map", "auto")
    dtype = _resolve_dtype(cfg.get("torch_dtype", "auto"))
    local_dir = cfg.get("local_dir")
    if local_dir:
        local_dir = str((REPO_ROOT / local_dir).resolve())

    pretrained_source = local_dir or model_id
    pretrained_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if local_dir:
        pretrained_kwargs["local_files_only"] = bool(cfg.get("local_files_only", True))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_source, **pretrained_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if local_dir:
        model_kwargs["local_files_only"] = bool(cfg.get("local_files_only", True))

    if "max_memory" in cfg:
        model_kwargs["max_memory"] = cfg["max_memory"]
    elif cfg.get("max_memory_gb") is not None:
        gb = int(cfg["max_memory_gb"])
        model_kwargs["max_memory"] = {0: f"{gb}GiB", "cpu": "16GiB"}

    if load_in_4bit:
        compute_dtype_name = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = _resolve_dtype(compute_dtype_name)
        if compute_dtype == "auto":
            compute_dtype = torch.bfloat16
        quant_type = cfg.get("bnb_4bit_quant_type", "nf4")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        )
    elif dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    if use_fallback:
        arch = (cfg.get("fallback_architecture") or "causal_lm").lower()
    else:
        arch = (cfg.get("architecture") or "auto").lower()
    if arch == "causal_lm":
        loader = AutoModelForCausalLM
    elif arch in ("qwen3_5", "image_text_to_text"):
        loader = AutoModelForImageTextToText
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(pretrained_source, **model_kwargs)
        except ValueError:
            loader = AutoModelForImageTextToText
            model = loader.from_pretrained(pretrained_source, **model_kwargs)
        else:
            model.eval()
            return LoadedModel(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                load_in_4bit=load_in_4bit,
            )

    model = loader.from_pretrained(pretrained_source, **model_kwargs)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        load_in_4bit=load_in_4bit,
    )
