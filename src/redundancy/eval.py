"""Baseline evaluation: perplexity and optional lm-evaluation-harness."""

from __future__ import annotations

import math
from typing import Any

import torch
from tqdm import tqdm

from redundancy.models import LoadedModel


@torch.inference_mode()
def compute_perplexity(
    loaded: LoadedModel,
    texts: list[str],
    *,
    max_length: int = 2048,
    stride: int = 512,
    batch_size: int = 1,
) -> dict[str, float]:
    """Sliding-window perplexity on raw text (GPT-2 / HF style)."""
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="perplexity", unit="doc"):
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - begin
            if target_len < 2:
                break

            chunk_ids = input_ids[:, begin:end]
            labels = chunk_ids.clone()
            if begin > 0:
                labels[:, : prev_end - begin] = -100

            outputs = model(chunk_ids, labels=labels)
            valid = (labels != -100).sum().item()
            if valid == 0:
                prev_end = end
                continue

            nll = outputs.loss.item() * valid
            total_nll += nll
            total_tokens += valid
            prev_end = end
            if end == seq_len:
                break

    if total_tokens == 0:
        raise RuntimeError("No tokens scored for perplexity.")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return {
        "perplexity": ppl,
        "avg_nll": avg_nll,
        "num_tokens": float(total_tokens),
        "num_documents": float(len(texts)),
    }


def run_lm_eval(
    model_id: str,
    tasks: list[str],
    *,
    limit: int | None = 100,
    batch_size: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run lm-evaluation-harness on a HF model id."""
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model_id, batch_size=batch_size)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        random_seed=seed,
    )
    return results.get("results", results)


def run_lm_eval_loaded(
    loaded: LoadedModel,
    tasks: list[str],
    *,
    limit: int | None = 100,
    batch_size: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run lm-evaluation-harness on an already-loaded model.

    Unlike :func:`run_lm_eval`, which re-instantiates the model from its hub id,
    this evaluates the in-memory model so any registered hooks (e.g. neuron
    masks) stay active.
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=loaded.model, tokenizer=loaded.tokenizer, batch_size=batch_size)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        random_seed=seed,
    )
    return results.get("results", results)
