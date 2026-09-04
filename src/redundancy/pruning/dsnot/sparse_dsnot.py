"""
SparseGPT initial mask with DSnoT-compatible per-row layout.

Same calibration / OBS metric as SparseGPT, but zeros exactly
``⌊sparsity · C_in⌋`` entries per output row (official DSnoT
``initial_method=sparsegpt``), so ``dsnot base=existing`` can refine it.

The classic ``sparsegpt`` algorithm keeps block-flatten thresholds and is
*not* safe as a DSnoT existing-mask base.
"""

from __future__ import annotations

from redundancy.pruning.sparsegpt import sparsegpt_prune_model

ALGORITHM_NAME = "sparse_dsnot"

PARAMETERS = {
    "sparsity": {
        "type": float,
        "prompt": "Sparsity (0-1)",
        "default": 0.5,
    },
    "nsamples": {
        "type": int,
        "prompt": "Calibration samples",
        "default": 128,
    },
    "seqlen": {
        "type": int,
        "prompt": "Calibration sequence length",
        "default": 2048,
    },
    "percdamp": {
        "type": float,
        "prompt": "Hessian dampening fraction",
        "default": 0.01,
    },
    "dataset": {
        "type": str,
        "prompt": "Calibration dataset (wikitext2|c4)",
        "default": "wikitext2",
    },
    "seed": {
        "type": int,
        "prompt": "Random seed",
        "default": 42,
    },
}


def sparse_dsnot_prune_model(model, sparsity: float, **kwargs):
    kwargs.pop("blocksize", None)
    kwargs.pop("prunen", None)
    kwargs.pop("prunem", None)
    kwargs.pop("mask_layout", None)
    return sparsegpt_prune_model(
        model,
        sparsity,
        mask_layout="per_row",
        prunen=0,
        prunem=0,
        **kwargs,
    )


def prune(model, **kwargs):
    return sparse_dsnot_prune_model(model, **kwargs)
