"""
DSnoT weight-level refinement (Zhang et al., ICLR 2024).

Dynamic Sparse No Training: training-free fine-tuning of an already-sparse LLM
by iterative grow-and-prune of the binary mask (arXiv:2310.08915).

It is not a standalone first-stage pruner. Point MODEL at a saved checkpoint
under experiments/pruned/ (and DENSE at the original HF model) to refine those
weights. base=wanda|magnitude|sparsegpt is only the fallback that prunes a
dense model first when no sparse checkpoint is provided.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

try:
    import transformers
except ImportError:  # pragma: no cover
    transformers = None

from redundancy.pruning.wanda import (
    _extract_hidden,
    _hidden_size,
    _move_embeddings_to,
    _resolve_seqlen,
    find_layers,
    get_calibration_loader,
    get_transformer_blocks,
)


ALGORITHM_NAME = "dsnot"

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
    "prunen": {
        "type": int,
        "prompt": "N for N:M semi-structured (0 = unstructured)",
        "default": 0,
    },
    "prunem": {
        "type": int,
        "prompt": "M for N:M semi-structured (0 = unstructured)",
        "default": 0,
    },
    "base": {
        "type": str,
        "prompt": "Init mask (wanda|magnitude|sparsegpt|existing)",
        "default": "wanda",
    },
    "dense": {
        "type": str,
        "prompt": "Dense original (HF id/path) if refining a saved sparse model",
        "default": "",
    },
    "cycles": {
        "type": int,
        "prompt": "Max grow-and-prune cycles T",
        "default": 50,
    },
    "epsilon": {
        "type": float,
        "prompt": "Stop threshold epsilon on |reconstruction error|",
        "default": 0.1,
    },
    "var_power": {
        "type": float,
        "prompt": "Power of activation variance in the grow score",
        "default": 1.0,
    },
    "same_sign": {
        "type": int,
        "prompt": "Reject swaps that flip error sign (1=yes, 0=no)",
        "default": 0,
    },
    "skip_layer": {
        "type": str,
        "prompt": "Skip DSnoT on this module prefix (none|mlp|self_attn)",
        "default": "none",
    },
}


def _is_conv1d(module: nn.Module) -> bool:
    return transformers is not None and isinstance(module, transformers.Conv1D)


def _weight_as_out_in(module: nn.Module) -> torch.Tensor:
    W = module.weight.data
    if isinstance(module, nn.Conv2d):
        return W.flatten(1)
    if _is_conv1d(module):
        return W.t()
    return W


def _write_weight(module: nn.Module, W: torch.Tensor) -> None:
    if _is_conv1d(module):
        W = W.t()
    module.weight.data = W.reshape(module.weight.shape).to(module.weight.data.dtype)


class ActivationStats:
    """Per-layer input statistics used by Wanda, DSnoT, and optional SparseGPT."""

    def __init__(self, layer: nn.Module, *, collect_hessian: bool = False):
        self.layer = layer
        self.dev = layer.weight.device
        W = _weight_as_out_in(layer)
        self.rows = int(W.shape[0])
        self.columns = int(W.shape[1])
        self.nsamples = 0
        self.ntokens = 0
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.sum_row = torch.zeros(self.columns, device=self.dev)
        self.mean = torch.zeros(self.columns, device=self.dev)
        self.var = torch.zeros(self.columns, device=self.dev)
        self.H: torch.Tensor | None = None
        if collect_hessian:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> None:
        del out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        is_linear = isinstance(self.layer, nn.Linear) or _is_conv1d(self.layer)
        if is_linear:
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.float()

        n_tok = inp.shape[1]
        mean_inp = torch.mean(inp, dim=1)
        var_inp = torch.var(inp, dim=1, unbiased=False)
        if self.ntokens == 0:
            self.mean = mean_inp
            self.var = var_inp
        else:
            total = self.ntokens + n_tok
            self.mean = (self.mean * self.ntokens + mean_inp * n_tok) / total
            self.var = (self.var * self.ntokens + var_inp * n_tok) / total
        self.ntokens += n_tok

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.sum_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        self.sum_row += torch.sum(inp, dim=1) / self.nsamples

        if self.H is not None:
            scaled = math.sqrt(2 / self.nsamples) * inp
            self.H += scaled.matmul(scaled.t())

    def free(self) -> None:
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _initial_metric(
    W: torch.Tensor,
    stats: ActivationStats,
    base: str,
) -> torch.Tensor:
    """Importance used for the dense-to-sparse initialization (paper §3)."""
    base = base.lower()
    if base in {"existing", "pretrained", "sparse"}:
        raise ValueError("existing masks do not use an initial prune metric")
    if base == "wanda":
        return torch.abs(W) * torch.sqrt(stats.scaler_row).reshape(1, -1)
    if base == "magnitude":
        return torch.abs(W)
    if base == "sparsegpt":
        if stats.H is None:
            raise ValueError("SparseGPT base requires Hessian accumulation")
        H = stats.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W = W.clone()
        W[:, dead] = 0
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(stats.columns, device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        return (W**2) / (torch.diag(H).reshape(1, -1)) ** 2
    raise ValueError(f"Unknown DSnoT base metric: {base}")


def _sign_pack_order(values: torch.Tensor) -> torch.Tensor:
    """Permutation that puts negative entries first (stable) and positives last
    in reverse original order.

    Combined with Wanda-sorted kept indices this implements Eq. (3): walking
    from the left yields smallest-Wanda negative-contribution weights; walking
    from the right yields smallest-Wanda positive-contribution weights.
    """
    rows, cols = values.shape
    arange = torch.arange(cols, device=values.device, dtype=torch.float64).expand(
        rows, -1
    )
    pos = arange.clone()
    neg = arange.clone()
    pos[values <= 0] = float("inf")
    neg[values >= 0] = float("inf")
    pos_sorted = torch.sort(pos, dim=1).values
    neg_sorted = torch.sort(neg, dim=1).values
    pos_sorted = torch.flip(pos_sorted, dims=[1])
    pos_sorted[pos_sorted == float("inf")] = 0
    neg_sorted[neg_sorted == float("inf")] = 0
    return (pos_sorted + neg_sorted).to(torch.int64)


def _nm_prune_indices(metric: torch.Tensor, prunen: int, prunem: int) -> torch.Tensor:
    """Per-row indices of the N lowest-metric weights in each group of M."""
    rows, cols = metric.shape
    n_blocks = cols // prunem
    grouped = metric[:, : n_blocks * prunem].reshape(rows, n_blocks, prunem)
    local = torch.topk(grouped, prunen, dim=-1, largest=False).indices
    block_start = (
        torch.arange(n_blocks, device=metric.device) * prunem
    ).reshape(1, n_blocks, 1)
    return (local + block_start).reshape(rows, n_blocks * prunen)


def _padded_row_split(
    prune_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row pruned/kept indices, padded to the max count in the batch."""
    n_prune = prune_mask.sum(dim=1).to(torch.long)
    n_keep = (~prune_mask).sum(dim=1).to(torch.long)
    pmax = max(int(n_prune.max().item()), 1)
    kmax = max(int(n_keep.max().item()), 1)
    prune_idx = torch.argsort(prune_mask.float(), dim=1, descending=True)[:, :pmax]
    keep_idx = torch.argsort((~prune_mask).float(), dim=1, descending=True)[:, :kmax]
    return prune_idx, keep_idx, n_prune, n_keep


@torch.no_grad()
def dsnot_mask(
    W: torch.Tensor,
    *,
    stats: ActivationStats,
    sparsity: float,
    base: str = "wanda",
    cycles: int = 50,
    epsilon: float = 0.1,
    var_power: float = 1.0,
    same_sign: bool = False,
    prunen: int = 0,
    prunem: int = 0,
    apply_dsnot: bool = True,
    existing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary prune mask (True = zero the weight) for one (C_out, C_in) matrix.

    `base=wanda|magnitude|sparsegpt` builds a fresh mask from a dense `W`.
    `base=existing` (or a provided `existing_mask`) refines a mask that already
    lives in a sparse checkpoint — DSnoT does not prune from scratch.
    """
    W = W.float()
    rows, cols = W.shape
    dsnot_metric = W * stats.sum_row.reshape(1, -1)
    base = base.lower()

    if existing_mask is not None or base in {"existing", "pretrained", "sparse"}:
        if existing_mask is None:
            raise ValueError("base=existing requires existing_mask from a sparse checkpoint")
        prune_mask = (
            existing_mask.reshape(W.shape).to(dtype=torch.bool, device=W.device).clone()
        )
        if not apply_dsnot:
            return prune_mask
        if not bool(prune_mask.any()):
            return prune_mask
        wanda = torch.abs(W) * torch.sqrt(stats.scaler_row).reshape(1, -1)
        if prunen != 0:
            return _refine_nm(
                prune_mask,
                dsnot_metric=dsnot_metric,
                initial=wanda,
                stats=stats,
                cycles=cycles,
                epsilon=epsilon,
                var_power=var_power,
                prunen=prunen,
                prunem=prunem,
            )
        init_prune, init_keep, n_prune, n_keep = _padded_row_split(prune_mask)
        if not torch.all(n_prune == n_prune[0]) or not torch.all(n_keep == n_keep[0]):
            raise ValueError(
                "Existing sparse mask has a different zero-count per output row "
                f"(min={int(n_prune.min())}, max={int(n_prune.max())}). "
                "DSnoT expects a Wanda/SparseGPT-style per-row mask."
            )
        init_prune = init_prune[:, : int(n_prune[0].item())]
        init_keep = init_keep[:, : int(n_keep[0].item())]
        return _refine_unstructured(
            prune_mask,
            dsnot_metric=dsnot_metric,
            init_prune=init_prune,
            init_keep=init_keep,
            W=W,
            stats=stats,
            cycles=cycles,
            epsilon=epsilon,
            var_power=var_power,
            same_sign=same_sign,
        )

    initial = _initial_metric(W, stats, base)
    prune_mask = torch.zeros_like(W, dtype=torch.bool)

    if prunen != 0:
        if prunem <= 0 or prunen >= prunem:
            raise ValueError("N:M pruning requires 0 < N < M")
        init_prune = _nm_prune_indices(initial, prunen, prunem)
        prune_mask.scatter_(1, init_prune, True)
        if not apply_dsnot:
            return prune_mask
        return _refine_nm(
            prune_mask,
            dsnot_metric=dsnot_metric,
            initial=initial,
            stats=stats,
            cycles=cycles,
            epsilon=epsilon,
            var_power=var_power,
            prunen=prunen,
            prunem=prunem,
        )

    n_prune = int(cols * sparsity)
    if n_prune <= 0:
        return prune_mask
    if n_prune >= cols:
        return torch.ones_like(W, dtype=torch.bool)

    sorted_idx = torch.sort(initial, dim=-1, stable=True).indices
    init_prune = sorted_idx[:, :n_prune]
    init_keep = sorted_idx[:, n_prune:]
    prune_mask.scatter_(1, init_prune, True)
    if not apply_dsnot:
        return prune_mask

    return _refine_unstructured(
        prune_mask,
        dsnot_metric=dsnot_metric,
        init_prune=init_prune,
        init_keep=init_keep,
        W=W,
        stats=stats,
        cycles=cycles,
        epsilon=epsilon,
        var_power=var_power,
        same_sign=same_sign,
    )


def _refine_unstructured(
    prune_mask: torch.Tensor,
    *,
    dsnot_metric: torch.Tensor,
    init_prune: torch.Tensor,
    init_keep: torch.Tensor,
    W: torch.Tensor,
    stats: ActivationStats,
    cycles: int,
    epsilon: float,
    var_power: float,
    same_sign: bool,
) -> torch.Tensor:
    """Eq. (2)–(3) grow-and-prune over the initial prune/keep split."""
    rows = prune_mask.shape[0]
    device = prune_mask.device

    grow_vals = dsnot_metric.gather(1, init_prune).clone()
    if var_power != 0:
        var = stats.var.reshape(1, -1).clamp_min(1e-12).expand_as(dsnot_metric)
        grow_vals = grow_vals / torch.pow(var.gather(1, init_prune), var_power)
    grow_order = init_prune.gather(1, torch.sort(grow_vals, dim=1, stable=True).indices)

    wanda = torch.abs(W) * torch.sqrt(stats.scaler_row).reshape(1, -1)
    wanda.scatter_(1, init_prune, float("inf"))
    n_keep = init_keep.shape[1]
    keep_by_wanda = torch.sort(wanda, dim=1, stable=True).indices[:, :n_keep]
    prune_order = keep_by_wanda.gather(1, _sign_pack_order(dsnot_metric.gather(1, keep_by_wanda)))

    # Expected reconstruction error of the initial sparse row: sum over pruned
    # of W_k * sum_tokens(A_k), i.e. the row-sum of Eq. (1) over calibration.
    error = torch.sum(dsnot_metric.gather(1, init_prune), dim=1, keepdim=True)
    init_sign = torch.sign(error)

    grow_lo = torch.zeros(rows, 1, device=device, dtype=torch.long)
    grow_hi = torch.full((rows, 1), grow_order.shape[1] - 1, device=device, dtype=torch.long)
    prune_lo = torch.zeros(rows, 1, device=device, dtype=torch.long)
    prune_hi = torch.full((rows, 1), prune_order.shape[1] - 1, device=device, dtype=torch.long)
    grow_last = grow_order.shape[1] - 1
    prune_last = prune_order.shape[1] - 1

    active = torch.ones(rows, 1, device=device, dtype=torch.bool)
    for _ in range(cycles):
        if not bool(active.any()):
            break

        error_pos = error > 0
        error_neg = error < 0

        # Eq. (2): revive argmax / argmin of W E[A] / Var(A) among pruned.
        grow_ptr = torch.where(error_pos, grow_hi, grow_lo).clamp(0, grow_last)
        grow_idx = grow_order.gather(1, grow_ptr)
        grow_val = dsnot_metric.gather(1, grow_idx)

        # Eq. (3): drop smallest Wanda among kept weights whose removal
        # moves reconstruction error toward zero (sign constraint).
        prune_ptr = torch.where(error_neg, prune_hi, prune_lo).clamp(0, prune_last)
        prune_idx = prune_order.gather(1, prune_ptr)
        prune_val = dsnot_metric.gather(1, prune_idx)

        error_after = error + prune_val - grow_val
        accept = active & (error.abs() > epsilon)
        accept &= grow_lo <= grow_hi
        accept &= prune_lo <= prune_hi
        if same_sign:
            accept &= init_sign == torch.sign(error_after)

        prune_mask.scatter_(1, prune_idx, accept)
        prune_mask.scatter_(1, grow_idx, ~accept)

        error = error + torch.where(accept, prune_val, torch.zeros_like(prune_val))
        error = error - torch.where(accept, grow_val, torch.zeros_like(grow_val))

        grow_hi = torch.where(error_pos, grow_hi - 1, grow_hi)
        grow_lo = torch.where(~error_pos, grow_lo + 1, grow_lo)
        prune_hi = torch.where(error_neg, prune_hi - 1, prune_hi)
        prune_lo = torch.where(~error_neg, prune_lo + 1, prune_lo)

        active = accept & (grow_lo <= grow_hi) & (prune_lo <= prune_hi)

    return prune_mask


def _refine_nm(
    prune_mask: torch.Tensor,
    *,
    dsnot_metric: torch.Tensor,
    initial: torch.Tensor,
    stats: ActivationStats,
    cycles: int,
    epsilon: float,
    var_power: float,
    prunen: int,
    prunem: int,
) -> torch.Tensor:
    """N:M DSnoT: grow a pruned weight and drop one in the same block of M."""
    del prunen
    rows, cols = prune_mask.shape
    device = prune_mask.device
    prune_idx_mat = prune_mask.nonzero(as_tuple=False)[:, 1].reshape(rows, -1)

    grow_vals = dsnot_metric.gather(1, prune_idx_mat).clone()
    if var_power != 0:
        var = stats.var.reshape(1, -1).clamp_min(1e-12).expand_as(dsnot_metric)
        grow_vals = grow_vals / torch.pow(var.gather(1, prune_idx_mat), var_power)
    grow_order = prune_idx_mat.gather(
        1, torch.sort(grow_vals, dim=1, stable=True).indices
    )
    grow_last = grow_order.shape[1] - 1

    error = torch.sum(
        torch.where(prune_mask, dsnot_metric, torch.zeros_like(dsnot_metric)),
        dim=1,
        keepdim=True,
    )
    init_sign = torch.sign(error)

    grow_lo = torch.zeros(rows, 1, device=device, dtype=torch.long)
    grow_hi = torch.full((rows, 1), grow_last, device=device, dtype=torch.long)
    active = torch.ones(rows, 1, device=device, dtype=torch.bool)

    # Partner prune is the lowest remaining initial-metric weight in the same
    # block of M as the revived index (paper §4.2, N:M).
    block_metric = initial.clone()
    block_metric.scatter_(1, prune_idx_mat, float("inf"))
    metric_ceiling = block_metric.max(dim=1, keepdim=True).values + 1

    for _ in range(cycles):
        if not bool(active.any()):
            break
        error_pos = error > 0
        grow_ptr = torch.where(error_pos, grow_hi, grow_lo).clamp(0, grow_last)
        grow_idx = grow_order.gather(1, grow_ptr)
        grow_val = dsnot_metric.gather(1, grow_idx)

        block_start = grow_idx - (grow_idx % prunem)
        offsets = torch.arange(prunem, device=device).reshape(1, -1)
        block_idx = (block_start + offsets).clamp(max=cols - 1)
        block_vals = block_metric.gather(1, block_idx)
        local = torch.topk(block_vals, 1, dim=1, largest=False).indices
        prune_idx = local + block_start
        prune_val = dsnot_metric.gather(1, prune_idx)

        error_after = error + prune_val - grow_val
        accept = active & (error.abs() > epsilon) & (grow_lo <= grow_hi)
        accept &= init_sign == torch.sign(error_after)

        prune_mask.scatter_(1, prune_idx, accept)
        prune_mask.scatter_(1, grow_idx, ~accept)
        block_metric.scatter_(1, prune_idx, metric_ceiling)

        error = error + torch.where(accept, prune_val, torch.zeros_like(prune_val))
        error = error - torch.where(accept, grow_val, torch.zeros_like(grow_val))

        grow_hi = torch.where(error_pos, grow_hi - 1, grow_hi)
        grow_lo = torch.where(~error_pos, grow_lo + 1, grow_lo)
        active = accept & (grow_lo <= grow_hi)

    return prune_mask


def _skip_dsnot(name: str, skip_layer: str) -> bool:
    skip_layer = (skip_layer or "none").lower()
    if skip_layer in ("", "none", "no_skip"):
        return False
    head = name.split(".")[0]
    tail = name.split(".")[1] if "." in name else ""
    return head == skip_layer or tail == skip_layer


@torch.no_grad()
def dsnot_prune_model(
    model: nn.Module,
    sparsity: float,
    *,
    tokenizer=None,
    nsamples: int = 128,
    seqlen: int | None = None,
    dataset: str = "wikitext2",
    seed: int = 42,
    prunen: int = 0,
    prunem: int = 0,
    base: str = "wanda",
    dense: str = "",
    cycles: int = 50,
    epsilon: float = 0.1,
    var_power: float = 1.0,
    same_sign: int | bool = 0,
    skip_layer: str = "none",
    device: str | torch.device | None = None,
    exclude_substrings: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """
    DSnoT mask refinement (paper: training-free fine-tune of a *sparse* LLM).

    - `base=wanda|magnitude|sparsegpt`: build a new mask on a dense `model`, then
      refine it. Use this only when you do not already have a pruned checkpoint.
    - `base=existing` (or a non-empty `dense=` path): `model` is an already
      pruned checkpoint. The dense original is loaded only to restore grown
      weights; Wanda/SparseGPT are not run again.
    """
    if prunen == 0 and not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")
    if tokenizer is None:
        raise ValueError("tokenizer is required for DSnoT calibration")

    dense = str(dense or "").strip()
    base = str(base).lower()
    if dense and base not in {"existing", "pretrained", "sparse"}:
        print(f"dense={dense} → base=existing (refine saved sparse weights)")
        base = "existing"
    if base in {"pretrained", "sparse"}:
        base = "existing"
    if base not in {"wanda", "magnitude", "sparsegpt", "existing"}:
        raise ValueError("base must be wanda, magnitude, sparsegpt, or existing")
    if base == "existing" and not dense:
        raise ValueError(
            "base=existing needs the original dense model: zeros in the sparse "
            "checkpoint cannot be grown back without the original values. "
            "Pass dense=gpt2 or dense=Qwen/Qwen3-1.7B (or DENSE=... in Make)."
        )
    same_sign_flag = bool(int(same_sign))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    seqlen = _resolve_seqlen(model, seqlen)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("n_positions", "max_position_embeddings"):
            if hasattr(cfg, attr) and getattr(cfg, attr):
                seqlen = min(seqlen, int(getattr(cfg, attr)))

    dense_blocks = None
    if base == "existing":
        from redundancy.models import load_model_and_tokenizer

        print(f"Loading dense original {dense} (restore grown weights only) …")
        dense_model, _ = load_model_and_tokenizer(
            dense, dtype="float32", device_map="cpu"
        )
        dense_model.eval()
        dense_blocks = get_transformer_blocks(dense_model)
        print(
            f"DSnoT: refine existing mask, cycles={cycles}, epsilon={epsilon}, "
            f"nsamples={nsamples}, seqlen={seqlen}, device={device}"
        )
    else:
        print(
            f"DSnoT: base={base}, sparsity={sparsity}, cycles={cycles}, "
            f"epsilon={epsilon}, nsamples={nsamples}, seqlen={seqlen}, device={device}"
        )

    calibration = get_calibration_loader(
        tokenizer,
        dataset=dataset,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
    )

    use_cache = getattr(model.config, "use_cache", False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()

    layers = get_transformer_blocks(model)
    dtype = next(iter(model.parameters())).dtype
    hidden = _hidden_size(model)

    _move_embeddings_to(model, device)
    layers[0] = layers[0].to(device)

    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=device)
    cache: dict[str, Any] = {"i": 0, "layer_args": (), "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inps[cache["i"]] = args[0]
            cache["i"] += 1
            cache["layer_args"] = args[1:]
            cache["layer_kwargs"] = dict(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calibration:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    _move_embeddings_to(model, torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_args = cache["layer_args"]
    layer_kwargs = cache["layer_kwargs"]

    def run_layer(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        out = layer(hidden_states.unsqueeze(0), *layer_args, **layer_kwargs)
        return _extract_hidden(out).squeeze(0)

    collect_hessian = base == "sparsegpt"
    for i, _ in enumerate(layers):
        layer = layers[i].to(device)
        subset = find_layers(layer)
        subset = {
            name: mod
            for name, mod in subset.items()
            if not any(ex in name for ex in exclude_substrings)
        }
        dense_subset: dict[str, nn.Module] = {}
        if dense_blocks is not None:
            dense_subset = find_layers(dense_blocks[i])

        stats_map = {
            name: ActivationStats(mod, collect_hessian=collect_hessian)
            for name, mod in subset.items()
        }

        def make_hook(name: str):
            def hook(_module, inp, out):
                stats_map[name].add_batch(
                    inp[0].data, out.data if out is not None else None
                )

            return hook

        handles = [
            subset[name].register_forward_hook(make_hook(name)) for name in stats_map
        ]
        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j])
        for handle in handles:
            handle.remove()

        for name, mod in subset.items():
            print(f"  layer {i} · {name} …")
            W_sparse = _weight_as_out_in(mod).float()
            existing_mask = None
            W_score = W_sparse
            if base == "existing":
                if name not in dense_subset:
                    raise KeyError(
                        f"Dense original is missing block module '{name}'"
                    )
                W_score = (
                    _weight_as_out_in(dense_subset[name])
                    .to(device=W_sparse.device, dtype=torch.float32)
                )
                existing_mask = (W_sparse == 0).clone()
                if not bool(existing_mask.any()):
                    print("    no zeros in checkpoint, skip")
                    stats_map[name].free()
                    continue
            mask = dsnot_mask(
                W_score,
                stats=stats_map[name],
                sparsity=sparsity,
                base=base,
                cycles=cycles,
                epsilon=epsilon,
                var_power=var_power,
                same_sign=same_sign_flag,
                prunen=prunen,
                prunem=prunem,
                apply_dsnot=not _skip_dsnot(name, skip_layer),
                existing_mask=existing_mask,
            )
            W_out = W_score.clone()
            if existing_mask is not None:
                # Keep SparseGPT (etc.) updates on weights that were already alive.
                W_out[~existing_mask] = W_sparse[~existing_mask]
            W_out[mask] = 0
            _write_weight(mod, W_out)
            stats_map[name].free()

        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j])

        layers[i] = layer.cpu()
        del layer, stats_map
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inps, outs = outs, inps

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache

    print("DSnoT pruning finished.")
    return model


def prune(model: nn.Module, **kwargs) -> nn.Module:
    """Pipeline entrypoint matching Redunformer pruning API."""
    return dsnot_prune_model(model, **kwargs)
