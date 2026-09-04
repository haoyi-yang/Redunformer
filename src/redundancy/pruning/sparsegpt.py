"""
SparseGPT weight-level pruning (Frantar & Alistarh, ICML 2023).

One-shot layer-wise pruning via approximate OBS reconstruction with
adaptive mask selection (Algorithm 1 in arXiv:2301.00774).
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn as nn

try:
    import transformers
except ImportError:  # pragma: no cover
    transformers = None


ALGORITHM_NAME = "sparsegpt"

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
    "blocksize": {
        "type": int,
        "prompt": "Adaptive mask block size Bs",
        "default": 128,
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
}


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_layers(
    module: nn.Module,
    layers: tuple[type, ...] | None = None,
    name: str = "",
) -> dict[str, nn.Module]:
    """Recursively find Linear / Conv1D / Conv2d modules."""
    if layers is None:
        layer_types: list[type] = [nn.Linear, nn.Conv2d]
        if transformers is not None:
            layer_types.append(transformers.Conv1D)
        layers = tuple(layer_types)

    if isinstance(module, layers):
        return {name: module}

    found: dict[str, nn.Module] = {}
    for child_name, child in module.named_children():
        child_path = f"{name}.{child_name}" if name else child_name
        found.update(find_layers(child, layers=layers, name=child_path))
    return found


def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the list of transformer blocks for common HF causal LMs."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # LLaMA / Qwen / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 / BLOOM-style
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        if hasattr(decoder, "layers"):
            return decoder.layers  # OPT
    raise ValueError(
        "Unsupported model architecture: could not locate transformer blocks"
    )


def _move_embeddings_to(model: nn.Module, device: torch.device) -> None:
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "embed_tokens"):
            inner.embed_tokens = inner.embed_tokens.to(device)
        if hasattr(inner, "norm") and inner.norm is not None:
            inner.norm = inner.norm.to(device)
        if hasattr(inner, "embed_positions"):
            inner.embed_positions = inner.embed_positions.to(device)
        if hasattr(inner, "decoder"):
            dec = inner.decoder
            if hasattr(dec, "embed_tokens"):
                dec.embed_tokens = dec.embed_tokens.to(device)
            if hasattr(dec, "embed_positions"):
                dec.embed_positions = dec.embed_positions.to(device)
            for attr in ("project_in", "project_out"):
                if hasattr(dec, attr) and getattr(dec, attr) is not None:
                    setattr(dec, attr, getattr(dec, attr).to(device))
    if hasattr(model, "transformer"):
        tr = model.transformer
        for attr in ("wte", "wpe", "word_embeddings", "word_embeddings_layernorm"):
            if hasattr(tr, attr) and getattr(tr, attr) is not None:
                setattr(tr, attr, getattr(tr, attr).to(device))


class SparseGPT:
    """Per-layer SparseGPT solver: Hessian accumulation + OBS reconstruction."""

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if transformers is not None and isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> None:
        del out  # only inputs are needed for XX^T
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        is_linear = isinstance(self.layer, nn.Linear)
        is_conv1d = transformers is not None and isinstance(
            self.layer, transformers.Conv1D
        )
        if is_linear or is_conv1d:
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def _prepare_w_hinv(self, percdamp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
        """Clone weight as (out, in) and build damped Cholesky factor of H^{-1}."""
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if transformers is not None and isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        return W, H

    def _write_w(self, W: torch.Tensor) -> None:
        if transformers is not None and isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    @torch.no_grad()
    def prune_per_row_obs(
        self,
        sparsity: float,
        *,
        percdamp: float = 0.01,
    ) -> float:
        """
        DSnoT-compatible SparseGPT *initial* mask (official ``initial_method=sparsegpt``).

        Uses the OBS / Hessian diagonal metric, then zeros exactly
        ``⌊sparsity · C_in⌋`` entries **per output row**. No block-wise flatten
        threshold and no iterative OBS weight updates — equal zeros per row so
        ``dsnot base=existing`` can refine the saved checkpoint.
        """
        W, Hinv = self._prepare_w_hinv(percdamp=percdamp)
        metric = (W**2) / (torch.diag(Hinv).reshape((1, -1))) ** 2
        k = int(self.columns * sparsity)
        if k > 0:
            indices = torch.sort(metric, dim=-1, stable=True).indices[:, :k]
            mask = torch.zeros_like(W, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            W[mask] = 0
        self._write_w(W)
        return 0.0

    @torch.no_grad()
    def fasterprune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        mask_layout: str = "block",
    ) -> float:
        """
        Prune the wrapped layer in-place.

        Matches Algorithm 1: Cholesky of H^{-1}, iterative blocking with
        adaptive OBS mask selection, lazy batch weight updates.
        Returns the accumulated OBS reconstruction error.

        ``mask_layout="per_row"`` switches to :meth:`prune_per_row_obs` (DSnoT init).
        """
        if mask_layout == "per_row":
            if prunen != 0:
                raise ValueError("per_row SparseGPT layout is unstructured only")
            return self.prune_per_row_obs(sparsity, percdamp=percdamp)
        if mask_layout != "block":
            raise ValueError("mask_layout must be 'block' or 'per_row'")

        W, Hinv = self._prepare_w_hinv(percdamp=percdamp)

        Losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                # Unstructured: select mask for this Bs-block by OBS error.
                tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    # Semi-structured n:m — prune n lowest-OBS weights per group of m.
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self._write_w(W)
        return float(torch.sum(Losses).item())
    def free(self) -> None:
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_calibration_loader(
    tokenizer,
    *,
    dataset: str = "wikitext2",
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Build a list of input_id tensors for SparseGPT calibration."""
    from datasets import load_dataset

    dataset = dataset.lower()
    random.seed(seed)

    if dataset in ("wikitext2", "wikitext"):
        # Sample documents incrementally — avoid tokenizing the full corpus at once.
        traindata = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="train"
        )
        samples: list[torch.Tensor] = []
        attempts = 0
        while len(samples) < nsamples and attempts < nsamples * 50:
            attempts += 1
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]["text"]
            if not text or not text.strip():
                continue
            # Grow a short buffer from consecutive rows until long enough.
            buf = text
            j = i + 1
            while j < len(traindata) and len(buf.split()) < seqlen * 2:
                nxt = traindata[j]["text"]
                if nxt and nxt.strip():
                    buf = f"{buf}\n\n{nxt}"
                j += 1
            enc = tokenizer(buf, return_tensors="pt", truncation=False, add_special_tokens=False)
            ids = enc.input_ids
            if ids.shape[1] <= seqlen:
                continue
            start = random.randint(0, ids.shape[1] - seqlen - 1)
            samples.append(ids[:, start : start + seqlen])
        if len(samples) < nsamples:
            raise ValueError(
                f"Could only build {len(samples)}/{nsamples} calibration samples "
                f"of length {seqlen} from wikitext2"
            )
        return samples

    if dataset == "c4":
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        samples = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            samples.append(trainenc.input_ids[:, i : i + seqlen])
        return samples

    raise ValueError(f"Unknown calibration dataset: {dataset}")


def _hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if cfg is not None and hasattr(cfg, "n_embd"):
        return int(cfg.n_embd)
    raise ValueError("Could not determine model hidden size")


def _resolve_seqlen(model: nn.Module, seqlen: int | None) -> int:
    if seqlen is not None and seqlen > 0:
        return seqlen
    if hasattr(model, "seqlen"):
        return int(model.seqlen)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            if hasattr(cfg, attr) and getattr(cfg, attr):
                return min(int(getattr(cfg, attr)), 2048)
    return 2048


def _extract_hidden(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def sparsegpt_prune_model(
    model: nn.Module,
    sparsity: float,
    *,
    tokenizer=None,
    nsamples: int = 128,
    seqlen: int | None = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
    dataset: str = "wikitext2",
    seed: int = 42,
    prunen: int = 0,
    prunem: int = 0,
    mask_layout: str = "block",
    device: str | torch.device | None = None,
    exclude_substrings: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """
    Apply SparseGPT unstructured (or n:m) pruning to all linear layers
    inside transformer blocks, sequentially (layer by layer).

    Embeddings and the language-model head are left dense, matching the
    SparseGPT paper setup.

    ``mask_layout="per_row"`` builds the official DSnoT SparseGPT *initial*
    mask (equal zeros per output row) instead of block-flatten thresholds.
    """
    if prunen == 0 and not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")
    if tokenizer is None:
        raise ValueError("tokenizer is required for SparseGPT calibration")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    seqlen = _resolve_seqlen(model, seqlen)
    # Cap seqlen for small models (e.g. GPT-2 max 1024).
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("n_positions", "max_position_embeddings"):
            if hasattr(cfg, attr) and getattr(cfg, attr):
                seqlen = min(seqlen, int(getattr(cfg, attr)))

    layout = "per-row OBS init" if mask_layout == "per_row" else f"block={blocksize}"
    print(
        f"SparseGPT: sparsity={sparsity}, nsamples={nsamples}, "
        f"seqlen={seqlen}, {layout}, device={device}"
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

    inps = torch.zeros(
        (nsamples, seqlen, hidden), dtype=dtype, device=device
    )
    cache: dict[str, Any] = {
        "i": 0,
        "layer_args": (),
        "layer_kwargs": {},
    }

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            # GPT-2 (transformers≥5) passes several positional args;
            # LLaMA/Qwen typically pass hidden + kwargs only.
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

    total_error = 0.0
    for i, _ in enumerate(layers):
        layer = layers[i].to(device)
        subset = find_layers(layer)
        subset = {
            name: mod
            for name, mod in subset.items()
            if not any(ex in name for ex in exclude_substrings)
        }

        gpts = {name: SparseGPT(mod) for name, mod in subset.items()}

        def make_hook(name: str):
            def hook(_module, inp, out):
                gpts[name].add_batch(inp[0].data, out.data if out is not None else None)

            return hook

        handles = [
            subset[name].register_forward_hook(make_hook(name)) for name in gpts
        ]

        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j])

        for handle in handles:
            handle.remove()

        for name, gpt in gpts.items():
            print(f"  layer {i} · {name} …")
            err = gpt.fasterprune(
                sparsity,
                prunen=prunen,
                prunem=prunem,
                percdamp=percdamp,
                blocksize=blocksize,
                mask_layout=mask_layout,
            )
            total_error += err
            if mask_layout != "per_row":
                print(f"    OBS error={err:.4g}")
            gpt.free()

        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j])

        layers[i] = layer.cpu()
        del layer, gpts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inps, outs = outs, inps

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache

    print(f"SparseGPT finished. Total OBS error={total_error:.4g}")
    return model


def prune(model: nn.Module, **kwargs) -> nn.Module:
    """Pipeline entrypoint matching Redunformer pruning API."""
    return sparsegpt_prune_model(model, **kwargs)

