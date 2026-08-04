"""
Wanda weight-level pruning (Sun et al., 2023).

Pruning by Weights and Activations: Evaluates weight importance via 
S_ij = |W_ij| * ||X_j||_2 without weight updates (arXiv:2306.11695).
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


ALGORITHM_NAME = "wanda"

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
}


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


class Wanda:
    """Per-layer Wanda solver: Accumulate activation L2 norm + prune weight matrix."""

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if transformers is not None and isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.scaler_row = torch.zeros((self.columns,), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> None:
        del out
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

        # Accumulate L2 norm squared of input activations per channel
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    @torch.no_grad()
    def prune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
    ) -> None:
        """Prune layer in-place using Wanda metric: S_ij = |W_ij| * ||X_j||_2."""
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if transformers is not None and isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Calculate input activation L2 norm: sqrt(sum(x^2))
        scaler_row = torch.sqrt(self.scaler_row).reshape((1, -1))
        
        # Wanda score metric: |W_ij| * ||X_j||_2
        W_metric = torch.abs(W) * scaler_row

        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

        if prunen == 0:
            # Unstructured pruning (per-row output channel sorting)
            sort_res = torch.sort(W_metric, dim=-1, descending=False)
            k = int(self.columns * sparsity)
            indices = sort_res.indices[:, :k]
            W_mask.scatter_(1, indices, True)
        else:
            # Semi-structured N:M pruning
            for j in range(0, self.columns, prunem):
                sub_metric = W_metric[:, j : j + prunem]
                indices = torch.topk(sub_metric, prunen, dim=-1, largest=False)[1]
                W_mask[:, j : j + prunem].scatter_(1, indices, True)

        W[W_mask] = 0

        if transformers is not None and isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def free(self) -> None:
        self.scaler_row = None
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
    """Build a list of input_id tensors for Wanda calibration."""
    from datasets import load_dataset

    dataset = dataset.lower()
    random.seed(seed)

    if dataset in ("wikitext2", "wikitext"):
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
def wanda_prune_model(
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
    device: str | torch.device | None = None,
    exclude_substrings: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """
    Apply Wanda unstructured (or n:m) pruning to all linear layers
    inside transformer blocks sequentially (layer by layer).
    """
    if prunen == 0 and not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")
    if tokenizer is None:
        raise ValueError("tokenizer is required for Wanda calibration")

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

    print(
        f"Wanda: sparsity={sparsity}, nsamples={nsamples}, "
        f"seqlen={seqlen}, device={device}"
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

    for i, _ in enumerate(layers):
        layer = layers[i].to(device)
        subset = find_layers(layer)
        subset = {
            name: mod
            for name, mod in subset.items()
            if not any(ex in name for ex in exclude_substrings)
        }

        wandas = {name: Wanda(mod) for name, mod in subset.items()}

        def make_hook(name: str):
            def hook(_module, inp, out):
                wandas[name].add_batch(inp[0].data, out.data if out is not None else None)

            return hook

        handles = [
            subset[name].register_forward_hook(make_hook(name)) for name in wandas
        ]

        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j])

        for handle in handles:
            handle.remove()

        for name, wanda_obj in wandas.items():
            print(f"  layer {i} · {name} …")
            wanda_obj.prune(
                sparsity,
                prunen=prunen,
                prunem=prunem,
            )
            wanda_obj.free()

        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j])

        layers[i] = layer.cpu()
        del layer, wandas
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inps, outs = outs, inps

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache

    print("Wanda pruning finished.")
    return model


def prune(model: nn.Module, **kwargs) -> nn.Module:
    """Pipeline entrypoint matching Redunformer pruning API."""
    return wanda_prune_model(model, **kwargs)