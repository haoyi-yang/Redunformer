#!/usr/bin/env python3
"""CLI for SparseGPT weight-level pruning (arXiv:2301.00774)."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redundancy.models import load_model_and_tokenizer
from redundancy.pruning.sparsegpt import sparsegpt_prune_model


def save_pretrained_low_mem(
    model, tokenizer, out: Path, max_shard_bytes: int = 800_000_000
) -> None:
    """Save HF weights in shards without peaking at 2× model RAM."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(out)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(out)
    tokenizer.save_pretrained(out)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    weight_map: dict[str, str] = {}
    total_size = 0
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_id = 1

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_id
        if not shard:
            return
        name = f"model-{shard_id:05d}-of-SHARDS.safetensors"
        save_file(shard, str(out / name))
        for key in shard:
            weight_map[key] = name
        print(f"  wrote {name} ({shard_bytes / 1e6:.0f} MB, {len(shard)} tensors)")
        shard = {}
        shard_bytes = 0
        shard_id += 1
        gc.collect()

    with torch.no_grad():
        for name, param in list(model.named_parameters()) + list(model.named_buffers()):
            tensor = param.detach().to("cpu").contiguous()
            nbytes = tensor.numel() * tensor.element_size()
            total_size += nbytes
            if shard and shard_bytes + nbytes > max_shard_bytes:
                flush()
            shard[name] = tensor
            shard_bytes += nbytes
            # Drop parameter storage so peak RAM stays near one shard.
            param.data = torch.empty(0, dtype=param.dtype, device="cpu")

    flush()

    n_shards = shard_id - 1
    final_map = {
        k: v.replace("SHARDS", f"{n_shards:05d}") for k, v in weight_map.items()
    }
    for i in range(1, n_shards + 1):
        old = out / f"model-{i:05d}-of-SHARDS.safetensors"
        new = out / f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {
            k: v.replace("SHARDS", f"{n_shards:05d}") for k, v in weight_map.items()
        },
    }
    # weight_map values already updated via final_map
    index["weight_map"] = final_map
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot SparseGPT unstructured / N:M weight pruning."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Target unstructured sparsity in [0, 1)",
    )
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model load dtype (float16 recommended on GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Pruning device (default: cuda if available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save the pruned model",
    )
    args = parser.parse_args()

    print(f"Loading dense checkpoint {args.model} (fresh weights) …")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        device_map="cpu",
    )

    sparsegpt_prune_model(
        model,
        sparsity=args.sparsity,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        dataset=args.dataset,
        seed=args.seed,
        prunen=args.prunen,
        prunem=args.prunem,
        device=args.device,
    )

    if args.output is None:
        safe = args.model.split("/")[-1].lower()
        pct = int(args.sparsity * 100)
        args.output = f"experiments/pruned/{safe}-sparsegpt{pct}"

    out = Path(args.output)
    print(f"Saving to {out} (low-mem sharded write) …")
    save_pretrained_low_mem(model, tokenizer, out)
    print("Done.")


if __name__ == "__main__":
    main()
