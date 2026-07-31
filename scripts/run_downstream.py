"""
Downstream task accuracy for pruned whole models (the capability test).

Applies a whole-model pruning config (method / sparsity / policy A or B) to an
in-memory model, then measures real task accuracy with the lm-evaluation-harness
(HellaSwag, PIQA, ARC-Easy). Perplexity can stay reasonable while task ability
degrades -- this is the check that catches that.

Examples
  # dense baseline (run this first, it is the reference)
  python scripts/run_downstream.py --dense

  # best method, sensitivity-aware, at 20% global sparsity
  python scripts/run_downstream.py --method sparsegpt_recon --prune-ratio 0.20 --policy sensitivity

  # quick smoke (200 examples/task instead of the full sets)
  python scripts/run_downstream.py --dense --limit 200
"""

import argparse
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # scripts/  (reuse run_pruning helpers)
sys.path.insert(0, os.path.join(HERE, "..", "src"))        # src/

import torch

from run_pruning import (
    MATRICES, build_target_name, get_target_weight, apply_pruning,
    collect_stats_for_targets, build_tile_counts, get_num_layers, save_json,
)
from redundancy.models import load_model_and_tokenizer
from redundancy.data import load_calibration_dataset
from redundancy.policy import load_sensitivity, classify, expand_to_model, allocate

TASKS = ["hellaswag", "piqa", "arc_easy"]
# wanda_recon needs the Gram matrix too: collect_stats_for_targets returns col-norms only for
# the literal "wanda", and the full H for everything else -- which is what wanda_recon wants
# (its Wanda scores come from sqrt(diag(H))).
NEEDS_CALIB = ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon", "random_recon")
METHODS = ["magnitude", "magnitude_high", "random", "wanda", "sparsegpt", "sparsegpt_recon",
           "wanda_recon", "random_recon"]


def prune_whole_model(model, args, layers, calib_samples, ratio_map=None):
    """Prune every matrix in every layer, in order, freeing each layer's stats.

    --matrices restricts which projection types are pruned (default: all seven). This is what
    lets us measure downstream accuracy of a targeted config like "o_proj on layers 17-21,32,34,35"
    -- the exact model whose WikiText perplexity beat dense, to test whether that win survives on
    real tasks or is WikiText-specific.
    """
    scan_matrices = getattr(args, "matrices", None) or list(MATRICES.keys())
    total_tiles = total_pruned = 0
    for layer in layers:
        stats = None
        if args.method in NEEDS_CALIB:
            names = [build_target_name(layer, m) for m in scan_matrices]
            stats = collect_stats_for_targets(model, calib_samples, names, args.method)

        for m in scan_matrices:
            target = build_target_name(layer, m)
            w = get_target_weight(model, target)
            stat = stats[target] if stats is not None else None
            ratio = ratio_map[(layer, m)] if ratio_map is not None else args.prune_ratio
            nt, npd = apply_pruning(w, args.method, args.tile_size, ratio, args.seed, stat=stat)
            total_tiles += nt
            total_pruned += npd

        del stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  layer {layer:2}: total pruned {total_pruned}")

    return total_tiles, total_pruned


def main():
    p = argparse.ArgumentParser(description="Downstream accuracy of a pruned whole model.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dense", action="store_true", help="Evaluate the unpruned model (baseline).")
    p.add_argument("--method", default="sparsegpt_recon", choices=METHODS)
    p.add_argument("--prune-ratio", type=float, default=0.20)
    p.add_argument("--policy", choices=["uniform", "sensitivity"], default="uniform")
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Restrict pruning to these layers (default: all).")
    p.add_argument("--matrices", type=str, nargs="+", default=None, choices=list(MATRICES.keys()),
                   help="Restrict pruning to these projection types (default: all seven).")
    p.add_argument("--seed", type=int, default=42, help="Only used by --method random.")
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--screen-dir", default="experiments/screen")
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--limit", type=int, default=None, help="Examples per task (for a quick smoke).")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    tag = "dense"
    pruned_info = None
    if not args.dense:
        layers = args.layers if args.layers is not None else list(range(get_num_layers(model)))

        calib = None
        if args.method in NEEDS_CALIB:
            calib = load_calibration_dataset(tokenizer, n_samples=args.calib_samples,
                                             seqlen=args.calib_seqlen)

        ratio_map, policy_info = None, None
        if args.policy == "sensitivity":
            sens = load_sensitivity(args.screen_dir, method="sparsegpt_recon", ref_ratio="0.20")
            classes = classify(sens)
            measured = sorted({l for (l, _) in sens})
            full = expand_to_model(classes, layers, list(MATRICES.keys()), measured)
            tiles = build_tile_counts(model, layers, args.tile_size)
            ratio_map, policy_info = allocate(full, tiles, args.prune_ratio)
            print(f"Policy B: target {policy_info['target']:.3f} -> achieved {policy_info['achieved']:.4f}")

        print(f"Pruning whole model: {args.method} @ {args.prune_ratio} ({args.policy} policy)")
        tt, tp = prune_whole_model(model, args, layers, calib, ratio_map)
        print(f"Pruned {tp}/{tt} tiles = {tp / tt:.4f}")
        pruned_info = {"total_tiles": tt, "total_pruned": tp, "achieved_sparsity": tp / tt,
                       "policy_info": policy_info}
        tag = f"{args.method}_p{int(args.prune_ratio * 100)}_{args.policy}"

        # Reconstruction allocates very large short-lived tensors (down_proj's Gram matrix
        # alone is 9728^2 fp32 = 378MB, plus its inverse and the Schur solves). Left behind,
        # they fragment the caching allocator and the eval then spills to host RAM -- on
        # Windows this is silent, and costs 3-6x wall-clock rather than raising OOM.
        # Drop the calibration set and hand the cached blocks back before evaluating.
        del calib, ratio_map
        calib = ratio_map = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            print(f"GPU free before eval: {free / 1e9:.2f} / {total / 1e9:.2f} GB")

    # Evaluate the (possibly pruned) in-memory model with lm-eval.
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"Running lm-eval on tasks: {args.tasks}")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit)

    out = {
        "tag": tag,
        "model": args.model,
        "dense": args.dense,
        "method": None if args.dense else args.method,
        "prune_ratio": None if args.dense else args.prune_ratio,
        "policy": None if args.dense else args.policy,
        "tile_size": args.tile_size,
        "limit": args.limit,
        "pruned": pruned_info,
        "results": res["results"],
    }
    path = args.output or os.path.join("experiments", "downstream", f"downstream_{tag}.json")
    save_json(path, out)

    print("\n=== downstream accuracy ===")
    for t, v in res["results"].items():
        acc = v.get("acc_norm,none", v.get("acc,none"))
        print(f"  {t:12} {acc}")


if __name__ == "__main__":
    main()
