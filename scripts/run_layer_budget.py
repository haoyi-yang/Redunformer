"""
Budget-matched layer-wide Wanda pruning -- Rathore's Wanda Strategy 3.

Uniform pruning forces every matrix in a layer to the same sparsity. W3 gives the whole LAYER one
tile budget and lets the split between its seven matrices emerge from the pooled Wanda ranking:
robust matrices surrender more tiles, sensitive ones fewer.

The fair comparison he specifies:
    1. uniform  N%-per-matrix pruning          (already measured: experiments/screen_wholelayer/)
    2. adaptive layer-wide pruning with the SAME total number of removed tiles   (this script)
Both remove the same tile count from the layer, so any difference is attributable purely to the
allocation.

Dense-model discipline: all seven matrices are restored before moving to the next layer.

Example
  python scripts/run_layer_budget.py --layers 0 9 18 27 35 --prune-ratio 0.20
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch

from run_pruning import (
    MATRICES, build_target_name, get_target_weight, collect_stats_for_targets,
    get_num_layers, save_json, ratio_short_name, zero_tiles,
)
from redundancy.models import load_model_and_tokenizer
from redundancy.data import load_evaluation_dataset, load_calibration_dataset
from redundancy.eval import evaluate_perplexity
from redundancy.scoring import wanda_tile_scores
from redundancy.layer_budget import pooled_allocation


def main():
    p = argparse.ArgumentParser(description="Rathore W3: budget-matched layer-wide Wanda pruning.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 9, 18, 27, 35])
    p.add_argument("--prune-ratio", type=float, default=0.20,
                   help="Total layer budget as a fraction of the layer's tiles.")
    p.add_argument("--mode", choices=["mean", "raw"], default="mean",
                   help="Score normalisation within each matrix before pooling.")
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--eval-frac", type=float, default=0.2)
    # Defaults match the whole-layer runs this is compared against -- W3's fair comparison is
    # against uniform at the same tile count, so the eval must be identical. Those runs recorded
    # dataset="wikitext" (not "Salesforce/wikitext"); matching the string rather than assuming
    # the two resolve identically.
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-2-raw-v1")
    p.add_argument("--experiment-dir", default="experiments/layer_budget")
    args = p.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = load_evaluation_dataset(args.dataset, args.subset, split="test")
    calib = load_calibration_dataset(tokenizer, n_samples=args.calib_samples,
                                     seqlen=args.calib_seqlen)

    layers = args.layers or list(range(get_num_layers(model)))
    ratio = ratio_short_name(args.prune_ratio)

    for layer in layers:
        print(f"\n########## Layer {layer} — W3 budget-matched (budget {args.prune_ratio}) ##########")

        names = [build_target_name(layer, m) for m in MATRICES.keys()]
        stats = collect_stats_for_targets(model, calib, names, "wanda")

        weights = {m: get_target_weight(model, build_target_name(layer, m)) for m in MATRICES.keys()}
        originals = {m: w.detach().clone() for m, w in weights.items()}

        # Score every tile in every matrix of the layer, then pool under one shared budget.
        scores_by_matrix = {
            m: wanda_tile_scores(weights[m], stats[build_target_name(layer, m)], args.tile_size)
            for m in MATRICES.keys()
        }
        pruned_by_matrix, info = pooled_allocation(scores_by_matrix, args.prune_ratio, mode=args.mode)

        print(f"  layer tiles {info['layer_total_tiles']:,} · budget {info['layer_pruned_tiles']:,} "
              f"· achieved {info['achieved_ratio']:.4f}")
        print("  emergent allocation (uniform would be flat at "
              f"{args.prune_ratio:.0%}):")
        for m, d in sorted(info["per_matrix"].items(), key=lambda x: -x[1]["ratio"]):
            print(f"    {m:11} {d['pruned']:>6,}/{d['tiles']:<6,} -> {d['ratio']:6.1%}")

        with torch.no_grad():
            for m, tiles in pruned_by_matrix.items():
                if tiles:
                    zero_tiles(weights[m], tiles, args.tile_size)

        ppl = evaluate_perplexity(model, tokenizer, dataset, eval_frac=args.eval_frac)
        print(f"  perplexity {ppl:.4f}")

        with torch.no_grad():
            for m in MATRICES.keys():
                weights[m].copy_(originals[m])   # dense-model discipline
        del stats, originals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_json(os.path.join(args.experiment_dir, f"layer{layer}_w3_p{ratio}.json"), {
            "model": args.model,
            "dataset": args.dataset,
            "subset": args.subset,
            "scope": "layer_budget_matched",
            "strategy": "rathore_W3",
            "layer": layer,
            "method": "wanda_layer_budget",
            "mode": args.mode,
            "tile_size": args.tile_size,
            "prune_ratio": args.prune_ratio,
            "calib_samples": args.calib_samples,
            "calib_seqlen": args.calib_seqlen,
            "eval_frac": args.eval_frac,
            "perplexity": ppl,
            "allocation": info,
        })


if __name__ == "__main__":
    main()
