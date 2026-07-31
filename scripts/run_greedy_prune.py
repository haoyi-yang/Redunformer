"""
Greedy sequential pruning (Seb's idea) -- the principled fix for the sequential-dependency /
stale-H problem that made isolate-then-compose fail.

Instead of measuring each layer against the DENSE model and then composing (where input drift
breaks the measurements), we march through the layers and make each accept/reject decision on the
ACTUAL current model, with all previously-accepted prunes applied and NO restore. Every decision
is therefore valid for the real activations that arrive -- which is exactly how SparseGPT already
processes a model, and the search form of "iterative calibration".

  for each candidate layer L (in order):
      collect the selection stat for L on the CURRENT (partially-pruned) model   <- sequential calib
      prune L's target matrices at prune_ratio
      re-evaluate the cheap metric (perplexity) on the CURRENT model
      if it improved  -> ACCEPT (keep pruned, update best)
      else            -> REJECT (restore L, move on)

The accept/reject metric is deliberately cheap (perplexity, optionally a subset). Confirm the FINAL
greedily-pruned model separately on a HELD-OUT suite (run_downstream) -- greedily accepting on
perplexity tunes the model to perplexity, so the genuine-improvement claim needs a metric the
search never saw.

Example
  python scripts/run_greedy_prune.py --matrices o_proj --prune-ratio 0.40 --direction forward
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
    apply_pruning, get_num_layers, save_json,
)
from redundancy.models import load_model_and_tokenizer
from redundancy.data import load_evaluation_dataset, load_calibration_dataset
from redundancy.eval import evaluate_perplexity

NEEDS_CALIB = ("wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon")


def main():
    p = argparse.ArgumentParser(description="Greedy sequential pruning with accept/reject.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--matrices", type=str, nargs="+", default=["o_proj"], choices=list(MATRICES.keys()),
                   help="Which projection type(s) to consider pruning at each layer.")
    p.add_argument("--prune-ratio", type=float, default=0.40)
    p.add_argument("--method", default="wanda",
                   choices=["magnitude", "random", "wanda", "sparsegpt", "sparsegpt_recon", "wanda_recon"])
    p.add_argument("--direction", choices=["forward", "backward"], default="forward")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Candidate layers to consider (default: all).")
    p.add_argument("--tile-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1, help="Only for --method random.")
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--eval-frac", type=float, default=1.0,
                   help="Perplexity fraction for the accept/reject decision. <1.0 = faster, noisier.")
    p.add_argument("--margin", type=float, default=0.0,
                   help="Require ppl to improve by at least this much to accept (guards against noise).")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-2-raw-v1")
    p.add_argument("--output", default="experiments/greedy/greedy_run.json")
    args = p.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = load_evaluation_dataset(args.dataset, args.subset, split="test")
    calib = None
    if args.method in NEEDS_CALIB:
        calib = load_calibration_dataset(tokenizer, n_samples=args.calib_samples, seqlen=args.calib_seqlen)

    layers = args.layers if args.layers is not None else list(range(get_num_layers(model)))
    if args.direction == "backward":
        layers = list(reversed(layers))

    def ppl():
        return evaluate_perplexity(model, tokenizer, dataset, eval_frac=args.eval_frac)

    baseline = ppl()
    best = baseline
    print(f"dense perplexity ({args.eval_frac:.0%} eval) = {baseline:.4f}\n")

    accepted, trace = [], []
    for L in layers:
        # Sequential calibration: the stat is collected on the CURRENT (partially-pruned) model,
        # so the selection sees the activations that actually arrive at L now -- not the dense ones.
        stats = None
        if args.method in NEEDS_CALIB:
            names = [build_target_name(L, m) for m in args.matrices]
            stats = collect_stats_for_targets(model, calib, names, args.method)

        originals = {}
        for m in args.matrices:
            w = get_target_weight(model, build_target_name(L, m))
            originals[m] = w.detach().clone()
            stat = stats[build_target_name(L, m)] if stats is not None else None
            apply_pruning(w, args.method, args.tile_size, args.prune_ratio, args.seed, stat=stat)

        cand = ppl()
        improved = cand < best - args.margin
        if improved:
            best = cand
            accepted.append(L)
            verdict = "ACCEPT"
        else:
            with torch.no_grad():
                for m in args.matrices:
                    get_target_weight(model, build_target_name(L, m)).copy_(originals[m])
            verdict = "reject"
        trace.append({"layer": L, "ppl_after": cand, "verdict": verdict, "best": best})
        print(f"  layer {L:2}: ppl {cand:8.4f}  {verdict}   (best {best:.4f}, accepted {len(accepted)})")

        del stats, originals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n=== greedy done ===")
    print(f"  baseline {baseline:.4f} -> final {best:.4f}  ({best - baseline:+.4f})")
    print(f"  accepted layers ({len(accepted)}): {accepted}")
    save_json(args.output, {
        "model": args.model, "matrices": args.matrices, "method": args.method,
        "prune_ratio": args.prune_ratio, "direction": args.direction, "margin": args.margin,
        "eval_frac": args.eval_frac, "baseline_ppl": baseline, "final_ppl": best,
        "delta": best - baseline, "accepted_layers": accepted, "trace": trace,
    })
    print(f"  saved {args.output}")
    print("  NEXT: confirm this accepted set on a HELD-OUT downstream suite (it was picked on perplexity).")


if __name__ == "__main__":
    main()
