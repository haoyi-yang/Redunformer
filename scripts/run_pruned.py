#!/usr/bin/env python3
"""One-shot block removal sweep using measured BI scores.

Evaluates lowest-BI pruning against a random-removal baseline at the same k,
reusing the existing WikiText-2 perplexity loop.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.data import load_wikitext, prepare_encodings
from redundancy.eval import compute_perplexity
from redundancy.models import (
    clone_and_prune_model,
    get_device,
    get_transformer_blocks,
    load_model,
)
from utils import build_config, build_result_dict, save_results, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="One-shot block pruning sweep.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file.",
    )
    p.add_argument(
        "--measurement",
        type=str,
        default=None,
        help="Path to a measurement JSON with bi_scores.",
    )
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--skip-lm-eval",
        action="store_true",
        help="Skip lm-eval-harness.",
    )
    p.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Maximum number of blocks to remove.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
    )
    return p.parse_args()


def load_bi_scores(path: str | None) -> list[float]:
    """Load BI scores from a measurement file.

    If no valid path is given, use the newest measurement JSON in experiments/.
    """
    if path is None or not Path(path).exists():
        experiments_dir = Path("experiments")

        measurement_files = sorted(
            experiments_dir.glob("measurement_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not measurement_files:
            raise FileNotFoundError(
                f"No measurement files found in {experiments_dir}"
            )

        path = str(measurement_files[0])

        print(
            "Measurement file not found or not specified. "
            f"Using newest measurement file: {path}"
        )

    with open(path) as f:
        payload = json.load(f)

    bi_scores = payload.get("bi_scores")

    if not isinstance(bi_scores, list) or not bi_scores:
        raise ValueError(
            f"No bi_scores found in measurement file: {path}"
        )

    return [float(x) for x in bi_scores]


def select_blocks(bi_scores: list[float], k: int) -> list[int]:
    """Return the indices of the k lowest-BI blocks."""

    ranked = sorted(
        range(len(bi_scores)),
        key=lambda idx: (bi_scores[idx], idx),
    )

    return ranked[:k]


def evaluate(
    model,
    input_ids,
    device,
    cfg,
    run_type,
    blocks_to_remove,
):
    """Clone, prune, and evaluate a model."""

    print(f"  Removing blocks: {blocks_to_remove}")

    pruned_model = clone_and_prune_model(
        model,
        blocks_to_remove,
    )

    n_remaining = len(get_transformer_blocks(pruned_model))

    print(
        f"  Remaining blocks: {n_remaining}"
    )

    print("\nComputing perplexity ...")

    ppl = compute_perplexity(
        pruned_model,
        input_ids,
        device=device,
        stride=cfg["stride"],
    )

    print(f"\n  Perplexity: {ppl:.2f}")

    results = build_result_dict(
        cfg,
        ppl,
        device,
    )

    results["pruning"] = {
        "run_type": run_type,
        "blocks_removed": blocks_to_remove,
        "n_blocks_removed": len(blocks_to_remove),
        "n_blocks_remaining": n_remaining,
    }

    # Do not run lm-eval here yet.
    #
    # run_lm_eval(model_name, ...) reloads a fresh model from Hugging Face,
    # so it would evaluate the original unpruned model rather than
    # pruned_model. Perplexity above is correctly evaluated on pruned_model.

    save_results(
        results,
        cfg["output_dir"],
        f"{cfg['model_name']}_{run_type}",
    )

    print_summary(
        cfg,
        ppl,
        None,
    )


def main():
    args = parse_args()
    cfg = build_config(args)

    bi_scores = load_bi_scores(args.measurement)

    print("=" * 50)
    print(
        f"PRUNING SWEEP — {cfg['model_name']} "
        "lowest-BI vs random removal"
    )
    print("=" * 50)
    print(
        f"  measurement={args.measurement or 'newest measurement file'}"
    )
    print(f"  max_k={args.max_k}")
    print(
        f"  max_length={cfg['max_length']}  "
        f"stride={cfg['stride']}  "
        f"seed={cfg['seed']}"
    )
    print("=" * 50)

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    device = get_device(cfg["device"])

    model, tokenizer = load_model(
        cfg["model_name"],
        device=str(device),
    )

    n_blocks = len(get_transformer_blocks(model))

    print(f"\nModel has {n_blocks} transformer blocks.")

    if len(bi_scores) != n_blocks:
        raise ValueError(
            "Measurement/model mismatch: measurement has "
            f"{len(bi_scores)} BI scores, but "
            f"{cfg['model_name']} has {n_blocks} blocks."
        )

    if args.max_k > n_blocks:
        raise ValueError(
            f"max_k={args.max_k} exceeds the number of blocks "
            f"({n_blocks})."
        )

    dataset = load_wikitext(split="test")

    input_ids = prepare_encodings(
        dataset,
        tokenizer,
        cfg["max_length"],
        cfg["stride"],
    )

    # Select all blocks needed for the maximum sweep once.
    # This makes the k experiments nested:
    # k=1 uses the first block,
    # k=2 uses the first two blocks, etc.
    blocks_to_remove_lowest = select_blocks(
        bi_scores,
        args.max_k,
    )

    blocks_to_remove_random = random.sample(
        range(n_blocks),
        args.max_k,
    )

    print(
        f"\nLowest-BI block order: {blocks_to_remove_lowest}"
    )
    print(
        f"Random block order:    {blocks_to_remove_random}"
    )

    for k in range(1, args.max_k + 1):
        print("\n" + "=" * 50)
        print(f"Evaluating lowest-BI pruning with k={k} ...")
        print("=" * 50)

        evaluate(
            model,
            input_ids,
            device,
            cfg,
            run_type=f"lowest_bi_k{k}",
            blocks_to_remove=blocks_to_remove_lowest[:k],
        )

        print("\n" + "=" * 50)
        print(f"Evaluating random pruning with k={k} ...")
        print("=" * 50)

        evaluate(
            model,
            input_ids,
            device,
            cfg,
            run_type=f"random_k{k}",
            blocks_to_remove=blocks_to_remove_random[:k],
        )


if __name__ == "__main__":
    main()