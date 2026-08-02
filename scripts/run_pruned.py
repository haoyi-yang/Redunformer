#!/usr/bin/env python3
"""One-shot GPT-2 block removal sweep using measured BI scores.

Evaluates lowest-BI pruning against a random-removal baseline at the same k,
reusing the existing WikiText-2 perplexity loop and lm-eval harness.
"""

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import torch
from transformers import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.data import load_wikitext, prepare_encodings
from redundancy.eval import compute_perplexity, run_lm_eval
from redundancy.models import get_device, load_model, clone_and_prune_model
from utils import build_config, build_result_dict, save_results, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="One-shot GPT-2 pruning sweep.")
    p.add_argument("--config", type=str, default=None, help="JSON config file.")
    p.add_argument("--measurement", type=str, default=None, help="Path to a measurement JSON with bi_scores.")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-lm-eval", action="store_true", help="Skip lm-eval-harness.")
    p.add_argument("--max-k", type=int, default=6, help="Maximum number of blocks to remove")
    p.add_argument("--output-dir", type=str, default="experiments")
    return p.parse_args()


def load_bi_scores(path: str) -> list[float]:

    #if path doesnt exist or empty, pick the newest measurement file in the experiments folder
    if path == None or not Path(path).exists() :
        experiments_dir = Path("experiments")
        measurement_files = sorted(experiments_dir.glob("measurement_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not measurement_files:
            raise FileNotFoundError(f"No measurement files found in {experiments_dir}")
        path = str(measurement_files[0])
        print(f"Measurement file not found or not specified. Using the newest measurement file: {path}")
    
    with open(path) as f:
        payload = json.load(f)
    bi_scores = payload.get("bi_scores")
    if not isinstance(bi_scores, list) or not bi_scores:
        raise ValueError(f"No bi_scores found in measurement file: {path}")
    return [float(x) for x in bi_scores]


def select_blocks(bi_scores: list[float], k: int) -> list[int]:
    def block_rank_key(idx: int) -> tuple[float, int]:
        return bi_scores[idx], idx

    ranked = sorted(range(len(bi_scores)), key=block_rank_key)
    return ranked[:k]

def evaluate(model, tokenizer, input_ids, device, cfg, type, blocks_to_remove):
    model = clone_and_prune_model(model, blocks_to_remove)
    print("\nComputing perplexity ...")
    ppl = compute_perplexity(model, input_ids, device=device, stride=cfg["stride"])
    print(f"\n  Perplexity: {ppl:.2f}")

    results = build_result_dict(cfg, ppl, device)

    lm_eval_summary = None
    if not cfg["skip_lm_eval"]:
        lm_results = run_lm_eval(
            cfg["model_name"],
            cfg["lm_eval_tasks"],
            device=str(device),
            batch_size=cfg["batch_size"],
            model=model,
            tokenizer=tokenizer,
        )
        if lm_results is None:
            raise RuntimeError("lm-eval returned no results")
        lm_eval_summary = {
            task: {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))}
            for task, res in lm_results.get("results", {}).items()
        }
        results["lm_eval"] = lm_eval_summary
    save_results(results, cfg["output_dir"], cfg["model_name"] + type + str(blocks_to_remove))
    print_summary(cfg, ppl, lm_eval_summary)

def main():
    args = parse_args()
    cfg = build_config(args)
    bi_scores = load_bi_scores(args.measurement)
    k = args.max_k

    print("=" * 50)
    print("GPT-2 PRUNING SWEEP — lowest-BI vs random removal")
    print("=" * 50)
    print(f"  measurement={args.measurement}")
    print(f"  k values={k}")
    print(f"  max_length={cfg['max_length']}  stride={cfg['stride']}  seed={cfg['seed']}")
    print("=" * 50)

    torch.manual_seed(cfg["seed"])
    device = get_device(cfg["device"])

    model, tokenizer = load_model(cfg["model_name"], device=str(device))
    dataset = load_wikitext(split="test")
    input_ids = prepare_encodings(dataset, tokenizer, cfg["max_length"], cfg["stride"])

    blocks_to_remove_lowest = select_blocks(bi_scores, k)
    blocks_to_remove_random = random.sample(range(1, 11),k)
    for k in range(1, args.max_k + 1):
        #evaluate axing k worst blocks by BI score
        print(f"\nEvaluating lowest-BI pruning with k={k} ...")
        #evaluate(model, tokenizer, input_ids, device, cfg, type=f"_lowest_bi_k{k}", blocks_to_remove=blocks_to_remove_lowest[:k])

        #evaluate axing k random blocks
        print(f"\nEvaluating random pruning with k={k} ...")
        evaluate(model, tokenizer, input_ids, device, cfg, type=f"_random_k{k}", blocks_to_remove=blocks_to_remove_random[:k])

if __name__ == "__main__":
    main()