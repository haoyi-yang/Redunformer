#!/usr/bin/env python3
"""Week 1-2 deliverable: baseline evaluation of GPT-2 on WikiText-2."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.models import get_device, load_model
from redundancy.data import load_wikitext, prepare_encodings
from redundancy.eval import compute_perplexity, run_lm_eval
from utils import build_config, build_result_dict, save_results, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Baseline GPT-2 evaluation.")
    p.add_argument("--config", type=str, default=None, help="JSON config file.")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-lm-eval", action="store_true", help="Skip lm-eval-harness.")
    p.add_argument("--output-dir", type=str, default="experiments")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)

    print("=" * 50)
    print("BASELINE EVALUATION — GPT-2 × WikiText-2")
    print("=" * 50)
    print(f"  max_length={cfg['max_length']}  stride={cfg['stride']}  seed={cfg['seed']}")
    print(f"  lm-eval: {'skip' if cfg['skip_lm_eval'] else cfg['lm_eval_tasks']}")
    print("=" * 50)

    torch.manual_seed(cfg["seed"])
    device = get_device(cfg["device"])

    model, tokenizer = load_model("gpt2", device=str(device))
    dataset = load_wikitext(split="test")
    input_ids = prepare_encodings(dataset, tokenizer, cfg["max_length"], cfg["stride"])

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
        lm_eval_summary = {
            task: {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))}
            for task, res in lm_results.get("results", {}).items()
        }
        results["lm_eval"] = lm_eval_summary

    save_results(results, cfg["output_dir"], cfg["model_name"])
    print_summary(cfg, ppl, lm_eval_summary)


if __name__ == "__main__":
    main()
