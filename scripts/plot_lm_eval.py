#!/usr/bin/env python3
"""Plot lm_eval results (e.g. HellaSwag acc_norm) across pruning steps."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from redundancy.plotting import plot_lm_eval_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Plot lm_eval results from pruning experiment JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/Qwen/baseline_Qwen_Qwen3-0.6B_random_run2_20260913_111529.json",
        help="Path to pruning experiment JSON file.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="experiments/Qwen/baseline_Qwen_Qwen3-0.6B_20260913_120550.json",
        help="Path to unpruned baseline JSON file for k=0 reference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (defaults to reports/group9/figures/lm_eval_<exp_name>.png).",
    )
    parser.add_argument(
        "--include-perplexity",
        action="store_true",
        help="Also create a 2-panel plot including perplexity.",
    )
    return parser.parse_args()


def load_experiment(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_experiment(input_path)
    model_name = data.get("model_name", "Model")
    pruning_data = data.get("pruning", {})
    blocks_removed = pruning_data.get("blocks_removed", [])
    lm_eval = pruning_data.get("lm_eval", {})
    acc_norm = lm_eval.get("acc_norm", [])
    acc_stderr = lm_eval.get("acc_stderr", [])
    perplexity = pruning_data.get("perplexity", [])

    baseline_acc = None
    baseline_stderr = None
    baseline_ppl = None
    if args.baseline:
        base_path = Path(args.baseline)
        if base_path.exists():
            base_data = load_experiment(base_path)
            baseline_ppl = base_data.get("perplexity")
            base_lm = base_data.get("lm_eval", {})
            if "lm_eval" in base_lm and isinstance(base_lm["lm_eval"], dict):
                baseline_acc = base_lm["lm_eval"].get("acc_norm,none")
                baseline_stderr = base_lm["lm_eval"].get("acc_stderr,none")

    exp_stem = input_path.stem
    if args.output:
        out_path = args.output
    else:
        out_path = f"reports/group9/figures/lm_eval_{exp_stem}.png"

    plot_lm_eval_curve(
        acc_norm=acc_norm,
        acc_stderr=acc_stderr,
        blocks_removed=blocks_removed,
        out_path=out_path,
        title=f"{model_name} Pruning: HellaSwag Accuracy vs. Blocks Removed\n({exp_stem})",
        baseline_acc=baseline_acc,
        baseline_stderr=baseline_stderr,
        perplexity=perplexity if args.include_perplexity else None,
        baseline_ppl=baseline_ppl if args.include_perplexity else None,
    )


if __name__ == "__main__":
    main()
