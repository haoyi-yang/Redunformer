#!/usr/bin/env python3
"""Weeks 5-8 deliverable: Block Influence redundancy measurement for GPT-2 on WikiText-2."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.models import get_device, load_model
from redundancy.data import load_wikitext, prepare_encodings
from redundancy.metrics import collect_layer_stats, compute_cosine_similarity_matrix
from redundancy.plotting import plot_bi_bar, plot_residual_norms, plot_similarity_heatmap
from utils import (
    build_measurement_config,
    build_measurement_result_dict,
    print_measurement_summary,
    save_measurement_results,
)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="Block Influence redundancy measurement.")
    p.add_argument("--config", type=str, default=None, help="JSON config file.")
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=list(DTYPES))
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Subsample calibration windows (for quick smoke tests).",
    )
    p.add_argument("--output-dir", type=str, default="experiments")
    p.add_argument("--figures-dir", type=str, default="reports/group9/figures")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_measurement_config(args)
    command = "python " + " ".join(sys.argv)

    print("=" * 50)
    print(
        f"BLOCK INFLUENCE MEASUREMENT — "
        f"{cfg['model_name']} × WikiText-2"
    )
    print("=" * 50)
    print(f"  max_length={cfg['max_length']}  stride={cfg['stride']}  dtype={cfg['dtype']}  seed={cfg['seed']}")
    print(f"  max_windows={cfg['max_windows'] or 'all'}")
    print("=" * 50)

    torch.manual_seed(cfg["seed"])
    device = get_device(cfg["device"])
    dtype = DTYPES[cfg["dtype"]]

    model, tokenizer = load_model(cfg["model_name"], device=str(device), dtype=dtype)
    dataset = load_wikitext(split="test")
    input_windows = prepare_encodings(dataset, tokenizer, cfg["max_length"], cfg["stride"])

    print("\nCollecting per-layer hidden-state statistics ...")
    stats = collect_layer_stats(model, input_windows, device, max_windows=cfg["max_windows"])
    similarity_matrix = compute_cosine_similarity_matrix(stats.layer_means)

    results = build_measurement_result_dict(cfg, stats, similarity_matrix, device, command)
    save_measurement_results(results, cfg["output_dir"], cfg["model_name"])
    print_measurement_summary(cfg, stats)

    print("\nSaving figures ...")
    figures_dir = cfg["figures_dir"]
    plot_bi_bar(stats.bi_scores, f"{figures_dir}/bi_bar_{cfg['model_name']}.png")
    plot_similarity_heatmap(
        similarity_matrix,
        f"{figures_dir}/cosine_heatmap_{cfg['model_name']}.png",
        title="Hidden-state cosine similarity (centered)",
    )
    plot_residual_norms(stats.relative_residual_norms, f"{figures_dir}/residual_norms_{cfg['model_name']}.png")


if __name__ == "__main__":
    main()
