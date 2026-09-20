#!/usr/bin/env python3
"""Compute only the pairwise hidden-state cosine-similarity heatmap."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.data import load_wikitext, prepare_encodings
from redundancy.metrics.block_influence import compute_tokenwise_cosine_similarity_matrix
from redundancy.models import get_device, load_model
from redundancy.plotting import plot_similarity_heatmap


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--figures-dir", default="reports/group9/figures")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_name = config["model_name"]
    device = get_device(config.get("device"))
    dtype_name = config.get("dtype", "float32")
    dtype = DTYPES[dtype_name]

    # Use TensorFloat-32 for the large normalized-vector matrix products on
    # recent NVIDIA GPUs while retaining float32 inputs and outputs.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model, tokenizer = load_model(model_name, device=str(device), dtype=dtype)
    model.config.use_cache = False
    dataset = load_wikitext(split="test")
    windows = prepare_encodings(
        dataset,
        tokenizer,
        config.get("max_length", 1024),
        config.get("stride", 1024),
    )

    print("\nComputing mean tokenwise cosine similarities ...")
    matrix = compute_tokenwise_cosine_similarity_matrix(
        model, windows, device, max_windows=args.max_windows
    )

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"similarity_{safe_name}_{timestamp}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset": "wikitext-2-raw-v1",
        "metric": "mean_tokenwise_cosine_similarity",
        "max_length": config.get("max_length", 1024),
        "stride": config.get("stride", 1024),
        "dtype": dtype_name,
        "device": str(device),
        "n_windows": min(windows.size(0), args.max_windows or windows.size(0)),
        "n_hidden_states": int(matrix.shape[0]),
        "cosine_similarity_matrix": matrix.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Results saved to {output_path}")

    figure_path = Path(args.figures_dir) / f"cosine_heatmap_{safe_name}.png"
    observed_min = float(matrix.min())
    plot_similarity_heatmap(
        matrix,
        str(figure_path),
        title="Mean tokenwise hidden-state cosine similarity",
        vmin=observed_min,
        vmax=1.0,
    )
    print(f"  Observed similarity range: {observed_min:.4f} to {matrix.max():.4f}")


if __name__ == "__main__":
    main()
