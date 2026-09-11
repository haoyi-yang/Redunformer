"""Shared utility functions."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def load_json_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path) as f:
        return json.load(f)


def build_config(args):
    """Merge JSON config with CLI overrides."""
    cfg = {}

    confg_path = args.config or "configs/baseline.json"
    cfg = load_json_config(confg_path)

    return {
        "model_name":    cfg.get("model_name", "gpt2"),
        "dataset":       "wikitext-2-raw-v1",
        "max_length":    args.max_length   or cfg.get("max_length", 1024),
        "stride":        args.stride       or cfg.get("stride", 512),
        "batch_size":    args.batch_size   or cfg.get("batch_size", 4),
        "seed":          args.seed if args.seed is not None else cfg.get("seed", 42),
        "device":        args.device       or cfg.get("device"),
        "lm_eval_tasks": cfg.get("lm_eval_tasks", ["hellaswag"]),
        "skip_lm_eval":  args.skip_lm_eval or cfg.get("skip_lm_eval", False),
        "output_dir":    args.output_dir,
    }


def build_result_dict(cfg, device):
    """Build a structured results dict with system info."""
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "model_name": cfg["model_name"],
        "dataset":    cfg["dataset"],
        "max_length": cfg["max_length"],
        "stride":     cfg["stride"],
        "seed":       cfg["seed"],
        "device":     str(device),
        "system": {
            "platform": platform.platform(),
            "python":   platform.python_version(),
            "torch":    torch.__version__,
        }
    }


def save_results(results: dict, output_dir: str, model_name: str):
    """Save results as a timestamped JSON file."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Hugging Face model names can contain "/" (e.g. Qwen/Qwen3-0.6B).
    # Make them safe for use in filenames.
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"baseline_{safe_model_name}_{ts}.json"

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {path}")
    return path


def print_summary(cfg, ppl, lm_eval_results=None):
    """Print a final summary block."""
    print("\n" + "=" * 50)
    print(f"  Model:       {cfg['model_name']}")
    print(f"  Perplexity:  {ppl:.2f}")
    if lm_eval_results:
        for task, metrics in lm_eval_results.items():
            acc = metrics.get("acc_norm,none", metrics.get("acc,none", "N/A"))
            print(f"  {task}: {acc}")
    print("=" * 50)


def build_measurement_config(args):
    """Merge JSON config with CLI overrides for the block-influence measurement script."""
    cfg = {}
    if args.config:
        cfg = load_json_config(args.config)

    return {
        "model_name":  args.model_name or cfg.get("model_name", "gpt2"),
        "dataset":     "wikitext-2-raw-v1",
        "max_length":  args.max_length  or cfg.get("max_length", 1024),
        "stride":      args.stride      or cfg.get("stride", 512),
        "seed":        args.seed if args.seed is not None else cfg.get("seed", 42),
        "device":      args.device      or cfg.get("device"),
        "dtype":       args.dtype       or cfg.get("dtype", "float32"),
        "max_windows": args.max_windows,
        "output_dir":  args.output_dir,
        "figures_dir": args.figures_dir,
    }


def build_measurement_result_dict(cfg, stats, similarity_matrix, device, command: str):
    """Build a structured results dict for a Block Influence measurement run."""
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "model_name": cfg["model_name"],
        "dataset":    cfg["dataset"],
        "metric":     "block_influence",
        "max_length": cfg["max_length"],
        "stride":     cfg["stride"],
        "seed":       cfg["seed"],
        "device":     str(device),
        "dtype":      cfg["dtype"],
        "command":    command,
        "system": {
            "platform": platform.platform(),
            "python":   platform.python_version(),
            "torch":    torch.__version__,
        },
        "n_layers":       stats.n_layers,
        "n_windows":      stats.n_windows,
        "n_tokens":       stats.n_tokens,
        "bi_scores":      [round(x, 6) for x in stats.bi_scores.tolist()],
        "relative_residual_norms": [round(x, 6) for x in stats.relative_residual_norms.tolist()],
        "cosine_similarity_matrix": [[round(x, 6) for x in row] for row in similarity_matrix.tolist()],
    }


def save_measurement_results(results: dict, output_dir: str, model_name: str):
    """Save measurement results as a timestamped JSON file."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Make Hugging Face model IDs safe for filenames.
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"measurement_{safe_model_name}_{ts}.json"

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {path}")
    return path


def print_measurement_summary(cfg, stats):
    """Print a final summary block for a measurement run."""
    bi = stats.bi_scores
    most_redundant = int(bi.argmin())
    least_redundant = int(bi.argmax())
    print("\n" + "=" * 50)
    print(f"  Model:             {cfg['model_name']}")
    print(f"  Layers:            {stats.n_layers}")
    print(f"  Windows / tokens:  {stats.n_windows} / {stats.n_tokens}")
    print(f"  Most redundant:    block {most_redundant}  (BI={bi[most_redundant]:.4f})")
    print(f"  Least redundant:   block {least_redundant}  (BI={bi[least_redundant]:.4f})")
    print("=" * 50)
