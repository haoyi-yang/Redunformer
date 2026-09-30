#!/usr/bin/env python
"""Weeks 1–2: load model + dataset, run baseline eval, save JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.config import REPO_ROOT as ROOT, load_yaml  # noqa: E402
from redundancy.data import load_text_dataset  # noqa: E402
from redundancy.eval import compute_perplexity, run_lm_eval  # noqa: E402
from redundancy.models import load_causal_lm  # noqa: E402


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run baseline LLM evaluation.")
    parser.add_argument(
        "--config",
        default="configs/eval/baseline.yaml",
        help="Path to baseline eval config (YAML).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override dataset max_samples for quick runs.",
    )
    parser.add_argument(
        "--use-fallback-model",
        action="store_true",
        help="Load fallback_model_id from model config (smaller Qwen3.5).",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    model_cfg = load_yaml(cfg["model_config"])
    if args.use_fallback_model or cfg.get("use_fallback_model"):
        model_cfg["_use_fallback"] = True

    dataset_cfg = load_yaml(cfg["dataset_config"])
    if cfg.get("max_samples") is not None:
        dataset_cfg["max_samples"] = cfg["max_samples"]
    if args.max_samples is not None:
        dataset_cfg["max_samples"] = args.max_samples

    eval_cfg = cfg.get("eval", {})
    output_dir = ROOT / cfg.get("output_dir", "experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    use_fallback = bool(model_cfg.get("_use_fallback"))
    display_id = model_cfg.get("fallback_model_id") if use_fallback else model_cfg.get("model_id")
    print(f"Loading model from {display_id} ...")
    loaded = load_causal_lm(model_cfg)

    print("Loading dataset ...")
    texts, dataset_meta = load_text_dataset(dataset_cfg)

    results: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_1_2_baseline",
        "seed": seed,
        "config_path": str(args.config),
        "command": command,
        "model": {
            "model_id": loaded.model_id,
            "load_in_4bit": loaded.load_in_4bit,
            "config": cfg["model_config"],
        },
        "dataset": dataset_meta,
        "metrics": {},
    }

    if eval_cfg.get("run_perplexity", True):
        print("Computing perplexity ...")
        ppl = compute_perplexity(
            loaded,
            texts,
            max_length=int(eval_cfg.get("max_length", 2048)),
            stride=int(eval_cfg.get("stride", 512)),
            batch_size=int(eval_cfg.get("batch_size", 1)),
        )
        results["metrics"]["perplexity"] = ppl

    if eval_cfg.get("run_lm_eval", False):
        print("Running lm-evaluation-harness ...")
        lm_results = run_lm_eval(
            loaded.model_id,
            list(eval_cfg.get("lm_eval_tasks", ["piqa"])),
            limit=eval_cfg.get("lm_eval_limit"),
            batch_size=int(eval_cfg.get("lm_eval_batch_size", 1)),
            seed=seed,
        )
        results["metrics"]["lm_eval"] = lm_results

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = loaded.model_id.replace("/", "__")
    out_path = output_dir / f"baseline_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {out_path}")
    if "perplexity" in results["metrics"]:
        print(f"Perplexity: {results['metrics']['perplexity']['perplexity']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
