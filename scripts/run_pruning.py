#!/usr/bin/env python
"""Weeks 9–10: mask FFN neurons and measure the cost.

Replays a Weeks 5–8 measurement artifact as a masking intervention: for each
ranking strategy and removal ratio, zero the selected post-SwiGLU neurons and
re-evaluate WikiText-2 perplexity (optionally PIQA). The headline deliverable is
the *gap* between a redundancy-guided curve and the random baseline at matched
per-layer sparsity, not the absolute degradation.

Masking only — parameter shapes never change, so no speed or FLOP claim.

Example:
    python scripts/run_pruning.py --config configs/pruning/neuron_masking_qwen3_0.6b.yaml
    python scripts/run_pruning.py --config configs/pruning/neuron_masking_smoke.yaml \
        --use-fallback-model
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.config import REPO_ROOT as ROOT, load_yaml  # noqa: E402
from redundancy.data import load_text_dataset  # noqa: E402
from redundancy.eval import compute_perplexity, run_lm_eval_loaded  # noqa: E402
from redundancy.models import load_causal_lm  # noqa: E402
from redundancy import plotting  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    NeuronMasker,
    find_latest_measurement,
    load_measurement,
    resolve_layer_filter,
    select_neurons,
    selection_stats,
)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_measurement(cfg: dict, model_id: str) -> Path:
    spec = cfg.get("measurement_path")
    if spec and spec != "auto":
        path = Path(spec)
        return path if path.is_absolute() else ROOT / path
    return find_latest_measurement(ROOT / cfg.get("output_dir", "experiments/results"), model_id)


def _accuracy_from_lm_eval(results: dict) -> dict[str, float]:
    """Flatten lm-eval output to ``{task.metric: value}`` for the summary JSON.

    lm-eval reports keys like ``acc,none`` / ``acc_stderr,none``; we keep the
    accuracies and drop the standard errors.
    """
    flat: dict[str, float] = {}
    for task, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or "stderr" in key:
                continue
            if key.startswith(("acc", "acc_norm")):
                flat[f"{task}.{key.split(',')[0]}"] = float(value)
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mask FFN neurons and re-evaluate (Weeks 9-10).")
    parser.add_argument(
        "--config",
        default="configs/pruning/neuron_masking.yaml",
        help="Path to pruning config (YAML).",
    )
    parser.add_argument("--measurement", default=None, help="Override measurement JSON path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override eval documents.")
    parser.add_argument("--tag", default=None, help="Suffix for the output filenames.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices overriding pruning.layers (e.g. 0,1,2,3).",
    )
    parser.add_argument(
        "--ratios",
        default=None,
        help="Comma-separated masking ratios overriding pruning.ratios (e.g. 0.10).",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated ranking strategies overriding pruning.strategies.",
    )
    parser.add_argument(
        "--no-lm-eval",
        action="store_true",
        help="Skip PIQA / lm-eval even if the config requests it.",
    )
    parser.add_argument(
        "--use-fallback-model",
        action="store_true",
        help="Load fallback_model_id from the model config (smaller model).",
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

    prune_cfg = dict(cfg.get("pruning", {}))
    eval_cfg = dict(cfg.get("eval", {}))
    if args.layers:
        prune_cfg["layers"] = [int(part.strip()) for part in args.layers.split(",") if part.strip()]
    if args.ratios:
        prune_cfg["ratios"] = [float(part.strip()) for part in args.ratios.split(",") if part.strip()]
    if args.strategies:
        prune_cfg["strategies"] = [part.strip() for part in args.strategies.split(",") if part.strip()]
    if args.no_lm_eval:
        eval_cfg["run_lm_eval"] = False
    strategies = list(prune_cfg.get("strategies", ["importance", "random"]))
    ratios = [float(r) for r in prune_cfg.get("ratios", [0.05, 0.10, 0.25, 0.50])]
    ratios = sorted({r for r in ratios if r > 0})

    output_dir = ROOT / cfg.get("output_dir", "experiments/results")
    figures_dir = ROOT / cfg.get("figures_dir", "experiments/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    use_fallback = bool(model_cfg.get("_use_fallback"))
    display_id = model_cfg.get("fallback_model_id") if use_fallback else model_cfg.get("model_id")
    print(f"Loading model from {display_id} ...")
    loaded = load_causal_lm(model_cfg)

    measurement_path = (
        Path(args.measurement) if args.measurement else _resolve_measurement(cfg, loaded.model_id)
    )
    print(f"Using measurement artifact {measurement_path}")
    artifact = load_measurement(measurement_path)
    print(f"  calibrated on: {artifact.calibration_label()}")
    layer_filter = resolve_layer_filter(artifact, prune_cfg.get("layers", "all"))

    print("Loading evaluation dataset ...")
    texts, dataset_meta = load_text_dataset(dataset_cfg)

    ppl_kwargs = {
        "max_length": int(eval_cfg.get("max_length", 2048)),
        "stride": int(eval_cfg.get("stride", 512)),
    }
    run_piqa = bool(eval_cfg.get("run_lm_eval", False))
    lm_eval_tasks = list(eval_cfg.get("lm_eval_tasks", ["piqa"]))

    def evaluate(tag: str) -> dict:
        print(f"  evaluating {tag} ...")
        entry = {"perplexity": compute_perplexity(loaded, texts, **ppl_kwargs)["perplexity"]}
        if run_piqa:
            lm_results = run_lm_eval_loaded(
                loaded,
                lm_eval_tasks,
                limit=eval_cfg.get("lm_eval_limit"),
                batch_size=int(eval_cfg.get("lm_eval_batch_size", 1)),
                seed=seed,
            )
            entry["lm_eval"] = _accuracy_from_lm_eval(lm_results)
        return entry

    with NeuronMasker(loaded.model, layer_names=artifact.layer_names) as masker, torch.inference_mode():
        maskable = [n for n in artifact.layer_names if n in set(masker.layer_names)]
        print(f"Masking {len(maskable)} of {len(artifact.layer_names)} measured FFN layers.")

        print("\nBaseline (no masking):")
        baseline = evaluate("baseline")
        print(f"  baseline perplexity = {baseline['perplexity']:.4f}")

        runs: list[dict] = []
        for strategy in strategies:
            for ratio in ratios:
                selected = select_neurons(
                    artifact, strategy, ratio, seed=seed, layer_filter=layer_filter
                )
                num_masked = masker.apply(selected)
                stats = selection_stats(artifact, selected)
                print(f"\n{strategy} @ {ratio:.0%} ({num_masked} neurons masked):")
                entry = evaluate(f"{strategy}@{ratio:.0%}")
                entry.update(
                    {
                        "strategy": strategy,
                        "ratio": ratio,
                        "num_masked": num_masked,
                        "masked_fraction": stats["masked_fraction"],
                        "importance_removed_fraction": stats["importance_removed_fraction"],
                        "delta_perplexity": entry["perplexity"] - baseline["perplexity"],
                    }
                )
                runs.append(entry)
                print(
                    f"  ppl = {entry['perplexity']:.4f} "
                    f"(delta {entry['delta_perplexity']:+.4f}), "
                    f"importance removed = {stats['importance_removed_fraction']:.2%}"
                )
                masker.clear()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = loaded.model_id.replace("/", "__")
    suffix = f"_{args.tag}" if args.tag else ""

    # ---- Plots ------------------------------------------------------------------
    fig_paths: dict[str, str] = {}
    curve_ratios = [0.0, *ratios]
    ppl_series = {
        strategy: [baseline["perplexity"]]
        + [r["perplexity"] for r in runs if r["strategy"] == strategy]
        for strategy in strategies
    }
    fig_paths["removal_curve_ppl"] = str(
        plotting.plot_removal_curves(
            curve_ratios,
            ppl_series,
            figures_dir / f"prune_ppl{suffix}_{safe_model}_{stamp}.png",
            title=f"WikiText-2 perplexity vs neuron masking — {loaded.model_id}",
            baseline=baseline["perplexity"],
            logy=bool(prune_cfg.get("log_y", True)),
        )
    )
    baseline_acc = baseline.get("lm_eval") or {}
    if run_piqa and baseline_acc:
        acc_key = next(iter(baseline_acc))
        acc_series = {
            strategy: [baseline_acc[acc_key]]
            + [
                r.get("lm_eval", {}).get(acc_key, float("nan"))
                for r in runs
                if r["strategy"] == strategy
            ]
            for strategy in strategies
        }
        fig_paths["removal_curve_accuracy"] = str(
            plotting.plot_removal_curves(
                curve_ratios,
                acc_series,
                figures_dir / f"prune_acc{suffix}_{safe_model}_{stamp}.png",
                title=f"{acc_key} vs neuron masking — {loaded.model_id}",
                ylabel=acc_key,
                baseline=baseline_acc[acc_key],
            )
        )

    # ---- Summary JSON -----------------------------------------------------------
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_9_10_neuron_masking",
        "seed": seed,
        "config_path": str(args.config),
        "command": " ".join(sys.argv),
        "model": {
            "model_id": loaded.model_id,
            "load_in_4bit": loaded.load_in_4bit,
            "config": cfg["model_config"],
        },
        "dataset": dataset_meta,
        "measurement_artifact": str(measurement_path),
        "intervention": {
            "kind": "activation masking (post-SwiGLU channel -> down_proj input)",
            "strategies": strategies,
            "ratios": ratios,
            "layers": prune_cfg.get("layers", "all"),
            "num_masked_layers": len(maskable),
        },
        "eval": {
            "max_length": ppl_kwargs["max_length"],
            "stride": ppl_kwargs["stride"],
            "num_documents": dataset_meta.get("num_samples"),
            "lm_eval_tasks": lm_eval_tasks if run_piqa else [],
        },
        "baseline": baseline,
        "runs": runs,
        "artifacts": {"figures": fig_paths},
    }
    out_path = output_dir / f"pruning{suffix}_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote summary -> {out_path}")
    for label, p in fig_paths.items():
        print(f"Wrote figure  -> {p}  ({label})")
    _print_table(baseline, runs, strategies, ratios)
    return 0


def _print_table(baseline: dict, runs: list[dict], strategies: list[str], ratios: list[float]) -> None:
    print(f"\nWikiText-2 perplexity (baseline {baseline['perplexity']:.4f}):")
    header = f"  {'ratio':>7}" + "".join(f"{s:>16}" for s in strategies)
    print(header)
    for ratio in ratios:
        row = f"  {ratio:>6.0%} "
        for strategy in strategies:
            match = next(
                (r for r in runs if r["strategy"] == strategy and r["ratio"] == ratio), None
            )
            row += f"{match['perplexity']:>16.3f}" if match else f"{'-':>16}"
        print(row)
    guided = [s for s in strategies if s != "random"]
    if "random" in strategies and guided:
        print("\nGap vs random (negative = guided is better):")
        for ratio in ratios:
            rnd = next(
                (r for r in runs if r["strategy"] == "random" and r["ratio"] == ratio), None
            )
            if rnd is None:
                continue
            parts = []
            for strategy in guided:
                match = next(
                    (r for r in runs if r["strategy"] == strategy and r["ratio"] == ratio), None
                )
                if match:
                    parts.append(f"{strategy}: {match['perplexity'] - rnd['perplexity']:+.3f}")
            print(f"  {ratio:>6.0%}  " + "   ".join(parts))

    baseline_acc = baseline.get("lm_eval") or {}
    for acc_key in baseline_acc:
        print(f"\n{acc_key} (baseline {baseline_acc[acc_key]:.4f}):")
        print(header)
        for ratio in ratios:
            row = f"  {ratio:>6.0%} "
            for strategy in strategies:
                match = next(
                    (r for r in runs if r["strategy"] == strategy and r["ratio"] == ratio), None
                )
                value = (match or {}).get("lm_eval", {}).get(acc_key)
                row += f"{value:>16.4f}" if value is not None else f"{'-':>16}"
            print(row)


if __name__ == "__main__":
    raise SystemExit(main())
