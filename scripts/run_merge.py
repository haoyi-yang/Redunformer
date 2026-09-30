#!/usr/bin/env python
"""Weeks 11-12: neuron replacement (merge) for near-duplicate pairs.

Weeks 9-10 left an open question: masking one twin of a near-duplicate pair costs
perplexity, but masking *both* costs vastly more, so the survivor clearly carries
much of the pair's function. This script tests whether handing the redundant
twin's contribution to that survivor removes the cost entirely.

For each pair it fits ``h_drop ~= alpha * h_keep + beta`` on calibration
activations, then compares:

    mask_only        drop the redundant twin (the Weeks 9-10 intervention)
    merge            drop it and add ``(alpha * h_keep + beta) x W[:, drop]``
    merge_no_bias    the same without the constant term

The merge wins if its delta perplexity is closer to zero than ``mask_only``.
``r2`` of each fit bounds how much a merge could possibly recover.

Example:
    python scripts/run_merge.py --config configs/pruning/merge_qwen3_0.6b.yaml
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

from redundancy import plotting  # noqa: E402
from redundancy.config import REPO_ROOT as ROOT, load_yaml  # noqa: E402
from redundancy.data import build_calibration_blocks, load_text_dataset  # noqa: E402
from redundancy.eval import compute_perplexity  # noqa: E402
from redundancy.models import load_causal_lm  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    NeuronMasker,
    NeuronMerger,
    PairActivationCollector,
    find_latest_measurement,
    fit_merge_plans,
    load_measurement,
    merge_stats,
    pairs_summary,
    plans_to_masked,
    resolve_layer_filter,
    select_disjoint_pairs,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duplicate-neuron replacement (merge) study.")
    parser.add_argument(
        "--config",
        default="configs/pruning/merge_qwen3_0.6b.yaml",
        help="Path to merge config (YAML).",
    )
    parser.add_argument("--measurement", default=None, help="Override measurement JSON path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override eval documents.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the study to these layer indices (default: config scope).",
    )
    parser.add_argument("--tag", default=None, help="Suffix for the output filenames.")
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

    pair_cfg = cfg.get("pairs", {})
    fit_cfg = cfg.get("fit", {})
    eval_cfg = cfg.get("eval", {})

    output_dir = ROOT / cfg.get("output_dir", "experiments/results")
    figures_dir = ROOT / cfg.get("figures_dir", "experiments/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    use_fallback = bool(model_cfg.get("_use_fallback"))
    display_id = model_cfg.get("fallback_model_id") if use_fallback else model_cfg.get("model_id")
    print(f"Loading model from {display_id} ...")
    loaded = load_causal_lm(model_cfg)
    device = next(loaded.model.parameters()).device

    measurement_path = (
        Path(args.measurement) if args.measurement else _resolve_measurement(cfg, loaded.model_id)
    )
    print(f"Using measurement artifact {measurement_path}")
    artifact = load_measurement(measurement_path)
    print(f"  calibrated on: {artifact.calibration_label()}")
    layer_scope = args.layers if args.layers is not None else pair_cfg.get("layers", "all")
    layer_filter = resolve_layer_filter(artifact, layer_scope)

    pairs = select_disjoint_pairs(
        artifact,
        min_abs_corr=float(pair_cfg.get("min_abs_corr", 0.9)),
        max_pairs_per_layer=pair_cfg.get("max_pairs_per_layer"),
        layer_filter=layer_filter,
    )
    if not pairs:
        print(
            "No duplicate pairs above the correlation threshold in this measurement. "
            "Lower pairs.min_abs_corr or re-run measurement with a larger reservoir."
        )
        return 1
    print(f"Selected {len(pairs)} disjoint duplicate pairs.")

    # --- fit the substitution on calibration activations -------------------
    fit_dataset_cfg = load_yaml(fit_cfg.get("dataset_config", cfg["dataset_config"]))
    if fit_cfg.get("max_samples") is not None:
        fit_dataset_cfg["max_samples"] = fit_cfg["max_samples"]
    print("Loading calibration text for the merge fit ...")
    fit_texts, fit_meta = load_text_dataset(fit_dataset_cfg)
    blocks = build_calibration_blocks(
        fit_texts,
        loaded.tokenizer,
        seq_len=int(fit_cfg.get("seq_len", 512)),
        max_blocks=int(fit_cfg.get("num_blocks", 32)),
    )
    neurons_by_layer: dict[str, set[int]] = {}
    for pair in pairs:
        neurons_by_layer.setdefault(pair.layer_name, set()).update((pair.drop, pair.keep))

    print(f"Fitting merge coefficients on {len(blocks)} calibration blocks ...")
    with PairActivationCollector(loaded.model, neurons_by_layer) as collector:
        with torch.inference_mode():
            for block in blocks:
                loaded.model(block.to(device))
    plans = fit_merge_plans(collector.activations(), pairs)
    if not plans:
        print("Could not fit any merge plan (pair layers not found in the model).")
        return 1

    stats = merge_stats(plans)
    print(f"Fitted {len(plans)} plans (mean r2 = {stats['mean_r2']:.4f}):")
    for plan in plans:
        print(
            f"  {plan.layer_name}: drop {plan.drop} -> keep {plan.keep} | "
            f"rho {plan.correlation:+.4f}  alpha {plan.alpha:+.4f}  "
            f"beta {plan.beta:+.4f}  r2 {plan.r2:.4f}"
        )

    # --- evaluate the arms -------------------------------------------------
    print("Loading evaluation dataset ...")
    texts, dataset_meta = load_text_dataset(dataset_cfg)
    ppl_kwargs = {
        "max_length": int(eval_cfg.get("max_length", 2048)),
        "stride": int(eval_cfg.get("stride", 512)),
    }
    masked_selection = plans_to_masked(plans)

    with torch.inference_mode():
        print("\nBaseline (no intervention):")
        baseline_ppl = compute_perplexity(loaded, texts, **ppl_kwargs)["perplexity"]
        print(f"  baseline perplexity = {baseline_ppl:.4f}")

        results: list[dict] = []

        with NeuronMasker(loaded.model, layer_names=artifact.layer_names) as masker:
            num_masked = masker.apply(masked_selection)
            print(f"\nmask_only ({num_masked} neurons masked):")
            ppl = compute_perplexity(loaded, texts, **ppl_kwargs)["perplexity"]
            results.append(
                {
                    "condition": "mask_only",
                    "description": "drop the redundant twin (Weeks 9-10 intervention)",
                    "num_neurons": num_masked,
                    "perplexity": ppl,
                    "delta_perplexity": ppl - baseline_ppl,
                }
            )
            print(f"  ppl = {ppl:.4f} (delta {ppl - baseline_ppl:+.4f})")

        for label, include_bias, description in (
            ("merge", True, "drop the twin and route its contribution through its partner"),
            ("merge_no_bias", False, "same, without the constant beta term"),
        ):
            with NeuronMerger(loaded.model, plans, include_bias=include_bias) as merger:
                print(f"\n{label} ({merger.num_merged()} pairs merged):")
                ppl = compute_perplexity(loaded, texts, **ppl_kwargs)["perplexity"]
                results.append(
                    {
                        "condition": label,
                        "description": description,
                        "num_neurons": merger.num_merged(),
                        "perplexity": ppl,
                        "delta_perplexity": ppl - baseline_ppl,
                    }
                )
                print(f"  ppl = {ppl:.4f} (delta {ppl - baseline_ppl:+.4f})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = loaded.model_id.replace("/", "__")
    suffix = f"_{args.tag}" if args.tag else ""
    scope_label = "all layers" if layer_filter is None else f"layers {sorted(layer_scope)}"

    fig_path = plotting.plot_condition_bars(
        [r["condition"] for r in results],
        [r["delta_perplexity"] for r in results],
        figures_dir / f"merge{suffix}_{safe_model}_{stamp}.png",
        title=f"Duplicate-neuron replacement — {loaded.model_id} ({scope_label})",
        ylabel="\u0394 WikiText-2 perplexity vs baseline",
        highlight=("merge",),
    )

    verdict = _merge_verdict(results, stats)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_11_12_duplicate_neuron_replacement",
        "question": "does routing a redundant twin's contribution through its "
        "partner remove the cost of masking it?",
        "seed": seed,
        "config_path": str(args.config),
        "command": " ".join(sys.argv),
        "model": {
            "model_id": loaded.model_id,
            "load_in_4bit": loaded.load_in_4bit,
            "config": cfg["model_config"],
        },
        "dataset": dataset_meta,
        "fit_dataset": {**fit_meta, "num_blocks": len(blocks)},
        "measurement_artifact": str(measurement_path),
        "pairs": {
            "min_abs_corr": float(pair_cfg.get("min_abs_corr", 0.9)),
            "layers": layer_scope,
            "num_pairs": len(pairs),
            "selected": pairs_summary(pairs),
        },
        "merge_plans": [plan.summary() for plan in plans],
        "merge_stats": stats,
        "baseline_perplexity": baseline_ppl,
        "conditions": results,
        "verdict": verdict,
        "artifacts": {"figures": {"merge": str(fig_path)}},
    }
    out_path = output_dir / f"merge{suffix}_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote summary -> {out_path}")
    print(f"Wrote figure  -> {fig_path}")
    print("\nDelta perplexity by condition:")
    for r in sorted(results, key=lambda r: abs(r["delta_perplexity"])):
        print(f"  {r['condition']:>16}  {r['delta_perplexity']:+.4f}")
    print(f"\nVerdict: {verdict['summary']}")
    return 0


def _merge_verdict(results: list[dict], stats: dict) -> dict:
    """Did replacement beat plain masking on the same neurons?"""
    by_label = {r["condition"]: r for r in results}
    mask = by_label.get("mask_only")
    merge = by_label.get("merge")
    if mask is None or merge is None:
        return {"supported": None, "summary": "conditions missing; cannot evaluate"}

    mask_delta = mask["delta_perplexity"]
    merge_delta = merge["delta_perplexity"]
    verdict = {
        "delta_mask_only": mask_delta,
        "delta_merge": merge_delta,
        "mean_fit_r2": stats.get("mean_r2"),
    }
    r2_note = f"; mean fit r2 = {stats.get('mean_r2', float('nan')):.4f}"

    if mask_delta <= 0:
        # Nothing to repair: these neurons were free (or better) to drop, so a
        # merge can only add back a contribution the model did not want.
        verdict.update(
            supported=None,
            fraction_of_mask_cost_recovered=None,
            summary=(
                f"masking these neurons already left perplexity at {mask_delta:+.4f}, "
                f"so replacement has nothing to recover; merging gives "
                f"{merge_delta:+.4f}" + r2_note
            ),
        )
        return verdict

    recovered = (mask_delta - merge_delta) / mask_delta
    verdict.update(
        supported=bool(abs(merge_delta) < mask_delta),
        fraction_of_mask_cost_recovered=recovered,
        summary=(
            f"masking costs {mask_delta:+.4f} PPL, merging {merge_delta:+.4f} "
            f"({recovered:.1%} of the masking cost recovered)" + r2_note
        ),
    )
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
