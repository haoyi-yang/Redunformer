#!/usr/bin/env python
"""Weeks 9–10: duplicate-neuron ablation study (hypothesis H5).

Measurement found pairs of FFN neurons whose activations are nearly collinear.
Correlation alone does not show the network only needs one of them, so this
script masks them and compares the cost against controls at the same neuron
count:

    duplicate_one       one neuron of each near-duplicate pair
    duplicate_both      both neurons of each pair
    importance_matched  uncorrelated neurons of matched importance  <- key control
    random_matched      random uncorrelated neurons
    lowest_importance   least-important neurons

H5 holds if ``duplicate_one`` costs clearly less than ``importance_matched``.
Because the removal budget is tiny (a handful of neurons), the evaluation set
should be large enough for the perplexity differences to be readable.

Example:
    python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation_qwen3_0.6b.yaml
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
from redundancy.eval import compute_perplexity  # noqa: E402
from redundancy.models import load_causal_lm  # noqa: E402
from redundancy import plotting  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    NeuronMasker,
    build_conditions,
    find_latest_measurement,
    load_measurement,
    pairs_summary,
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
    parser = argparse.ArgumentParser(description="Duplicate-neuron ablation study (H5).")
    parser.add_argument(
        "--config",
        default="configs/pruning/pair_ablation_qwen3_0.6b.yaml",
        help="Path to pair-ablation config (YAML).",
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
    eval_cfg = cfg.get("eval", {})

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
    print(f"Selected {len(pairs)} disjoint duplicate pairs:")
    for p in pairs:
        print(f"  {p.layer_name}: drop {p.drop} / keep {p.keep} (rho = {p.correlation:+.4f})")

    conditions = build_conditions(artifact, pairs, seed=seed)

    print("Loading evaluation dataset ...")
    texts, dataset_meta = load_text_dataset(dataset_cfg)
    ppl_kwargs = {
        "max_length": int(eval_cfg.get("max_length", 2048)),
        "stride": int(eval_cfg.get("stride", 512)),
    }

    with NeuronMasker(loaded.model, layer_names=artifact.layer_names) as masker, torch.inference_mode():
        print("\nBaseline (no masking):")
        baseline_ppl = compute_perplexity(loaded, texts, **ppl_kwargs)["perplexity"]
        print(f"  baseline perplexity = {baseline_ppl:.4f}")

        results: list[dict] = []
        for cond in conditions:
            num_masked = masker.apply(cond.masked)
            print(f"\n{cond.label} ({num_masked} neurons masked):")
            ppl = compute_perplexity(loaded, texts, **ppl_kwargs)["perplexity"]
            delta = ppl - baseline_ppl
            results.append(
                {
                    "condition": cond.label,
                    "description": cond.description,
                    "num_masked": num_masked,
                    "mean_importance": cond.meta.get("mean_importance"),
                    "perplexity": ppl,
                    "delta_perplexity": delta,
                }
            )
            print(f"  ppl = {ppl:.4f} (delta {delta:+.4f})")
            masker.clear()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = loaded.model_id.replace("/", "__")
    suffix = f"_{args.tag}" if args.tag else ""
    scope_label = "all layers" if layer_filter is None else f"layers {sorted(layer_scope)}"

    fig_path = plotting.plot_condition_bars(
        [r["condition"] for r in results],
        [r["delta_perplexity"] for r in results],
        figures_dir / f"pair_ablation{suffix}_{safe_model}_{stamp}.png",
        title=f"Duplicate-neuron ablation (H5) — {loaded.model_id} ({scope_label})",
        ylabel="\u0394 WikiText-2 perplexity vs baseline",
        highlight=("duplicate_one", "duplicate_both"),
    )

    verdict = _h5_verdict(results)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_9_10_duplicate_pair_ablation",
        "hypothesis": "H5: one neuron of a near-duplicate pair is cheaper to mask "
        "than a matched-importance uncorrelated neuron",
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
        "pairs": {
            "min_abs_corr": float(pair_cfg.get("min_abs_corr", 0.9)),
            "layers": layer_scope,
            "num_pairs": len(pairs),
            "selected": pairs_summary(pairs),
        },
        "baseline_perplexity": baseline_ppl,
        "conditions": results,
        "verdict": verdict,
        "artifacts": {"figures": {"pair_ablation": str(fig_path)}},
    }
    out_path = output_dir / f"pair_ablation{suffix}_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote summary -> {out_path}")
    print(f"Wrote figure  -> {fig_path}")
    print("\nDelta perplexity by condition:")
    for r in sorted(results, key=lambda r: r["delta_perplexity"]):
        print(f"  {r['condition']:>20}  {r['delta_perplexity']:+.4f}  ({r['num_masked']} neurons)")
    print(f"\nH5: {verdict['summary']}")
    return 0


def _h5_verdict(results: list[dict]) -> dict:
    """Compare duplicate_one against the matched-importance control.

    Also reports how well the control matched: duplicates can sit at the very
    top of a layer's importance distribution, in which case no uncorrelated
    neuron of equal importance exists and the control is weaker than the
    condition it stands in for.
    """
    by_label = {r["condition"]: r for r in results}
    one = by_label.get("duplicate_one")
    matched = by_label.get("importance_matched")
    if one is None or matched is None:
        return {"supported": None, "summary": "conditions missing; cannot evaluate"}
    margin = matched["delta_perplexity"] - one["delta_perplexity"]
    supported = margin > 0
    one_imp = one.get("mean_importance") or 0.0
    matched_imp = matched.get("mean_importance") or 0.0
    match_ratio = matched_imp / one_imp if one_imp else float("nan")
    return {
        "supported": bool(supported),
        "delta_duplicate_one": one["delta_perplexity"],
        "delta_importance_matched": matched["delta_perplexity"],
        "margin": margin,
        "importance_match_ratio": match_ratio,
        "summary": (
            f"duplicate_one costs {one['delta_perplexity']:+.4f} PPL vs "
            f"{matched['delta_perplexity']:+.4f} for matched-importance controls "
            f"(margin {margin:+.4f}, control held {match_ratio:.2f}x the duplicates' "
            f"mean importance) - " + ("supported" if supported else "not supported")
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
