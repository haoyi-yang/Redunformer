#!/usr/bin/env python
"""Weeks 5–8: measure neuron / FFN-unit redundancy.

Captures post-SwiGLU FFN activations on a calibration set and computes, per
neuron: activation magnitude (mean |a|, RMS), RMS-relative firing frequency at
several thresholds, activation-aware **importance** (``RMS x ||W_down[:, i]||``,
the expected contribution to the residual), and within-layer correlation
(duplicate detection). Saves a JSON summary, a per-neuron ``.npz`` for the later
masking/pruning stage, and plots.

Why importance: post-SwiGLU activations rarely sit near zero, so firing
frequency saturates near 1.0 and is a poor redundancy signal for these models.
Importance is heavy-tailed and gives a usable ranking for ablation.

Example:
    python scripts/run_measurement.py --config configs/measurement/neuron_activations.yaml
    python scripts/run_measurement.py \
        --config configs/measurement/neuron_activations_smoke.yaml --use-fallback-model
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.config import REPO_ROOT as ROOT, load_yaml  # noqa: E402
from redundancy.data import build_calibration_blocks, load_text_dataset  # noqa: E402
from redundancy.hooks import NeuronActivationCollector  # noqa: E402
from redundancy.metrics import (  # noqa: E402
    build_correlation_neighborhood,
    dense_correlation_matrix,
    duplication_score,
    NeuronActivationStats,
    top_k_correlated_pairs,
)
from redundancy.models import load_causal_lm  # noqa: E402
from redundancy import plotting  # noqa: E402


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_corr_layers(results, stats, spec, top_n: int, rank_max_neurons: int) -> list[int]:
    """Resolve the correlation-layer spec to a list of layer indices."""
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if spec == "tertiles":
        idxs = sorted(r.layer_index for r in results)
        if not idxs:
            return []
        return sorted({idxs[0], idxs[len(idxs) // 2], idxs[-1]})
    if spec == "auto":
        # Rank by per-layer duplication score (mean of top per-neuron max|corr|),
        # which is meaningful even when firing frequency is saturated.
        scored: list[tuple[float, int]] = []
        for r in tqdm(results, desc="dup-score", unit="layer"):
            res_v = stats.reservoir(r.name)
            if res_v is None or res_v.shape[0] < 3:
                continue
            scored.append((duplication_score(res_v, max_neurons=rank_max_neurons), r.layer_index))
        scored.sort(reverse=True)
        return sorted(idx for _, idx in scored[:top_n])
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure neuron/FFN redundancy (Weeks 5-8).")
    parser.add_argument(
        "--config",
        default="configs/measurement/neuron_activations.yaml",
        help="Path to measurement config (YAML).",
    )
    parser.add_argument("--max-blocks", type=int, default=None, help="Override calibration blocks.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override calibration block length.")
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Label inserted into the output filenames (also settable as output_tag "
            "in the config). Required for non-default calibration corpora: a tagged "
            "artifact is excluded from measurement_path: 'auto' resolution, so a "
            "PIQA-calibrated ranking is never silently used for a WikiText run."
        ),
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

    calib_cfg = cfg.get("calibration", {})
    seq_len = int(args.seq_len or calib_cfg.get("seq_len", 512))
    max_blocks = args.max_blocks if args.max_blocks is not None else calib_cfg.get("max_blocks", 256)

    meas_cfg = cfg.get("measurement", {})
    corr_cfg = cfg.get("correlation", {})

    output_dir = ROOT / cfg.get("output_dir", "experiments/results")
    figures_dir = ROOT / cfg.get("figures_dir", "experiments/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    use_fallback = bool(model_cfg.get("_use_fallback"))
    display_id = model_cfg.get("fallback_model_id") if use_fallback else model_cfg.get("model_id")
    print(f"Loading model from {display_id} ...")
    loaded = load_causal_lm(model_cfg)
    model = loaded.model
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    print("Loading calibration dataset ...")
    texts, dataset_meta = load_text_dataset(dataset_cfg)
    blocks = build_calibration_blocks(
        texts,
        loaded.tokenizer,
        seq_len=seq_len,
        max_blocks=max_blocks,
        add_special_tokens=bool(calib_cfg.get("add_special_tokens", False)),
    )
    print(f"Built {len(blocks)} calibration blocks of {seq_len} tokens.")

    stats = NeuronActivationStats(
        fixed_eps=float(meas_cfg.get("fixed_eps", 1e-3)),
        rms_alpha=float(meas_cfg.get("rms_alpha", 1e-3)),
        rms_alphas=meas_cfg.get("rms_alphas"),
        max_reservoir_tokens=int(meas_cfg.get("max_reservoir_tokens", 4096)),
        seed=seed,
    )

    print("Capturing FFN activations ...")
    with NeuronActivationCollector(model, stats) as collector, torch.inference_mode():
        print(f"Hooked {len(collector.layers)} FFN down-projections.")
        for block in tqdm(blocks, desc="calibration", unit="block"):
            model(block.to(device))
        print("Reading down-projection weight column norms ...")
        col_norms = collector.input_weight_norms()

    results = stats.finalize()
    for r in results:
        r.set_weight_col_norm(col_norms.get(r.name))
    col_norm_available = any(r.col_norm is not None for r in results)
    by_index = {r.layer_index: r for r in results}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag if args.tag is not None else cfg.get("output_tag")
    # The tag goes *before* the model id so tagged runs fall outside the
    # "measurement_<model>_*.json" glob that `measurement_path: auto` uses.
    safe_model = f"{tag}_{loaded.model_id.replace('/', '__')}" if tag else loaded.model_id.replace("/", "__")

    # ---- Correlation on selected layers ----------------------------------------
    corr_results: dict[str, list] = {}
    corr_layers: list[int] = []
    corr_fig_paths: dict[str, str] = {}
    if corr_cfg.get("enabled", True):
        corr_layers = _select_corr_layers(
            results,
            stats,
            corr_cfg.get("layers", "auto"),
            int(corr_cfg.get("top_n", 3)),
            int(corr_cfg.get("rank_max_neurons", 512)),
        )
        for layer_index in corr_layers:
            res = by_index.get(layer_index)
            if res is None:
                continue
            reservoir = stats.reservoir(res.name)
            if reservoir is None or reservoir.shape[0] < 3:
                continue
            pairs = top_k_correlated_pairs(
                reservoir,
                k=int(corr_cfg.get("k", 50)),
                max_neurons=int(corr_cfg.get("max_neurons", 2048)),
                abs_threshold=corr_cfg.get("abs_threshold"),
            )
            corr_results[str(layer_index)] = [
                {"neuron_i": p.neuron_i, "neuron_j": p.neuron_j, "correlation": p.correlation}
                for p in pairs
            ]
            if corr_cfg.get("save_heatmaps", True):
                neighborhood = build_correlation_neighborhood(
                    reservoir, pairs, size=int(corr_cfg.get("heatmap_neurons", 40))
                )
                if len(neighborhood) >= 2:
                    corr = dense_correlation_matrix(reservoir, neighborhood)
                    corr_path = figures_dir / f"corr_layer{layer_index}_{safe_model}_{stamp}.png"
                    plotting.plot_correlation_heatmap(
                        corr,
                        corr_path,
                        title=f"Neuron correlation neighborhood — layer {layer_index}",
                    )
                    corr_fig_paths[f"corr_layer{layer_index}"] = str(corr_path)

    # ---- Plots ------------------------------------------------------------------
    layer_idx_sorted = sorted(by_index)

    freqs_by_layer = {r.layer_index: r.freq_rms for r in results}
    importance_by_layer = {r.layer_index: r.importance_vector for r in results}

    fig_paths = {}
    fig_paths["frequency_histograms"] = str(
        plotting.plot_frequency_histograms(
            freqs_by_layer, figures_dir / f"freq_hist_{safe_model}_{stamp}.png"
        )
    )
    fig_paths["freq_distribution_heatmap"] = str(
        plotting.plot_freq_distribution_heatmap(
            freqs_by_layer, figures_dir / f"freq_depth_{safe_model}_{stamp}.png"
        )
    )
    sweep = {
        alpha: [by_index[i].lazy_fraction_at_alpha(alpha, 0.01) for i in layer_idx_sorted]
        for alpha in stats.rms_alphas
    }
    fig_paths["lazy_fraction_sweep"] = str(
        plotting.plot_lazy_fraction_sweep(
            layer_idx_sorted, sweep, figures_dir / f"lazy_sweep_{safe_model}_{stamp}.png"
        )
    )
    fig_paths["importance_histograms"] = str(
        plotting.plot_importance_histograms(
            importance_by_layer, figures_dir / f"imp_hist_{safe_model}_{stamp}.png"
        )
    )
    fig_paths["importance_share_by_depth"] = str(
        plotting.plot_metric_by_depth(
            layer_idx_sorted,
            [by_index[i].importance_share_bottom(0.10) for i in layer_idx_sorted],
            figures_dir / f"imp_depth_{safe_model}_{stamp}.png",
            title="Removability proxy: importance share of bottom-10% neurons vs depth",
            ylabel="importance share of bottom 10%",
            color="#dd8452",
        )
    )

    # ---- Per-neuron arrays (.npz) for the later masking/pruning stage ----------
    npz_path = output_dir / f"measurement_{safe_model}_{stamp}.npz"
    np.savez_compressed(
        npz_path,
        layer_index=np.array([r.layer_index for r in results]),
        names=np.array([r.name for r in results]),
        freq_rms=_stack_ragged([r.freq_rms for r in results]),
        freq_fixed=_stack_ragged([r.freq_fixed for r in results]),
        mean_abs=_stack_ragged([r.mean_abs for r in results]),
        rms=_stack_ragged([r.rms for r in results]),
        importance=_stack_ragged([r.importance_vector for r in results]),
        col_norm=_stack_ragged(
            [r.col_norm if r.col_norm is not None else np.full(r.width, np.nan) for r in results]
        ),
    )

    # ---- Headline JSON ----------------------------------------------------------
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_5_8_neuron_measurement",
        "seed": seed,
        "config_path": str(args.config),
        "command": " ".join(sys.argv),
        "model": {
            "model_id": loaded.model_id,
            "load_in_4bit": loaded.load_in_4bit,
            "config": cfg["model_config"],
            "num_ffn_layers": len(results),
        },
        "dataset": dataset_meta,
        "calibration": {
            "seq_len": seq_len,
            "num_blocks": len(blocks),
            "num_tokens": int(results[0].num_tokens) if results else 0,
        },
        "measurement": {
            "fixed_eps": stats.fixed_eps,
            "rms_alpha": stats.rms_alpha,
            "rms_alphas": stats.rms_alphas,
            "max_reservoir_tokens": stats.max_reservoir_tokens,
            "importance_definition": "rms * down_proj_input_column_norm (Wanda/LLM-Pruner style)",
            "col_norm_available": col_norm_available,
        },
        "layers": [r.summary() for r in results],
        "depth_tertiles": _depth_tertiles(results),
        "correlation": {
            "selection": corr_cfg.get("layers", "auto"),
            "layers": corr_layers,
            "abs_threshold": corr_cfg.get("abs_threshold"),
            "top_pairs": corr_results,
        },
        "artifacts": {"npz": str(npz_path), "figures": {**fig_paths, **corr_fig_paths}},
    }
    out_path = output_dir / f"measurement_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote summary  -> {out_path}")
    print(f"Wrote arrays   -> {npz_path}")
    for label, p in fig_paths.items():
        print(f"Wrote figure   -> {p}  ({label})")
    _print_console_summary(results, col_norm_available)
    return 0


def _stack_ragged(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack equal-width arrays, or return an object array if widths differ."""
    if arrays and all(a.shape == arrays[0].shape for a in arrays):
        return np.stack(arrays)
    return np.array(arrays, dtype=object)


def _depth_tertiles(results) -> dict:
    if not results:
        return {}
    ordered = sorted(results, key=lambda r: r.layer_index)
    n = len(ordered)
    third = max(n // 3, 1)
    groups = {
        "early": ordered[:third],
        "middle": ordered[third : 2 * third] or ordered[third:],
        "deep": ordered[2 * third :] or ordered[-third:],
    }
    return {
        name: {
            "layers": [r.layer_index for r in grp],
            "mean_freq_rms": float(np.mean([r.freq_rms.mean() for r in grp])),
            "mean_importance_share_bottom_10pct": float(
                np.mean([r.importance_share_bottom(0.10) for r in grp])
            ),
            "mean_lazy_fraction_le_1pct": float(np.mean([r.lazy_fraction(0.01) for r in grp])),
        }
        for name, grp in groups.items()
        if grp
    }


def _print_console_summary(results, col_norm_available: bool) -> None:
    src = "RMS x weight-col-norm" if col_norm_available else "RMS only (weights unavailable)"
    print(f"\nPer-layer redundancy (importance = {src}):")
    print(f"  {'layer':>5} {'width':>6} {'bot10% share':>12} {'lazy@a=0.1':>10}")
    for r in sorted(results, key=lambda r: r.layer_index):
        share = r.importance_share_bottom(0.10)
        lazy = r.lazy_fraction_at_alpha(0.1, 0.01)
        bar = "#" * int(round(share * 100))
        print(f"  L{r.layer_index:>4} {r.width:>6} {share:>11.2%} {lazy:>10.2%} {bar}")


if __name__ == "__main__":
    raise SystemExit(main())
