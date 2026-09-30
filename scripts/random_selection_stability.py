#!/usr/bin/env python
"""#18: how much does the matched-sparsity random baseline depend on its seed?

Every ``random`` column in Weeks 9-14 comes from a single seeded draw. This
script resamples that draw offline - it reads only the measurement ``.npz`` and
calls ``select_neurons``, so there is no model, no tokenizer, and no forward
pass. It runs on a laptop CPU in seconds.

What it reports per ratio, over N seeds:

    importance removed   mean / sd / range / 95% bootstrap CI of the fraction of
                         layer importance the random draw removes
    guided arm           the same quantity for the importance ranking, and how
                         many seed standard deviations separate the two
    pairwise Jaccard     agreement between two draws vs the analytic chance
                         floor - a self-check that the seeding is independent

This bounds the *selection* half of the random baseline's variance only. Turning
removed importance into a perplexity interval still needs forward passes, so do
not quote a CI on any PPL number from this output. See
``reports/group_5/random_selection_stability.md``.

Example:
    python scripts/random_selection_stability.py \\
        --measurement experiments/results/measurement_Qwen__Qwen3-0.6B_20260629T182308Z.json \\
        --ratios 0.05 0.10 0.25 0.50 --seeds 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.config import REPO_ROOT as ROOT  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    find_latest_measurement,
    load_measurement,
    random_selection_stability,
    resolve_layer_filter,
    stability_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed stability of the random masking baseline (offline, CPU-only)."
    )
    parser.add_argument(
        "--measurement",
        default="auto",
        help="measurement_*.json to resample; 'auto' picks the newest in --results-dir.",
    )
    parser.add_argument("--model-id", default=None, help="Restrict 'auto' to one model id.")
    parser.add_argument(
        "--results-dir", default="experiments/results", help="Where artifacts live."
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50],
        help="Removal ratios to resample.",
    )
    parser.add_argument("--seeds", type=int, default=20, help="Number of random draws per ratio.")
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="First seed; the production runs used 42, and it is included as draw 1.",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Restrict to these layer indices (default: every measured layer).",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="Bootstrap resamples for the CI."
    )
    parser.add_argument("--tag", default=None, help="Suffix for the output filename.")
    parser.add_argument(
        "--no-write", action="store_true", help="Print the table without writing JSON."
    )
    args = parser.parse_args(argv)

    results_dir = ROOT / args.results_dir
    if args.measurement == "auto":
        measurement_path = find_latest_measurement(results_dir, args.model_id)
    else:
        measurement_path = Path(args.measurement)
        if not measurement_path.is_absolute():
            measurement_path = ROOT / measurement_path

    artifact = load_measurement(measurement_path)
    layer_filter = resolve_layer_filter(artifact, args.layers)
    scope = layer_filter if layer_filter is not None else artifact.layer_names

    print(f"Model        : {artifact.model_id}")
    print(f"Measurement  : {measurement_path.name}")
    print(f"Calibration  : {artifact.calibration_label()}")
    print(f"Layers       : {len(scope)} ({artifact.total_neurons(layer_filter):,} neurons)")
    print(f"Seeds        : {args.seeds} from base {args.base_seed}\n")

    results = [
        random_selection_stability(
            artifact,
            ratio,
            num_seeds=args.seeds,
            base_seed=args.base_seed,
            layer_filter=layer_filter,
            num_resamples=args.bootstrap,
        )
        for ratio in args.ratios
    ]

    header = (
        f"{'ratio':>6} {'masked':>9} {'imp.removed':>12} {'sd':>9} {'range':>9} "
        f"{'95% CI':>19} {'guided':>9} {'gap (sd)':>9} {'jaccard/chance':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        lo, hi = r.bootstrap_ci
        print(
            f"{r.ratio:>6.0%} {r.num_masked:>9,} {r.mean:>12.5f} {r.std:>9.5f} "
            f"{r.spread:>9.5f} {f'[{lo:.5f}, {hi:.5f}]':>19} "
            f"{r.guided_importance_removed:>9.5f} {r.guided_gap_in_sds:>9.1f} "
            f"{f'{r.mean_pairwise_jaccard:.4f}/{r.chance_jaccard:.4f}':>15}"
        )

    summary = stability_summary(results)
    print()
    if summary["guided_outside_seed_noise"]:
        print(
            "Verdict: at every ratio the guided arm removes less importance than any "
            "plausible random draw\n         (closest: "
            f"{summary['min_guided_gap_in_seed_sds']:.1f} sd at "
            f"{summary['min_gap_at_ratio']:.0%}), so the single-seed random baselines "
            "in the\n         committed reports are safe as *selection* controls."
        )
    else:
        print(
            "Verdict: at "
            f"{summary['min_gap_at_ratio']:.0%} the guided arm is only "
            f"{summary['min_guided_gap_in_seed_sds']:.1f} sd from the random mean, so that "
            "ratio's\n         single-seed baseline cannot carry a claim - re-run it with "
            "several seeds."
        )
    print(
        "Note   : this bounds which neurons get picked, not the perplexity that "
        "follows.\n         No PPL confidence interval can be read off this table."
    )

    if args.no_write:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = artifact.model_id.replace("/", "__")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = results_dir / f"random_stability{suffix}_{safe_model}_{stamp}.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "issue_18_random_baseline_seed_stability",
        "command": " ".join(sys.argv),
        "model_id": artifact.model_id,
        "measurement": str(measurement_path),
        "calibration": artifact.calibration_label(),
        "layers": list(scope),
        "num_seeds": args.seeds,
        "base_seed": args.base_seed,
        "bootstrap_resamples": args.bootstrap,
        "scope": "selection only; no model forward passes",
        "by_ratio": [r.summary() for r in results],
        "summary": summary,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
