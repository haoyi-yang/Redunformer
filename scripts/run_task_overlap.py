#!/usr/bin/env python
"""Weeks 11-12: is redundancy task-specific? (hypothesis H3)

H3 says neurons that look redundant under WikiText-2 are *not* redundant under
PIQA. This script compares two measurement artifacts for the same model - one
calibrated on each corpus - and reports, per layer:

    jaccard_bottom  agreement on which neurons to mask (the pruning-relevant set)
    jaccard_top     agreement on the most active neurons (the proposal's wording)
    spearman        agreement across the whole importance ranking

Set overlap is judged against the agreement two *independent* random selections
of the same size would reach, since at a 10% budget that floor is already ~5%.

This is the correlational half of H3. The behavioural half - masking the
WikiText-redundant set and scoring PIQA - is run by ``run_pruning.py`` on each
artifact in turn; low overlap here predicts a gap there.

Example:
    python scripts/run_task_overlap.py --config configs/measurement/task_overlap_qwen3_0.6b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy import plotting  # noqa: E402
from redundancy.config import REPO_ROOT as ROOT, load_yaml  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    compare_artifacts,
    expected_random_jaccard,
    load_measurement,
    overlap_summary,
)


def _resolve(spec: str) -> Path:
    path = Path(spec)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-task neuron overlap (H3).")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML with reference/comparison artifact paths and ratios.",
    )
    parser.add_argument("--reference", default=None, help="Measurement JSON for corpus A.")
    parser.add_argument("--comparison", default=None, help="Measurement JSON for corpus B.")
    parser.add_argument(
        "--reference-label", default=None, help="Display name for corpus A (e.g. wikitext2)."
    )
    parser.add_argument(
        "--comparison-label", default=None, help="Display name for corpus B (e.g. piqa)."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="Removal ratios at which to compare the selected sets.",
    )
    parser.add_argument("--strategy", default=None, help="Ranking metric to compare.")
    parser.add_argument("--tag", default=None, help="Suffix for the output filenames.")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config) if args.config else {}
    ref_spec = args.reference or cfg.get("reference")
    cmp_spec = args.comparison or cfg.get("comparison")
    if not ref_spec or not cmp_spec:
        parser.error("Both --reference and --comparison measurement artifacts are required.")

    ref_label = args.reference_label or cfg.get("reference_label", "reference")
    cmp_label = args.comparison_label or cfg.get("comparison_label", "comparison")
    ratios = [float(r) for r in (args.ratios or cfg.get("ratios", [0.05, 0.1, 0.25]))]
    strategy = args.strategy or cfg.get("strategy", "importance")

    output_dir = ROOT / cfg.get("output_dir", "experiments/results")
    figures_dir = ROOT / cfg.get("figures_dir", "experiments/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference = load_measurement(_resolve(ref_spec))
    comparison = load_measurement(_resolve(cmp_spec))
    if reference.model_id != comparison.model_id:
        print(
            f"Warning: comparing artifacts from different models "
            f"({reference.model_id} vs {comparison.model_id}); overlap is not meaningful."
        )

    print(f"Comparing {ref_label} vs {cmp_label} on {reference.model_id} ({strategy} ranking)")

    per_ratio: dict[str, dict] = {}
    for ratio in ratios:
        overlaps = compare_artifacts(reference, comparison, ratio=ratio, strategy=strategy)
        if not overlaps:
            print("No layers in common between the two artifacts.")
            return 1
        summary = overlap_summary(overlaps)
        per_ratio[f"{ratio}"] = {
            "ratio": ratio,
            "summary": summary,
            "layers": [o.summary() for o in overlaps],
        }
        print(
            f"\n{ratio:.0%} budget — mean Jaccard(bottom) {summary['mean_jaccard_bottom']:.4f} "
            f"vs chance {summary['mean_chance_jaccard']:.4f} "
            f"({summary['bottom_over_chance']:.2f}x), "
            f"mean Spearman {summary['mean_spearman']:+.4f}"
        )
        print(f"  {'layer':>6} {'width':>7} {'bottom':>9} {'top':>9} {'spearman':>10}")
        for o in overlaps:
            print(
                f"  {o.layer_index:>6} {o.width:>7} {o.jaccard_bottom:>9.4f} "
                f"{o.jaccard_top:>9.4f} {o.spearman:>+10.4f}"
            )

    # Depth curves at the mid ratio, which is the one the sweep masks at.
    focus = ratios[len(ratios) // 2]
    focus_overlaps = compare_artifacts(reference, comparison, ratio=focus, strategy=strategy)
    depth = [o.layer_index for o in focus_overlaps]
    chance = float(
        np.mean([expected_random_jaccard(o.width, o.count) for o in focus_overlaps])
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = reference.model_id.replace("/", "__")
    suffix = f"_{args.tag}" if args.tag else ""

    fig_path = plotting.plot_overlap_by_depth(
        depth,
        {
            "masked set (bottom)": [o.jaccard_bottom for o in focus_overlaps],
            "most active (top)": [o.jaccard_top for o in focus_overlaps],
        },
        figures_dir / f"task_overlap{suffix}_{safe_model}_{stamp}.png",
        title=f"{ref_label} vs {cmp_label} neuron agreement at {focus:.0%} — {reference.model_id}",
        chance=chance,
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_11_12_cross_task_overlap",
        "hypothesis": "H3: neurons that look redundant under WikiText-2 are not "
        "redundant under PIQA",
        "command": " ".join(sys.argv),
        "model_id": reference.model_id,
        "strategy": strategy,
        "artifacts_compared": {
            ref_label: str(_resolve(ref_spec)),
            cmp_label: str(_resolve(cmp_spec)),
        },
        "calibration": {
            ref_label: reference.summary.get("dataset"),
            cmp_label: comparison.summary.get("dataset"),
        },
        "ratios": ratios,
        "by_ratio": per_ratio,
        "artifacts": {"figures": {"task_overlap": str(fig_path)}},
    }
    out_path = output_dir / f"task_overlap{suffix}_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote summary -> {out_path}")
    print(f"Wrote figure  -> {fig_path}")
    mid = per_ratio[f"{focus}"]["summary"]
    agreement = mid["mean_jaccard_bottom"]
    print(
        f"\nH3 evidence at {focus:.0%}: the two corpora agree on {agreement:.1%} of the "
        f"masked set ({mid['bottom_over_chance']:.2f}x chance), and disagree on "
        f"{1 - agreement:.1%}; mean Spearman {mid['mean_spearman']:+.3f}."
    )
    print(
        "  Well above chance, so redundancy is largely a property of the model "
        "rather than the corpus"
        if mid["bottom_over_chance"] > 2
        else "  Near chance, so the redundant set is essentially corpus-specific"
    )
    print(
        f"  — but {1 - agreement:.0%} of the set still differs, which is what a "
        "behavioural test has to settle: mask by each ranking and score the task."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
