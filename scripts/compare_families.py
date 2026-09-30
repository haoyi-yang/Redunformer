#!/usr/bin/env python
"""Does FFN neuron redundancy generalize beyond one model family?

Everything measured so far comes from Qwen, so "LLMs carry redundant FFN
neurons" is not separable from "Qwen carries redundant FFN neurons". This
script joins the per-model measurement and masking artifacts into one table and
one overlaid figure, over two contrasts:

    architecture  GPT-2 has a plain GELU FFN with no gate, Qwen and Llama-2 use
                  SwiGLU, so a shared profile cannot be a gating artifact.
    size          GPT-2 small vs large, and Qwen 0.6B vs 4B, give two
                  within-family size axes that do not confound size with
                  architecture.

Absolute perplexity is not comparable across families, so the curves are each
model's perplexity divided by its own unmasked baseline. The headline number is
the *gap* between the importance-guided and random curves at matched sparsity:
that is what says a ranking found real redundancy rather than just slack.

Example:
    python scripts/compare_families.py --model gpt2 --model Qwen/Qwen3-0.6B \
        --out reports/group_5/cross_family_week13-14.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy import plotting  # noqa: E402
from redundancy.config import REPO_ROOT as ROOT  # noqa: E402

# Ratios every model in the table is expected to share.
DEFAULT_RATIOS = (0.05, 0.10, 0.25, 0.50)


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _latest(results_dir: Path, prefix: str, model_id: str) -> Path | None:
    """Newest untagged ``<prefix>_<model>_*.json``.

    Tagged artifacts (``pruning_piqaranked_...``) belong to other experiments
    and must not be picked up here, so the model slug has to follow the prefix
    immediately.
    """
    matches = sorted(results_dir.glob(f"{prefix}_{_slug(model_id)}_*.json"))
    return matches[-1] if matches else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def measurement_row(doc: dict) -> dict:
    """Redundancy profile of one model, from its measurement artifact."""
    layers = doc["layers"]
    widths = [int(layer["width"]) for layer in layers]
    lazy = np.array([layer["lazy_fraction_le_1pct"] for layer in layers], dtype=float)
    bottom10 = np.array([layer["importance_share_bottom_10pct"] for layer in layers], dtype=float)
    bottom25 = np.array([layer["importance_share_bottom_25pct"] for layer in layers], dtype=float)

    pairs = [
        abs(entry["correlation"])
        for group in doc.get("correlation", {}).get("top_pairs", {}).values()
        for entry in group
    ]
    return {
        "layers": len(layers),
        "width": widths[0] if widths else 0,
        "neurons": int(sum(widths)),
        "col_norms": all(layer.get("col_norm_available") for layer in layers),
        "lazy": float(lazy.mean()),
        "bottom10": float(bottom10.mean()),
        "bottom25": float(bottom25.mean()),
        "max_corr": float(max(pairs)) if pairs else float("nan"),
        "dup_pairs": int(sum(1 for value in pairs if value >= 0.9)),
        "scanned_pairs": len(pairs),
    }


def pruning_row(doc: dict, ratios) -> dict:
    """Masking cost of one model, normalized by its own unmasked baseline."""
    baseline = float(doc["baseline"]["perplexity"])
    base_acc = doc["baseline"].get("lm_eval", {}).get("piqa.acc")

    curves: dict[str, dict[float, float]] = {}
    accs: dict[str, dict[float, float]] = {}
    for run in doc["runs"]:
        strategy = run["strategy"]
        curves.setdefault(strategy, {})[float(run["ratio"])] = float(run["perplexity"])
        acc = run.get("lm_eval", {}).get("piqa.acc")
        if acc is not None:
            accs.setdefault(strategy, {})[float(run["ratio"])] = float(acc)

    relative = {
        strategy: [points.get(r, float("nan")) / baseline for r in ratios]
        for strategy, points in curves.items()
    }
    return {
        "baseline_ppl": baseline,
        "baseline_acc": base_acc,
        "curves": curves,
        "relative": relative,
        "accs": accs,
    }


def pair_row(doc: dict) -> dict:
    """Duplicate-pair ablation (H5) for one model.

    Two things are worth reading separately. ``second_twin`` vs ``first_twin``
    is the within-pair evidence: if the survivor really absorbs its partner's
    job, removing the second twin must cost more than removing the first did.
    ``control_coverage`` is the health check on the comparison H5 is actually
    scored against - if the matched-importance control could not reach the
    duplicates' importance, the verdict is measuring the mismatch, not H5.
    """
    conditions = {c["condition"]: c for c in doc["conditions"]}
    one = conditions.get("duplicate_one")
    both = conditions.get("duplicate_both")
    matched = conditions.get("importance_matched")

    first = one["delta_perplexity"] if one else float("nan")
    second = (both["delta_perplexity"] - first) if (both and one) else float("nan")
    coverage = (
        matched["mean_importance"] / one["mean_importance"]
        if one and matched and one["mean_importance"]
        else float("nan")
    )
    verdict = doc.get("verdict")
    if isinstance(verdict, dict):
        verdict = verdict.get("supported")
    return {
        "baseline_ppl": float(doc["baseline_perplexity"]),
        "num_pairs": int(doc["pairs"]["num_pairs"]),
        "first_twin": first,
        "second_twin": second,
        "matched_control": matched["delta_perplexity"] if matched else float("nan"),
        "control_coverage": coverage,
        "verdict": verdict,
    }


def _gap(prune: dict, ratio: float) -> float:
    """How much cheaper importance-guided masking is than random, in relative PPL."""
    guided = prune["curves"].get("importance", {}).get(ratio)
    random_ = prune["curves"].get("random", {}).get(ratio)
    if guided is None or random_ is None:
        return float("nan")
    return (random_ - guided) / prune["baseline_ppl"]


def _fmt(value, spec: str = ".3f", dash: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return format(value, spec)


def build_markdown(rows: list[dict], ratios, figure: Path | None, out_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Cross-family neuron redundancy")
    lines.append("")
    lines.append(
        "Generated by `scripts/compare_families.py`. Every model uses the same "
        "calibration budget (128 x 512-token WikiText-2 blocks) and the same "
        "masking sweep, so the columns are comparable."
    )
    lines.append("")

    lines.append("## Redundancy profile (measurement)")
    lines.append("")
    lines.append(
        "| model | layers x width | neurons | lazy frac | bottom-10% importance | "
        "bottom-25% importance | max \\|r\\| | pairs >= 0.9 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        meas = row.get("measurement")
        if not meas:
            continue
        lines.append(
            f"| {row['label']} | {meas['layers']} x {meas['width']} | {meas['neurons']:,} | "
            f"{_fmt(meas['lazy'], '.4f')} | {_fmt(meas['bottom10'] * 100, '.2f')}% | "
            f"{_fmt(meas['bottom25'] * 100, '.2f')}% | {_fmt(meas['max_corr'])} | "
            f"{meas['dup_pairs']}/{meas['scanned_pairs']} |"
        )
    lines.append("")

    lines.append("## Masking cost (intervention)")
    lines.append("")
    lines.append(
        "`rel PPL` is perplexity divided by that model's own unmasked baseline. "
        "`gap` is the random-minus-importance difference at the same sparsity, "
        "in units of the baseline: larger means the ranking found more genuinely "
        "spare capacity."
    )
    lines.append("")
    header = "| model | baseline PPL | " + " | ".join(
        f"rel PPL @ {int(r * 100)}% | gap @ {int(r * 100)}%" for r in ratios
    ) + " |"
    lines.append(header)
    lines.append("| --- |" + " --- |" * (1 + 2 * len(ratios)))
    for row in rows:
        prune = row.get("pruning")
        if not prune:
            continue
        cells = [row["label"], _fmt(prune["baseline_ppl"], ".2f")]
        for ratio in ratios:
            guided = prune["curves"].get("importance", {}).get(ratio)
            rel = guided / prune["baseline_ppl"] if guided else float("nan")
            cells.append(_fmt(rel, ".3f"))
            cells.append(_fmt(_gap(prune, ratio), "+.3f"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if any(row.get("pruning", {}).get("accs") for row in rows):
        lines.append("## PIQA accuracy under masking")
        lines.append("")
        lines.append(
            "| model | baseline acc | "
            + " | ".join(f"importance @ {int(r * 100)}%" for r in ratios)
            + " |"
        )
        lines.append("| --- |" + " --- |" * (1 + len(ratios)))
        for row in rows:
            prune = row.get("pruning")
            if not prune or not prune["accs"]:
                continue
            cells = [row["label"], _fmt(prune["baseline_acc"], ".3f")]
            guided = prune["accs"].get("importance", {})
            cells += [_fmt(guided.get(r), ".3f") for r in ratios]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if any(row.get("pairs") for row in rows):
        lines.append("## Duplicate-pair ablation (H5)")
        lines.append("")
        lines.append(
            "`1st twin` is the cost of masking one neuron of each near-duplicate "
            "pair; `2nd twin` is the extra cost of then masking its partner. "
            "`2nd > 1st` is the direct evidence that the survivor was carrying "
            "the load. `control` is the matched-importance arm H5 is scored "
            "against, and `coverage` is how much of the duplicates' importance "
            "that control actually reached - well below 1.0 means the control is "
            "not really matched and the verdict should not be read as a refutation."
        )
        lines.append("")
        lines.append(
            "| model | pairs | 1st twin | 2nd twin | control | coverage | verdict |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            pair = row.get("pairs")
            if not pair:
                continue
            lines.append(
                f"| {row['label']} | {pair['num_pairs']} | "
                f"{_fmt(pair['first_twin'], '+.3f')} | {_fmt(pair['second_twin'], '+.3f')} | "
                f"{_fmt(pair['matched_control'], '+.3f')} | "
                f"{_fmt(pair['control_coverage'], '.2f')}x | {pair['verdict']} |"
            )
        lines.append("")

    if figure is not None:
        rel = os.path.relpath(figure, out_path.parent).replace(os.sep, "/")
        lines.append(f"![Cross-family removal curves]({rel})")
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    for row in rows:
        lines.append(f"- **{row['label']}** (`{row['model_id']}`)")
        for key in ("measurement_path", "pruning_path", "pair_path"):
            value = row.get(key)
            if value:
                lines.append(f"  - `{Path(value).name}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-family redundancy comparison.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model id to include; repeatable. Artifacts are auto-resolved by id.",
    )
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out", default="reports/group_5/cross_family.md")
    parser.add_argument("--figure", default="experiments/results/figures/cross_family_removal.png")
    parser.add_argument(
        "--ratios",
        default=",".join(str(r) for r in DEFAULT_RATIOS),
        help="Comma-separated removal ratios to tabulate.",
    )
    parser.add_argument("--logy", action="store_true", help="Log-scale the figure's y axis.")
    args = parser.parse_args(argv)

    results_dir = ROOT / args.results_dir
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]

    rows: list[dict] = []
    for model_id in args.model:
        row: dict = {"model_id": model_id, "label": model_id.split("/")[-1]}
        meas_path = _latest(results_dir, "measurement", model_id)
        prune_path = _latest(results_dir, "pruning", model_id)
        if meas_path is None and prune_path is None:
            print(f"  !! no artifacts found for {model_id}, skipping")
            continue
        if meas_path is not None:
            row["measurement"] = measurement_row(_load(meas_path))
            row["measurement_path"] = str(meas_path)
        if prune_path is not None:
            row["pruning"] = pruning_row(_load(prune_path), ratios)
            row["pruning_path"] = str(prune_path)
        pair_path = _latest(results_dir, "pair_ablation", model_id)
        if pair_path is not None:
            row["pairs"] = pair_row(_load(pair_path))
            row["pair_path"] = str(pair_path)
        rows.append(row)
        print(
            f"  {row['label']:<16} measurement={'yes' if meas_path else 'MISSING':<7} "
            f"pruning={'yes' if prune_path else 'MISSING'}"
        )

    if not rows:
        print("No artifacts matched; nothing to compare.")
        return 1

    figure_path = None
    series = {
        row["label"]: row["pruning"]["relative"] for row in rows if row.get("pruning")
    }
    if series:
        figure_path = ROOT / args.figure
        plotting.plot_family_removal_curves(
            ratios, series, figure_path, logy=args.logy
        )
        print(f"Wrote figure  -> {figure_path}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_markdown(rows, ratios, figure_path, out_path), encoding="utf-8"
    )
    print(f"Wrote table   -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
