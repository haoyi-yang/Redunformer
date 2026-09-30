"""Dump H1 depth-tertile stats from existing measurement artifacts (no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.config import REPO_ROOT as ROOT  # noqa: E402


def _latest(pattern: str) -> Path | None:
    matches = sorted((ROOT / "experiments" / "results").glob(pattern))
    return matches[-1] if matches else None


DEFAULT_ARTIFACTS = [
    ("gpt2", "measurement_gpt2_*.json"),
    ("gpt2-medium", "measurement_gpt2-medium_*.json"),
    ("gpt2-large", "measurement_gpt2-large_*.json"),
    ("Llama-2-7B", "measurement_meta-llama__Llama-2-7b-hf_*.json"),
    ("Qwen3-0.6B", "measurement_Qwen__Qwen3-0.6B_*.json"),
    ("Qwen3.5-4B", "measurement_Qwen__Qwen3.5-4B_*.json"),
]

# Slug used in the pruning_h1<tertile>_<slug>_<stamp>.json filenames.
ABLATION_SLUGS = [
    ("gpt2", "gpt2"),
    ("gpt2-medium", "gpt2-medium"),
    ("gpt2-large", "gpt2-large"),
    ("Llama-2-7B", "meta-llama__Llama-2-7b-hf"),
    ("Qwen3-0.6B", "Qwen__Qwen3-0.6B"),
    ("Qwen3.5-4B", "Qwen__Qwen3.5-4B"),
]


def _tertile_delta(slug: str, tertile: str) -> float | None:
    """ΔPPL of masking 10% of the lowest-importance neurons in one tertile."""
    path = _latest(f"pruning_h1{tertile}_{slug}_*.json")
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for run in payload.get("runs") or []:
        if run.get("strategy") == "importance":
            delta = run.get("delta_perplexity")
            if delta is None and run.get("perplexity") is not None:
                base = (payload.get("baseline") or {}).get("perplexity")
                if base is not None:
                    delta = run["perplexity"] - base
            return delta
    return None


def _ablation_section() -> list[str]:
    lines = [
        "",
        "## Ablation (10% lowest-importance neurons, tertile layers only)",
        "",
        "The static share above is only a *proxy*. This section masks 10% of the",
        "neurons in each tertile and measures the actual cost, which is what H1",
        "is really about. H1 predicted the **middle** tertile would hurt most.",
        "",
        "| model | early ΔPPL | middle ΔPPL | deep ΔPPL | most expensive tertile |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, slug in ABLATION_SLUGS:
        deltas = {t: _tertile_delta(slug, t) for t in ("early", "middle", "deep")}
        cells = []
        for tertile in ("early", "middle", "deep"):
            value = deltas[tertile]
            cells.append(f"{value:+.2f}" if isinstance(value, (int, float)) else "—")
        known = {k: v for k, v in deltas.items() if isinstance(v, (int, float))}
        if len(known) == 3:
            worst = max(known, key=lambda k: known[k])
            spread = max(known.values()) - min(known.values())
            # A spread this small is inside run-to-run noise on these slices.
            verdict = "flat" if spread < 0.10 else worst
        else:
            verdict = "—"
        lines.append(f"| {label} | {' | '.join(cells)} | {verdict} |")
    lines += [
        "",
        "Eval slice matches each model's own masking config, so ΔPPL is comparable",
        "within a row but not across rows (baselines differ by up to 5×).",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise H1 depth tertiles from measurement JSON.")
    parser.add_argument(
        "--out",
        default="reports/group_5/artifacts/h1_depth_tertiles.md",
        help="Markdown path relative to the repo root.",
    )
    args = parser.parse_args(argv)

    lines = [
        "# H1 — depth tertiles (from existing measurement artifacts)",
        "",
        "H1 predicted that **middle** FFN layers are *less* redundant than early/late.",
        "The measurement proxy is the importance share of the bottom 10% of neurons:",
        "a **smaller** share means those neurons are cheaper to drop (more redundant).",
        "So H1 is supported if the middle tertile's bottom-10% share is **larger** than",
        "early and late (middle less redundant).",
        "",
        "| model | tertile | layers | bottom-10% importance share | mean firing freq |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for label, pattern in DEFAULT_ARTIFACTS:
        path = _latest(pattern)
        if path is None:
            lines.append(f"| {label} | — | missing `{pattern}` | | |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tertiles = payload.get("depth_tertiles") or {}
        for name in ("early", "middle", "deep"):
            block = tertiles.get(name) or {}
            layers = block.get("layers") or []
            share = block.get("mean_importance_share_bottom_10pct")
            freq = block.get("mean_freq_rms")
            layer_s = ",".join(str(i) for i in layers) if layers else "—"
            share_s = f"{100 * share:.2f}%" if isinstance(share, (int, float)) else "—"
            freq_s = f"{freq:.4f}" if isinstance(freq, (int, float)) else "—"
            lines.append(f"| {label} | {name} | {layer_s} | {share_s} | {freq_s} |")
        src = path.name
        lines.append(f"| {label} | *source* | `{src}` | | |")

    lines.extend(_ablation_section())

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
