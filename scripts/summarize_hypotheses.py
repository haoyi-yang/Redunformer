"""Generate the per-hypothesis tables for H3, H5/H5' and H8 (no GPU).

Reads only the result JSONs already on disk and writes one markdown table per
hypothesis under ``reports/group_5/artifacts/``. Re-run this after any new
experiment arm lands so the final report never quotes a stale number.

Closes #18.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from redundancy.config import REPO_ROOT as ROOT  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
ARTIFACTS = ROOT / "reports" / "group_5" / "artifacts"

# (label, filename slug, FFN design)
MODELS = [
    ("gpt2", "gpt2", "GELU"),
    ("gpt2-medium", "gpt2-medium", "GELU"),
    ("gpt2-large", "gpt2-large", "GELU"),
    ("Qwen3-0.6B", "Qwen__Qwen3-0.6B", "SwiGLU"),
    ("Qwen3.5-4B", "Qwen__Qwen3.5-4B", "SwiGLU"),
    ("Llama-2-7B", "meta-llama__Llama-2-7b-hf", "SwiGLU"),
]


def _latest(pattern: str) -> Path | None:
    matches = sorted(RESULTS.glob(pattern))
    return matches[-1] if matches else None


def _load(pattern: str) -> dict | None:
    path = _latest(pattern)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["__file__"] = path.name
    return payload


def _num(value: object, fmt: str = "{:.3f}", dash: str = "—") -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else dash


def _write(name: str, lines: list[str]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


# --------------------------------------------------------------------------
# H3 — task specificity
# --------------------------------------------------------------------------


def _prune_row(payload: dict, ratio: float, strategy: str) -> dict | None:
    for run in payload.get("runs") or []:
        if run.get("strategy") == strategy and abs(float(run.get("ratio", -1)) - ratio) < 1e-9:
            return run
    return None


def h3() -> None:
    lines = [
        "# H3 — is redundancy task-specific?",
        "",
        "H3 predicted that neurons which look redundant under WikiText-2 are **not**",
        "redundant under PIQA. Tested in two halves: do the two calibration corpora",
        "nominate the same neurons (set overlap), and does the disagreement change",
        "behaviour (mask by each ranking, score both metrics).",
        "",
        "## 3a. Set agreement between the two rankings",
        "",
        "Jaccard overlap of the masked set, against the chance floor for the same",
        "budget. A ratio near 1x would mean the two corpora pick unrelated neurons.",
        "",
        "| model | FFN | layers | budget | Jaccard | chance | x chance | Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, slug, ffn in MODELS:
        payload = None
        for path in sorted(RESULTS.glob("task_overlap_*.json")):
            if path.name.startswith(f"task_overlap_{slug}_"):
                payload = json.loads(path.read_text(encoding="utf-8"))
        if payload is None:
            lines.append(f"| {label} | {ffn} | — | — | not run | | | |")
            continue
        for ratio_key in sorted(payload.get("by_ratio") or {}, key=float):
            s = payload["by_ratio"][ratio_key]["summary"]
            lines.append(
                "| {model} | {ffn} | {layers} | {budget:.0f}% | {jac:.3f} | {chance:.3f} "
                "| {x:.2f}x | {sp:.3f} |".format(
                    model=label,
                    ffn=ffn,
                    layers=s.get("num_layers", "—"),
                    budget=100 * float(ratio_key),
                    jac=s["mean_jaccard_bottom"],
                    chance=s["mean_chance_jaccard"],
                    x=s["bottom_over_chance"],
                    sp=s["mean_spearman"],
                )
            )

    lines += [
        "",
        "## 3b. Does the disagreement change behaviour?",
        "",
        "Mask by each ranking at matched sparsity, then score both metrics. The",
        "`random` arm is ranking-independent, so it must come out identical in the",
        "two runs — that is the consistency check that the rows are comparable.",
        "",
        "H3 needs a **double dissociation**: the WikiText ranking should win on",
        "perplexity while the PIQA ranking wins on PIQA accuracy.",
        "",
        "| model | budget | ranking | WikiText-2 PPL | PIQA acc | PIQA acc_norm |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    verdicts: list[tuple[str, str, str]] = []
    for label, slug, ffn in MODELS:
        wiki = _load(f"pruning_{slug}_*.json")
        piqa = _load(f"pruning_piqaranked_{slug}_*.json")
        if wiki is None or piqa is None:
            lines.append(f"| {label} | — | not run | | | |")
            verdicts.append((label, ffn, "not run"))
            continue
        base = wiki.get("baseline") or {}
        base_eval = base.get("lm_eval") or {}
        lines.append(
            "| {m} | — | none (baseline) | {ppl} | {acc} | {accn} |".format(
                m=label,
                ppl=_num(base.get("perplexity"), "{:.2f}"),
                acc=_num(base_eval.get("piqa.acc")),
                accn=_num(base_eval.get("piqa.acc_norm")),
            )
        )
        for ratio in (0.10, 0.25):
            arms = [
                ("WikiText-2", _prune_row(wiki, ratio, "importance")),
                ("PIQA", _prune_row(piqa, ratio, "importance")),
                ("random", _prune_row(wiki, ratio, "random")),
            ]
            for name, run in arms:
                if run is None:
                    continue
                ev = run.get("lm_eval") or {}
                lines.append(
                    "| {m} | {b:.0f}% | {n} | {ppl} | {acc} | {accn} |".format(
                        m=label,
                        b=100 * ratio,
                        n=name,
                        ppl=_num(run.get("perplexity"), "{:.2f}"),
                        acc=_num(ev.get("piqa.acc")),
                        accn=_num(ev.get("piqa.acc_norm")),
                    )
                )
        # Verdict from the 25% arm, where the rankings diverge most.
        w25 = _prune_row(wiki, 0.25, "importance")
        p25 = _prune_row(piqa, 0.25, "importance")
        if w25 and p25:
            wiki_wins_ppl = w25["perplexity"] < p25["perplexity"]
            w_acc = (w25.get("lm_eval") or {}).get("piqa.acc")
            p_acc = (p25.get("lm_eval") or {}).get("piqa.acc")
            piqa_wins_acc = (
                isinstance(w_acc, (int, float))
                and isinstance(p_acc, (int, float))
                and p_acc > w_acc
            )
            if wiki_wins_ppl and piqa_wins_acc:
                gap = 100 * (p_acc - w_acc)
                verdicts.append((label, ffn, f"supported (double dissociation, +{gap:.1f} pp PIQA)"))
            elif wiki_wins_ppl:
                verdicts.append((label, ffn, "not supported (PIQA ranking loses on both metrics)"))
            else:
                verdicts.append((label, ffn, "unclear"))
        else:
            verdicts.append((label, ffn, "not run"))

    lines += [
        "",
        "## Verdict per model (from the 25% arm, where the rankings diverge most)",
        "",
        "| model | FFN | H3 |",
        "| --- | --- | --- |",
    ]
    for label, ffn, verdict in verdicts:
        lines.append(f"| {label} | {ffn} | {verdict} |")
    lines += [
        "",
        "Generated by `scripts/summarize_hypotheses.py` from `task_overlap_*.json`",
        "and `pruning_*.json` / `pruning_piqaranked_*.json`.",
    ]
    _write("h3_task_specificity.md", lines)


# --------------------------------------------------------------------------
# H5 / H5' — dropping vs replacing a duplicate twin
# --------------------------------------------------------------------------


def _condition(payload: dict, name: str) -> dict | None:
    for block in payload.get("conditions") or []:
        if block.get("condition") == name:
            return block
    return None


def _coverage(payload: dict) -> float | None:
    """How much of the duplicates' importance the matched control reached.

    Runs from before the Weeks 9-10 polish pass do not record this in the
    verdict, so recompute it from the per-condition means.
    """
    verdict = payload.get("verdict") or {}
    recorded = verdict.get("importance_match_ratio")
    if isinstance(recorded, (int, float)):
        return recorded
    one = _condition(payload, "duplicate_one") or {}
    ctrl = _condition(payload, "importance_matched") or {}
    dup_imp = one.get("mean_importance")
    ctrl_imp = ctrl.get("mean_importance")
    if isinstance(dup_imp, (int, float)) and isinstance(ctrl_imp, (int, float)) and dup_imp:
        return ctrl_imp / dup_imp
    return None


def h5() -> None:
    lines = [
        "# H5 vs H5' — is a duplicate twin cheap to *drop*, or cheap to *replace*?",
        "",
        "Both experiments use the same near-duplicate pairs (|rho| >= 0.9), the same",
        "eval slice, and the same seed, so the two tables are directly comparable.",
        "",
        "## H5 — dropping one twin (`run_pair_ablation.py`)",
        "",
        "H5 is supported only if `duplicate_one` costs **less** than the",
        "matched-importance control. `coverage` is how much of the duplicates' mean",
        "importance the control arm actually reached: below ~0.9x the control is an",
        "easier ablation and the comparison is biased against H5.",
        "",
        "| model | pairs | drop one | matched control | margin | coverage | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, slug, _ffn in MODELS:
        payload = _load(f"pair_ablation_{slug}_*.json")
        if payload is None:
            lines.append(f"| {label} | — | not run | | | | |")
            continue
        verdict = payload.get("verdict") or {}
        lines.append(
            "| {m} | {n} | {one} | {ctrl} | {margin} | {cov} | {v} |".format(
                m=label,
                n=(payload.get("pairs") or {}).get("num_pairs", "—"),
                one=_num(verdict.get("delta_duplicate_one"), "{:+.3f}"),
                ctrl=_num(verdict.get("delta_importance_matched"), "{:+.3f}"),
                margin=_num(verdict.get("margin"), "{:+.3f}"),
                cov=_num(_coverage(payload), "{:.2f}x"),
                v="**supported**" if verdict.get("supported") else "not supported",
            )
        )

    lines += [
        "",
        "### Both twins — the 'jointly critical, individually redundant' signature",
        "",
        "Masking one twin against masking both. A large jump means the surviving",
        "twin was genuinely covering for its partner.",
        "",
        "| model | drop one | drop both | ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, slug, _ffn in MODELS:
        payload = _load(f"pair_ablation_{slug}_*.json")
        if payload is None:
            continue
        one = _condition(payload, "duplicate_one") or {}
        both = _condition(payload, "duplicate_both") or {}
        d_one = one.get("delta_perplexity")
        d_both = both.get("delta_perplexity")
        if isinstance(d_one, (int, float)) and isinstance(d_both, (int, float)) and d_one > 1e-6:
            ratio = f"{d_both / d_one:,.0f}x"
        else:
            ratio = "n/a (dropping one was already free)"
        lines.append(
            f"| {label} | {_num(d_one, '{:+.3f}')} | {_num(d_both, '{:+.3f}')} | {ratio} |"
        )

    lines += [
        "",
        "## H5' — replacing one twin (`run_merge.py`)",
        "",
        "Instead of losing the dropped twin's contribution, hand it to its partner",
        "via the fitted map `h_drop ~= alpha * h_keep + beta`. This compares two",
        "treatments of the **same** neuron, so unlike H5 it is free of the",
        "importance confound.",
        "",
        "| model | pairs | mean r2 | min r2 | mask only | after merge | cost undone |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, slug, _ffn in MODELS:
        payload = _load(f"merge_{slug}_*.json")
        if payload is None:
            lines.append(f"| {label} | — | not run | | | | |")
            continue
        stats = payload.get("merge_stats") or {}
        verdict = payload.get("verdict") or {}
        frac = verdict.get("fraction_of_mask_cost_recovered")
        if isinstance(frac, (int, float)):
            undone = f"{100 * frac:.0f}%"
        else:
            undone = "n/a (masking was free)"
        lines.append(
            "| {m} | {n} | {mean} | {mn} | {mask} | {merge} | {undone} |".format(
                m=label,
                n=_num(stats.get("num_plans"), "{:.0f}"),
                mean=_num(stats.get("mean_r2")),
                mn=_num(stats.get("min_r2")),
                mask=_num(verdict.get("delta_mask_only"), "{:+.3f}"),
                merge=_num(verdict.get("delta_merge"), "{:+.3f}"),
                undone=undone,
            )
        )

    lines += [
        "",
        "**How to read this.** The `cost undone` column is only meaningful where",
        "masking cost something in the first place. On `gpt2-medium`, `gpt2-large`",
        "and Qwen3.5-4B the drop-one arm was already free (<= 0.01 PPL), so there is",
        "nothing for a merge to recover and the percentage is noise. On the three",
        "models where masking did cost something, the recovered fraction tracks the",
        "fit quality — which is the mechanistic claim H5' makes.",
        "",
        "Generated by `scripts/summarize_hypotheses.py` from `pair_ablation_*.json`",
        "and `merge_*.json`.",
    ]
    _write("h5_drop_vs_merge.md", lines)


# --------------------------------------------------------------------------
# H8 — recovery by short LoRA
# --------------------------------------------------------------------------


def _lora_budget(payload: dict) -> tuple[str, int | None]:
    """Training budget as ``steps x tokens/step``, so rows stay comparable.

    ``grad_accum`` is absent from the transcribed Colab artifact; treating a
    missing value as 1 is the reading that flatters that run least, and the
    caller flags any row that is not the modal budget.
    """
    cfg = payload.get("recovery_config") or {}
    train = payload.get("train_data") or {}
    steps = cfg.get("steps")
    batch = cfg.get("batch_size")
    accum = cfg.get("grad_accum", 1)
    seq = cfg.get("seq_len") or train.get("seq_len")
    if not all(isinstance(v, int) for v in (steps, batch, accum, seq)):
        return "unknown", None
    tokens = steps * batch * accum * seq
    return f"{steps}s x{batch}x{accum} x{seq} = {tokens / 1000:.0f}k tok", tokens


def h8() -> None:
    lines = [
        "# H8 — does short LoRA undo the cost of masking?",
        "",
        "**Adjusted recovery is the only honest number.** A LoRA budget spent on",
        "WikiText-2 improves perplexity on its own, so recovery measured against the",
        "untrained baseline is inflated. Every run therefore trains an *unmasked*",
        "control at the same budget and reports",
        "`1 - (recovered - control) / damage`.",
        "",
        "**Read the `budget` column before comparing rows.** Adjusted recovery is",
        "internally fair within a row — masked and unmasked arms always share a",
        "budget — but the budget itself is not constant *across* models, so a",
        "difference between rows may be a difference in training tokens rather than",
        "in the model.",
        "",
        "| model | ratio | budget | untrained base | masked | masked+LoRA | unmasked+LoRA | residual | raw rec. | **adj. rec.** |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    extra = [
        (
            "Qwen3.5-4B (Colab, 4-bit)",
            ROOT
            / "reports"
            / "group_5"
            / "artifacts"
            / "recovery_colab_Qwen__Qwen3.5-4B_20250812T115950Z.json",
        )
    ]
    jobs: list[tuple[str, dict | None]] = [
        (label, _load(f"recovery_{slug}_*.json")) for label, slug, _ffn in MODELS
    ]
    for label, path in extra:
        if path.exists():
            jobs.append((label, json.loads(path.read_text(encoding="utf-8"))))

    budgets: dict[str, int | None] = {}
    for label, payload in jobs:
        if payload is None:
            lines.append(f"| {label} | — | — | not run | | | | | | |")
            continue
        base = (payload.get("baseline") or {}).get("perplexity")
        ctrl = (payload.get("unmasked_control") or {}).get("perplexity")
        budget, tokens = _lora_budget(payload)
        budgets[label] = tokens
        for run in payload.get("runs") or []:
            masked = (run.get("masked") or {}).get("perplexity")
            rec = (run.get("masked_recovered") or {}).get("perplexity")
            if not all(isinstance(v, (int, float)) for v in (base, ctrl, masked, rec)):
                continue
            damage = masked - base
            residual = rec - ctrl
            raw = (masked - rec) / damage if damage else float("nan")
            adj = 1.0 - residual / damage if damage else float("nan")
            lines.append(
                "| {m} | {r:.0f}% | {budget} | {base:.2f} | {masked:.2f} | {rec:.2f} | {ctrl:.2f} "
                "| {res:+.2f} | {raw:.0f}% | **{adj:.0f}%** |".format(
                    m=label,
                    r=100 * float(run.get("ratio", 0)),
                    budget=budget,
                    base=base,
                    masked=masked,
                    rec=rec,
                    ctrl=ctrl,
                    res=residual,
                    raw=100 * raw,
                    adj=100 * adj,
                )
            )

    lines += [
        "",
        "`residual` is what still separates the masked model from the unmasked one",
        "after both got the same adapter budget — the number to quote if only one is",
        "quoted. A raw recovery above 100% just means masked+LoRA beat the untrained",
        "unmasked model, which says more about in-domain adaptation than about",
        "recovery.",
    ]

    known = {label: tok for label, tok in budgets.items() if tok}
    if len(set(known.values())) > 1:
        modal = max(set(known.values()), key=lambda t: list(known.values()).count(t))
        odd = {label: tok for label, tok in known.items() if tok != modal}
        lines += [
            "",
            "**Budgets are not matched across models.** Most runs share "
            f"{modal / 1000:.0f}k training tokens; "
            + ", ".join(
                f"`{label}` got {tok / 1000:.0f}k ({modal / tok:.0f}x less)"
                if tok < modal
                else f"`{label}` got {tok / 1000:.0f}k ({tok / modal:.0f}x more)"
                for label, tok in sorted(odd.items())
            )
            + ". Any cross-model claim about recovery therefore confounds model",
            "size with training budget, and cannot be read as a scale effect on its",
            "own.",
        ]

    lines += [
        "",
        "Not run: Llama-2-7B (4-bit 7B QLoRA evaluation trips",
        "`CUBLAS_STATUS_EXECUTION_FAILED` on 8 GB) and Qwen3.5-9B (#15).",
        "",
        "PIQA accuracies are recorded for every local run but **not** for the",
        "Colab 4B run: `lm_eval_limit: 200` was configured, yet no `lm_eval` block",
        "was transcribed into the artifact and none appears in #16. Those numbers",
        "are lost unless the original Colab JSON resurfaces.",
        "",
        "Generated by `scripts/summarize_hypotheses.py` from `recovery_*.json`.",
    ]
    _write("h8_recovery.md", lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate per-hypothesis tables.")
    parser.add_argument(
        "--only",
        choices=["h3", "h5", "h8"],
        default=None,
        help="Generate a single table.",
    )
    args = parser.parse_args(argv)
    for name, fn in (("h3", h3), ("h5", h5), ("h8", h8)):
        if args.only in (None, name):
            fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
