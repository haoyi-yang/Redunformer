#!/usr/bin/env python
"""Weeks 11-12: recover a masked model with short LoRA fine-tuning.

Weeks 9-10 established that importance-guided masking is far cheaper than random,
but still costs perplexity at every ratio. This script asks whether that cost is
permanent: it masks the least-important neurons, fine-tunes LoRA adapters on a
small WikiText-2 train slice with the masks live, and re-evaluates.

Arms per removal ratio:

    baseline          unmasked model, no training
    masked            masked, no training (the Weeks 9-10 number)
    masked_recovered  masked, after ``recovery.steps`` LoRA steps
    unmasked_control  unmasked, same LoRA budget   <- separates recovery from
                                                     ordinary in-domain gain

The headline number is the recovery fraction: how much of the masking damage the
adapters undo. It only means something next to the unmasked control, since some
of the improvement is just adaptation to the fine-tuning slice.

Example:
    python scripts/run_recovery.py --config configs/recovery/lora_qwen3_0.6b.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
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
from redundancy.eval import compute_perplexity, run_lm_eval_loaded  # noqa: E402
from redundancy.models import load_causal_lm  # noqa: E402
from redundancy.pruning import (  # noqa: E402
    NeuronMasker,
    find_latest_measurement,
    load_measurement,
    resolve_layer_filter,
    select_neurons,
    selection_stats,
)
from redundancy.recovery import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    RecoveryConfig,
    build_lora_model,
    recovery_fraction,
    train_lora,
    trainable_parameter_count,
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


def _recovery_config(cfg: dict, seed: int) -> RecoveryConfig:
    section = dict(cfg.get("recovery", {}))
    section.setdefault("seed", seed)
    known = {f.name for f in dataclasses.fields(RecoveryConfig)}
    unknown = set(section) - known
    if unknown:
        raise KeyError(f"Unknown recovery options in config: {sorted(unknown)}")
    if "target_modules" not in section:
        section["target_modules"] = list(DEFAULT_TARGET_MODULES)
    return RecoveryConfig(**section)


def _accuracy_from_lm_eval(results: dict) -> dict[str, float]:
    """Flatten lm-eval output to ``{task.metric: value}``, dropping stderrs."""
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
    parser = argparse.ArgumentParser(description="LoRA recovery after neuron masking.")
    parser.add_argument(
        "--config",
        default="configs/recovery/lora_qwen3_0.6b.yaml",
        help="Path to recovery config (YAML).",
    )
    parser.add_argument("--measurement", default=None, help="Override measurement JSON path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override eval documents.")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="Override the removal ratios to recover from.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override LoRA steps.")
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="Skip the unmasked LoRA control (halves runtime, weakens the claim).",
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

    train_cfg = cfg.get("train_data", {})
    eval_cfg = cfg.get("eval", {})
    rec_cfg = _recovery_config(cfg, seed)
    if args.steps is not None:
        rec_cfg.steps = args.steps

    ratios = [float(r) for r in (args.ratios or cfg.get("ratios", [0.1]))]
    strategy = str(cfg.get("strategy", "importance"))
    run_control = not args.skip_control and bool(cfg.get("train_unmasked_control", True))

    output_dir = ROOT / cfg.get("output_dir", "experiments/results")
    figures_dir = ROOT / cfg.get("figures_dir", "experiments/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation dataset ...")
    texts, dataset_meta = load_text_dataset(dataset_cfg)
    ppl_kwargs = {
        "max_length": int(eval_cfg.get("max_length", 2048)),
        "stride": int(eval_cfg.get("stride", 512)),
    }
    lm_eval_tasks = list(eval_cfg.get("lm_eval_tasks", ["piqa"]))
    run_lm = bool(eval_cfg.get("run_lm_eval", False))

    # Each arm needs a *fresh* model: LoRA adapters cannot be cleanly unwound
    # between ratios, so we reload rather than risk carrying weights over.
    def fresh_model():
        loaded = load_causal_lm(model_cfg)
        return loaded, next(loaded.model.parameters()).device

    print("Loading model for the untrained baselines ...")
    loaded, device = fresh_model()
    model_id = loaded.model_id

    measurement_path = (
        Path(args.measurement) if args.measurement else _resolve_measurement(cfg, loaded.model_id)
    )
    print(f"Using measurement artifact {measurement_path}")
    artifact = load_measurement(measurement_path)
    print(f"  calibrated on: {artifact.calibration_label()}")
    layer_filter = resolve_layer_filter(artifact, cfg.get("layers", "all"))

    print("Loading fine-tuning text ...")
    train_dataset_cfg = load_yaml(train_cfg.get("dataset_config", "configs/datasets/wikitext2_calib.yaml"))
    if train_cfg.get("max_samples") is not None:
        train_dataset_cfg["max_samples"] = train_cfg["max_samples"]
    train_texts, train_meta = load_text_dataset(train_dataset_cfg)
    blocks = build_calibration_blocks(
        train_texts,
        loaded.tokenizer,
        seq_len=int(train_cfg.get("seq_len", 512)),
        max_blocks=int(train_cfg.get("num_blocks", 256)),
    )
    print(f"  {len(blocks)} training blocks of {blocks[0].shape[-1]} tokens")

    def evaluate(model_holder) -> dict:
        out = {"perplexity": compute_perplexity(model_holder, texts, **ppl_kwargs)["perplexity"]}
        if run_lm:
            out["lm_eval"] = _accuracy_from_lm_eval(
                run_lm_eval_loaded(
                    model_holder,
                    lm_eval_tasks,
                    limit=eval_cfg.get("lm_eval_limit", 200),
                    batch_size=int(eval_cfg.get("lm_eval_batch_size", 1)),
                    seed=seed,
                )
            )
        return out

    # --- untrained arms ----------------------------------------------------
    selections = {
        ratio: select_neurons(artifact, strategy, ratio, seed=seed, layer_filter=layer_filter)
        for ratio in ratios
    }

    with NeuronMasker(loaded.model, layer_names=artifact.layer_names) as masker:
        print("\n=== baseline (unmasked, untrained) ===")
        baseline = evaluate(loaded)
        print(f"  ppl = {baseline['perplexity']:.4f}")

        masked_arms: dict[float, dict] = {}
        for ratio in ratios:
            num_masked = masker.apply(selections[ratio])
            print(f"\n=== masked, untrained ({strategy} @ {ratio:.0%}, {num_masked} neurons) ===")
            masked_arms[ratio] = evaluate(loaded)
            print(f"  ppl = {masked_arms[ratio]['perplexity']:.4f}")
            masker.clear()

    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- trained arms ------------------------------------------------------
    def train_arm(label: str, selection: dict | None) -> dict:
        print(f"\n=== {label}: LoRA {rec_cfg.steps} steps "
              f"(batch {rec_cfg.batch_size} x accum {rec_cfg.grad_accum}) ===")
        arm_loaded, arm_device = fresh_model()
        peft_model = build_lora_model(
            arm_loaded.model, rec_cfg, is_4bit=bool(arm_loaded.load_in_4bit)
        )
        trainable, total = trainable_parameter_count(peft_model)
        print(f"  trainable {trainable:,} / {total:,} parameters ({trainable / total:.3%})")
        arm_loaded = dataclasses.replace(arm_loaded, model=peft_model)

        masker = NeuronMasker(peft_model, layer_names=artifact.layer_names)
        try:
            if selection is not None:
                num_masked = masker.apply(selection)
                print(f"  masks live during training: {num_masked} neurons")
            history = train_lora(
                peft_model,
                blocks,
                rec_cfg,
                device=arm_device,
                progress=lambda step, loss: (
                    print(f"    step {step:>4}/{rec_cfg.steps}  loss {loss:.4f}")
                    if step == 1 or step % max(1, rec_cfg.steps // 10) == 0
                    else None
                ),
            )
            metrics = evaluate(arm_loaded)
        finally:
            masker.remove()
        print(f"  ppl = {metrics['perplexity']:.4f}")
        result = {**metrics, "training": history.summary()}
        result["trainable_parameters"] = trainable
        del peft_model, arm_loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    recovered_arms = {ratio: train_arm(f"masked+LoRA @ {ratio:.0%}", selections[ratio]) for ratio in ratios}
    control_arm = train_arm("unmasked+LoRA control", None) if run_control else None

    # --- assemble ----------------------------------------------------------
    rows: list[dict] = []
    for ratio in ratios:
        masked = masked_arms[ratio]
        recovered = recovered_arms[ratio]
        damage = masked["perplexity"] - baseline["perplexity"]
        row = {
            "ratio": ratio,
            "strategy": strategy,
            "selection": selection_stats(artifact, selections[ratio]),
            "masked": masked,
            "masked_recovered": recovered,
            "delta_masked": damage,
            "delta_recovered": recovered["perplexity"] - baseline["perplexity"],
            "recovery_fraction": recovery_fraction(
                baseline["perplexity"], masked["perplexity"], recovered["perplexity"]
            ),
        }
        if control_arm is not None:
            # The fair comparison at equal training budget: how much worse is the
            # masked-and-tuned model than the unmasked-and-tuned one? Fine-tuning
            # also buys plain in-domain gain, which this subtracts out.
            residual = recovered["perplexity"] - control_arm["perplexity"]
            row["residual_vs_control"] = residual
            row["adjusted_recovery_fraction"] = (
                1.0 - residual / damage if abs(damage) > 1e-9 else None
            )
        rows.append(row)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_id.replace("/", "__")
    suffix = f"_{args.tag}" if args.tag else ""

    series = {
        "masked (no recovery)": [r["masked"]["perplexity"] for r in rows],
        "masked + LoRA recovery": [r["masked_recovered"]["perplexity"] for r in rows],
    }
    fig_path = plotting.plot_removal_curves(
        ratios,
        series,
        figures_dir / f"recovery_ppl{suffix}_{safe_model}_{stamp}.png",
        title=f"LoRA recovery after {strategy}-guided masking — {model_id}",
        ylabel="WikiText-2 perplexity",
        baseline=baseline["perplexity"],
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "weeks_11_12_lora_recovery",
        "question": "does short LoRA fine-tuning undo the perplexity cost of neuron masking?",
        "seed": seed,
        "config_path": str(args.config),
        "command": " ".join(sys.argv),
        "model": {"model_id": model_id, "config": cfg["model_config"]},
        "dataset": dataset_meta,
        "train_data": {**train_meta, "num_blocks": len(blocks), "seq_len": int(blocks[0].shape[-1])},
        "measurement_artifact": str(measurement_path),
        "strategy": strategy,
        "ratios": ratios,
        "recovery_config": rec_cfg.summary(),
        "baseline": baseline,
        "unmasked_control": control_arm,
        "runs": rows,
        "artifacts": {"figures": {"recovery_ppl": str(fig_path)}},
    }
    out_path = output_dir / f"recovery{suffix}_{safe_model}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote summary -> {out_path}")
    print(f"Wrote figure  -> {fig_path}")
    _print_table(baseline, rows, control_arm)
    return 0


def _print_table(baseline: dict, rows: list[dict], control: dict | None) -> None:
    print(f"\nWikiText-2 perplexity (unmasked baseline {baseline['perplexity']:.4f}):")
    header = f"  {'ratio':>6} {'masked':>12} {'+recovery':>12} {'raw rec.':>10}"
    if control is not None:
        header += f" {'vs control':>11} {'adj. rec.':>10}"
    print(header)
    for row in rows:
        frac = row["recovery_fraction"]
        line = (
            f"  {row['ratio']:>6.0%} {row['masked']['perplexity']:>12.4f} "
            f"{row['masked_recovered']['perplexity']:>12.4f} "
            + (f"{frac:>10.1%}" if frac is not None else f"{'-':>10}")
        )
        if control is not None:
            adjusted = row.get("adjusted_recovery_fraction")
            line += f" {row['residual_vs_control']:>+11.4f} " + (
                f"{adjusted:>10.1%}" if adjusted is not None else f"{'-':>10}"
            )
        print(line)
    if control is not None:
        delta = control["perplexity"] - baseline["perplexity"]
        print(
            f"\nUnmasked LoRA control: {control['perplexity']:.4f} ({delta:+.4f} vs the "
            "untrained baseline). The same adapter budget on an unmasked model, so "
            "this is the share of any gain that is plain in-domain adaptation."
        )
        print(
            "  'raw rec.' credits that adaptation to recovery; 'adj. rec.' does not, "
            "and compares masked+LoRA against this control instead."
        )
    accuracy_keys = sorted(baseline.get("lm_eval", {}))
    for key in accuracy_keys:
        print(f"\n{key} (baseline {baseline['lm_eval'][key]:.4f}):")
        print(f"  {'ratio':>6} {'masked':>12} {'+recovery':>12}")
        for row in rows:
            masked = row["masked"].get("lm_eval", {}).get(key)
            rec = row["masked_recovered"].get("lm_eval", {}).get(key)
            print(
                f"  {row['ratio']:>6.0%} "
                + (f"{masked:>12.4f}" if masked is not None else f"{'-':>12}")
                + (f"{rec:>12.4f}" if rec is not None else f"{'-':>12}")
            )


if __name__ == "__main__":
    raise SystemExit(main())
