"""Run remaining cross-family experiments (baselines, H1, H3, H5/H5′, recovery).

Skips a stage when a matching result JSON already exists. Qwen3.5-9B is out of
scope. Intended for a long GPU pass after ``herai/model_families`` landed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


def _latest(pattern: str) -> Path | None:
    matches = sorted(RESULTS.glob(pattern))
    return matches[-1] if matches else None


def _run(argv: list[str]) -> None:
    print("\n>>>", " ".join(argv), flush=True)
    subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, check=True)


def _tertile_layers(measurement: Path) -> dict[str, list[int]]:
    payload = json.loads(measurement.read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {}
    for name, block in (payload.get("depth_tertiles") or {}).items():
        layers = [int(i) for i in block.get("layers") or []]
        if layers:
            out[name] = layers
    return out


def _overlap_exists(model_id: str) -> bool:
    for path in sorted(RESULTS.glob("task_overlap_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("model_id") == model_id:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-family follow-up experiments.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-h1", action="store_true")
    parser.add_argument("--skip-h3", action="store_true")
    parser.add_argument("--skip-h5", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument(
        "--include-llama-recovery",
        action="store_true",
        help="Attempt Llama-2-7B QLoRA recovery (fails with CUBLAS on 8 GB).",
    )
    parser.add_argument(
        "--only",
        choices=["baselines", "h1", "h3", "h5", "recovery"],
        default=None,
        help="Run a single stage.",
    )
    args = parser.parse_args(argv)

    def want(stage: str) -> bool:
        if args.only:
            return args.only == stage
        skips = {
            "baselines": args.skip_baselines,
            "h1": args.skip_h1,
            "h3": args.skip_h3,
            "h5": args.skip_h5,
            "recovery": args.skip_recovery,
        }
        return not skips[stage]

    def try_run(argv_: list[str], what: str) -> None:
        """Run a stage but let the pass continue if this arm does not fit."""
        try:
            _run(argv_)
        except subprocess.CalledProcessError as exc:
            print(f"!! {what} failed (exit {exc.returncode}); continuing", flush=True)

    if want("baselines"):
        for cfg, pattern in [
            ("configs/eval/baseline_gpt2.yaml", "baseline_gpt2_*.json"),
            ("configs/eval/baseline_gpt2-medium.yaml", "baseline_gpt2-medium_*.json"),
            ("configs/eval/baseline_gpt2-large.yaml", "baseline_gpt2-large_*.json"),
            ("configs/eval/baseline_llama2-7b.yaml", "baseline_meta-llama__Llama-2-7b-hf_*.json"),
        ]:
            if _latest(pattern):
                print(f"skip baseline, have {pattern}")
                continue
            _run(["scripts/run_baseline.py", "--config", cfg])

    if want("h1"):
        _run(["scripts/summarize_h1_depth.py"])
        h1_models = [
            ("configs/pruning/neuron_masking_gpt2.yaml", "measurement_gpt2_*.json", "gpt2"),
            ("configs/pruning/neuron_masking_gpt2-medium.yaml", "measurement_gpt2-medium_*.json", "gpt2-medium"),
            ("configs/pruning/neuron_masking_gpt2-large.yaml", "measurement_gpt2-large_*.json", "gpt2-large"),
            (
                "configs/pruning/neuron_masking_llama2-7b.yaml",
                "measurement_meta-llama__Llama-2-7b-hf_*.json",
                "meta-llama__Llama-2-7b-hf",
            ),
            # The two Qwen models had only the measurement-side tertile proxy;
            # these close the same ablation arm the GPT-2/Llama models got.
            (
                "configs/pruning/neuron_masking_qwen3_0.6b.yaml",
                "measurement_Qwen__Qwen3-0.6B_*.json",
                "Qwen__Qwen3-0.6B",
            ),
            (
                "configs/pruning/neuron_masking.yaml",
                "measurement_Qwen__Qwen3.5-4B_*.json",
                "Qwen__Qwen3.5-4B",
            ),
        ]
        for prune_cfg, meas_pat, slug in h1_models:
            measurement = _latest(meas_pat)
            if measurement is None:
                print(f"skip H1, no {meas_pat}")
                continue
            for tertile, layers in _tertile_layers(measurement).items():
                out_tag = f"h1{tertile}"
                if _latest(f"pruning_{out_tag}_{slug}_*.json"):
                    print(f"skip H1 {out_tag} {slug}")
                    continue
                try_run(
                    [
                        "scripts/run_pruning.py",
                        "--config",
                        prune_cfg,
                        "--measurement",
                        str(measurement),
                        "--layers",
                        ",".join(str(i) for i in layers),
                        "--ratios",
                        "0.10",
                        "--strategies",
                        "importance",
                        "--no-lm-eval",
                        "--tag",
                        out_tag,
                    ],
                    f"H1 {out_tag} {slug}",
                )

    if want("h3"):
        jobs = [
            (
                "gpt2",
                "configs/measurement/neuron_activations_piqa_gpt2.yaml",
                "measurement_piqacalib_gpt2_*.json",
                "measurement_gpt2_*.json",
                "configs/measurement/task_overlap_gpt2.yaml",
                "configs/pruning/neuron_masking_piqa_ranked_gpt2.yaml",
                "pruning_piqaranked_gpt2_*.json",
            ),
            (
                "meta-llama/Llama-2-7b-hf",
                "configs/measurement/neuron_activations_piqa_llama2-7b.yaml",
                "measurement_piqacalib_meta-llama__Llama-2-7b-hf_*.json",
                "measurement_meta-llama__Llama-2-7b-hf_*.json",
                "configs/measurement/task_overlap_llama2-7b.yaml",
                "configs/pruning/neuron_masking_piqa_ranked_llama2-7b.yaml",
                "pruning_piqaranked_meta-llama__Llama-2-7b-hf_*.json",
            ),
            # H3 on the primary model. Qwen3-0.6B was done in Weeks 11-12; the
            # 4B arm was the conspicuous hole in the task dimension.
            (
                "Qwen/Qwen3.5-4B",
                "configs/measurement/neuron_activations_piqa_qwen3.5-4b.yaml",
                "measurement_piqacalib_Qwen__Qwen3.5-4B_*.json",
                "measurement_Qwen__Qwen3.5-4B_*.json",
                "configs/measurement/task_overlap_qwen3.5-4b.yaml",
                "configs/pruning/neuron_masking_piqa_ranked_qwen3.5-4b.yaml",
                "pruning_piqaranked_Qwen__Qwen3.5-4B_*.json",
            ),
        ]
        for model_id, meas_cfg, piqa_pat, wiki_pat, overlap_cfg, prune_cfg, prune_pat in jobs:
            if not _latest(piqa_pat):
                try_run(
                    ["scripts/run_measurement.py", "--config", meas_cfg],
                    f"H3 measurement {model_id}",
                )
            wiki = _latest(wiki_pat)
            piqa = _latest(piqa_pat)
            if wiki and piqa and not _overlap_exists(model_id):
                try_run(
                    [
                        "scripts/run_task_overlap.py",
                        "--config",
                        overlap_cfg,
                        "--reference",
                        str(wiki),
                        "--comparison",
                        str(piqa),
                    ],
                    f"H3 overlap {model_id}",
                )
            elif not wiki or not piqa:
                print(f"skip overlap for {model_id}: missing WikiText or PIQA measurement")
            if not _latest(prune_pat):
                piqa = _latest(piqa_pat)
                if piqa is None:
                    print(f"skip PIQA-ranked prune, no {piqa_pat}")
                    continue
                try_run(
                    [
                        "scripts/run_pruning.py",
                        "--config",
                        prune_cfg,
                        "--measurement",
                        str(piqa),
                        "--tag",
                        "piqaranked",
                    ],
                    f"H3 piqa-ranked prune {model_id}",
                )

    if want("h5"):
        for cfg, pattern, script in [
            (
                "configs/pruning/pair_ablation_gpt2-medium.yaml",
                "pair_ablation_gpt2-medium_*.json",
                "scripts/run_pair_ablation.py",
            ),
            (
                "configs/pruning/merge_gpt2-medium.yaml",
                "merge_gpt2-medium_*.json",
                "scripts/run_merge.py",
            ),
            (
                "configs/pruning/pair_ablation_gpt2-large.yaml",
                "pair_ablation_gpt2-large_*.json",
                "scripts/run_pair_ablation.py",
            ),
            (
                "configs/pruning/merge_gpt2-large.yaml",
                "merge_gpt2-large_*.json",
                "scripts/run_merge.py",
            ),
        ]:
            if _latest(pattern):
                print(f"skip {pattern}")
                continue
            _run([script, "--config", cfg])

    if want("recovery"):
        jobs = [
            ("configs/recovery/lora_gpt2.yaml", "recovery_gpt2_*.json"),
            # Same adapter budget across the GPT-2 size arm, so "does recovery
            # depend on size?" is asked with the recipe held fixed.
            ("configs/recovery/lora_gpt2-medium.yaml", "recovery_gpt2-medium_*.json"),
            ("configs/recovery/lora_gpt2-large.yaml", "recovery_gpt2-large_*.json"),
        ]
        if args.include_llama_recovery:
            jobs.append(
                (
                    "configs/recovery/lora_llama2-7b.yaml",
                    "recovery_meta-llama__Llama-2-7b-hf_*.json",
                )
            )
        else:
            print("skip Llama-2-7B recovery (pass --include-llama-recovery to try)")
        for cfg, pattern in jobs:
            if _latest(pattern):
                print(f"skip {pattern}")
                continue
            try:
                _run(["scripts/run_recovery.py", "--config", cfg])
            except subprocess.CalledProcessError as exc:
                print(
                    f"recovery failed for {cfg} (exit {exc.returncode}); "
                    "continuing — 4-bit 7B QLoRA eval is known to trip CUBLAS on 8 GB"
                )

    print("=== follow-ups finished ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
