# Wanda 20% on Qwen3-4B — mlsp4

Date: 2026-09-04 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** Wanda unstructured, sparsity **0.2**
- **Calibration:** wikitext2, `nsamples=128`, `seqlen=2048`, `seed=42`
- **Pruned model:** `experiments/pruned/qwen3-4b-wanda20`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp4 (`130.83.166.154`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/wanda20-after-magnitude50-20260904-211641.log`

Chained after Magnitude50 eval via watcher: prune → save → lm-eval.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Wanda 20%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-wanda20_2026-09-04T21-17-43.879224.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5158 | 0.6800   | 10042 |
| piqa      | 0.7454 | 0.7492   | 1838  |
| arc_easy  | 0.7984 | 0.7786   | 2376  |

## Delta (wanda20 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0056 | −0.0044    |
| piqa      | −0.0038 | +0.0016    |
| arc_easy  | −0.0076 | −0.0063    |

## Reference: Magnitude / SparseGPT / Random 20% (same model / tasks)

| Method        | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|---------------|----------------------|-----------------|---------------------|
| magnitude20   | 0.5153 / 0.6833      | 0.7470 / 0.7465 | 0.8005 / 0.7630     |
| wanda20       | 0.5158 / 0.6800      | 0.7454 / 0.7492 | 0.7984 / 0.7786     |
| sparsegpt20   | 0.5214 / 0.6843      | 0.7486 / 0.7514 | 0.8026 / 0.7790     |
| random20      | 0.2551 / 0.2649      | 0.5185 / 0.5071 | 0.2513 / 0.2492     |

## Takeaway

At **20%** Wanda on Qwen3-4B, scores stay near dense (≈0.4–0.8 pp). In line with magnitude20; SparseGPT20 is slightly closer to dense on these tasks. All three informed methods crush random20. Wanda20 masks are per-row and suitable as a DSnoT base.
