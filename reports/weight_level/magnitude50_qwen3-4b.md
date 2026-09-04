# Magnitude 50% on Qwen3-4B — mlsp4

Date: 2026-09-04 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** magnitude (unstructured), sparsity **0.5** (50% smallest-|w| zeros per `nn.Linear`, `lm_head` excluded)
- **Threshold:** `torch.kthvalue` (not `torch.quantile`)
- **Pruned model:** `experiments/pruned/qwen3-4b-magnitude50`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp4 (`130.83.166.154`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/magnitude50-after-sparsegpt20-20260904-194523.log`

Chained after SparseGPT20 eval via watcher: prune → save → lm-eval.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Magnitude 50%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-magnitude50_2026-09-04T19-39-29.415918.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.3353 | 0.4325   | 10042 |
| piqa      | 0.6556 | 0.6469   | 1838  |
| arc_easy  | 0.5484 | 0.4903   | 2376  |

## Delta (magnitude50 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.1861 | −0.2519    |
| piqa      | −0.0936 | −0.1007    |
| arc_easy  | −0.2576 | −0.2946    |

## Reference: Magnitude 20% / SparseGPT 50% (same model / tasks)

| Method        | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|---------------|----------------------|-----------------|---------------------|
| magnitude20   | 0.5153 / 0.6833      | 0.7470 / 0.7465 | 0.8005 / 0.7630     |
| magnitude50   | 0.3353 / 0.4325      | 0.6556 / 0.6469 | 0.5484 / 0.4903     |
| sparsegpt50   | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |

## Takeaway

At **50%** magnitude, quality drops hard vs dense and vs magnitude20 (hellaswag ≈−19 pp acc / −25 pp acc_norm; arc_easy ≈−26 / −29 pp). **SparseGPT50 clearly beats magnitude50** on all three tasks (~13–20 pp on hellaswag/arc_easy). Magnitude remains a weak mid-sparsity baseline next to SparseGPT at the same ratio.
