# Magnitude 20% on Qwen3-4B — mlsp4

Date: 2026-09-04 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** magnitude (unstructured), sparsity **0.2** (20% smallest-|w| zeros per `nn.Linear`, `lm_head` excluded)
- **Threshold:** `torch.kthvalue` (not `torch.quantile` — quantile fails on large Qwen linear weights)
- **Pruned model:** `experiments/pruned/qwen3-4b-magnitude20`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto`, device_map=auto
- **Machine:** mlsp4 (`130.83.166.154`), podman + NVIDIA lib bind-mount workaround (CDI `additionalGids` broken for rootless podman)
- **Log:** `experiments/logs/magnitude20-qwen3-4b-20260904-064019.log`

Pipeline: prune → save → **lm-eval immediately** (no separate step).

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Magnitude 20%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-magnitude20_2026-09-04T06-09-50.536717.json`  
Meta timestamp: `2026-09-04T06:09:57Z`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5153 | 0.6833   | 10042 |
| piqa      | 0.7470 | 0.7465   | 1838  |
| arc_easy  | 0.8005 | 0.7630   | 2376  |

## Delta (magnitude20 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0061 | −0.0011    |
| piqa      | −0.0022 | −0.0011    |
| arc_easy  | −0.0055 | −0.0219    |

## Reference: Random 20% (same model / tasks)

From `experiments/random_pruning/experiments_pruned_qwen3-4b-random20_2026-06-16T22-06-02.715263.json`

| Task      | acc    | acc_norm |
|-----------|--------|----------|
| hellaswag | 0.2551 | 0.2649   |
| piqa      | 0.5185 | 0.5071   |
| arc_easy  | 0.2513 | 0.2492   |

## Takeaway

At **20%** magnitude pruning on Qwen3-4B, hellaswag/piqa stay near dense baseline (≈0.1–0.6 pp). `arc_easy` acc_norm drops more (≈2.2 pp). Magnitude clearly beats random 20%, which collapses accuracy toward chance on these tasks.
