# Wanda 50% on Qwen3-4B — mlsp4

Date: 2026-09-05 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** Wanda unstructured, sparsity **0.5**
- **Calibration:** wikitext2, `nsamples=128`, `seqlen=2048`, `seed=42`
- **Pruned model:** `experiments/pruned/qwen3-4b-wanda50`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp4 (`130.83.166.154`), podman + NVIDIA lib bind-mount workaround
- **Prune log:** `experiments/logs/wanda50-then-dsnot-20260904-232557.log`
- **Eval log:** `experiments/logs/wanda50-verified-dsnot-20260904-234644.log`

Prune finished ~23:36; per-row mask preflight `WANDA50_PER_ROW_OK`; lm-eval finished 2026-09-05 ~01:12.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Wanda 50%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-wanda50_2026-09-04T23-11-49.658164.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4474 | 0.5891   | 10042 |
| piqa      | 0.7106 | 0.7214   | 1838  |
| arc_easy  | 0.7340 | 0.7066   | 2376  |

## Delta (wanda50 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0740 | −0.0953    |
| piqa      | −0.0386 | −0.0262    |
| arc_easy  | −0.0720 | −0.0783    |

## Reference: Wanda 20% / other 50% methods (same model / tasks)

| Method        | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|---------------|----------------------|-----------------|---------------------|
| wanda20       | 0.5158 / 0.6800      | 0.7454 / 0.7492 | 0.7984 / 0.7786     |
| wanda50       | 0.4474 / 0.5891      | 0.7106 / 0.7214 | 0.7340 / 0.7066     |
| sparsegpt50   | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |
| magnitude50   | 0.3353 / 0.4325      | 0.6556 / 0.6469 | 0.5484 / 0.4903     |
| random50      | 0.2547 / 0.2617      | 0.5207 / 0.5136 | 0.2471 / 0.2551     |

## Takeaway

At **50%** Wanda drops clearly vs dense and vs wanda20 (hellaswag ≈−7.4 pp acc / −9.5 pp acc_norm), but stays in the same ballpark as **SparseGPT50** (slightly behind on all three tasks) and far above **magnitude50** / **random50**. Mask is per-row equal zeros and is a valid DSnoT base (refine+eval already chained on mlsp4).
