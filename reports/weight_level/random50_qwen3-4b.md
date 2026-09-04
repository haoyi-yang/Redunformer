# Random 50% on Qwen3-4B — mlsp2

Date: 2026-09-04 (mlsp2, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** random unstructured, sparsity **0.5** (50% weights zeroed at random, seed=42)
- **Module:** `src/redundancy/pruning/random_pruning.py` via `scripts/run_pruning_pipeline.py`
- **Pruned model:** `experiments/pruned/qwen3-4b-random50`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp2 (`130.83.166.152`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/random50-after-sparsegpt50-20260904-184056.log`

Chained after SparseGPT50 eval via watcher: prune → save → lm-eval.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Random 50%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-random50_2026-09-04T20-45-04.140256.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.2547 | 0.2617   | 10042 |
| piqa      | 0.5207 | 0.5136   | 1838  |
| arc_easy  | 0.2471 | 0.2551   | 2376  |

## Delta (random50 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.2667 | −0.4227    |
| piqa      | −0.2285 | −0.2340    |
| arc_easy  | −0.5589 | −0.5298    |

## Reference: Random 20% / Magnitude 50% / SparseGPT 50%

| Method        | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|---------------|----------------------|-----------------|---------------------|
| random20      | 0.2551 / 0.2649      | 0.5185 / 0.5071 | 0.2513 / 0.2492     |
| random50      | 0.2547 / 0.2617      | 0.5207 / 0.5136 | 0.2471 / 0.2551     |
| magnitude50   | 0.3353 / 0.4325      | 0.6556 / 0.6469 | 0.5484 / 0.4903     |
| sparsegpt50   | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |

## Takeaway

At **50%** random pruning, accuracy is already at the chance floor (same collapse as random20 on hellaswag/arc_easy). Extra sparsity beyond 20% does not change the qualitative picture. Random remains a weak control vs magnitude50 and especially SparseGPT50. Note: DSnoT cannot refine this checkpoint — random masks are not per-row equal-sparsity.
