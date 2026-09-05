# Magnitude-row 50% + DSnoT on Qwen3-4B — mlsp2

Date: 2026-09-05 (mlsp2, RTX 2080 Ti 11 GB)

## Setup

- **Base sparse model:** `experiments/pruned/qwen3-4b-magnitude_row50` (per-row magnitude, sparsity **0.5** — equal zeros per output row, DSnoT-compatible; not classic global-threshold magnitude)
- **Refiner:** DSnoT on the saved magnitude_row50 checkpoint
- **Calibration / DSnoT:** wikitext2, `nsamples=128`, `seqlen=2048`, `seed=42` (same family as other weight-level runs)
- **Pruned model:** `experiments/pruned/qwen3-4b-magnitude_row50-dsnot`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp2 (`130.83.166.152`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/magnitude-row50-dsnot-20260904-235106.log`

Chain: magnitude_row50 prune → per-row preflight → DSnoT refine (~23:55–00:11) → lm-eval finished ~02:20. No separate lm-eval JSON for the pre-DSnoT magnitude_row50 checkpoint.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Magnitude-row 50% + DSnoT

Output: `experiments/baseline/experiments_pruned_qwen3-4b-magnitude_row50-dsnot_2026-09-05T00-19-41.432254.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4198 | 0.5675   | 10042 |
| piqa      | 0.7051 | 0.7051   | 1838  |
| arc_easy  | 0.6814 | 0.6473   | 2376  |

## Delta (magnitude_row50-dsnot − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.1016 | −0.1169    |
| piqa      | −0.0441 | −0.0425    |
| arc_easy  | −0.1246 | −0.1376    |

## Reference: classic magnitude50 / Wanda50±DSnoT / SparseGPT50

| Method                | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|-----------------------|----------------------|-----------------|---------------------|
| magnitude50 (classic) | 0.3353 / 0.4325      | 0.6556 / 0.6469 | 0.5484 / 0.4903     |
| magnitude_row50-dsnot | 0.4198 / 0.5675      | 0.7051 / 0.7051 | 0.6814 / 0.6473     |
| wanda50               | 0.4474 / 0.5891      | 0.7106 / 0.7214 | 0.7340 / 0.7066     |
| wanda50-dsnot         | 0.4404 / 0.5803      | 0.7100 / 0.7247 | 0.7365 / 0.7125     |
| sparsegpt50           | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |

## Takeaway

Per-row magnitude + DSnoT at **50%** is **much stronger than classic magnitude50** (~8–16 pp depending on task/metric) but still trails Wanda50 / Wanda50-DSnoT and SparseGPT50. Without a standalone magnitude_row50 eval, the gain from DSnoT vs the per-row magnitude base alone is unknown.
