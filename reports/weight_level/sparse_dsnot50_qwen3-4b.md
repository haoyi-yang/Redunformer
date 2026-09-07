# SparseGPT-row (sparse_dsnot) 50% on Qwen3-4B — mlsp2

Date: 2026-09-05 (mlsp2, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** `sparse_dsnot` — SparseGPT OBS metric with **per-row** mask (`⌊sparsity · C_in⌋` zeros per output row). Not classic block-flatten SparseGPT.
- **Calibration:** wikitext2, `nsamples=128`, `seqlen=2048`, `percdamp=0.01`, `seed=42`
- **Pruned model:** `experiments/pruned/qwen3-4b-sparse_dsnot50`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp2 (`130.83.166.152`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/sparse-dsnot50-dsnot-20260905-031539.log`

Prune finished ~03:28; per-row preflight `SPARSE_DSNOT50_PER_ROW_OK` (252/252); lm-eval finished ~05:33.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## sparse_dsnot 50%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-sparse_dsnot50_2026-09-05T03-33-27.300800.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4539 | 0.5989   | 10042 |
| piqa      | 0.6942 | 0.6980   | 1838  |
| arc_easy  | 0.7311 | 0.6940   | 2376  |

## Delta (sparse_dsnot50 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0675 | −0.0855    |
| piqa      | −0.0550 | −0.0496    |
| arc_easy  | −0.0749 | −0.0909    |

## Reference: classic SparseGPT50 / Wanda50 (same model / tasks)

| Method          | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|-----------------|----------------------|-----------------|---------------------|
| sparsegpt50     | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |
| sparse_dsnot50  | 0.4539 / 0.5989      | 0.6942 / 0.6980 | 0.7311 / 0.6940     |
| wanda50         | 0.4474 / 0.5891      | 0.7106 / 0.7214 | 0.7340 / 0.7066     |

## Takeaway

Per-row SparseGPT at **50%** is a valid DSnoT base but scores **below classic SparseGPT50** on all three tasks (~1–3 pp). Roughly in the Wanda50 ballpark (hellaswag a bit higher, piqa a bit lower). Use this checkpoint when DSnoT refine is required; keep classic SparseGPT for non-DSnoT comparisons.
