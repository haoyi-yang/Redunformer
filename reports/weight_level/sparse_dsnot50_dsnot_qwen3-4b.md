# sparse_dsnot 50% + DSnoT on Qwen3-4B — mlsp2

Date: 2026-09-05 (mlsp2, RTX 2080 Ti 11 GB)

## Setup

- **Base sparse model:** `experiments/pruned/qwen3-4b-sparse_dsnot50` (per-row SparseGPT, sparsity **0.5**, preflight `SPARSE_DSNOT50_PER_ROW_OK`)
- **Refiner:** DSnoT (`algorithm=dsnot`, `base=sparsegpt`), dense `Qwen/Qwen3-4B`
- **DSnoT / calib:** wikitext2, `nsamples=128`, `seqlen=2048`, `seed=42`
- **Pruned model:** `experiments/pruned/qwen3-4b-sparse_dsnot50-dsnot`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp2 (`130.83.166.152`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/sparse-dsnot50-dsnot-20260905-031539.log`

Chain: sparse_dsnot50 eval (~05:33) → DSnoT refine → lm-eval finished ~08:00.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## sparse_dsnot50 + DSnoT

Output: `experiments/baseline/experiments_pruned_qwen3-4b-sparse_dsnot50-dsnot_2026-09-05T06-00-23.219823.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4500 | 0.5982   | 10042 |
| piqa      | 0.7084 | 0.7258   | 1838  |
| arc_easy  | 0.7412 | 0.7071   | 2376  |

## Delta (sparse_dsnot50-dsnot − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0714 | −0.0862    |
| piqa      | −0.0408 | −0.0218    |
| arc_easy  | −0.0648 | −0.0778    |

## Delta (sparse_dsnot50-dsnot − sparse_dsnot50)

Base from `reports/weight_level/sparse_dsnot50_qwen3-4b.md`

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0039 | −0.0007    |
| piqa      | +0.0142 | +0.0278    |
| arc_easy  | +0.0101 | +0.0131    |

## Reference: sparse_dsnot50 / classic SparseGPT50 / Wanda±DSnoT

| Method                   | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|--------------------------|----------------------|-----------------|---------------------|
| sparsegpt50 (classic)    | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |
| sparse_dsnot50           | 0.4539 / 0.5989      | 0.6942 / 0.6980 | 0.7311 / 0.6940     |
| sparse_dsnot50-dsnot     | 0.4500 / 0.5982      | 0.7084 / 0.7258 | 0.7412 / 0.7071     |
| wanda50                  | 0.4474 / 0.5891      | 0.7106 / 0.7214 | 0.7340 / 0.7066     |
| wanda50-dsnot            | 0.4404 / 0.5803      | 0.7100 / 0.7247 | 0.7365 / 0.7125     |

## Takeaway

DSnoT on sparse_dsnot50 is **mixed**: hellaswag essentially flat / slightly down; **piqa and arc_easy improve** (~1–3 pp). Still below classic SparseGPT50 on hellaswag/arc_easy and near Wanda50-DSnoT. Per-row SparseGPT + DSnoT does not recover classic SparseGPT50 quality on this setup.
