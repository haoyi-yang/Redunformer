# Wanda 50% + DSnoT on Qwen3-4B — mlsp4

Date: 2026-09-05 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Base sparse model:** `experiments/pruned/qwen3-4b-wanda50` (Wanda unstructured, sparsity **0.5**, per-row mask verified `WANDA50_PER_ROW_OK`)
- **Refiner:** DSnoT (`algorithm=dsnot`, `base=wanda`), dense `Qwen/Qwen3-4B`
- **DSnoT params:** wikitext2, `nsamples=128`, `seqlen=2048`, `seed=42`, `cycles=50`, `epsilon=0.1`, `var_power=1.0`, `same_sign=0`, `skip_layer=none`
- **Pruned model:** `experiments/pruned/qwen3-4b-wanda50-dsnot`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp4 (`130.83.166.154`), podman + NVIDIA lib bind-mount workaround
- **Log:** `experiments/logs/wanda50-verified-dsnot-20260904-234644.log`

Chain: Wanda50 eval → DSnoT refine (~01:12–01:24) → lm-eval finished ~02:49.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Wanda 50% + DSnoT

Output: `experiments/baseline/experiments_pruned_qwen3-4b-wanda50-dsnot_2026-09-05T00-49-04.634235.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4404 | 0.5803   | 10042 |
| piqa      | 0.7100 | 0.7247   | 1838  |
| arc_easy  | 0.7365 | 0.7125   | 2376  |

## Delta (wanda50-dsnot − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0810 | −0.1041    |
| piqa      | −0.0392 | −0.0229    |
| arc_easy  | −0.0695 | −0.0724    |

## Delta (wanda50-dsnot − wanda50)

Wanda50 from `reports/weight_level/wanda50_qwen3-4b.md`

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0070 | −0.0088    |
| piqa      | −0.0006 | +0.0033    |
| arc_easy  | +0.0025 | +0.0059    |

## Reference: Wanda50 / SparseGPT50 / magnitude_row50-dsnot

| Method               | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|----------------------|----------------------|-----------------|---------------------|
| wanda50              | 0.4474 / 0.5891      | 0.7106 / 0.7214 | 0.7340 / 0.7066     |
| wanda50-dsnot        | 0.4404 / 0.5803      | 0.7100 / 0.7247 | 0.7365 / 0.7125     |
| sparsegpt50          | 0.4648 / 0.6132      | 0.7203 / 0.7307 | 0.7462 / 0.7277     |
| magnitude_row50-dsnot| 0.4198 / 0.5675      | 0.7051 / 0.7051 | 0.6814 / 0.6473     |

## Takeaway

DSnoT on Wanda50 is **mixed / near-flat** vs the Wanda50 base: hellaswag slightly down (~0.7–0.9 pp), piqa/arc_easy slightly up. Still well below SparseGPT50 and above magnitude_row50-dsnot on these tasks. No clear win for this DSnoT setting over plain Wanda50.
