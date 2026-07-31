# SparseGPT 20% on Qwen3-1.7B — local run

Date: 2026-08-01 (local RTX 3060 Laptop 6 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-1.7B` from Hugging Face (not a previously pruned checkpoint)
- **Pruner:** SparseGPT unstructured, sparsity **0.2** (20% zeros)
- **Calibration:** wikitext2, `nsamples=128`, `seqlen=2048`, `seed=42`, `blocksize=128`
- **Pruned model:** `experiments/pruned/qwen3-1.7b-sparsegpt20`
- **Eval:** hellaswag + piqa, seed=42, batch_size=4, float16 (same tasks as baseline report)

Measured linear/Conv weight sparsity after prune: **~20%** (embeddings / norms / `lm_head` left dense).

## Baseline (from `reports/weight_level/baseline_runs_2026-06-06.md`)

Command: `make qwen-small` · dtype bfloat16 · full eval

| Task      | acc    | acc_norm |
|-----------|--------|----------|
| hellaswag | 0.4612 | 0.6038   |
| piqa      | 0.7258 | 0.7203   |

## SparseGPT 20%

Output: `experiments/baseline/experiments_pruned_qwen3-1.7b-sparsegpt20_2026-08-01T00-20-21.702659.json`

| Task      | acc    | acc_norm |
|-----------|--------|----------|
| hellaswag | 0.4594 | 0.6045   |
| piqa      | 0.7231 | 0.7165   |

## Delta (pruned − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0018 | **+0.0007** |
| piqa      | −0.0027 | −0.0038     |

## Takeaway

At **20%** unstructured SparseGPT on Qwen3-1.7B, hellaswag/piqa stay essentially at dense baseline (changes within ~0.4 pp). Good local smoke that SparseGPT is working end-to-end on the Weight-branch pipeline models.
