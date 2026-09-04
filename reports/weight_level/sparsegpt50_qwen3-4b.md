# SparseGPT 50% on Qwen3-4B — mlsp2

Date: 2026-09-04 (mlsp2, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** SparseGPT unstructured, sparsity **0.5**
- **Calibration:** wikitext2, `nsamples=128`, `seqlen=2048`, `blocksize=128`, `percdamp=0.01`, `seed=42`
- **Pruned model:** `experiments/pruned/qwen3-4b-sparsegpt50`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → **13**, device_map=auto
- **Machine:** mlsp2 (`130.83.166.152`), podman + NVIDIA lib bind-mount workaround
- **Prune log:** `experiments/logs/sparsegpt50-qwen3-4b-20260904-174439.log`
- **Eval log:** `experiments/logs/sparsegpt50-eval-auto-20260904-182932.log`

Prune ran with `--skip-eval`; lm-eval was a separate step (first attempt used `batch_size=4`, then restarted with `auto`).

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## SparseGPT 50%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-sparsegpt50_2026-09-04T18-35-09.908963.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.4648 | 0.6132   | 10042 |
| piqa      | 0.7203 | 0.7307   | 1838  |
| arc_easy  | 0.7462 | 0.7277   | 2376  |

## Delta (sparsegpt50 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.0566 | −0.0712    |
| piqa      | −0.0289 | −0.0169    |
| arc_easy  | −0.0598 | −0.0572    |

## Reference: SparseGPT 20% (same model / tasks)

From `reports/weight_level/sparsegpt20_qwen3-4b.md`

| Task      | sparsegpt20 acc / norm | sparsegpt50 acc / norm |
|-----------|------------------------|------------------------|
| hellaswag | 0.5214 / 0.6843        | 0.4648 / 0.6132        |
| piqa      | 0.7486 / 0.7514        | 0.7203 / 0.7307        |
| arc_easy  | 0.8026 / 0.7790        | 0.7462 / 0.7277        |

## Takeaway

At **50%** SparseGPT, quality drops clearly vs dense and vs SparseGPT20: hellaswag ≈−5.7 pp acc / −7.1 pp acc_norm, arc_easy ≈−6.0 / −5.7 pp, piqa milder (≈−2.9 / −1.7 pp). Still far above random-collapse levels seen at 20% random pruning; usable as a mid-sparsity SparseGPT point on Qwen3-4B once magnitude50 / random50 finish for head-to-head.
