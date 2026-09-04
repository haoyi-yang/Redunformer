# SparseGPT 20% on Qwen3-4B — mlsp4

Date: 2026-09-04 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** fresh `Qwen/Qwen3-4B` from Hugging Face
- **Pruner:** SparseGPT unstructured, sparsity **0.2**
- **Calibration:** wikitext2, `nsamples=128`, `seqlen=2048`, `blocksize=128`, `percdamp=0.01`, `seed=42`
- **Pruned model:** `experiments/pruned/qwen3-4b-sparsegpt20`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=4`, device_map=auto
- **Machine:** mlsp4 (`130.83.166.154`), podman + NVIDIA lib bind-mount workaround
- **Prune log:** `experiments/logs/sparsegpt20-qwen3-4b-20260904-083319.log`
- **Eval log:** `experiments/logs/sparsegpt20-eval-when-free-20260904-174717.log`

Prune finished earlier the same day; first immediate lm-eval OOM’d (SparseGPT parent still held CUDA). Eval was restarted from scratch after GPU free (watcher), not resumed mid-run.

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## SparseGPT 20%

Output: `experiments/baseline/experiments_pruned_qwen3-4b-sparsegpt20_2026-09-04T18-11-02.905805.json`

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6843   | 10042 |
| piqa      | 0.7486 | 0.7514   | 1838  |
| arc_easy  | 0.8026 | 0.7790   | 2376  |

## Delta (sparsegpt20 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | +0.0000 | −0.0001    |
| piqa      | −0.0006 | +0.0038    |
| arc_easy  | −0.0034 | −0.0059    |

## Reference: Magnitude / Random 20% (same model / tasks)

| Method        | hellaswag acc / norm | piqa acc / norm | arc_easy acc / norm |
|---------------|----------------------|-----------------|---------------------|
| magnitude20   | 0.5153 / 0.6833      | 0.7470 / 0.7465 | 0.8005 / 0.7630     |
| sparsegpt20   | 0.5214 / 0.6843      | 0.7486 / 0.7514 | 0.8026 / 0.7790     |
| random20      | 0.2551 / 0.2649      | 0.5185 / 0.5071 | 0.2513 / 0.2492     |

## Takeaway

At **20%** SparseGPT on Qwen3-4B, scores stay at dense baseline (≤0.6 pp on acc; piqa `acc_norm` slightly **up**). SparseGPT20 matches or slightly beats magnitude20 on these tasks and far exceeds random20, which collapses toward chance.
