# Random 20% on Qwen3-4B

Date: 2026-06-16 (eval JSON timestamp `2026-06-16T22:06:02`)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** random unstructured, sparsity **0.2** (20% weights zeroed at random, seed=42)
- **Module:** `src/redundancy/pruning/random_pruning.py` / `scripts/run_random_pruning.py`
- **Pruned model:** `experiments/pruned/qwen3-4b-random20`
- **Eval:** hellaswag + piqa + arc_easy, seed=42, dtype bfloat16, `batch_size=auto` → 9, device_map=auto
- **Runtime:** ~6792 s (~1.9 h) lm-eval
- **Raw output:** `experiments/random_pruning/experiments_pruned_qwen3-4b-random20_2026-06-16T22-06-02.715263.json`

## Baseline (dense Qwen3-4B)

From `reports/weight_level/baseline_runs_2026-06-06.md` · mlsp4 · seed=42 · full eval

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.5214 | 0.6844   | 10042 |
| piqa      | 0.7492 | 0.7476   | 1838  |
| arc_easy  | 0.8060 | 0.7849   | 2376  |

## Random 20%

| Task      | acc    | acc_norm | n     |
|-----------|--------|----------|-------|
| hellaswag | 0.2551 | 0.2649   | 10042 |
| piqa      | 0.5185 | 0.5071   | 1838  |
| arc_easy  | 0.2513 | 0.2492   | 2376  |

## Delta (random20 − dense)

| Task      | Δ acc   | Δ acc_norm |
|-----------|---------|------------|
| hellaswag | −0.2663 | −0.4195    |
| piqa      | −0.2307 | −0.2405    |
| arc_easy  | −0.5547 | −0.5357    |

## Takeaway

At **20%** random pruning on Qwen3-4B, accuracy collapses: hellaswag/arc_easy fall near chance (~0.25), piqa roughly halves. Random is a weak baseline — magnitude/SparseGPT at the same sparsity should be compared against this floor, not treated as interchangeable with informed pruning.
