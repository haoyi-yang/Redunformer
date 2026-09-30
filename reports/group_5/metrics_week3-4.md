# Evaluation metrics — Group 5 (Neuron / FFN-unit)

**Weeks 3–4 deliverable for issue #3 — Proposal question 4 (metrics half).**
**Status:** reviewed (Sooraj Rathore, 2026-06-21) — outcome/internal metric wording and ε threshold agreed.

---

## Summary

Two metric families:

1. **Outcome metrics** — overall model performance before and after neuron masking, measured with WikiText-2 test perplexity and PIQA accuracy.
2. **Internal redundancy metrics** — statistics over FFN activations to *identify* redundant neurons (activation frequency, magnitude, correlation, ablation sensitivity).

The outcome metrics are already wired (`src/redundancy/eval.py::compute_perplexity` + `run_lm_eval`). The internal metrics will live under `src/redundancy/metrics/` (Week 5).

---

## Primary metric bundle

| Phase | Metric | Computed by | Reported how | Used for |
|-------|--------|-------------|--------------|----------|
| **Weeks 1–2 (done)** | **WikiText-2 perplexity** | `compute_perplexity` (sliding window, `max_length=2048`, `stride=512`, batch 1) | Single PPL number per (model, sparsity) | Outcome — LM baseline |
| **Weeks 3–4** | **PIQA accuracy / `acc_norm`** | `run_lm_eval` (lm-eval-harness) | Single accuracy per (model, sparsity) | Outcome — downstream sanity |
| **Weeks 5–8** | **Per-neuron activation frequency** *f*<sub>i,ℓ</sub> | Hook on post-SwiGLU activation; fraction of (seq, position) where \|act\| > ε | Per-layer histogram + threshold counts | Redundancy signal (lazy neurons) |
| **Weeks 5–8** | **Per-neuron activation magnitude** (mean \|act\|, RMS) | Same hook | Per-layer distribution | Complement to frequency |
| **Weeks 5–8** | **Pairwise neuron correlation** (Pearson on activations across calibration set) | Activation matrix → corr per layer | Top-correlated pairs table; correlation heatmap (selected layers) | Redundant-pair candidates |
| **Weeks 5–8** | **Ablation importance** ΔPPL when masking neuron set *S* | Mask + `compute_perplexity` on WikiText-2 calibration subset | ΔPPL per layer / per neuron-group | "Removability" ground truth |
| **Weeks 9–10** | **Performance vs. removal ratio curve** | Sweep masking 5/10/25/50% by **importance** (frequency kept as the second arm), re-eval PPL and PIQA | Plot: x = % masked, y = PPL or accuracy | Headline intervention result |
| **Weeks 9–10** | **Random-baseline gap** | Same removal ratio, neurons chosen uniformly at random (matched per layer) | Multi-curve plot (guided vs random) | Fair comparison |
| **Weeks 9–10** | **Duplicate-pair ablation** (H5) | Mask one twin of a near-duplicate pair vs matched-importance controls | Δ-PPL bar chart per condition | Does correlation imply removability? |
| **Weeks 11–12** | **Recovery PPL after short fine-tuning** | LoRA on a small WikiText slice with masks live → re-eval, **plus the same budget on an unmasked model** | PPL/accuracy before vs after recovery; recovery fraction against the control | Does redundancy stay redundant? |
| **Weeks 11–12** | **Replacement (merge) ΔPPL** | Fit `h_drop ≈ α·h_keep + β` on calibration activations, re-inject via hook | Δ-PPL bar chart: mask vs merge vs merge-without-bias | Can a twin absorb its duplicate? |
| **Weeks 11–12** | **Cross-task neuron agreement** (H3) | Measure on PIQA prompts, compare rankings to WikiText-2 | Per-layer Jaccard (vs chance floor) and Spearman by depth | Is redundancy task-specific? |
| **Weeks 11–12** | **Cross-task ablation** (H3) | Mask by each corpus's ranking, score both metrics | PPL + PIQA table per ranking at matched sparsity | Behavioural test of H3 |

---

## Metric details

### Outcome metrics

Overall model performance before and after neuron masking is measured with **WikiText-2 test perplexity** and **PIQA accuracy**. WikiText-2 perplexity is computed with a sliding-window setup using context length 2,048 and stride 512 (`compute_perplexity`). PIQA is reported with the standard `acc` and `acc_norm` metrics from `lm-evaluation-harness` (`run_lm_eval`).

### WikiText-2 perplexity

- **Definition:** \( \text{PPL} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N}\text{NLL}(x_i)\right) \), token-averaged.
- **Implementation:** `src/redundancy/eval.py::compute_perplexity` (sliding window with `prev_end` masking to avoid double-counting).
- **Reference number:** Qwen3.5-4B (4-bit) = **17.06** on full test split (297,053 tokens) — see `reports/group_5/baseline_week1-2.md`. Qwen3.5-9B = **12.90** (per #11, Henrik 2026-06-20).

### PIQA accuracy

- **Definition:** standard `lm-evaluation-harness` task `piqa`; we report `acc` and `acc_norm` (length-normalized).
- **Implementation:** `run_lm_eval(model_id, ["piqa"], limit=…, batch_size=1)`.
- **Calibration size:** PIQA `validation` has 1,838 items; we will run full validation for final numbers, `limit=200` for fast iteration.

### Activation frequency *f*<sub>i,ℓ</sub>

- **Definition:** fraction of (sequence, token) positions on the calibration set where neuron *i* in layer ℓ has \|activation\| > ε.
- **Threshold ε (agreed):** **RMS-relative per layer** as the primary setup — ε<sub>ℓ</sub> = α × RMS<sub>ℓ</sub> (e.g. α = 1e-3) so that "lazy" is not biased by layer scale. A **fixed ε** (e.g. 1e-3) is kept only as a sanity-check alternative.
- **Why:** directly implements the **Lazy Neuron** test (Li et al. 2022) — the histogram of *f* across neurons is heavy-tailed if there is exploitable redundancy.
- **Hook target:** post-SwiGLU activation tensor inside each FFN block, captured via `ActivationStore.register` (already in `src/redundancy/hooks.py`).

### Internal redundancy metrics (summary)

For each neuron *i* in layer ℓ, we compute activation frequency *f*<sub>i,ℓ</sub>, mean and RMS activation magnitude, pairwise correlation within each layer (top-*k* pairs), and ablation sensitivity as ΔPPL when masking a neuron set *S*. Our initial success criterion is that, when masking 10% of neurons per layer, ΔPPL stays below +1 and PIQA loss remains below 2 percentage points; the main quantity of interest is the performance gap between redundancy-guided and random masking at matched sparsity.

### Activation magnitude

- **Definition:** per-neuron mean of \|activation\| and RMS, over the same calibration set. Used together with frequency, because a neuron can fire often with tiny magnitude or rarely with huge magnitude.

### Pairwise correlation (later)

- **Definition:** Pearson correlation between neuron activation vectors across the calibration set, computed within a layer (cross-layer is too expensive on 8 GB). For redundancy we are interested in pairs with \|ρ\| > 0.9.
- **Memory note:** for a layer of width *d*, the corr matrix is *d × d*; we only keep the top-*k* pairs per layer, not the full matrix.

### Ablation importance

- **Definition:** ΔPPL = PPL(with neurons in *S* zeroed) − PPL(baseline).
- **Granularity:** report per-neuron (when affordable) and per-set (e.g. "bottom 10% by frequency in layer ℓ"). The set-level number is what feeds the removal-ratio curve.

### Performance-vs-removal curve

- **Sweep:** mask the bottom 5%, 10%, 25%, 50% of neurons (by *f*) per layer, re-evaluate WikiText-2 PPL **and** PIQA accuracy.
- **Random baseline:** for each ratio, sample the same count of neurons uniformly at random *per layer* (matched layer-wise sparsity is the apples-to-apples comparison — global random is too easy to beat).

---

## Success criteria (what counts as "acceptable")

These are first-pass thresholds for the Week 9–10 discussion — we can sharpen them once Marcel's measurement plan (#7) produces real numbers.

| Removal ratio | Acceptable WikiText-2 PPL increase vs baseline 17.06 | Acceptable PIQA accuracy drop |
|---------------|-----------------------------------------------------|-------------------------------|
| 10% | < +1 PPL (≈ +6 %) | < 2 percentage points |
| 25% | < +3 PPL (≈ +18 %) | < 5 percentage points |
| 50% | exploratory — large degradation expected | exploratory |

These are deliberately loose — the *interesting* number is not the absolute, it is the **gap between redundancy-guided and random** at the same ratio.

---

## What produces what (mapping to repo)

| Metric | Model / dataset | Script / config |
|--------|-----------------|-----------------|
| WikiText-2 PPL (4B) | Qwen3.5-4B + WikiText-2 test | `scripts/run_baseline.py --config configs/eval/baseline_full.yaml` |
| WikiText-2 PPL (9B) | Qwen3.5-9B + WikiText-2 test | same script with 9B model config (#11, Henrik) |
| PIQA accuracy | Qwen3.5-4B + PIQA validation | `run_lm_eval` (`src/redundancy/eval.py`) — config to add |
| Activation freq / magnitude / importance | Qwen3.5-4B or Qwen3-0.6B + WikiText-2 train | `scripts/run_measurement.py --config configs/measurement/neuron_activations.yaml` (or `neuron_activations_qwen3_0.6b.yaml`) |
| Pairwise correlation | same | `scripts/run_measurement.py` (auto layer selection by duplication score) |
| Ablation ΔPPL | same | `scripts/run_pruning.py --config configs/pruning/neuron_masking.yaml` |
| Removal-ratio curve | same | `scripts/run_pruning.py` → `plot_removal_curves` (`src/redundancy/plotting.py`) |
| Duplicate-pair ablation (H5) | same | `scripts/run_pair_ablation.py --config configs/pruning/pair_ablation.yaml` |
| Recovery after masking | Qwen3-0.6B / 4B + WikiText-2 train → test | `scripts/run_recovery.py --config configs/recovery/lora_qwen3_0.6b.yaml` |
| Duplicate replacement (merge) | Qwen3-0.6B + WikiText-2 | `scripts/run_merge.py --config configs/pruning/merge_qwen3_0.6b.yaml` |
| PIQA-prompt calibration (H3) | Qwen3-0.6B + PIQA train prompts | `scripts/run_measurement.py --config configs/measurement/neuron_activations_piqa_qwen3_0.6b.yaml` |
| Cross-task overlap (H3) | two measurement artifacts, same model | `scripts/run_task_overlap.py --config configs/measurement/task_overlap_qwen3_0.6b.yaml` |
| Cross-task ablation (H3) | Qwen3-0.6B + PIQA-ranked artifact | `scripts/run_pruning.py --config configs/pruning/neuron_masking_piqa_ranked_qwen3_0.6b.yaml` |

---

## Deliverable checklist (issue #3)

- [x] Final metric list (primary + secondary) with formulas / references
- [x] Each metric mapped to model / dataset / script
- [x] First-pass success criteria for acceptable degradation
- [x] Fed into one-page proposal (#10) — see `proposal_week3-4.md` → Q4 / Q6 / Q7
- [x] Reviewed by assignee / group (Sooraj Rathore, 2026-06-21)

## References

- Seminar §2.3, Weeks 5–12
- Li et al. 2022 — *The Lazy Neuron Phenomenon*
- Geva et al. 2021 — *FFN as Key-Value Memories*
- Ma et al. 2023 — *LLM-Pruner*
- Marcel's measurement / intervention plan in #7 (2026-06-17)
- `src/redundancy/eval.py`, `src/redundancy/hooks.py`, `configs/eval/baseline_full.yaml`
