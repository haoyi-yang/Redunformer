# Weeks 11–12 — Recovery, Neuron Replacement, and the Task Dimension

**Group 5 deliverable: the three things Weeks 9–10 left open.**
**Status:** implementation complete and tested (47 offline tests); GPU runs done
on 2026-07-30 on Qwen3-0.6B, and the 4-bit Qwen3.5-4B QLoRA arm on **2026-08-12
in Colab** (Marcel, #16) — see §5. Artifacts: `experiments/results/recovery_*.json`,
`merge_*.json`, `task_overlap_*.json`, `pruning_piqaranked_*.json`, and
`experiments/results/figures/`.

Weeks 9–10 ended with three unanswered questions. This stage answers each with
its own experiment:

| # | Question left open | Experiment | Script |
|---|---|---|---|
| 1 | Is the perplexity cost of masking permanent? | short LoRA fine-tuning with masks live | `run_recovery.py` |
| 2 | Can a near-duplicate's twin *absorb* it instead of losing it? | fit `h_drop ≈ α·h_keep + β` and route the contribution | `run_merge.py` |
| 3 | Is redundancy task-specific (H3)? | PIQA-calibrated ranking vs WikiText-2 ranking | `run_task_overlap.py`, `run_pruning.py` |

Headline: **on 0.6B, masking damage is ~80% repairable**; **on 4B, this QLoRA
budget overfits and leaves the gap intact (~2% adj.)**; **duplicate pairs are
~90% repairable without training**; **H3 holds behaviourally**.

## 1. What is new in the code

```
src/redundancy/recovery.py            LoRA config, training loop, recovery accounting
src/redundancy/pruning/merge.py       fit + apply neuron replacement (hook-based)
src/redundancy/pruning/overlap.py     cross-task set/rank agreement with a chance floor
src/redundancy/data.py                text_builder: PIQA prompts as calibration text
scripts/run_recovery.py               masked -> LoRA -> re-eval, with an unmasked control
scripts/run_merge.py                  mask_only vs merge vs merge_no_bias
scripts/run_task_overlap.py           per-layer Jaccard / Spearman between two artifacts
tests/test_recovery.py                24 offline tests, no GPU and no downloads
```

Two design decisions carry most of the weight.

**Replacement is applied as a hook, not a weight edit.** For a pair (drop `i`,
keep `j`), masking `i` removes `h_i · W[:, i]` from the FFN output. Substituting
the fitted `h_i ≈ α·h_j + β` gives back `(α·h_j + β) · W[:, i]`, which is
algebraically the same as setting `W[:, j] ← W[:, j] + α·W[:, i]` plus a
constant. Adding it in a forward hook instead of rewriting the weight keeps the
"masking, not surgery" contract from the proposal (Q1/Q8) *and* works on the
4-bit model, where a merged column would have to be re-quantized.

**LoRA does not target `down_proj`.** A masked channel feeds `down_proj` zeros,
so its columns receive no gradient and cannot be repaired there anyway. Leaving
`down_proj` unwrapped also keeps the mask hooks attached to plain `Linear`
modules. Recovery therefore happens in the projections that *feed* the surviving
neurons. `NeuronMasker` gained suffix-tolerant layer-name resolution so a
measurement artifact recorded on the bare model still matches after PEFT
re-prefixes every module path.

## 2. Experiment 1 — does fine-tuning undo the damage?

**Setup.** Qwen3-0.6B. Masking by `importance` (the ranking that won Weeks
9–10), at 10% and 25% per layer. LoRA rank 16 on
`q,k,v,o,gate,up`, 200 steps, effective batch 8 × 512 tokens, lr 2e-4 cosine,
trainable 8.3 M / 604 M parameters (1.37%). Fine-tuning text is WikiText-2
**train**; evaluation is WikiText-2 **test** (256 rows) plus PIQA (`lm_eval`,
limit 500). Masks stay active during training and evaluation.

**The control is the point.** 200 LoRA steps on the *unmasked* model take
perplexity from 33.75 to **23.08** — a 10.7-point gain that has nothing to do
with recovery and everything to do with adapting to WikiText-2. Any recovery
claim measured against the untrained baseline is inflated by that amount, so two
numbers are reported:

* **raw recovery** — share of the damage undone relative to the untrained baseline.
* **adjusted recovery** — the honest one: compares masked+LoRA against
  unmasked+LoRA at the same adapter budget, i.e. `1 − (recovered − control) / damage`.

### Qwen3-0.6B — WikiText-2 perplexity (untrained baseline **33.75**, unmasked+LoRA control **23.08**)

| masked / layer | masked | masked + LoRA | residual vs control | raw rec. | **adj. rec.** |
|---:|---:|---:|---:|---:|---:|
| 10% | 38.15 (+4.40) | 24.15 | **+1.07** | 318% | **75.8%** |
| 25% | 52.86 (+19.11) | 26.33 | **+3.25** | 139% | **83.0%** |

### Qwen3-0.6B — PIQA accuracy (baseline **0.698**, unmasked+LoRA control **0.694**)

| masked / layer | masked | masked + LoRA | residual vs control |
|---:|---:|---:|---:|
| 10% | 0.656 | 0.672 | −2.2 pp |
| 25% | 0.594 | 0.652 | −4.2 pp |

(`acc_norm`: baseline 0.692, control 0.686; 10% recovers 0.674 → **0.692**, i.e.
back to baseline; 25% recovers 0.598 → 0.638.)

### Findings

1. **Most of the masking damage is recoverable.** At equal adapter budget the
   masked model ends up only **+1.07 PPL** behind the unmasked one at 10%
   masking, down from +4.40 untrained — **76% of the damage absorbed**. At 25%
   the absolute residual is larger (+3.25) but the *share* recovered is higher
   (**83%**), because there is more headroom to reclaim. Adapters routing around
   the removed channels is enough; the capacity was genuinely redundant.
2. **Raw recovery numbers are meaningless without the control.** The raw figure
   at 10% is **318%** — masked+LoRA (24.15) beats the untrained unmasked model
   (33.75) outright. Reporting that as "recovery" would be indefensible: the
   unmasked control shows most of it is in-domain adaptation. This is the single
   biggest methodological trap in this stage and the reason the control is on by
   default (`train_unmasked_control: true`).
3. **Recovery helps PIQA too, and disproportionately at high sparsity.** PIQA
   accuracy at 25% masking goes 0.594 → 0.652, recovering most of the gap to the
   0.694 control even though the adapters never saw a PIQA prompt. At 10%,
   `acc_norm` returns exactly to the 0.692 baseline.
4. **WikiText LoRA slightly *hurts* PIQA.** The control lands at 0.694/0.686
   versus a 0.698/0.692 baseline. Small, but it means perplexity gains from
   in-domain tuning do not transfer to downstream accuracy — the same
   PPL/accuracy divergence seen in the Weeks 9–10 ranking comparison.
5. **The Q4 criteria are nearly met once recovery is allowed.** The proposal
   asked for ΔPPL < +1 and PIQA drop < 2 pp at 10% masking. Against the control:
   **+1.07 PPL and −2.2 pp** — just outside both, versus +4.40 PPL and −4.2 pp
   without recovery. Masking plus a 200-step adapter is close to the promised
   operating point; it does not quite reach it.

## 3. Experiment 2 — neuron replacement instead of removal

**Setup.** Qwen3-0.6B, the *same* duplicate pairs, eval slice, and seed as the
Weeks 9–10 pair ablation, so the arms are directly comparable. `α, β` are fitted
per pair by least squares on 32 calibration blocks (~16 k token positions) of
WikiText-2 train — only the paired neurons' columns are captured, so the pass is
cheap. `r²` of each fit bounds how much a merge can possibly recover.

**Sanity check:** `mask_only` reproduces the Weeks 9–10 `duplicate_one` numbers
exactly (+2.2409 vs +2.241 all-layers; +1.8395 vs +1.840 for layer 2; −0.1894 vs
−0.189 for layer 27), and the baseline matches at 38.3869. The two studies are
measuring the same thing.

### Results — Δ perplexity vs unmasked (baseline **38.39**)

| Condition | all layers (14 pairs) | layer 2 (5 pairs) | layer 27 (7 pairs) |
|-----------|---:|---:|---:|
| `mask_only` (Weeks 9–10) | +2.2409 | +1.8395 | −0.1894 |
| **`merge`** | **+0.2838** | **+0.1359** | +0.1415 |
| `merge_no_bias` | +0.3001 | +0.1398 | +0.1416 |
| *mean fit r²* | 0.892 | 0.895 | 0.965 |
| *masking cost recovered* | **87.3%** | **92.6%** | n/a (masking was free) |

### Findings

1. **Replacement works, and it needs no training.** Merging recovers **87%** of
   the masking cost across all pairs and **93%** in layer 2 — the massive-activation
   layer whose pairs were the expensive ones. Two fitted scalars per pair buy
   most of what 200 LoRA steps buy, at essentially zero cost.
2. **This resolves the Weeks 9–10 puzzle.** That stage found layer-2 pairs to be
   *jointly critical but individually redundant*: one twin costs +1.8 PPL, both
   cost +2898. The prediction was that the survivor could carry the pair if
   properly rescaled. It can: +1.84 → **+0.14**. The pair really does encode one
   signal twice, and the network only needs it once at the right gain.
3. **The scale factor is what matters, not the constant.** `merge_no_bias` is
   within 0.02 PPL of the full merge everywhere, so the `β` term contributes
   almost nothing on real activations. The fitted `α` values are far from 1
   (layer 2: 0.13–0.80, mean |α| 0.36), which is exactly why plain masking fails
   and why `α` has to be fitted rather than read off the correlation — correlation
   fixes the shape of the relationship but not the gain.
4. **Correlation alone does not predict merge quality.** Layer 27's fits are the
   *best* (mean r² 0.965, |ρ| up to 0.997) yet merging there is pointless: those
   neurons were already free to drop, so re-injecting their contribution moves
   perplexity the wrong way (−0.19 → +0.14). Merge where masking hurts; drop by
   importance where it does not.
5. **Where the residual 7–13% lives.** The two lowest-r² pairs in layer 2 (0.714
   and 0.777) are the ones a scalar map cannot fully express. A richer
   substitution — regressing the dropped neuron on several survivors — is the
   obvious next step, and beyond this stage's scope.

## 4. Experiment 3 — is redundancy task-specific? (H3)

H3 predicted that neurons which look redundant under WikiText-2 are **not**
redundant under PIQA. Testing it needs a PIQA-derived ranking, so PIQA train
rows are rendered as `Question: {goal}\nAnswer: {correct solution}` — the same
prompt shape `lm_eval` scores — and pushed through the unchanged Weeks 5–8
measurement pipeline. The artifact is tagged `piqacalib` so it can never be
picked up by `measurement_path: "auto"` in a WikiText run.

### 4a. Do the two corpora nominate the same neurons? (all 28 layers)

| budget | Jaccard (masked set) | chance floor | ×chance | Jaccard (most active) | Spearman |
|---:|---:|---:|---:|---:|---:|
| 5% | 0.298 | 0.026 | 11.6× | 0.283 | +0.582 |
| 10% | 0.333 | 0.053 | 6.3× | 0.315 | +0.582 |
| 25% | 0.416 | 0.143 | 2.9× | 0.392 | +0.582 |

Per-layer agreement at 10% ranges from 0.233 (layer 11) to 0.438 (layer 18).

### 4b. Does the disagreement change behaviour?

Masking by each ranking at matched sparsity and scoring both metrics. The
`random` arms are ranking-independent and come out **bit-identical** across the
two runs (57.659 / 198.736 PPL), confirming the runs are comparable.

| masked / layer | ranking | WikiText-2 PPL | PIQA acc | PIQA acc_norm |
|---:|---|---:|---:|---:|
| — | none (baseline) | 33.75 | 0.698 | 0.692 |
| 10% | WikiText-2 | **38.15** | 0.656 | **0.674** |
| 10% | PIQA | 43.29 | **0.666** | 0.666 |
| 10% | random | 57.66 | 0.628 | 0.638 |
| 25% | WikiText-2 | **52.86** | 0.594 | 0.598 |
| 25% | PIQA | 94.80 | **0.630** | **0.632** |
| 25% | random | 198.74 | 0.594 | 0.566 |

### Findings

1. **H3 is supported behaviourally — a clean double dissociation.** Each ranking
   wins on the corpus it was calibrated on. At 25%, the WikiText ranking is far
   better on perplexity (52.9 vs 94.8) while the PIQA ranking is clearly better
   on PIQA accuracy (0.630 vs 0.594, **+3.6 pp**). Neither is "the" redundancy
   ranking; redundancy is defined relative to a distribution.
2. **But redundancy is mostly a property of the model, not the corpus.** Set
   agreement is 6.3× chance at a 10% budget and rank agreement is +0.58, so the
   two corpora are far from independent. The correct reading is *partial*
   task-specificity: a shared core of genuinely cheap neurons, plus a
   task-dependent margin — and the margin is where the 3.6 pp lives.
3. **The behavioural gap is bigger than the set overlap suggests.** At 10% the
   rankings differ on 67% of the masked set yet the PIQA difference is only
   1 pp; at 25% they differ on 58% and the difference is 3.6 pp. Disagreement
   about *which* neurons to drop matters more as the budget grows.
4. **A WikiText-calibrated ranking still beats random on PIQA**, at every ratio
   and on both metrics. Calibrating on the wrong corpus costs accuracy; it does
   not make the ranking worthless.

> **Write-up version (#18).** The half-page H3 summary for the final analysis,
> with the open questions this stage leaves, is
> [`task_specificity_h3_summary.md`](task_specificity_h3_summary.md).

## 5. QLoRA recovery on the 4B

The laptop 8 GB card could not finish a 4-bit backward pass (training sat at
7.8 / 8.2 GB and never completed an optimizer step). Marcel ran the Colab
config on **2026-08-12** (`#16`): LoRA rank 16, 100 steps, 128 blocks × 512
tokens, batch 1, PIQA limit 200, unmasked control on. Untrained arms match
Weeks 9–10 to three decimals, so this is the same masking as the intervention
sweep, now with adapters.

Source: `experiments/results/recovery_colab_Qwen_Qwen3.5-4B_20250812T115950Z.json`
(gitignored; provenance copy in
`reports/group_5/artifacts/recovery_colab_Qwen__Qwen3.5-4B_20250812T115950Z.json`).

### Qwen3.5-4B (4-bit) — WikiText-2 perplexity

| arm | PPL | vs untrained baseline |
|---|---:|---:|
| unmasked, untrained | **18.877** | — |
| `importance` @ 10% masked, untrained | 20.708 | **+1.831** |
| masked + QLoRA | 27.587 | +8.710 |
| unmasked + QLoRA (control) | 25.799 | +6.922 |

| ratio | masked | +QLoRA | vs control | raw rec. | **adj. rec.** |
|---:|---:|---:|---:|---:|---:|
| 10% | 20.708 | 27.587 | **+1.788** | −376% | **2.3%** |

### Findings

1. **This LoRA budget overfits the 4B, masked or not.** Train loss fell, eval
   PPL rose by ~7 points on *both* arms. That is the opposite of the 0.6B run,
   where the same recipe *improved* eval PPL. The control is doing its job:
   without it, 27.59 would look like “masking became unrecoverable”; with it,
   the rise is mostly in-domain overfit, not a failure of the mask.
2. **Adjusted recovery is ~0.** After the same adapter budget the masked model
   is still **+1.79 PPL** behind the unmasked one, versus **+1.83** before
   training — 2% of the original damage closed. On this hyper-parameter setting,
   QLoRA does not route around the removed 4B neurons the way it did on 0.6B
   (76–83% there).
3. **H2 for recovery is therefore a split, not a blank.** Short LoRA undoes
   most of the 0.6B masking cost and almost none of the 4B cost *under the
   budget we could actually run*. A milder schedule (fewer steps / dropout /
   earlier stop) might still recover the 4B; that is a hyper-parameter question,
   not a proof that 4B masking is permanent.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geeeeenccc/Redunformer-Neuron/blob/main/notebooks/group_5/week11_12_qlora_4b_colab.ipynb)

## 6. How to run

```powershell
# 1. Recovery (needs a measurement artifact; ~30 min on a 0.6B, 8 GB GPU)
uv run python scripts/run_recovery.py --config configs/recovery/lora_qwen3_0.6b.yaml
uv run python scripts/run_recovery.py --config configs/recovery/lora_smoke.yaml      # fast wiring check
uv run python scripts/run_recovery.py --config configs/recovery/lora_smoke_4b.yaml   # fast 4-bit check
# 4-bit full run: laptop 8 GB cannot finish training; Colab numbers are in §5
uv run python scripts/run_recovery.py --config configs/recovery/lora_qwen3.5-4b.yaml

# 2. Neuron replacement, all pairs or one layer at a time
uv run python scripts/run_merge.py --config configs/pruning/merge_qwen3_0.6b.yaml
uv run python scripts/run_merge.py --config configs/pruning/merge_qwen3_0.6b.yaml --layers 2 --tag L2

# 3. Task dimension (H3): measure on PIQA prompts, then compare and mask by it
uv run python scripts/run_measurement.py --config configs/measurement/neuron_activations_piqa_qwen3_0.6b.yaml
uv run python scripts/run_task_overlap.py --config configs/measurement/task_overlap_qwen3_0.6b.yaml `
    --reference experiments/results/measurement_Qwen__Qwen3-0.6B_<stamp>.json `
    --comparison experiments/results/measurement_piqacalib_Qwen__Qwen3-0.6B_<stamp>.json
uv run python scripts/run_pruning.py --config configs/pruning/neuron_masking_piqa_ranked_qwen3_0.6b.yaml --tag piqaranked
```

Offline tests (no GPU, no downloads): `uv run pytest tests/`.

## 7. Hypothesis scoreboard after this stage

> **Superseded by [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md).** This
> table is what we believed on 2026-07-30, from Qwen alone. Since then H1 was
> refuted by the depth-tertile ablation, H2 was refuted by the GPT-2 size arm,
> H3 turned out to hold only on gated FFNs, and the frequency half of H4 became
> its own hypothesis (H7). The replacement result below stands and is now named
> **H5′**. Kept as the record of this stage.

| Hypothesis | Verdict *(as of this stage)* | Now |
|---|---|---|
| H1 (depth) | supported (Weeks 5–8 / 9–10) | **refuted** |
| H2 (size) | partial — larger model degrades less relative to itself; **recovery does not transfer** (0.6B 76–83% vs 4B ~2% adj., §5) | **refuted as stated**; the recovery split is now **H8** |
| **H3 (task)** | **supported** — double dissociation in §4b, though overlap is 6.3× chance | supported on **gated FFNs only** |
| H4 (component) | supported on PPL, mixed on PIQA (Weeks 9–10) | supported; frequency split out as **H7** |
| H5 (duplication) | rejected as stated in Weeks 9–10, **but the follow-up prediction it generated is confirmed**: duplicate pairs are ~93% repairable by replacement (§3) | H5 **not supported**; the replacement result is promoted to **H5′**, supported |
| Q1 item 4 (recoverable) | **supported on 0.6B**, **not on 4B** under this LoRA budget — 4B QLoRA overfits and leaves the +1.83 masking gap intact (§5) | **H8**, split — also ~80–87% on GPT-2 124M/355M/774M |

## 8. Limitations

- **Masking ≠ removal, still.** Shapes are unchanged and the merge is a forward
  hook, so no speed, memory, or FLOP claim is made anywhere in this report. A
  structural version of the merge (fold `α·W[:, i]` into `W[:, j]`, delete the
  column) is a straightforward follow-up on an unquantized model.
- **Recovery is measured on the fine-tuning domain.** Adapters trained on
  WikiText-2 train and evaluated on WikiText-2 test will flatter recovery. The
  unmasked control bounds this, and PIQA (never trained on) is the out-of-domain
  check — but a third corpus would be better.
- **One LoRA budget, one hyper-parameter setting.** 200 steps at rank 16 on
  0.6B; 100 steps at rank 16 on 4B. The 4B run overfit, so "76–83% recoverable"
  is a 0.6B point estimate, not a ceiling, and **not** a 4B result.
- **Merge coefficients are fitted on WikiText-2 activations** and evaluated on
  WikiText-2 test. A pair whose `α` shifts across corpora would merge worse than
  reported here; the PIQA-calibrated equivalent was not run.
- **Scalar substitution only.** One survivor per dropped neuron, one gain, one
  offset. The two low-r² layer-2 pairs show the limit of that model.
- **Pair inventory is not exhaustive.** Correlation pairs still come only from
  the three measurement drill-down layers and from a reservoir subsample.
- **PIQA limits.** 500 examples for the 0.6B runs; 200 for the 4B Colab recovery.
  Absolute accuracies are not full-validation estimates; within-run comparisons
  hold. 4B recovery PIQA numbers were not copied into this report from the
  Colab JSON.
- **4B recovery overfits this schedule.** The masking gap survived QLoRA
  (~2% adjusted recovery). A milder recipe is untested.
- **H3 tested on one task pair and one model.** WikiText-2 vs PIQA on the 0.6B.
  Whether the double dissociation widens on the 4B is untested.
- **Replacement tested on the 0.6B only.** The merge is 4-bit-safe by
  construction (it never touches stored weights), but the 4B artifact has only
  four pairs above |ρ| ≥ 0.9, and Weeks 9–10 showed those are already free to
  drop — so there was nothing there for a merge to recover.
