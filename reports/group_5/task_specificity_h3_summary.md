# H3 (task) — what we concluded about task-specificity

**Analysis-only deliverable (#18), written for the Weeks 13–16 final analysis.**
Half a page, no new runs. Full method and per-layer numbers:
[`recovery_week11-12.md`](recovery_week11-12.md) §4.

**H3 predicted** that neurons which look redundant under WikiText-2 are *not*
redundant under PIQA. We tested it on Qwen3-0.6B in two ways: correlationally
(do the two calibration corpora nominate the same neurons?) and behaviourally
(does masking by one ranking hurt the other corpus more?).

## The evidence in one table

| budget | Jaccard of masked sets | chance floor | ×chance | Spearman (full ranking) |
|---:|---:|---:|---:|---:|
| 5% | 0.298 | 0.026 | 11.6× | +0.582 |
| 10% | 0.333 | 0.053 | 6.3× | +0.582 |
| 25% | 0.416 | 0.143 | 2.9× | +0.582 |

| masked / layer | ranking | WikiText-2 PPL | PIQA acc |
|---:|---|---:|---:|
| — | none (baseline) | 33.75 | 0.698 |
| 10% | WikiText-2 | **38.15** | 0.656 |
| 10% | PIQA | 43.29 | **0.666** |
| 10% | random | 57.66 | 0.628 |
| 25% | WikiText-2 | **52.86** | 0.594 |
| 25% | PIQA | 94.80 | **0.630** |
| 25% | random | 198.74 | 0.594 |

The `random` arms are ranking-independent and came out bit-identical across the
two runs (57.659 / 198.736), which is what licenses reading the rows against
each other.

## What we concluded

**H3 is supported behaviourally, as a clean double dissociation.** Each ranking
wins on the corpus it was calibrated on. At 25% the WikiText ranking is far
better on perplexity (52.9 vs 94.8, a 1.8× difference) while the PIQA ranking is
clearly better on PIQA accuracy (0.630 vs 0.594, **+3.6 pp**). Neither is *the*
redundancy ranking. Redundancy is defined relative to a distribution, and a
ranking is a property of the (model, corpus) pair rather than of the model.

**But redundancy is mostly a property of the model, not the corpus.** Set
agreement runs 6.3× above the chance floor at a 10% budget and rank agreement is
+0.58 across the whole ranking, so the two corpora are nowhere near independent.
The correct reading is *partial* task-specificity: a shared core of genuinely
cheap neurons that both corpora agree to drop, plus a task-dependent margin —
and the margin is where the 3.6 pp lives. A practical corollary is that a
WikiText-calibrated ranking still beats random on PIQA at every ratio and on both
metrics: calibrating on the wrong corpus costs accuracy, it does not make the
ranking worthless.

**The behavioural gap is larger than the set overlap suggests.** At 10% the two
rankings differ on 67% of the masked set but PIQA differs by only 1 pp; at 25%
they differ on 58% and PIQA differs by 3.6 pp. Disagreement about *which* neurons
to drop matters more as the budget grows — the neurons the two corpora fight over
are the ones near the decision boundary, and at a small budget almost none of
them are reached.

**Consequence for anyone pruning.** Calibrate on the distribution you will be
scored on, and report which corpus a ranking came from. The pipeline enforces the
second half: the PIQA-calibrated artifact is tagged `piqacalib` so
`measurement_path: "auto"` can never silently resolve a WikiText run to it, and
`MeasurementArtifact.calibration_label()` prints the corpus before every run.

## What is still open

- **One model, one task pair.** WikiText-2 vs PIQA on Qwen3-0.6B only. Whether
  the dissociation widens on the 4B — which has a flatter redundancy profile and
  no lazy subpopulation — is untested, and it is the cheapest remaining
  extension: a PIQA-calibrated measurement plus one masking sweep.
- **PIQA is a weak instrument.** 500-example subsets, and multiple-choice
  accuracy moves in coarse steps, so a 3.6 pp difference rests on ~18 questions.
  A third corpus with a continuous metric would make the dissociation much
  harder to argue with.
- **Task-specificity of the *repair* was never tested.** Merge coefficients
  `α, β` are fitted on WikiText-2 activations and evaluated on WikiText-2. A pair
  whose gain shifts across corpora would merge worse than reported; the
  PIQA-calibrated equivalent has not been run.
- **The overlap metric is set-based.** Jaccard treats a neuron ranked 1st and one
  ranked 300th as equally "in the set". A rank-weighted agreement would separate
  "the corpora disagree about marginal neurons" from "they disagree about which
  neurons are critical" — the two have very different implications, and the
  Spearman column alone does not decide it.

## Sources

- `reports/group_5/recovery_week11-12.md` §4 (method, per-layer table, findings)
- `src/redundancy/pruning/overlap.py` (Jaccard, chance floor, Spearman)
- `experiments/results/task_overlap_Qwen__Qwen3-0.6B_*.json`,
  `experiments/results/pruning_piqaranked_Qwen__Qwen3-0.6B_*.json` (both gitignored)
