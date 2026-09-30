# Weeks 9–10 — Neuron Masking Intervention and Evaluation

**Group 5 deliverable: the intervention promised in the proposal (Q7), plus the
duplicate-neuron experiment added after review.**
**Status:** implementation complete; GPU runs done on 2026-07-26 on Qwen3-0.6B
and Qwen3.5-4B, with a **2026-07-29 polish pass** adding primary-model PIQA and
4B pair ablation. See `experiments/results/pruning_*.json`,
`experiments/results/pair_ablation_*.json`, and `experiments/results/figures/`.

> **Errata (2026-09-09).** §3's H5 discussion argues that the failure was
> largely a badly matched control and that "the mechanism replicates". Both
> claims are now retired. Llama-2-7B provides a genuinely matched control
> (0.94×) and H5 still fails there, and across all six models the verdict splits
> 3–3 with noise-floor margins — so the effect has **no consistent sign**. The
> within-pair asymmetry in §3 finding 2 is also no longer offered as evidence,
> because the procedure masks the lower-importance twin first and so confounds
> "duplicate" with "less important". The confound-free version is the
> replacement experiment (**H5′**), which does hold. Frozen verdicts:
> [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md).

This stage turns the Weeks 5–8 *measurement* into an *ablation*. Two experiments:

1. **Removal-ratio sweep** — mask 5/10/25/50% of neurons per layer under three
   rankings and compare against random at matched sparsity (proposal Q7).
2. **Duplicate-neuron ablation (H5)** — mask one twin of a near-duplicate pair
   and compare against matched-importance controls. This closes the gap Haoyi
   raised in review: the proposal *measured* duplication but never tested it.

---

## 1. What the intervention does

We **mask**, we do not structurally remove. A neuron is one post-SwiGLU
activation channel, and it reaches the residual stream only through `down_proj`.
Zeroing that channel on the way into `down_proj`
(`src/redundancy/pruning/masks.py::NeuronMasker`, a forward pre-hook) therefore
removes the neuron's entire contribution while leaving every parameter shape
untouched. Consequently **no runtime or FLOP claim is made** (proposal Q1/Q8).

Ratios are applied **per layer**, not globally. A global ranking would
concentrate removals in whichever layer happens to have the smallest activation
scale, and the random baseline would no longer be an apples-to-apples control.

### Ranking strategies

| Strategy | Neuron score (lowest masked first) | Role |
|----------|-----------------------------------|------|
| `importance` | `RMS(h_i) · ‖W_down[:, i]‖₂` | primary, from the Weeks 5–8 revision |
| `frequency` | RMS-relative firing frequency `f_i` | the originally-planned *Lazy Neuron* ranking (Q6/Q7) |
| `magnitude` | `E[|a_i|]` | available, not used in the headline runs |
| `random` | uniform sample at the same per-layer count | matched-sparsity baseline |

## 2. Experiment 1 — removal-ratio sweep

**Setup.** WikiText-2 test, sliding window (context 2048, stride 512). 0.6B:
157 documents + PIQA (`lm_eval`, limit 500) on the masked in-memory model. 4B
(4-bit): 79 documents + PIQA (limit 200, batch 4) — PPL numbers match the
2026-07-26 run; the polish pass adds the accuracy curves. Baselines are
measured in the same process with masks cleared, so every Δ is within-run.

### Qwen3-0.6B — WikiText-2 perplexity (baseline **33.75**)

| masked / layer | importance | frequency | random |
|---:|---:|---:|---:|
| 5% | **35.17** | 35.97 | 42.02 |
| 10% | **38.15** | 39.73 | 57.66 |
| 25% | **52.86** | 66.58 | 198.74 |
| 50% | **100.03** | 242.14 | 2900.15 |

### Qwen3-0.6B — PIQA accuracy (baseline **0.698**)

| masked / layer | importance | frequency | random |
|---:|---:|---:|---:|
| 5% | 0.678 | **0.698** | 0.682 |
| 10% | 0.656 | **0.666** | 0.628 |
| 25% | 0.594 | **0.630** | 0.594 |
| 50% | 0.548 | **0.568** | 0.540 |

### Qwen3.5-4B (4-bit) — WikiText-2 perplexity (baseline **18.87**)

| masked / layer | importance | frequency | random |
|---:|---:|---:|---:|
| 5% | **19.51** | 20.68 | 20.41 |
| 10% | **20.70** | 23.18 | 26.36 |
| 25% | **24.98** | 35.25 | 47.54 |
| 50% | **39.83** | 128.50 | 334.24 |

### Qwen3.5-4B (4-bit) — PIQA accuracy (baseline **0.770**, limit 200)

| masked / layer | importance | frequency | random |
|---:|---:|---:|---:|
| 5% | **0.755** | 0.750 | 0.735 |
| 10% | 0.745 | **0.755** | 0.730 |
| 25% | **0.705** | 0.695 | 0.675 |
| 50% | 0.645 | **0.695** | 0.605 |

(`acc_norm` baseline 0.790; same ranking pattern.)

### Findings

1. **Importance-guided masking beats random by a wide margin, and the gap grows
   with the ratio.** At 25% the 4B sits at 24.98 PPL guided versus 47.54 random;
   the 0.6B at 52.86 versus 198.74. This is the headline deliverable from Q7 —
   the *gap*, not the absolute drop.
2. **Importance beats frequency on perplexity everywhere**, confirming the
   Weeks 5–8 revision. The margin is large at high ratios (4B at 50%: 39.83 vs
   128.50). Frequency masks ~28% more of the layer's total importance mass than
   the importance ranking at the same neuron count (4B at 50%: 45.7% vs 35.8%),
   which is exactly the failure mode the saturated frequency signal predicted.
3. **Frequency is barely better than random at low ratios**, and on the 4B at 5%
   it is marginally *worse* (20.68 vs 20.41). The lazy-neuron ranking carries
   almost no signal on SwiGLU models until the budget is large.
4. **PIQA and perplexity disagree about which ranking is best — but both beat
   random.** On the 0.6B, `frequency` holds PIQA accuracy better than
   `importance` at every ratio, even though it is clearly worse on perplexity.
   On the 4B the picture is mixed (importance wins at 5%/25%, frequency at
   10%/50%) and both guided rankings beat random everywhere. Importance is
   tuned to preserve residual-stream L2 contribution (what perplexity rewards);
   multiple-choice accuracy is more forgiving of a shifted output distribution.
   Report both metrics rather than picking the flattering one.
5. **Larger model, more absolute headroom, smaller relative gap (H2, partial).**
   Relative to its own baseline, the 4B degrades much less than the 0.6B at every
   ratio (25% guided: 1.32× vs 1.57×; 25% random: 2.52× vs 5.89×). But the
   *guided-vs-random ratio* is larger for the **small** model (3.76× vs 1.90×),
   so the second clause of H2 — that the guided-vs-random gap grows with size —
   is **not** supported within the Qwen lineage on this pair of models.

> **Superseded (#18).** The size evidence from this stage, Weeks 11–12, and the
> Weeks 13–14 cross-family run is consolidated in
> [`size_h2_summary.md`](size_h2_summary.md), including the GPT-2 size arm that
> varies size with architecture held fixed. The verdict there is **H2 not
> supported**, not "partial": clause (a) fails at two of three ratios in the
> clean within-family test, and clause (b) is contradicted everywhere. Quote that
> file in the final report rather than this paragraph.

### Against the proposal's success criteria

The Q4 target was ΔPPL < +1 and PIQA drop < 2 pp at 10% masking. **Not met at
10% on perplexity:** 4B is +1.82 PPL, 0.6B is +4.40 PPL. On PIQA the 4B is
closer (−2.5 pp at 10% importance; −1.5 pp for frequency) and the 0.6B is −4.2
pp. It **is** met on perplexity at **5% on the 4B** (+0.64 PPL; PIQA −1.5 pp).
Pure masking without any recovery is a stricter setting than the criteria
assumed; that is what Weeks 11–12 exist for.

## 3. Experiment 2 — duplicate-neuron ablation (H5)

**Hypothesis (added after the Weeks 3–4 review).** For a near-duplicate pair
(|ρ| ≥ 0.9), masking **one** twin should cost less than masking an uncorrelated
neuron of the **same importance**, because their contributions are nearly
collinear.

**Conditions**, all at the same neuron count except `duplicate_both`:

| Condition | What is masked |
|-----------|----------------|
| `duplicate_one` | the lower-importance twin of each disjoint pair |
| `duplicate_both` | both twins (2× the neurons) |
| `importance_matched` | uncorrelated neurons with importance matched to `duplicate_one` — **the key control** |
| `random_matched` | random uncorrelated neurons |
| `lowest_importance` | the least-important neurons in the same layers |

Pairs are made **disjoint** first: the reported top-k often forms a small clique
of near-identical units, and masking one neuron from each of two overlapping
pairs would silently change the removal budget.

**Setup.** WikiText-2 test. **0.6B:** 512 rows (baseline PPL 38.39); correlation
pairs from the Weeks 5–8 drill-down layers (1, 2, 27). **4B (polish):** 332
rows (baseline 16.54); only **4** disjoint pairs above |ρ| ≥ 0.9 across layers
0 / 1 / 31 (weaker correlations than the 0.6B massive-activation clique).

### Results — Δ perplexity vs unmasked

| Condition | 0.6B all (14 pairs) | 0.6B L27 (7) | 0.6B L2 (5) | **4B all (4 pairs)** |
|-----------|---:|---:|---:|---:|
| `duplicate_one` | +2.241 | −0.189 | +1.840 | **−0.013** |
| `duplicate_both` | **+2884.764** | −0.857 | **+2898.431** | **+0.001** |
| `importance_matched` | −0.279 | −0.257 | −0.026 | **+0.033** |
| `random_matched` | +0.170 | −0.007 | +0.133 | −0.009 |
| `lowest_importance` | +0.017 | −0.001 | +0.059 | −0.001 |
| *control / duplicate importance ratio* | 0.60× | 0.99× | 0.02× | **1.00×** |

### Findings

1. **H5 as stated is not supported on 0.6B, and only weakly on 4B.** On 0.6B,
   `duplicate_one` never costs less than `importance_matched`. In layer 27,
   where the control is essentially perfectly matched (0.99×), the two are
   indistinguishable (−0.189 vs −0.257): being a duplicate buys **nothing**
   beyond being unimportant. On the 4B the control matches perfectly (1.00×)
   and `duplicate_one` edges the control by +0.046 PPL — technically
   "supported", but both absolute effects are within noise of a 4-neuron
   ablation. Treat this as **inconclusive / weakly consistent**, not a strong
   confirmation.
2. **A refined version *is* supported on 0.6B, and it is the interesting
   result.** In layer 2, masking one twin costs +1.84 PPL but masking **both**
   costs +2898 — a factor of ~1,500. The pair is *jointly critical* and
   *individually redundant*: the surviving twin genuinely absorbs the function.
   That is the "one can carry the other" signature, and it is the strongest
   argument yet for a **replacement/merge** step rather than plain masking. The
   4B drill-down pairs do **not** show this (both twins are free), so the
   phenomenon is tied to the massive-activation clique, not to high |ρ| alone.
3. **Duplicates are not cheap neurons — they are among the most important.** In
   0.6B layer 2 no uncorrelated neuron comes within 50× of the duplicates'
   importance (match ratio 0.02×). These are the massive-activation neurons
   flagged in the Weeks 5–8 measurement (layer 2 RMS ≈ 40× its neighbours).
   Duplication there looks like *redundant encoding of a critical signal*, not
   spare capacity. The 4B pairs sit mid-distribution (matchable at 1.00×) and
   behave like ordinary low-impact units.
4. **Deep-layer duplicates are genuinely free.** In 0.6B layer 27, masking all
   14 neurons of 7 pairs slightly *improves* perplexity (−0.857). They are
   removable — just not because they are duplicates.
5. **The whole catastrophic effect in the 0.6B all-layers run comes from
   layer 2.** The per-layer decomposition (+2898 for 5 pairs in L2 vs −0.86 for
   7 pairs in L27) shows an aggregate duplicate-pair result would have been
   meaningless without it.

**Implication for Weeks 11–12.** Do not merge duplicates blindly. The productive
target is layer-2-style pairs: mask one twin and rescale/absorb its
`down_proj` column into the survivor, then check whether the +1.84 PPL cost
disappears. Mid-correlation pairs like the 4B drill-down can just be dropped by
importance ranking like any other cheap neuron.

> **Outcome (2026-07-30, `recovery_week11-12.md`).** It disappears. Fitting
> `h_drop ≈ α·h_keep + β` on calibration activations and re-injecting the
> contribution takes layer 2 from **+1.84 to +0.14 PPL** (92.6% recovered) and all
> 14 pairs from +2.24 to +0.28 (87.3%), with no training. The prediction in the
> paragraph above was right, including the "do not merge blindly" part: in layer
> 27, where masking was already free, merging makes perplexity slightly *worse*
> (−0.19 → +0.14). Separately, 200 LoRA steps recover 76–83% of the cost of
> ordinary importance-guided masking at 10–25%.

## 4. How to run

```powershell
# Removal-ratio sweep
uv run python scripts/run_pruning.py --config configs/pruning/neuron_masking.yaml
uv run python scripts/run_pruning.py --config configs/pruning/neuron_masking_qwen3_0.6b.yaml

# Fast smoke run (few documents, one ratio)
uv run python scripts/run_pruning.py --config configs/pruning/neuron_masking_smoke.yaml

# Duplicate-pair ablation, all drill-down layers or one layer at a time
uv run python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation.yaml
uv run python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation_qwen3_0.6b.yaml
uv run python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation_qwen3_0.6b.yaml `
    --layers 27 --tag L27
```

Both scripts resolve `measurement_path: "auto"` to the newest
`measurement_<model>_*.json` for the loaded model; `--measurement <path>` pins a
specific one. Key config knobs: `pruning.strategies` / `pruning.ratios` /
`pruning.layers`, `eval.run_lm_eval` (PIQA on the masked model), and
`pairs.min_abs_corr` / `pairs.max_pairs_per_layer`.

## 5. Outputs

- **`pruning_<model>_<ts>.json`** — baseline, one entry per (strategy, ratio)
  with perplexity, ΔPPL, neurons masked, the fraction of layer importance
  removed, and PIQA accuracies when enabled. Records model id, seed, config
  path, full command, and the measurement artifact replayed.
- **`pair_ablation[_<tag>]_<model>_<ts>.json`** — selected pairs (drop/keep/ρ),
  per-condition perplexity, and a machine-readable H5 verdict including how well
  the importance control matched.
- **Figures**: `prune_ppl_*` (removal curves, log y), `prune_acc_*` (PIQA
  curves), `pair_ablation_*` (Δ-PPL bars per condition).

## 6. Tests

`tests/test_pruning.py` (13 tests) runs fully offline against a toy SwiGLU model
and a synthetic measurement artifact:

- the masker zeroes exactly the selected `down_proj` input channels, restores the
  original outputs on `clear()`, and removes its hooks on exit;
- out-of-range indices raise, unknown layer names are ignored (hybrid models);
- selection picks the lowest-scoring neurons, honours the per-layer budget, and
  `random` is count-matched and seed-reproducible;
- ratio 0 masks nothing; layer filters scope the selection;
- duplicate pairs come out disjoint and oriented (cheaper twin dropped);
- all five ablation conditions are count-matched, controls avoid paired neurons,
  and every condition applies cleanly to a real module;
- both new plotting helpers write non-empty files.

```powershell
uv run pytest tests/ -q    # 23 tests (metrics + pruning)
```

## 7. Mapping to the proposal

| Proposal item | Where |
|---------------|-------|
| Q7 mask lowest-ranked neurons, sweep 10→25→50% | `run_pruning.py`, `pruning.ratios` |
| Q7 random baseline at matched per-layer sparsity | `random` strategy, `prune_ppl_*` |
| Q4 outcome metrics (WikiText-2 PPL + PIQA) | `compute_perplexity`, `run_lm_eval_loaded` |
| Q4 success criteria at 10% | §2, "Against the proposal's success criteria" |
| H2 (size) | 0.6B vs 4B curves, §2 finding 5 — **since refuted**, see the scoreboard |
| H4 (post-activation magnitude ranks better than frequency) | `importance` vs `frequency`, §2 finding 2 — **supported on PPL; mixed on PIQA**. The frequency half is now **H7**: worse than random on every GELU model |
| H5 (duplicates, added in review) | `run_pair_ablation.py`, §3 — rejected on 0.6B; weakly/noise-level on 4B. Across all six models the verdict has **no consistent sign**; **H5′** (replacement) is the claim that holds |

## 8. Limitations

- **Masking ≠ removal.** Shapes are unchanged; no speed or memory claim.
- **Evaluation subsets.** 157 (0.6B) / 79 (4B) WikiText-2 documents, not the full
  test split, so absolute baselines differ from Weeks 1–2 (4B: 18.87 here vs
  17.06 on the full split). All comparisons are within-run, so the *gaps* are
  unaffected, but absolute PPL values are not comparable across reports.
- **One random draw per ratio.** The random baseline is a single seeded sample,
  not an average over repeats with error bars. The gaps are large enough that
  this is unlikely to flip any conclusion, but it should be repeated for the
  final write-up. `scripts/random_selection_stability.py` resamples the
  *selection* half of that variance offline from the measurement `.npz`, with no
  GPU time; see [`random_selection_stability.md`](random_selection_stability.md)
  for what it can and cannot bound.
- **The importance control in H5 is only as good as the layer allows.** Where
  duplicates are extreme outliers (layer 2, 0.02×) no fair control exists inside
  that layer, and the comparison is biased against H5. This is reported per run
  as `importance_match_ratio` rather than hidden.
- **Correlation pairs come from the measurement drill-down layers only** (3 per
  model) and from a reservoir subsample, so the pair inventory is not exhaustive.
  The 4B artifact has only four pairs above |ρ| ≥ 0.9.
- **PIQA limits.** 0.6B used limit 500; 4B polish used limit 200. Absolute
  accuracies are not full-validation estimates; ranking comparisons within a
  run remain valid.
- **Single calibration corpus.** Rankings come from WikiText-2 activations; the
  H3 (task) test needs a PIQA-prompt calibration run through the same pipeline.

## References

- Ma et al. 2023 — *LLM-Pruner* (importance → structured mask recipe)
- Sun et al. 2023 — *Wanda* (activation-aware weight importance)
- Li et al. 2022 — *The Lazy Neuron Phenomenon* (the frequency ranking we test)
- Geva et al. 2021 — *FFN Layers Are Key-Value Memories*
- `proposal_week3-4.md` (Q1/Q4/Q7/Q8, H2/H4), `measurement_week5-8.md`
