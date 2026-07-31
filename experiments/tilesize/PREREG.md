# Pre-registration: Tile-size as the correlation length of LLM tile redundancy

**Model:** Qwen/Qwen3-4B  **Written:** 2026-07-18, BEFORE any Stage-2 downstream
number has returned (no GPU job in this study has run yet). The purity probe is
proxy-only, so its prediction *is* part of the pre-registration. This document
freezes the hypotheses, the frozen researcher-degrees-of-freedom, and the
decision rules, to prevent retrofitting conclusions.

---

## 1. Hypothesis

The 32×32 whole-model findings — the **~5% retained-ability ceiling** (finding 1),
**selection ≈ random** (findings 4d/4e), and **magnitude < random** (finding 8) —
are hypothesized to be **artifacts of coarse tile granularity**, not intrinsic
properties of Qwen3-4B. If so, three findings move **together** as the tile size
`T` shrinks:

1. **H1 — ceiling rises:** the ~5%-sparsity retained-ability ceiling increases as `T→1`.
2. **H2 — selection regains power:** the SELECTION gap (`wanda_recon − random_recon`)
   grows from ~0 at `T=32` to > the random seed-SD at some `T<32`.
3. **H3 — magnitude<random vanishes:** the magnitude−random gap shrinks toward 0 as `T→1`.

And all three crossovers coincide with the probe's **correlation length S**.

## 2. Correlation length S (probe prediction — pre-committed)

> **S = ____________  (TO BE FILLED BY THE PURITY PROBE, `experiments/tilesize/probe/purity_probe.json`, before any Stage-2 downstream result is read.)**
>
> Tolerance band: **S ± 1 octave** (one step on the log2 tile axis).
> Anisotropy A = xi_col/xi_row = ______  (A≫1 ⇒ 1×N strips would pay; A≈1 ⇒ square is right).
> Probe verdict (DIFFUSE / CLUSTERED / COLUMNAR_ONLY / WEAK): ______
> Predicted sweep-priority order (descending marginal-gain-over-32): ______

The probe is a **hypothesis generator**, validated **only** if S matches where the
downstream ceiling/selection-gap peaks. A probe/ladder mismatch is itself the
finding ("removability is set by reconstruction, not importance clustering") and
must be reported as a discordance, not smoothed into a tidy story.

## 3. Four pre-registered outcomes and what each PROVES

- **DIFFUSE** — probe harvestable ≈ null at all T>1, z≈0, xi≈1; ceiling-vs-T rises
  monotonically toward 1×1 with **no knee**; SELECTION gap regains power **smoothly**
  as T falls; magnitude<random shrinks smoothly to 0 at 1×1.
  → **Proves** the 32×32 ~5% ceiling, "selection≈random", and "magnitude<random" are
  **granularity artifacts**. LLM tile redundancy has **no natural scale**; the true
  weight-level removable fraction is set by the 1×1 ceiling; coarse tile pruning
  systematically **under-estimates** redundancy. Quantifies the "structure tax" and
  kills the tile hyperparameter. Does **not** prove the 5% statement was wrong (it was
  correct *at 32×32*), nor that selection is useless (unstructured Wanda still works at 1×1).

- **CLUSTERED at S** — probe z-spike at T≈S predicted **in advance**; ceiling-vs-T shows
  a **knee** near S; SELECTION and REPAIR gaps peak near S; magnitude<random shrinks toward S.
  → **Stronger result.** Proves LLM weight redundancy has a **measurable correlation
  length S**, the optimal square tile equals S, and a cheap Wanda-purity probe predicts
  it **without pruning** (validated out-of-sample). Makes best-tile-size a measured model
  property + a front-end that replaces the sweep. Does not prove *why* redundancy clusters.

- **WEAK / GRADED** — mild excess dispersion, no sharp peak; ceiling rises gently 32→4 with
  diminishing returns; gaps grow modestly.
  → Proves granularity matters quantitatively but there is **no magic size**; recommend
  the smallest feasible structured tile; 5%@32 is a conservative floor.

- **KILL / NULL** — probe DIFFUSE-consistent **AND** downstream C(T) **flat** within noise
  across 32/16/8/4 **AND** selection gap ≈0 at every T.
  → Proves the ~5% ceiling is a **REAL property of Qwen3-4B**; findings 1/4/8 generalize
  across tile sizes; 32×32 is not the culprit. **SHIP THIS NULL as the headline.**
  If the three co-moving predictions **dissociate** (e.g. ceiling rises but selection gap
  stays 0), the coarse-granularity story is incomplete and "repair-not-selection is the
  whole method at every scale" is the publishable result.

## 4. Decision rules (committed in advance)

- **"Ceiling rises"** counts only if macro/min retained-ability@5% exceeds the 32×32 value
  by **> 2× SE** at bracketing points **AND** is monotone / single-peaked in T.
- **"Selection regains power"** counts only if SELECTION gap(T) exceeds the random seed-SD
  at some T<32 while ≈0 at T=32.
- The **H1 knee, H2 crossover, and H3 crossover must all coincide with S** (register S + tolerance above).
- A 90%-retained **crossing must be bracketed by real downstream points**, never extrapolated
  (extend a size's ladder up to {10,15,20,30}% if Stage 0/1 shows a raised ceiling).
- **Probe-vs-sweep concordance** is a pre-registered VALIDITY test: probe says diffuse but
  sweep shows a knee (or vice versa) ⇒ the Wanda/OBS importance measures the wrong notion of
  redundancy; report the discordance.

## 5. Exactly two primary tests (1 df each)

1. **Macro/min retained-ability ceiling vs T at fixed 5%.**
2. **SELECTION gap vs T at fixed 5%.**

Both judged as an **ordered (monotone-in-T) hypothesis** across the whole pre-declared grid
— **not** a cherry-picked winning cell (guards multiple-comparison inflation over 4 sizes ×
5 sparsities × 3 tasks). Everything else (per-task breakdowns, full sparsity ladder, ppl
screen, KL, isolation diagnostic) is **secondary / exploratory** — no significance stars.

---

## 6. Frozen choices (researcher-degrees-of-freedom locked)

| Item | Frozen value |
|---|---|
| **New tile sizes** | {16, 8, 4} via `run_pruning.py --tile-size`; **1×1** via `prune_unstructured.py` (vectorized). **T=2 held in reserve** — run only if the probe places S < 4. |
| **Reference (NOT re-run)** | 32×32 whole-model curves from `experiments/wholemodel/` + `experiments/downstream/`. |
| **Hold constant** | **PERCENT OF TOTAL WEIGHTS ZEROED** (not tiles-removed). Every matrix dim (2560/4096/1024/9728) divides every swept T ∈ {1,4,8,16,32}, so `--prune-ratio r` zeroes exactly weight-fraction r at every T with **zero drift** at the exact rungs. |
| **Sparsity rungs** | Exact/primary: **{5, 10, 20}%** (integer tile counts, drift < 1e-4). Shape-only: **{1, 2}%** (non-integer; curve shape, not primary claims). 1×1 ladder: {5,10,20,30,40,50}%. |
| **Operating-point priority** | 5% first (only point where 32×32 still works, ~90% retained), then 10/20%, then 1/2%. |
| **Scope** | **WHOLE-MODEL, uniform (Policy A), all 7 matrices × all 36 layers**, at every T. (Bounded L0 / L35-MLP isolation diagnostic is a cross-check, NOT the headline.) |
| **Anchor method** | **`sparsegpt_recon`** — never swapped mid-sweep (the 32×32 downstream ladder is sparsegpt_recon uniform, so new sizes overlay directly). |
| **Controls per T** | random (seeded floor), wanda (selection-only), wanda_recon (selection+repair), magnitude (aggregation-loss probe); paired same-tile ablation random / random_recon / wanda_recon. |
| **Calibration** | **ppl/KL screen: `--calib-samples 128 --calib-seqlen 512 --seed 0`**, held identical across every T. **Downstream ladder: `--calib-samples 64 --calib-seqlen 512`** to MATCH the archived 32×32 reference. col_norms/H depend only on activations, so any calib drift contaminates the size axis — calib seed logged and fixed on every run. |
| **Eval** | WikiText-2 for perplexity (dense PPL 13.22), `--eval-frac 0.2` for the Stage-1 screen (same fraction every comparable run); full lm-eval task sets (no `--limit`) for downstream. |
| **Damping** | `damp = 1e-2` (SparseGPT / OBS damped inverse). |
| **Model revision** | frozen (config: hidden 2560, intermediate 9728, q_proj out 4096, k/v out 1024, 36 layers, bf16). |

## 7. Metrics (ranked by trust)

1. **DOWNSTREAM retained-above-chance ability** = `(acc − chance)/(acc_dense − chance)` — the
   ONLY citable claim. `acc_norm` for hellaswag/arc_easy, `acc` for piqa; chance 0.25/0.50/0.25.
   Report **per-task AND min-across-tasks** (headline), macro-mean secondary. Effects must clear **~2× SE**.
2. **KL(dense‖pruned) + top-1 agreement** — emitted free on every whole-model `run_pruning` run;
   fills between downstream points and acts as a direction gate. Never a standalone ceiling claim.
3. **WikiText-2 perplexity** — SCREEN and point-selection ONLY; every cell labeled **"PROXY"**;
   forbidden as a final cross-tile capability claim (finding 2b: capability can move opposite to ppl).
4. **PROXY-FIDELITY GATE** — at each T, measure downstream AND ppl AND KL at ≥3 shared weight-fractions;
   test whether the per-T ppl→retained-ability curves **coincide** across sizes. If they diverge, every
   ppl-only tile claim is void.

Two GAP statistics tracked vs T: **REPAIR gap(T) = metric(random_recon) − metric(random)**
(paired, identical tiles) and **SELECTION gap(T) = metric(wanda_recon) − metric(random_recon)**
(both repaired, tiles differ). Control-relative rankings (H2, H3) MAY use perplexity honestly
(shared metric bias cancels); only the ABSOLUTE ceiling (H1) requires the downstream ladder.

## 8. Denominators / anchors (frozen, verified on disk)

- **Dense retained-ability denominator** (`downstream_dense.json`, tile_size 32):
  hellaswag acc_norm **0.6836**, piqa acc **0.7492**, arc_easy acc_norm **0.7828**.
- **Comparability anchor** — reproduce 32×32 sparsegpt_recon p5 uniform under tonight's config
  (calib 64) and assert it lands within seed noise of the archived
  (`downstream_sparsegpt_recon_p5_uniform.json`): hellaswag **0.6412**, piqa **0.7242**,
  arc_easy **0.7294** (≈90% retained). *(Note: plan text quoted piqa ≈0.719; the on-disk
  archived value is 0.7242 — the on-disk value is authoritative.)* A mismatch ⇒ **STOP** and
  reconcile calib / lm-eval / model revision before any cross-size claim.

## 9. Seeds & statistics

- **Random floor:** perplexity — 5 seeds (1,2,3,4,5), reported **non-parametrically** (median +
  seed-win-counts, NOT mean±σ; the floor is heavy-tailed). Downstream — 1 representative seed (seed 1).
- `--whole-model` consumes only `seeds[0]` (verified run_pruning line 943) ⇒ each seed is a
  **SEPARATE** invocation (the queue encodes this).
- The **seed-SD across the 5 random seeds is the noise floor** any selection/magnitude gap must clear.
- Downstream effects use lm-eval binomial stderr (hellaswag ~0.005, piqa/arc ~0.009–0.010; macro/min ~0.005)
  and must exceed **~2× SE**.
- Assert `achieved_sparsity` within **1e-4** of target across all T before any comparison.
- Purity-probe permutation nulls: **B ≥ 200**; excess reported as a z-score against BOTH the
  within-matrix and the within-column null bands (plus the closed-form binomial).

## 10. Circularity guard

The probe shares an importance field with the wanda/sparsegpt selectors, and diagonal-H purity
ignores the cross-column correlations reconstruction exploits. Therefore **purity predicts the
MASK-ONLY ceiling; the repaired ceiling is higher and less shape-sensitive.** The probe
predicts + prioritizes; the **downstream ladder decides.** Both are reported.
