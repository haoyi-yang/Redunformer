# Tile-Level Redundancy in LLMs — Findings

Living log. Last updated: **2026-07-17** (after the overnight downstream + ablation marathon).
Numbers come from the runs in `experiments/`; plots in `experiments/*/plots/`.

## The answer, in one line

> **At 32×32 tiles, ~5% of Qwen3-4B is genuinely redundant** — removable while keeping 90% of
> the model's learned ability. Not the 40–70% our perplexity curves implied.

## For the presentation — the four things worth saying

1. **We can now put a number on the project's question: ~5%** (measured on real tasks, not a proxy).
2. **Perplexity nearly fooled us.** At 20% sparsity it reads "2× worse"; the model has lost ~60%
   of its ability. Two of our conclusions changed once we measured accuracy.
3. **Repair is what matters, not the metric you select with** — worth 1.8×–105×, while the
   selection rule is worth ~nothing. **Rathore's combined variant (Wanda select + SparseGPT
   repair) is our best method**, and cheaper than what we were using.
4. **Rathore's Strategy 5 works, conditionally** — sensitivity-aware allocation wins at low
   sparsity, but it flatters perplexity ~3× more than it improves actual capability, because its
   map was built from perplexity.

**Status of the plan: complete.** All 10 of Rathore's strategies and all 5 Fiebiger stages are
done, including the whole-model control floor. The three headline findings (1, 2, 3) are
downstream-anchored — see the reliability audit below.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-4B` (36 layers, bf16), RTX 4080 Super 16 GB |
| Tile size | 32 × 32 (never varied — see "next marathon") |
| Screened layers | 0, 9, 18, 27, **32, 33, 34**, 35 |
| Matrices | `q/k/v/o_proj`, `gate/up/down_proj` (7 per layer, 98,560 tiles each, 3,548,160 total) |
| Selection methods | random, magnitude, wanda, sparsegpt (eq-22/23) |
| Repair | SparseGPT reconstruction (eq-23), exact per output-row-block |
| Eval | WikiText-2 perplexity · output divergence (KL) · **lm-eval accuracy (HellaSwag/PIQA/ARC-Easy)** |
| Dense reference | ppl **13.22** · acc **0.6836 / 0.7492 / 0.7828** (reproduces published Qwen3-4B) |

**All methods are tile pruning.** Contrasts with "weight-level" refer to the *literature's*
unstructured setting, not our runs.

---

# Findings — and where each came from

**Sequencing note, stated plainly.** The adopted plan should have been completed *before* any
follow-up experiments. It wasn't: the overnight work jumped to questions the early results
suggested, while four of Rathore's strategies and three stages of the extensions plan were still
open. The findings are tagged below so the planned results can be read on their own. Awkwardly,
the unplanned work produced the project's headline answer (~5%) — that does not make the ordering
right, and the planned gaps are now scheduled first.

| # | finding | provenance |
|---|---|---|
| 1 | Tile redundancy is ~5% | ⚠️ **unplanned** — extends E4b/Stage 5 to a 1/2/5/10% ladder |
| 2 | Perplexity is a nonlinear proxy | ✅ planned — E4b (downstream accuracy) |
| 2b | Perplexity is adversarially manipulable | ⚠️ **unplanned** — Seb's targeted `o_proj` + downstream |
| 3 | Policy B games its own metric | ✅ planned — W5/S5 (Policy A vs B) + E4b |
| 4 | Repair ≫ selection | ✅ planned — S5's **combined Wanda→SparseGPT variant** |
| 4b | Repair backfires at the final layers' **MLP** (not the layer) | ✅ planned — **S2** + **S4** |
| 4c | Data-aware selection only helps at layer 0 | ✅ planned — **E1/E2 controls at Stage 2** |
| 4d | At 5% (usable point) selection loses to a coin flip | ✅ planned — **E1/E2 controls at Stage 4** |
| 4e | Selection worthless *given repair* in the usable regime | ⚠️ **unplanned** — `random_recon` ablation |
| 5 | "Robust" does not compose | ✅ planned — synthesis of W1/W2 vs Stage 4 |
| 6 | Reallocation has an optimum | ⚠️ **mixed** — Policy B planned; depth concentration unplanned |
| 6b | Layer-wide budget matching helps only where structure exists | ✅ planned — **W3** |
| 7 | Robustness is the depth × matrix interaction | ✅ planned — **W4 cluster scan** |
| 8 | Magnitude is worse than random | ✅ planned — E1/E2 (controls + seeds) |
| 9 | Damage is sub-additive within a layer | ✅ planned — W2/S3 (whole-layer) |

✅ **Findings 1 and 4 now have their control floor** (E1/E2 at whole-model scale) — see finding
4d. It did not confirm finding 4; it sharpened it: at 5%, selection is *worse than random*,
and repair is the whole method.

## What each finding rests on — the reliability audit
Finding 2b (perplexity is manipulable) raises the fair question: *are our findings unreliable
because we used perplexity?* No — and here is why, made checkable. Each finding is one of three
kinds, in decreasing robustness:

- **Downstream-anchored** — measured on task accuracy, not perplexity. Immune by construction.
- **Control-relative** — a comparison of methods at *identical* conditions, usually against the
  random control. A shared metric bias cancels in the comparison, so the *ranking* holds even if
  absolute perplexity is biased. (Strongest: 4e's `random` vs `random_recon` prune the *same
  tiles* — any bias is identical on both sides.)
- **Perplexity-absolute** — depends on absolute perplexity or the perplexity-derived sensitivity
  map. Carries the caveat; finding 2b is live proof that some map cells are perplexity-specific.

| finding | rests on | robust to the perplexity problem? |
|---|---|---|
| 1 tile redundancy ~5% | **downstream** | ✅ fully |
| 2 / 2b perplexity unreliable & manipulable | **downstream** (the ppl↔acc gap *is* the finding) | ✅ self-reinforcing |
| 3 Policy B games its metric | **downstream** | ✅ fully |
| 4 / 4e repair ≫ selection | control-relative (same-tile ablation) | ✅ ranking robust |
| 4b repair backfires at final MLP | control-relative (mask vs recon, same tiles) | ✅ direction robust |
| 4c / 4d selection vs random floor | control-relative (vs the control that shares the bias) | ✅ robust |
| 5 "robust" does not compose | downstream (~60% capability loss) + perplexity | ✅ mostly |
| 6 / 6b reallocation optimum | control-relative (budget-matched) | ✅ ranking robust |
| 7 depth × matrix interaction | control-relative (o_proj vs up_proj across depth) | ⚠️ mostly; fine cell labels are perplexity-based |
| 8 magnitude < random | control-relative (vs random control) | ✅ robust |
| 9 sub-additive within a layer | perplexity-absolute | ⚠️ carries the caveat |

**The three headline findings (1, 2, 3) are all downstream-anchored** — we ran ~11 downstream
evals precisely so the conclusions we lead with would not rest on perplexity. Finding 2b does not
undermine the project; it **vindicates the design choice it was built on.**

# Headline findings

## 1. Tile redundancy is ~5% — measured on real tasks ⭐
`sparsegpt_recon`, uniform. **Retained = (acc − chance) / (acc_dense − chance)** — the share of
*above-chance* ability kept. A broken model still scores chance by guessing, so raw accuracy
flatters it.

| sparsity | HellaSwag | PIQA | ARC-Easy | perplexity |
|---|---|---|---|---|
| 1% | 100% | 103% | 98% | — |
| 2% | 97% | 98% | 93% | — |
| **5%** | **90%** | **90%** | **90%** | 15.62 |
| 10% | 73% | 81% | 82% | 18.78 |
| 20% | 37% | 51% | 41% | 30.95 |
| 30% | 17% | 29% | 24% | 61.56 |

**5% is the sweet spot** (90% retained, strikingly consistent across all three tasks); 10% is the
aggressive edge (~79%); past that it collapses. This is the project's core answer — modest, but
measured and defensible.

## 2. Perplexity is a *nonlinear* proxy — and dangerous exactly where you'd rely on it ⭐
Same configs as above:

| perplexity | vs dense | ability retained |
|---|---|---|
| 15.62 | 1.18× | ~90% (proportionate) |
| 18.78 | 1.42× | ~79% |
| **30.95** | **2.34×** | **~43%** ← "only 2× worse" = more than half destroyed |
| 61.56 | 4.66× | ~20% |

Perplexity is monotonic with capability — it doesn't point the wrong way. The failure is that
the mapping is **brutally nonlinear**: a 2.3× perplexity rise *reads* as mild degradation and
*means* the model is mostly gone. It is a fine proxy in the usable regime (≤5%) and misleading
precisely where you would use it to judge an aggressive method.

## 2b. Perplexity is adversarially manipulable — improve it 13% while degrading the model ⭐⭐
The sharpest form of finding 2. Blanket `o_proj` pruning (all 36 layers) never beats dense. But
restricted to the layers where `o_proj` *improved in isolation* (17–21, 32, 34, 35), mask-only
Wanda beats dense on WikiText — and the two metrics move in **opposite directions** with dose:

| dose | WikiText ppl | HellaSwag | PIQA | ARC-Easy |
|---|---|---|---|---|
| 20% | **12.24 (−0.98)** | −0.3σ (flat) | +0.8σ (flat) | +0.7σ (flat) |
| 40% | **11.50 (−1.72, 13% better)** | **−3.6σ (real drop)** | +0.8σ (flat) | −1.5σ |

Perplexity improves monotonically (−0.98 → −1.72) while capability degrades monotonically (flat →
a statistically-significant HellaSwag loss). At 20% it is pure gaming (perplexity up, capability
untouched); at 40% it is gaming **plus** real damage — if you trusted perplexity you would ship a
measurably worse model believing it 13% better.

**Repair erases the perplexity win** (recon +0.07 / +0.44, not −0.98 / −1.72). Coherent: repair
reconstructs the dense output, so it faithfully restores whatever masking removed. The "gain"
exists only because you did *not* reconstruct.

**Caveat (and it is the point):** the layers were selected on WikiText, so the perplexity win is
partly WikiText-specific — but that is exactly what the downstream measurement exposes. Perplexity
alone would have called this an improvement. *(Credit: Seb's experiment — a refinement of the
blanket `o_proj` test that turned a null result into the capstone example for finding 2.)*

## 3. Policy B games the metric it was built from ⭐⭐
Two separate results, both damning for reading perplexity as capability.

**(a) At high damage, a perplexity gain buys nothing.** Policy B vs A, same tile budget:

| sparsity | ppl gain | HellaSwag | PIQA | ARC-Easy |
|---|---|---|---|---|
| 20% | 1.31× | **+6pp** | +1pp | **+7pp** | ← real capability |
| **30%** | **1.58× (its peak)** | +3pp | **−1pp** | **0pp** | ← **nothing** |

Policy B's headline number — its 1.58× peak — **buys no capability at all**. Both models are
already near the guessing floor at 30%; improving a broken model's perplexity does not un-break
it. The point where a perplexity-optimizing method looks *most* impressive is the point where its
gain is most likely worthless. **We would have made that 1.58× the paper's headline.**

**(b) At *matched perplexity*, Policy B delivers 10–16pp LESS capability than uniform.**
Interpolating the recon+uniform curve to Policy B's exact perplexity:

| config | uniform at same ppl | Policy B | gap |
|---|---|---|---|
| p20 sens (ppl 23.6) | 56.5 / 67.3 / 63.3 | 43 / 52 / 48 | **−13.5 / −15.3 / −15.3 pp** |
| p30 sens (ppl 39.0) | 30.2 / 43.6 / 35.3 | 20 / 28 / 24 | **−10.2 / −15.6 / −11.3 pp** |

**6/6 measurements, same direction, large.** At 20% Policy B improves perplexity 1.31×; an honest
gain that size implies HellaSwag 37→56.5. It delivers 43. **Policy B buys ~31% of the capability
its perplexity advertises.**

**Why — and it is slightly circular:** our sensitivity map was *derived from perplexity*. We
measured which matrices hurt perplexity, then built a policy protecting exactly those. Perplexity
is dominated by the final layers (they feed the LM head); downstream tasks depend on the whole
computation. So the policy protects what perplexity cares about and flatters the metric it was
fit to. **Policy B's advantage is partly an artifact of how its map was built.**

Both things remain true: at a fixed sparsity budget B *is* genuinely better (43 vs 37). Its
perplexity number simply overstates that by ~3×.

→ **Next marathon:** build the sensitivity map from *downstream accuracy* instead of perplexity
and re-run A/B. If the gap closes, this is confirmed as a metric artifact.

## 4. Repair is everything; *which tiles you pick* barely matters ⭐
**Credit: the combined variant is Rathore's.** His strategy document specified "select tiles with
Wanda, reconstruct the survivors with SparseGPT" as part of the backbone from the start — it sat
unrun on the backlog for weeks. What we add here is the *framing* (that it isolates selection
from repair) and the measurement. **His method turns out to be the best one we have.**

Controlled ablation — the repair path is **bit-identical** between `wanda_recon` and
`sparsegpt_recon` (CPU-verified), so selection is the only variable:

| sparsity | wanda | **wanda_recon** | sparsegpt_recon | repair adds | eq-23 selection adds |
|---|---|---|---|---|---|
| 5% | 28.31 | **15.59** | 15.62 | 1.82× | 1.00× |
| 10% | 52.22 | 19.81 | **18.78** | 2.64× | 1.05× |
| 20% | 156.51 | **25.95** | 30.95 | 6.03× | 0.84× |
| 30% | 1,145 | **38.45** | 61.56 | 29.78× | 0.62× |
| 40% | 7,812 | **74.42** | 76.30 | **104.97×** | 0.98× |

- **Repair dominates, monotonically**: 1.82× → **104.97×** as damage grows.
- **Selection is ~neutral**: 0.62–1.05×, no trend.
- **`wanda_recon` wins 4/5 sparsities** and is *cheaper* (no per-tile Schur complements).

**This corrects our own headline.** "SparseGPT reconstruction is best" was never about SparseGPT's
*selection* — it was about repair. **Recommended method: `wanda_recon`** — not because Wanda picks
better, but because selection barely matters and Wanda's is cheap.

Capability confirms it at 20%: `wanda` retains 14/12/16%, `sparsegpt_recon` 37/51/41% — repair is
worth ~3× in real ability, not just perplexity.

## 4b. Repair backfires exactly where it is needed most (Rathore's S2) ⭐
His **reconstruction-benefit classification**, run on data we already had. Thresholds are set by
the random control's spread across seeds (median per-cell σ = 0.0619 ppl → "meaningful damage"
bar = 2σ = **0.124**), not hand-picked — which is what extension E2 was specified for.
Plot: `experiments/screen/plots/reconstruction_classes.png`

| class | meaning | cells | share |
|---|---|---|---|
| **A** | naturally redundant — barely hurts even unrepaired | 21 | 60.0% |
| **B** | compensatable — hurts, but repair fixes ≥60% of it | 9 | 25.7% |
| **C** | essential — hurts, and repair cannot fix it | 5 | 14.3% |

**Category C is layer 35's MLP, and there repair makes things *worse*:**

| cell | mask-only | after repair | damage repaired |
|---|---|---|---|
| L35 `up_proj` | +2.71 | **+3.37** | **−24.5%** |
| L35 `gate_proj` | +1.57 | +1.52 | +3.1% |
| L35 `down_proj` | +0.34 | **+0.43** | **−26.4%** |

Meanwhile in the middle it is spectacular: L18 `up_proj` repairs **124%** (ends *below* dense),
L35 `v_proj` 85%, L9 `down_proj` 91%.

**It is the final layer's MLP — not the final layer.** Rathore's **S4** cluster scan (task 2b)
tested this directly, and *refuted the depth explanation we first reached for*. Repair fraction
across the sensitive cluster (29–35), by matrix type:

| matrix | @10% | @20% | @40% | layer 35 @40% |
|---|---|---|---|---|
| **`k_proj`** (attention) | +70.9% | +65.9% | +49.6% | **0.88 → 0.32 = +63.8%** |
| **`up_proj`** (MLP) | +14.1% | **−18.1%** | **−40.7%** | **3.88 → 6.82 = −75.5%** |

**`k_proj` repairs at +64% at the very same layer 35 where `up_proj` fails at −75%.** Cross-checked
against S2, the split is by component, not depth:

- attention at L35: `k_proj` **+72%**, `v_proj` **+85%** — repair works
- MLP at L35: `gate_proj` +3%, `up_proj` **−25%**, `down_proj` **−26%** — repair backfires

> **Repair is everything (finding 4) — except in the final layers' MLP, where it actively hurts.**

⚠️ **What we first claimed, and why it was wrong.** The initial reading was "at layer 35 the output
feeds the LM head directly, so no downstream layer absorbs the error, so the local least-squares
objective stops proxying next-token loss." That is a *depth* argument, and `k_proj` at layer 35
disproves it — same layer, same absence of downstream layers, repair works fine.

**Mechanism: open.** Two candidates, neither separable with current data:
1. `up_proj` and `gate_proj` feed the **SwiGLU multiplicative gate**, so a *linear* least-squares
   objective may be misaligned with the gated output. But `down_proj` sits *after* the
   nonlinearity and also fails (−26%), which cuts against this.
2. The final-layer MLP Hessian is **ill-conditioned**, making the damped inverse produce a bad
   update.

A third observation any explanation must cover: **the failure deepens with sparsity**
(−38% → −59% → −75% at layer 35), which a simple "no absorption downstream" story does not
predict. Recording this as open rather than substituting another tidy narrative — the last one
survived exactly one experiment.

**Practical (unchanged, and now better targeted):** exclude the final layers' **MLP** from
reconstruction — mask it instead. Attention there should still be repaired.

It also shows finding #5 in Rathore's own taxonomy: 60% of cells are "naturally redundant"
*individually*, yet pruning them together at 20% costs ~60% of the model's ability.

**Practical:** exclude the final layer's MLP from reconstruction — mask it instead, or leave it
dense. Repairing it is strictly worse than not repairing it.

## 4c. Data-aware selection only earns its keep at layer 0 — the control says so ⭐⭐
Whole-layer (Stage 2) with the controls E1 specified: random × **5 seeds** (the floor) and
magnitude (the naive reference), same layers, same sparsities, same eval data. Each data-aware
method is scored against the random control's per-layer seed spread (σ from 5 seeds):
**better** = ≤ −1σ, **~tie** = |z| < 1 (indistinguishable from chance), **WORSE** = ≥ +1σ.

| zone | better | ~tie | **WORSE** |
|---|---|---|---|
| layer 0 | **8/12** | 1/12 | 3/12 *(all magnitude)* |
| middle (9/18/27) | 10/36 | **23/36** | 3/36 |
| **layer 35** | **0/12** | 3/12 *(all magnitude)* | **9/12** |

**Three regimes, replicated across 10/20/40%:**

- **Layer 0 — data-aware is essential.** At 40% the random control *explodes* to **169.90 ± 163.59**
  while recon holds at **0.89**. This is precisely the region E3 predicted would be most
  informative: "where a data-aware method continues to hold while the random control has already
  collapsed."
- **Middle (9/18/27) — indistinguishable from chance** at 10/20% (23/36 ~tie), separating only at
  40%. Also as E3 predicted: "methods are nearly indistinguishable at low sparsity and separate
  only as sparsity increases."
- **Layer 35 — every calibrated method LOSES to random, 9/9.** And at 40% the ranking **inverts
  completely**:

| L35 @40% | ΔPPL | calibration used |
|---|---|---|
| magnitude | **2.62** | none |
| random | **3.01** | none |
| sparsegpt (mask) | 4.53 | selection |
| wanda | 4.80 | selection |
| **sparsegpt_recon** | **5.39** | selection **+** repair |

> **Damage at layer 35 scales with how much calibration a method uses.** The globally *best*
> method is the *worst* one there; the globally worst (magnitude) is among the best. At the final
> layer, calibration signal is not merely uninformative — it is **anti-correlated**, and using
> more of it hurts more.

This unifies with **4b** (repair backfires at L35) into one mechanism covering *both halves* of
the method: at the final layer, calibration-local objectives — the Wanda/eq-23 ranking *and* the
reconstruction least-squares — stop proxying next-token loss, because no downstream layer remains
to absorb their error.

**It also refines finding 4.** "Selection is near-irrelevant" was too broad. Correctly: selection
is *indistinguishable from chance* in the middle at low sparsity, *genuinely useful* at layer 0,
and *harmful* at layer 35. That also explains why `wanda` ≈ `eq-23` — in the bulk of the network
they are not tied with each other, they are both tied with **random**.

⚠️ **Statistics, honestly:** 5 seeds gives a noisy σ, so no individual cell is significant. The
evidence is the *consistency* — layer 35 lands +1.5σ…+3.2σ in **9 of 9** calibrated
method×sparsity cells, and the middle sits within noise in 23 of 36. A pattern, not a p-value.

**None of this was visible before the control ran.** Without a floor, "wanda @10%, layer 18 =
0.441" looks like a data-aware method working; it is chance. E1's rationale — *"has no meaning
without a floor"* — was load-bearing, not a formality.

## 4d. At the only usable operating point, selection loses to a coin flip ⭐⭐
The **whole-model floor** (Fiebiger Stage 4 / E1+E2) — random ×5 seeds and magnitude, at model
scale, the control findings 1 and 4 were missing. Reported **non-parametrically** (how many of the
5 random seeds each method beats) because the random distribution is heavy-tailed: one seed
detonates and drags the mean, so mean±σ is actively misleading here.

Random floor, seed by seed (ppl, dense 13.22):

| sparsity | s1 | s2 | s3 | s4 | s5 | median |
|---|---|---|---|---|---|---|
| 5% | 25.0 | 24.0 | 20.7 | **73.2** | 22.2 | **24.0** |
| 10% | 344.6 | 335.8 | 83.9 | **2,278.8** | 92.3 | **335.8** |
| 20% | 815k | 39.6k | 95.0k | 462k | 16.9k | **95.0k** |

**How many of the 5 random seeds does each method beat?**

| method | 5% | 10% | 20% |
|---|---|---|---|
| `magnitude` | **0/5** (4,942) | 0/5 | 0/5 |
| `wanda` (selection only) | **1/5** (28.31) | 5/5 (52.2) | 5/5 (156.5) |
| `sparsegpt` (selection only) | **1/5** (28.03) | 5/5 (36.0) | 5/5 (74.7) |
| `sparsegpt_recon` (repair) | **5/5** (15.62) | 5/5 (18.8) | 5/5 (31.0) |
| `wanda_recon` (repair) | **5/5** (15.59) | 5/5 (19.8) | 5/5 (26.0) |

**Two regimes, and the crossover is between 5% and 10%:**

- **At 5% — the only sparsity where the model still works (finding 1)** — data-aware selection is
  **worse than a coin flip**: wanda loses to 4 of 5 random seeds. Repair still wins (5/5, 1.53× vs
  the median floor). So at the usable operating point, **repair is the entire method and selection
  is worse than nothing.**
- **At 10%+**, random collapses (336 → 95,000) while the data-aware methods hold (52 → 157).
  Selection becomes essential — 6.4× at 10%, 607× at 20%. But this is the regime where the model
  is already broken.

**E3 predicted this before the data existed:** *"methods are nearly indistinguishable at low
sparsity and separate only as sparsity increases"*, and the informative region is *"where a
data-aware method continues to hold while the random control has already collapsed."* Both land
exactly.

**Consistent with 4c** (whole-layer: selection loses to random at layer 35, ties through the
middle) — the same story at model scope.

⚠️ **Methodological note.** E2 specifies "mean together with a range", but the mean is
outlier-driven here (5%: mean 33.0 vs median 24.0, because s4 = 73.2). We report seed-win counts
and the median instead. Twice during analysis a mean-based summary flipped the conclusion — the
non-parametric statement needs no such choice.

## 4e. Selection is worthless *given repair* — but only in the usable regime ⭐⭐
`random_recon` (random selection + SparseGPT repair) prunes the **identical tiles** as `random` at
each seed, so `random` vs `random_recon` isolates **repair** and `random_recon` vs `wanda_recon`
isolates **selection**. 5 seeds per sparsity.

| sparsity | REPAIR (paired, same tiles) | random_recon | wanda_recon | sparsegpt_recon |
|---|---|---|---|---|
| **5%** | **5/5**, median 1.57× | **14.86** | 15.59 (beats 0/5) | 15.62 (beats 0/5) |
| **10%** | **5/5**, median 11.92× | **17.36** | 19.81 (beats 1/5) | 18.78 (beats 1/5) |
| 20% | 5/5, median 3006× | 31.59 | **25.95 (beats 5/5)** | 30.95 (beats 3/5) |

**Repair is unconditional: 15/15 paired improvements**, every seed, every sparsity. It behaves like
an *attractor* — it lands at ~15 / ~17 whether random selection started it at 24 or at 2,279, which
is why it erases the selection-induced variance (the ~5% floor spans 20.7–73.2; after repair,
14.6–15.3).

**Selection is scale-dependent, and my first framing was too strong:**
- **At 5% — the only sparsity where the model is still usable (finding 1)** — data-aware selection
  beats **0 of 10** coin-flip seeds. Repair is the entire method; selection is worse than nothing.
- **At 10%** it still loses (2/10).
- **At 20%** it *reverses*: `wanda_recon` beats all 5 random seeds. But this is the regime where the
  model already retains only ~40% of its ability, so it is a real ordering outside the deployable
  range.

**Unified with 4d:** masking's selection crossover (loses to random ↔ beats random) sat between 5%
and 10%; **repair pushes that crossover out to between 10% and 20%.** Repair widens the range where
selection does not matter — the same way it widened the whole-model collapse point. So the honest
one-liner is not "selection is worthless" but: **at any operating point you would actually deploy,
repair is the whole method and the tile-selection metric buys nothing.**

**Practical:** `random_recon` is the cheapest possible method (no scoring, no calibration for
selection) *and* the best at 5%. ⚠️ It has no downstream-accuracy run yet — its perplexity edge over
`sparsegpt_recon` is small, and per finding 2 that must be measured, not assumed, before it is
recommended.

## 5. "Robust" is measured in isolation and does not compose ⭐
Per our labels, **~81% of the model is individually "robust"** (ΔPPL ≈ 0 when pruned alone). Prune
all of it together at 20% and ~60% of the model's ability is gone.

> **The redundancy map measures MARGINAL damage. We were reading it as JOINT damage.**

A matrix being harmless alone says nothing about it being harmless when the other 204 are also
pruned. This single confusion explains the whole-model collapse, the capability numbers, *and*
why Policy B only buys 1.1–1.6× — you cannot fix a compounding problem by reallocating budget
among regions that are all mislabelled the same way. It is the empirical, quantified form of the
"sequential dependency" limitation Rathore flagged in §4.8.

## 6. Reallocation has an optimum — and Policy B + depth are ONE mechanism ⭐
Depth concentration, **identical 20% budget** (every layer holds the same tile count, so
N × local = 36 × 0.20 = 7.2 layer-equivalents is exact arithmetic):

| layers | local | perplexity | vs uniform-36 (30.95) |
|---|---|---|---|
| 32 | 22.5% | 24.94 | 1.24× |
| **24** | **30%** | **24.17** | **1.28× ← peak** |
| 16 | 45% | 30.42 | 1.02× |
| 12 | 60% | 54.51 | 0.57× |
| **8** | **90%** | **1,890** | **0.02× (61× worse)** |

A clean inverted-U. **Two competing forces explain both this and Policy B:**

- damage **compounds across layers** → concentrate
- damage is **super-linear within a layer** past ~30–45% → spread

The optimum balances them. Policy B reverses at high sparsity for exactly the same reason: it
pushes "robust" regions to 67% local, straight into the super-linear regime. **Two findings that
looked independent are one mechanism.**

Note: N=32 (prune only the robust zone, sensitive tail untouched) gives 1.24× — "prune only the
redundant part" works, but modestly. 24.17 is still ~1.8× dense: concentration optimizes *within*
the broken regime, it does not unlock a new one.

## 6b. Layer-wide budget matching helps only where there is structure to exploit (Rathore W3)
His W3: hand a whole LAYER one tile budget and let the split between its seven matrices emerge
from a pooled, mean-normalised Wanda ranking, instead of forcing every matrix to the same
sparsity. Compared against uniform at the **identical tile count** (his fair comparison).

| layer | @10% | @20% | @40% |
|---|---|---|---|
| 0 | 0.87× | **0.66×** | **0.61×** |
| 9 | 1.56× | 1.10× | 0.64× |
| 18 | 0.70× | 1.41× | 0.92× |
| 27 | 0.52× | 0.46× | 1.01× |
| **35** | **2.75×** | **2.18×** | **1.17×** |

**7/15 cells — a coin flip overall.** But not noise: it wins big at layer 35, where real sensitivity
structure exists, and loses at layer 0, where everything is robust so reallocation merely
rearranges damage into a worse shape.

**The allocation does NOT discover sensitivity.** ⚠️ Tempting claim, tested and refuted. Spearman
between each matrix's allocated ratio and its *measured* sensitivity, across all 15 layer×sparsity
cells: **mean ρ = −0.15**, mostly non-significant (p 0.25–0.88). At layer 18 it is **+0.79
(p=0.036)** — there the ranking prunes the *sensitive* matrices **more**.

What it actually does: `up_proj` is the least-pruned matrix in **12/15 cells regardless of whether
`up_proj` is sensitive at that layer**. Since scores are mean-normalised within each matrix, that
is not a scale artifact — `up_proj` simply has a **lighter left tail** than its peers. A
distributional-shape property, not sensitivity.

> **The layer-35 win is the ranking getting lucky where it matters most.** It starves `up_proj`
> everywhere; at layer 35 `up_proj` happens to be the worst thing in the model to prune.

**Consequence: the screening map is not redundant.** The seductive reading — "the scores already
know, skip the expensive map" — is what ρ = −0.15 rules out.

## 7. Robustness is the INTERACTION of depth × projection type ⭐⭐
Rathore's **W4 cluster scan** (his design, exactly): the most robust and most sensitive matrix
types from Strategy 1 (`o_proj` +0.018, `up_proj` +0.676 — measured, not assumed), run across a
robust cluster (**15–21**, around L18) and a one-sided boundary cluster (**29–35**, around L35).
14 layers, 3 sparsities, both `wanda` and `sparsegpt_recon`.

His question was whether robustness belongs to the layer region, the projection type, or their
interaction. **The interaction dominates both main effects, in every cell measured:**

| method | sparsity | region effect | matrix effect | **interaction** |
|---|---|---|---|---|
| wanda | 10% | +0.281 | +0.288 | **+0.442** |
| wanda | 20% | +0.404 | +0.505 | **+0.692** |
| wanda | 40% | +0.575 | +0.795 | **+0.971** |
| recon | 10% | +0.221 | +0.190 | **+0.424** |
| recon | 20% | +0.419 | +0.405 | **+0.812** |

It replicates across both methods, so it is a property of **the model**, not of a method.

**Neither "late layers are sensitive" nor "up_proj is sensitive" is true alone** (@40%):

| | 15 | 17 | 19 | 21 | 29 | 31 | 33 | **35** |
|---|---|---|---|---|---|---|---|---|
| `o_proj` | 0.02 | −0.11 | −0.24 | −0.25 | 0.13 | 0.03 | −0.01 | **−0.48** |
| `up_proj` | 0.20 | 0.21 | 0.04 | −0.06 | 0.06 | 0.32 | 0.72 | **4.70** |

**`o_proj` is robust at every depth — including layer 35, where pruning it *improves* perplexity
(−0.48).** So layer 35 is not sensitive; **layer 35's MLP is.** Its attention is better than fine.

**The boundary is measured, not assumed.** `up_proj` @40% through the sensitive cluster:

> L29 **0.06** → L30 **−0.02** → L31 **0.32** → L32 **0.79** → L33 **0.72** → L34 **1.92** → L35 **4.70**

Layers 29–30 are as robust as the middle. The rise begins at **31**, and the last two layers carry
almost all of it. Our 5-layer map could only say "somewhere between 27 and 35", and Policy B
guessed "32–35 are sensitive" — the truth is **34–35**, with 32–33 barely above the middle.

**Mechanism:** layer 35 feeds the LM head with no downstream layer left to absorb its error
(consistent with 4b and 4c: at the final layer both repair *and* selection are anti-correlated
with what the model needs).

**Two consequences:**
- **Vindicates the per-(layer, matrix) design.** The interaction result means any depth-only or
  matrix-type-only heuristic is wrong in principle. Our policy classifies per cell — correct.
- **Kills the depth inheritance.** Policy B assigns unmeasured layers their nearest measured
  neighbour's class, which is exactly what mislabelled 32–33 as sensitive. It spends 0.30×
  multipliers protecting layers that never needed it. A corrected Policy B should beat everything
  measured here.

## 8. Magnitude is worse than random — and we know why
Median ΔPPL over 35 layer×matrix cells (screening subset, dense 13.559):

| method | 10% | 20% | 40% |
|---|---|---|---|
| magnitude | 0.072 | 0.123 | 0.301 |
| random *(floor)* | 0.038 | 0.083 | 0.241 |
| wanda | 0.030 | 0.035 | 0.123 |
| sparsegpt | 0.026 | 0.063 | 0.115 |
| **sparsegpt_recon** | **0.006** | **0.016** | **0.050** |

Magnitude sits **above the random floor** — worse than chance. Replicated independently on two
model sizes (0.6B random 144-52, 4B random 63-42). `o_proj` is the reproducible exception
(magnitude wins in *both* models: 4B 10-5, 0.6B 21-7).

**Why: aggregation loss.** How you turn a per-weight score into a tile score matters more than
which score you start from — magnitude (weights → block norm) is worst, wanda (weights → tile
mean) better, sparsegpt (measures the tile's actual output effect) is tile-native. This
empirically confirms Rathore's §3.8 prediction ("aggregation may hide important weights").

## 9. Damage is sub-additive within a layer, compounding across layers
Whole-layer ΔPPL is only **0.69–0.81×** the sum of its 7 individual matrices — degradations
overlap. Across layers it compounds, which is why uniform whole-model pruning collapses.

---

# The unplanned experiments — what prompted each, and what it did

Four experiments were run that no plan scoped. Each was prompted by something in the planned
results; none was scoped in advance, and **all of them should have waited until the plan was
finished** (they didn't — see the schedule). Recording the reasoning so a reader can judge
whether each was worth its detour.

### 1. Low-sparsity downstream ladder — 1 / 2 / 5 / 10%
**Prompt.** The planned downstream run (E4b) at 20% came back at **37 / 51 / 41%** retained
ability, while perplexity read a mild-sounding 30.95 (2.3× dense). The planned grid's lowest
whole-model point was 5%, and downstream had only been run at 20/30%. So the entire region where
the model might still *work* had never been measured on any capability metric — we had a
collapse and no idea where it started.
**Did.** Extended the downstream ladder below the planned grid: 1 / 2 / 5 / 10% uniform
`sparsegpt_recon`, plus two extra whole-model perplexity runs at 1/2% so each point had a
perplexity to pair with.
**Found.** 1% → 100% retained; 5% → **90%**; 10% → ~79%; 20% → ~43%. **Finding 1** (the ~5%
answer) and the nonlinearity that sharpens **finding 2**.
**Verdict.** The project's headline number came from here. It should still have run second.

### 2. Depth concentration — same budget, 32/24/16/12/8 layers
**Prompt.** Seb asked whether pruning only the redundant middle would avoid the damage. The
planned data said the sensitive tail is only ~8% of the model, so avoiding it changes little —
but it raised a sharper question the plan does not cover: since damage compounds *across* layers
(finding 5), is it better to concentrate the same tiles into fewer layers?
**Did.** Five whole-model runs at an identical 7.2 layer-equivalent budget (every layer holds the
same 98,560 tiles, so N × ratio = 36 × 0.20 is exact), drawing only from layers 0–31.
**Found.** An inverted-U: peak **1.28× at N=24**, parity at 16, **0.02× at N=8**. Fused with
Policy B's reversal into one mechanism — compounding across layers vs super-linear damage within
one (**finding 6**).
**Verdict.** Genuinely new, and it explains a planned result (Policy B's reversal) that we had
only described.

### 3. `wanda_recon` ablation framing
**Prompt.** The combined variant is **Rathore's** (S5) and was always scoped — but reading the
code showed our ladder *confounds* selection and repair: `sparsegpt_recon` changes both at once
versus `wanda`, so "recon is best" could not distinguish "eq-23 picks better tiles" from
"repairing at all is what matters".
**Did.** Ran his combined variant as a controlled ablation — the repair path is **bit-identical**
between `wanda_recon` and `sparsegpt_recon` (CPU-verified), so selection is the only variable.
**Found.** Repair is worth 1.8×–105×; selection 0.62–1.05×, i.e. nothing (**finding 4**).
**Verdict.** The *experiment* was planned; only the framing was ours. It should have run weeks
ago as part of S5.

### 4. Layers 32/33/34 screening
**Prompt.** Policy B's labels come from screening only 5 layers; every other layer inherits its
nearest measured neighbour. That meant layers 32–34 were labelled *sensitive* purely because
layer 35 is — an assumption underpinning every Policy B result, never checked.
**Did.** Screened layers 32/33/34, all 7 matrices, 10/20/40%, `sparsegpt_recon`.
**Found.** 32 and 33 are **robust**; 34 ramps; only 35 spikes. **Two of three were mislabelled**,
so Policy B has been over-protecting layers that never needed it (**finding 7**).
**Verdict.** This is a partial, differently-scoped **W4** — his design calls for a robust *and* a
sensitive cluster using the two extreme matrix types. Doing it properly is task 2 of the
schedule.

# Coverage of the planned strategies
Audited against the source documents, not from memory:
**Rathore, *Pruning Strategies Using Metrics: Wanda and SparseGPT*** (2026-07-15) — the adopted
plan; and **Fiebiger, *Evaluation and Comparison Extensions*** (2026-07-16) — the extensions.

### Rathore's plan — the backbone

| | strategy | status |
|---|---|---|
| W1 | Representative-layer / individual-matrix screening (5 layers × 7 matrices) | ✅ done (extended to 10/20/40%) |
| W2 | Complete-layer uniform Wanda pruning | ✅ done |
| W3 | Budget-matched layer-wide Wanda pruning (one layer, one budget, allocation varies between matrices) | ✅ **done** → finding 6b |
| W4 | Robust/sensitive region zoom-in (clusters around a robust and a sensitive candidate) | ✅ **done** — 14 layers × 3 sparsities → finding 7 |
| W5 | Final whole-model Wanda: Policy A vs B, budget-matched | ✅ done |
| S1 | Representative matrix reconstruction scan (mask-only vs reconstructed) | ✅ done |
| S2 | Reconstruction-benefit classification (A naturally redundant / B compensatable / C essential) | ✅ **done** — see finding 4b; `scripts/classify_reconstruction.py` |
| S3 | Complete-layer SparseGPT reconstruction | ✅ done |
| S4 | SparseGPT cluster & depth analysis (easiest + hardest matrix types across clusters) | ✅ **done** — 84 runs → refutes 4b's mechanism; see finding 4b |
| S5 | Final whole-model SparseGPT + **combined Wanda→SparseGPT variant** | ✅ done |

**The gap is the cluster/zoom-in stage (W3, W4, S2, S4).** It is also Stage 3 of the Fiebiger
execution plan (~4 h), and it was skipped — the work jumped from whole-layer straight to
whole-model. Notably, **S2 needs no GPU at all**: we already have mask-only and reconstructed
perplexity for all 35 layer×matrix combinations, so the A/B/C classification is a pure analysis
of existing JSON.

### Extensions (Fiebiger) — beyond the adopted plan

| | extension | status |
|---|---|---|
| E1 | random + magnitude controls at identical settings | ✅ screening **+ whole-layer**; ⚠️ whole-model outstanding |
| E2 | 5 seeds for random, mean + range | ✅ screening **+ whole-layer**; ⚠️ whole-model outstanding |
| E3 | sparsity as a primary axis (coarse 10/20/40, fine 8-point) | ✅ done, widened to 1/2% |
| E4a | output-distribution divergence (KL, top-1, cosine) | ✅ done |
| E4b | downstream task accuracy | ✅ done (⚠️ not yet for `wanda_recon`) |

### Execution plan (Fiebiger §7) — stage coverage

| stage | spec | status |
|---|---|---|
| 0 Dense baseline | full model | ✅ |
| 1 Per-matrix screen | all + controls, 10/20/40% | ✅ |
| 2 Whole-layer | all + controls, 10/20/40% | ✅ **done** — controls added, see finding 4c |
| **3 Cluster analysis** | key methods, 10/20/40%, ~4 h | ❌ **never run** |
| **4 Whole-model sweep** | **all + controls**, 8-point grid | ⚠️ **controls missing** |
| 5 Final + downstream | best methods | ✅ (bar `wanda_recon`) |

> ⚠️ **The whole-model results currently have no control floor**, and E1 states the standard they
> fail: *"The controls must be evaluated at identical settings... run within the same sweeps
> rather than deferred to a separate later phase, because a control produced at mismatched
> settings is not a valid comparison."* Screening-scale controls do **not** cover the whole-model
> claims — Rathore §2.5 makes the same point, that different pruning scopes are different
> experiments. Until random and magnitude run at whole-model scale, findings 1 and 4 lack the
> floor and naive reference that make them interpretable.
>
> Note this is the **same experiment** as the proposed `random_recon` test: random selection at
> whole-model scale, with and without repair, is both the floor E1 demands and the decisive test
> of finding #4.

### Deviations from Rathore's logging template (§6)
His template requires fields we do not record: **calibration seed**, **runtime**, and a saved
**mask file** per run. The missing seed is the material one — it makes the runs
non-reproducible in the sense his template intends.

# Caveats
- **One model** (Qwen3-4B), **one tile size** (32×32), **one calibration seed**.
- ⚠️ Downstream measured for `sparsegpt_recon` (+ `wanda` @20%). **`wanda_recon` — the method we
  recommend — has no capability data yet, so that recommendation currently rests on perplexity
  alone, which is exactly the error finding #3 documents.** Runs are queued. Until they land,
  present finding #4 as a perplexity result.
- `wanda_recon`'s perplexity grid covers 5–40% where other methods cover 5–70% (50/60/70% queued).
- Layers 32–34 measured for `sparsegpt_recon` only, so aggregate figures use the canonical 5
  layers to keep every method on identical cells.
- Whole-model runs have no `magnitude`/`random` floor — those controls exist only at screening
  scale.
- **Calibration seed is neither varied nor recorded** in the result JSONs.
- Screening uses a 20% eval subset (5.1× faster); whole-model/downstream use full eval.
- Layers 1–8, 10–17, 19–26, 28–31 still inherit labels from their nearest measured neighbour.
- **One-shot calibration**: Gram matrices come from the dense model per layer, never re-derived
  after upstream pruning (Rathore §4.8).
- Comparisons above ppl ~1000 are meaningless (both models gibberish) and are excluded, not
  reported — this caught a fake "694× win" for Policy B at 70%.
- All methods deterministic (bit-identical reruns); only `random` uses seeds.

# Schedule — finishing the adopted plan
Run in **plan order** (screen → whole-layer → cluster → sweep), not value order. Everything below
was already scoped; none of it is a new idea. Nothing from the "later" list starts until this is
done. **≈ 11 h total, sequential on one GPU.**

| # | task | plan ref | what it settles | est. |
|---|---|---|---|---|
| ~~1~~ | ~~Whole-layer controls~~ — **DONE** (90 runs, 6/6 exit 0) → finding **4c** | Fiebiger Stage 2 · E1/E2 | the floor immediately overturned an unexamined assumption | ✅ |
| ~~2~~ | ~~Cluster / depth zoom-in (W4)~~ — **DONE** (84 runs) → finding **7**: the interaction dominates | Rathore W4 · Stage 3 | answered: neither depth nor matrix type — their interaction | ✅ |
| ~~2b~~ | ~~S4 proper~~ — **DONE** (84 runs, 6/6 exit 0) → **refuted 4b's depth mechanism** | Rathore S4 | answered: reconstructability is component-specific, not depth-specific | ✅ |
| ~~3~~ | ~~Layer-wide budget matching (W3)~~ — **DONE** (15 runs, 3/3 exit 0) → finding **6b** | Rathore W3 | answered: conditional — 7/15, wins where sensitivity structure exists | ✅ |
| **4** | **Whole-model controls** — random (5 seeds) + magnitude, 8-point grid | Fiebiger Stage 4 · E1/E2 | **the floor for findings 1 and 4.** Also answers whether data-aware selection beats random *at all* at whole-model scale — untested, and finding #4 hinges on it. Subsumes the `random_recon` idea | ~5 h |

**Then, and only then:** the list below.

# Later — flagged for a future marathon

Everything here is **unplanned** — it starts only once the schedule above is complete.

1. **Capability-derived sensitivity map** ⭐ — rebuild the map from *downstream accuracy* rather
   than perplexity, then re-run A/B. Finding #3(b) shows Policy B flatters the metric it was fit
   to; this tests that directly and, if it works, produces a policy that optimizes the thing we
   actually care about. **Highest value, and the most interesting.**
2. **Corrected Policy B** — relabel using the now-measured layers 32/33/34 and re-run the A/B
   sweep. It has been protecting robust layers (32–33) for free. Cheap, and should beat every
   policy result we have. Combines naturally with (1).
3. **Tile size 8 / 16 vs 32** ⭐ — the big one. Coarse structure is the prime suspect for why only
   ~5% is redundant. Finding #1 turns this from a nice-to-have into a sharp hypothesis: if
   redundancy is real but the 32×32 grid is too coarse to find it, 8×8 should recover
   substantially more. Needs a re-screen at the new tile sizes.
4. **`wanda_recon` downstream** — our recommended method, no capability data. Run the same
   1/2/5/10/20% ladder to confirm the perplexity win is real ability. Note finding #3(b): a
   perplexity win is not evidence of a capability win until measured.
5. **Depth optimum at *usable* sparsity** — the inverted-U (peak N=24) was measured at a 20%
   budget, i.e. inside the broken regime. Redo at 5%, where the model still works, and the
   optimum may sit elsewhere.
6. **Iterative / sequential calibration** — re-derive H after upstream layers are pruned. Directly
   attacks finding #5 (marginal ≠ joint) and Rathore §4.8. The most scientifically interesting
   of these, and a candidate mechanism for **4b**: whole-model repair computes each layer's H
   from the *dense* model, but once upstream layers are pruned the activations arriving at a
   layer have drifted, so its H is **stale** — repair confidently solves the least-squares
   problem for inputs that no longer arrive. The staleness compounds with depth, so the final
   MLP (end of the longest chain) repairs against the most fictional H. Iterative calibration
   re-measures H against the *actual* pruned inputs at each layer; comparing it to one-shot would
   split our residual error into "information genuinely lost" (irreducible — deleted signal not
   spanned by survivors) vs "H drifted" (fixable). Cheap relative to its value.
7. **Calibration-seed robustness** — never tested; cheap insurance against a seed artifact.
8. **Second model** (Qwen3-0.6B) for the downstream story — the perplexity findings replicate
   across sizes; the capability findings are single-model.
9. **Fill the depth profile** — measure the remaining unmeasured layers, or at least confirm the
   nearest-neighbour assumption holds somewhere in the middle.

# Where things live
```
experiments/screen/              per-matrix screening + baselines (+ plots/)
experiments/screen_wholelayer/   whole-layer (+ plots/)
experiments/wholemodel/          whole-model: uniform, Policy B, wanda_recon (+ plots/)
experiments/depth/               depth concentration (+ plots/)
experiments/downstream/          lm-eval accuracy (+ plots/)
docs/metrics.md                  every metric, and why
scripts/plot_{screening,wholelayer,wholemodel,policy,depth,downstream}.py
src/redundancy/{scoring,recovery,combined,policy,hooks,eval}.py
```
