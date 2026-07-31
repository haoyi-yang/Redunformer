# Tile-Level Redundancy in LLMs — Findings (Qwen3-4B)

**One line:** You can delete about **5% of the model** (as 32×32 weight-tiles) before its real-task ability starts to break — *far* less than perplexity implies — and the thing that makes pruning work is **repairing the surviving weights, not cleverly choosing which tiles to cut.** *(That 5% is the ceiling for hardware-usable **block** pruning specifically — the model's underlying redundancy is ~8× larger but too **diffuse** to cash as blocks; see Finding 10.)*

*Model: Qwen/Qwen3-4B. Tile size: 32×32 (project standard, Rathore §2.4). Dense WikiText perplexity = 13.22. Downstream ability measured on HellaSwag / PIQA / ARC-Easy via lm-eval, normalised to above-chance range: retained = (acc − chance) / (acc_dense − chance).*

---

## Scope, contribution & what we're asking (for the supervisor)

**What this is.** A mid-project **checkpoint**, not a finished paper: 10 findings on tile-level (T×T block) pruning of Qwen3-4B, all at the corrected tile-32 standard, downstream-anchored where it matters, cleared by an internal multi-reviewer QC pass.

**What's genuinely ours (contribution).** *Not a new pruning method* — a careful, honest **characterization** of tile-level pruning: (i) the downstream-anchored **~5% structured budget** (perplexity implied 40–70%); (ii) **repair ≫ tile-selection** (random-select + repair beats calibrated selection); (iii) a concrete demonstration that perplexity is **adversarially gameable**; (iv) the **purity probe** showing the redundancy is *diffuse*; (v) Finding 10 placing our 5% against the unstructured ceiling to show it's a **structured-pruning tax**, not a redundancy limit.

**What we are *not* claiming.** 1×1 (unstructured) pruning to ~50% is **standard SparseGPT / Wanda (2023)** — we reproduce it only as a *known reference point* to bound the tax; it is not a discovery and gives no hardware speedup. "Structured is harder than unstructured" is also known; our value is the specific, per-finding, downstream-anchored measurement on this model.

**Three caveats we state out loud:** (1) every capability finding is **n=1 model** (Qwen3-4B) — the generality gate; (2) F1's headline ladder uses `sparsegpt_recon`, which our own F4 shows is the **weakest** repair variant; (3) F2b's sharpest claim rests on **one task (HellaSwag) at one dose**.

**Three forks we're holding for your steering** (deliberately not pre-run): **(A)** second model (Llama-3.2-3B) for generality; **(B)** iterative/sequential calibration — our one shot at *raising* the 5% ceiling; **(C)** 1×N row strips — could convert some diffuse redundancy into hardware-usable structure. *Which should we spend the GPU on first?*

---

## How to read this doc

Every finding below is written as **What we test → Why we ran it → Result → Reliability**. The *Reliability* tag is the single most important label:

- **downstream-anchored** — judged on real task accuracy. Trust these most.
- **control-relative** — a comparison against a matched control (e.g. random tiles) that shares the same biases, so the *ranking* is trustworthy even if absolute numbers aren't.
- **perplexity-absolute** — rests on raw perplexity; directionally useful but carries the caveat that perplexity misleads (that's Finding 2).

> **Data note:** all headline numbers are at the corrected **32×32** tile size. A prior bug ran some controls at 64×64; those were re-run (2b, 6, 7 re-run clean; commit `a1be30e`). The only residual archived-64×64 evidence is Finding 8's Qwen3-0.6B cross-model check — flagged where it occurs.
>
> **Two dense baselines (important for re-deriving Δppl):** whole-model and downstream numbers use **full eval, dense = 13.22**. The screening-scale numbers (`screen` / `screen_wholelayer` / `cluster`, i.e. the per-layer and per-matrix Δppl in Findings 4b/4c/4d/5/7) use a **fixed 20% eval subset, dense = 13.56**. A screening Δppl is measured against 13.56, not 13.22.

---

## The five things worth saying in the presentation

1. **~5% is the real *structured* redundancy budget** (downstream-anchored) — perplexity suggested 40–70%, and it was wrong.
2. **Perplexity is not just a loose proxy — it is *gameable*.** You can push perplexity *below dense* while the model gets measurably worse on real tasks.
3. **Repair is the whole method.** Which tiles you delete barely matters; reconstructing the surviving weights is what preserves ability. Calibrated tile-selection loses to a coin flip at the usable operating point.
4. **Redundancy is not a per-part property.** Pieces that are safe alone are not safe together, and safety is the *interaction* of depth × matrix-type, not either alone. *(Scope: this non-composition is a **32×32 block** effect — the same 20% budget removed as 1×1 composes fine; Finding 10.)*
5. **The 5% is a *structured-pruning tax*, not the model's redundancy limit.** Delete the *same* fraction as scattered **1×1** weights (same SparseGPT-family repair) and the model keeps ~100% of ability to 20% and ~82% at 50% — the redundancy is real and ~8× larger than 5%, just **diffuse**, so only *block* pruning pays the tax. Honest caveat: 1×1 sparsity buys no hardware speedup, so 5% stays the *usable* structured ceiling (Finding 10).

---

## Reliability at a glance

| Finding | Rests on | Status |
|---|---|---|
| 1 — ~5% redundancy ladder | **downstream-anchored** | ✅ holds |
| 2 — perplexity is nonlinear | **downstream-anchored** | ✅ holds |
| 2b — perplexity is *gameable* | **now single-tile: gaming + drop both tile-32** | ✅ holds (cross-tile gap closed) |
| 3 — Policy B games its own metric | **downstream-anchored** | ✅ holds |
| 4 — repair ≫ selection (incl. same-tile ablation) | control-relative | ✅ holds-stronger (clean tile-32) |
| 4b — repair backfires at final MLP | control-relative | ✅ holds |
| 4c / 4d — selection loses to random | control-relative (corrected tile-32 floor) | ✅ holds-stronger |
| 5 — isolation does not compose | **downstream-anchored** | ✅ holds |
| 6 / 6b — depth concentration | control-relative; **now tile-32**; exploratory | ⚠️ monotonic (no interior optimum); confounds disclosed |
| 7 — depth × matrix interaction | control-relative; **now tile-32**, perplexity-only, marginal | ⚠️ holds as a marginal perplexity-sensitivity map |
| 8 — magnitude < random | control-relative (vs random control) | ✅ holds |
| 9 — damage sub-additive in-layer, compounds across | perplexity-absolute | ⚠️ direction holds, carries caveat |
| 10 — 5% is a *structured-pruning tax* (1×1 ≫ 32×32) | **downstream-anchored** (+ perplexity) | ✅ holds — 1×1 keeps ~100% to 20%, ~82% at 50% |

---

# Findings

## 1 — About 5% of tiles are removable before real ability breaks ⭐ (downstream-anchored)

**What we test.** How much of the model we can delete in 32×32 tiles before it fails on actual reasoning tasks — not just on a text-prediction score. We prune 1/2/5/10/20/30% with `sparsegpt_recon` (SparseGPT-select + SparseGPT-repair) and measure HellaSwag / PIQA / ARC-Easy. *(Note: Finding 4 shows `sparsegpt_recon` is actually the weakest of the three repair variants at 5% — all repair variants cluster near dense, so the ~5% number is unaffected, but this ladder is not "the best method.")*

**Why we ran it.** Our perplexity curves implied 40–70% was removable. Perplexity is only a proxy; we wanted the honest number judged on abilities a person cares about.

**Result.** ~5% of tiles can be removed while keeping **~90%** of above-chance ability — and it lands at ~90% on all three tasks (the exact 90/90/90 is a rounding coincidence, not a law). **5% is the capability-*preserving* operating point** (~90% retained); it is not a cliff — 10% is still *usable-but-degraded* (~79%), and the real collapse is between 10% and 20% (79% → 43%). This 5%-vs-10% distinction matters for Findings 4c/4d/8, whose "selection loses to a coin flip" claim is scoped specifically to the 5% capability-preserving point. **(This 5% is the *structured* ceiling — see Finding 10: the same weights removed as scattered 1×1 keep ~100% ability to 20%; the model's redundancy is ~8× larger but diffuse.)**

**Numbers (retained above-chance ability, HellaSwag/PIQA/ARC-Easy; tile-32):** 1% → 100/103/98; 2% → 97/98/93; **5% → 90/90/90 (avg 90%)**; 10% → 73/81/82 (avg 79%); 20% → 37/51/41 (avg 43%); 30% → 17/29/24 (avg 23%). Paired WikiText ppl: 5%=15.62, 10%=18.78, 20%=30.95, 30%=61.56 (all tile-32).

**Figure:** `figures/f1_redundancy_ladder.png` — retained ability vs sparsity, three tasks, 90% line + 5% marker.

---

## 2 — Perplexity is a *nonlinear* proxy, dangerous exactly where you'd rely on it ⭐ (downstream-anchored)

**What we test.** Whether WikiText perplexity actually tracks real ability across the sparsity range.

**Why we ran it.** Perplexity is the fast, cheap number we'd instinctively use to rank methods. If it flatters an aggressive setting, any perplexity-only headline is unsafe.

**Result.** Perplexity never points the *wrong* way, but the mapping to capability is brutally nonlinear. At 5% ppl reads 1.18× dense and the model keeps ~90% of ability; at 20% ppl reads only 2.34× ("about twice as bad") yet **over half the model's ability is gone** (43% retained); at 30%, 4.66× leaves 23%. So perplexity is fine in the usable regime (≤5%) and dangerously flattering exactly where you'd use it to justify aggressive pruning.

**Numbers (tile-32):** ppl-ratio → retained-avg: 1.18×→90%, 1.42×→79%, 2.34×→43%, 4.66×→23%.

**Figure:** `figures/f2_perplexity_scissors.png` (shared with 2b).

---

## 2b — Perplexity is *adversarially gameable*: push it below dense while the model gets worse ⭐⭐ (now single-tile at tile-32)

**What we test.** Whether you can deliberately make perplexity look *better* while real ability drops — by masking o_proj only in the layers where it helps in isolation.

**Why we ran it.** This is the sharpest possible statement of Finding 2, and the reason we ran ~11 downstream evals: if perplexity can be *driven the opposite direction* from capability, the field's default yardstick is not just loose but exploitable.

**Result (now at the project-standard tile-32).** Restricting Wanda o_proj masking to the 8 layers where o_proj improved in isolation (17–21, 32, 34, 35) drives WikiText perplexity **below dense at tile-32** — **12.21 at 20% dose (−1.01)** and **11.45 at 40% (−1.77, i.e. 13% "better" than dense)** — while real ability goes flat then down: at 40% HellaSwag drops a genuine **−3.6σ** (dense-stderr convention; ≈−2.5σ two-sample, ~1.6 pp absolute), ARC-Easy −1.5σ, PIQA flat. Every point reproduces the archived tile-64 run within noise (tile-64 was 12.24 / 11.50). Two controls make it airtight, **also re-run at tile-32**: blanket o_proj across all 36 layers never beats dense (13.93 at 20%), and SparseGPT **repair erases the perplexity win entirely** (13.28 / +0.06 at 20%, 13.64 / +0.42 at 40%) — because repair reconstructs the dense output. The "gain" exists *only because you didn't reconstruct*: a textbook proxy exploit.

**⚠️ Caveat (updated).** (1) **Cross-tile gap — now CLOSED.** The gaming *and* the downstream drop are both **tile-32**: the exact gamed model (Wanda o_proj, 8 layers, 40%) reads **11.45 ppl (looks better than dense)** while its HellaSwag ability drops **−3.6σ** — one single tile-32 network, both metrics, opposite directions. (Originally the ppl was archived tile-64 and the drop tile-32; the tile-32 ppl re-run — `experiments/oproj_targeted_tile32/` — removes the splice.) (2) **Still one task at one dose** — the opposite-direction effect is carried mainly by HellaSwag at 40% (−3.6σ, ~1.6 pp absolute); PIQA is flat and ARC −1.5σ — and the below-dense magnitude is partly select-on-WikiText / measure-on-WikiText overfitting. The *direction* and the controls are sound; single-task dependence is the remaining honest limit.

**Figure:** `figures/f2_perplexity_scissors.png` — capability-vs-perplexity-ratio scatter; the o_proj points sit left of the dense line ("looks improved") yet at/below dense ability.

---

## 3 — "Smart" layer budgeting (Policy B) games the metric it was built from ⭐⭐ (downstream-anchored)

**What we test.** Whether a sensitivity-aware budget that protects fragile regions (Policy B) beats plain uniform pruning (Policy A) at the same compression — *and* whether its perplexity win shows up on real tasks.

**Why we ran it.** The sensitivity map was built *from perplexity*, so Policy B risks just flattering the number it was tuned on. Downstream accuracy at matched budgets separates a genuine win from a mirage.

**Result.** Both halves hold. Policy B genuinely edges uniform on perplexity by ~1.1–1.6× (peaking 1.58× at 30%), but that **badly overstates the capability win**: at its 1.58× peak it buys essentially **zero** real ability (+3.2/−0.7/−0.5 retained-ability pp = noise). It only wins for real at lower sparsity (20%: +5.2/+1.3/+7.0 retained-ability pp). At *matched perplexity* it delivers ~10–18 pp **less** capability than uniform on all 6 measurements. Confirmed in metadata: it protects the perplexity-flagged "sensitive" class (pruned 0.042) and hammers the "robust" class (0.224) — and because that map came from perplexity (dominated by final layers feeding the LM head), it flatters the metric it was fit to.

**Numbers (tile-32, sparsegpt_recon):** ppl gain 1.10/1.19/1.31/1.58/1.11× at 5/10/20/30/40%; downstream Δ(B−A) in **retained-ability pp** = +5.2/+1.3/+7.0 at 20%, +3.2/−0.7/−0.5 at 30%.

**Figure:** `figures/f3_policyB_divergence.png` — perplexity-gain bars vs capability-gain line; they diverge at 30%.

---

## 4 — Repair is everything; *which tiles you pick* barely matters ⭐ (control-relative)

**What we test.** Whether the model's recovery comes from *selecting* good tiles to delete or from *repairing* (least-squares reconstruction of) the surviving weights after deletion.

**Why we ran it.** These are the two levers. Knowing which one carries the method tells us where to spend effort — and it's the opposite of the field's instinct (which obsesses over selection).

**Result (now proven with the full tile-32 paired ablation).** Repair carries the entire method — and *which tiles you pick contributes nothing*. At 5% whole-model, **repairing after deleting *random* tiles (`random_recon`, 14.64) is the best result of all** — better than repairing after "smart" Wanda selection (`wanda_recon` 15.38) or SparseGPT selection (`sparsegpt_recon` 15.62), and all three land near dense (13.22). Selection *without* repair (Wanda 28.31, SparseGPT 28.03) is beaten by a plain coin flip (random 22.87). The causal proof is the same-tile pair: `random` and `random_recon` prune the **identical** tiles at the same seed (verified byte-identical in code), so adding repair alone drops perplexity sharply at *every* seed — e.g. seed 1: **22.87 → 14.31** (same tiles, repair the only difference); across all 5 seeds the medians are 22.87 → 14.64. Repair is the whole lever; calibrated selection not only fails to beat random, random-select-then-repair actually *edges out* calibrated-select-then-repair.

**Numbers (tile-32, 5% whole-model, dense 13.22):** repair — random_recon 14.64 (median of 5 seeds, 14.2–14.9), wanda_recon 15.38, sparsegpt_recon 15.62; no-repair — random 22.87; selection-only — wanda 28.31, sparsegpt 28.03; magnitude 3449.

**Figure:** `figures/f4_repair_vs_selection.png` — repair methods (near dense) vs selection-only (2× worse) vs no-repair random, at 5%.

## 4b — Repair backfires exactly where it's needed most (final-layer MLP) (control-relative)

**Result.** At the deepest layer (L35), adding *more* calibration makes things *worse*. At 40%, ordered by how much a method leans on the calibration signal, the L35 damage rises: random 2.38 < magnitude 3.20 < sparsegpt 4.53 < wanda 4.80 < `sparsegpt_recon` 5.39 — so the most calibration-dependent method (`sparsegpt_recon`, the only one that actually reconstructs) is the **single worst** choice there. (Magnitude, with zero calibration, sits *inside* the random seed spread [2.18–3.37], so the clean contrast is the three calibrated methods vs the random floor.) Repairing against a stale/ill-conditioned final-layer signal actively harms — the one place the "always repair" rule inverts.

## 4c / 4d — At the usable operating point, tile-selection loses to a coin flip ⭐⭐ (control-relative, corrected tile-32)

**What we test.** Whether data-aware tile scores (Wanda, SparseGPT) beat deleting *random* tiles — at 5% whole-model, and layer by layer.

**Why we ran it.** A calibrated selection rule only earns its complexity if it beats chance. The earlier controls ran at the wrong tile size (64); we re-ran random + magnitude at tile-32 to give the claim a valid floor.

**Result (sharper after correction).** At **5% whole-model** — the setting where the pruned model still works — Wanda and SparseGPT each beat only **1 of 5** random seeds and lose to the random median (22.87). Per layer: **L0** data-aware is essential (random blows up at 40%); **middle layers** ~indistinguishable from random until 40%; **L35** selection **backfires** — all 9 calibrated method×sparsity cells are worse than random (z = +3.5 to +6.6; the load-bearing claim). At 40% the damage also grows monotonically with how much a method relies on the calibration signal; at 10–20% the three calibrated methods are within noise of each other.

**⚠️ Scope note (load-bearing — say it exactly this way):** "selection loses to random" is a claim about the **5% capability-preserving operating point** (the same 5% as Finding 1). Above it (10–20%), selection *does* start beating random — but the model is already past the capability cliff there *for these no-repair ladders* (random median 221 ppl at 10%, 178k at 20%), so it's a race between broken models. Consistent framing across Findings 1/4c/4d/8: *at the sparsity where capability is preserved (5%), calibrated selection is worthless; above it, everything without repair is degrading anyway.* Also note the honest **low power**: 5 seeds, and the calibrated points (wanda 28.3) sit inside the random spread (18–48), so this is "selection ≤ a coin flip," not a large-margin loss.

**Numbers (tile-32):** 5% whole-model — random seeds 18.0/20.2/22.9/26.7/48.0 (median 22.9); Wanda 28.3, SparseGPT 28.0 (each beat 1/5); magnitude 3,449 (0/5); sparsegpt_recon 15.6 (5/5). L35@40% dPPL: random 2.38 (best) < magnitude 3.20 < sparsegpt 4.53 < wanda 4.80 < sparsegpt_recon 5.39 (worst).

**Figure:** `figures/f4_L35_backfire.png` — L35 dPPL by method ordered by calibration amount, random floor shaded.

---

## 5 — "Robust in isolation" does not compose: marginal safety ≠ joint safety ⭐ (downstream-anchored)

**What we test.** Whether a piece being prunable *on its own* tells you it's safe to prune once *many* pieces go at once.

**Why we ran it.** Our redundancy map is built one matrix at a time. The entire strategy leans on trusting that map jointly — this checks whether that assumption holds.

**Result.** It doesn't. One at a time, most of the model is harmless: at 20% sparsity 83% of screened matrices (with repair) barely move perplexity, and even a whole single layer's 7 matrices together stay near dense (except final-layer MLP). Yet pruning the **whole model** at 20% destroys most real ability (37/51/41% retained, ~60% gone). The map measures **marginal** damage; we were reading it as **joint**. The non-composition is an across-layer effect: 36 individually-fine hits compound. This is the quantified sequential-dependency limit. **(Scope — see Finding 10:** this is a property of **32×32 block** pruning — the *same* 20% removed as scattered 1×1 composes fine (~100% retained), so it is marginal-vs-joint *for block deletion*, not a fundamental limit of the weights.)

**Numbers:** isolated @20% (screening dense 13.56) — 83% of matrices ΔPPL<0.12; whole-layer joint @20% (screening dense 13.56) — **L0 13.72 / L9 13.95 / L18 13.61 / L27 13.65** (all within ~0.4 of screening dense) vs **L35 17.16**; whole-model joint @20% (full eval, dense 13.22) — ppl 30.95, ~43% ability retained (downstream).

**Figure:** `figures/f5_scope_escalation.png` — perplexity vs pruning scope (matrix → layer → whole model), whole-model bar annotated "only 43% ability retained."

---

## 6 / 6b — Concentrating a fixed budget hurts monotonically; over-concentration collapses (control-relative, tile-32, ⚑ exploratory)

*⚑ Exploratory follow-up — **not** in the adopted plan. We added it because the reallocation result (Finding 3) raised the question: given a fixed budget, is it better to spread cuts across many layers or pack them into fewer? Labeled exploratory to avoid dressing a post-hoc probe as pre-registered.*

**What we test.** (6) Given a fixed tile budget, spread it thinly across many layers or pack it densely into fewer? (6b) When a layer gets one budget, does letting its 7 matrices share it unevenly beat uniform sparsity?

**Result (re-run clean at tile-32 — corrects an earlier tile-64 artifact).** The budget is packed into N ∈ {32, 24, 16, 12, 8} evenly-spaced layers (the recovered original design; local sparsity rises as N falls to hold the budget fixed). Tile-32 perplexities: **21.65 (N=32), 22.31 (N=24), 29.38 (N=16), 47.02 (N=12), 1185 (N=8, destroyed)**. The two most-spread settings (N=32, N=24) are **statistically indistinguishable** — a ~3% gap, and `sparsegpt_recon` is deterministic here (the 3 seeds returned identical perplexity, so there is no variance estimate). Quality then degrades **clearly** for N ≤ 16 and **collapses** at N=8. The earlier tile-64 run reported a "sweet spot" at N=24 (24.17 vs N=32's 24.94) — but that ordering **flips sign at tile-32** (N=32 now edges N=24), at the same ~3% magnitude, so the "optimum" was never real: **there is no interior optimum.** The robust, keepable claim is only: *spreading a fixed budget is at least as good as concentrating it, and extreme concentration collapses* — and that collapse is largely a restatement of Findings 5/9 (cross-layer compounding + the within-layer super-linear cliff), not independent evidence of a depth penalty.

For **6b** (layer-budget matching): a coin flip overall (7/15), and on inspection the allocation is **essentially uncorrelated with measured sensitivity** (Spearman |ρ| ≈ 0.15 ≈ 0) — so it does *not* discover which matrices are fragile; any win is a fixed "always starve up_proj" bias getting lucky, not intelligent structure exploitation. Its only real signal is the L35-vs-L0 effect-size contrast, not the win count. 6b is clean tile-32.

**⚠️ Disclosed confounds (why this stays a secondary, exploratory finding):** (a) all depth configs prune only layers **0–31**, sparing the fragile tail 32–35 — so any "beats uniform" comparison is invalid (uniform prunes all 36 incl. the tail) and is **dropped**; (b) the over-concentration collapse is confounded with Finding 9's within-layer super-linear cliff (the fixed budget forces N=8 to 90% local sparsity). We present only the within-experiment monotonic trend, at tile-32.

**Figure:** `figures/f6_inverted_u.png` — perplexity vs concentration (log y) at tile-32: monotonic rise from the spread end (N=32) to the N=8 collapse, no interior minimum.

---

## 7 — Robustness is the depth × matrix-type *interaction*, not either alone ⭐⭐ (control-relative, tile-32, perplexity-only)

**What we test.** Whether prunability is set by depth, by matrix type, or specifically by the *combination* — pruning o_proj (attention-out) vs up_proj (MLP) across a mid band (15–21) and final layers (29–35).

**Why we ran it.** If it were purely "late layers fragile" or "MLP fragile," a one-line rule would suffice. If it's the pairing, every 1-D rule is wrong in principle and per-(layer, matrix) budgeting is justified.

**Result (re-run clean at tile-32).** On the perplexity axis, o_proj is safe to prune at *every* depth (flat-to-negative, even at the final layer), while up_proj is safe everywhere *except* the last one or two layers, where it explodes: at 40% Wanda, **up_proj L35 +4.61 dPPL** (L34 +1.97, L33 +0.79) vs **o_proj L35 −0.43**. So "late layers sensitive" and "MLP sensitive" are each false alone — only *the final layers' MLP* is. The pattern is essentially unchanged from the earlier tile-64 run (up_proj L35 was +4.70), so it's a real model property, not a tile artifact, and it replicates across Wanda (mask) and sparsegpt_recon (repair).

**⚠️ Caveat + reconciliation (present as a marginal map, not a safety claim):** this is **perplexity-only, no downstream anchor**, and a *marginal* (one-matrix-at-a-time) result. Two honesty points the reviewers stressed: (i) "o_proj is safe" is certified by a perplexity *decrease* — the exact signal Finding 2b proves is **gameable** in the o_proj direction, so it is an internal tension unless scoped as marginal-only; (ii) "the interaction wins in all 6 cells" is **pseudo-replication** — 6 correlated views (2 methods × 3 sparsities) of a single late-layer up_proj effect, not 6 independent confirmations. Present this as a **marginal perplexity-sensitivity map**, not a deployment/safety recommendation. (Also: 2b's "8 o_proj-improving layers" come from the 14-layer cluster scan; the "32/36 improving" from the full-model scan — different scans, name both.)

**Figure:** `figures/f7_depth_matrix_interaction.png` — dPPL vs depth, o_proj vs up_proj lines fanning apart only at L34–35.

---

## 8 — Magnitude pruning is *worse than random* — and we know why (control-relative)

**What we test.** When deleting 32×32 tiles, is picking the smallest-magnitude tiles (the textbook default) better or worse than random?

**Why we ran it.** Magnitude is the obvious rule. If it can't even beat random block selection, tile pruning genuinely needs a smarter signal and naive intuition is a trap.

**Result.** At 5% (the capability-preserving point) magnitude **destroys the model** (ppl 3449 — well into the "destroyed" zone) while random stays roughly functional (median 22.9, ~1.7× dense). State this **qualitatively**: *at the sparsity where random still works, magnitude is already destroyed* — the exact "×N-worse" ratio is not meaningful because its numerator sits in the destroyed zone (>1000 ppl). The direction is robust and reproduces on the smaller **Qwen3-0.6B** model across 196 layer×matrix cells (*archived tile-64 cross-model check*): random beats magnitude 144–52; the one consistent exception is o_proj. Mechanism: per cell, magnitude loses because **within-tile magnitude-averaging hides the few important weights** in an otherwise low-norm tile ("aggregation loss" — a per-tile *selection* effect that explains the per-cell ordering); those per-cell deficits then **compound across layers** (this is Finding 9, a separate effect) into the model-scale collapse. We do not infer the exact model-scale factor from the per-cell median — the collapse is stated qualitatively.

**Figure:** `figures/f8_magnitude_vs_random.png` — ppl (log) vs sparsity, magnitude vs random-median-with-band, 5% point annotated.

## 9 — Damage is sub-additive within a layer, compounding across layers (perplexity-absolute)

**Result.** Within a layer, damage overlaps (whole-layer ≈ 0.5–0.8× the sum of its 7 matrices for the controls) → sub-additive. Across layers it super-compounds (whole-model damage runs 17× to thousands× the sum of per-layer damage). **Caveats:** the exact 0.5–0.8 band is one slice — sub-additivity is method-dependent (data-aware methods sit near 1.0, ~additive), L0 is a super-additive exception, and this is perplexity-absolute. F8 (a magnitude-vs-random comparison at identical settings) is robust to that caveat; F9 carries it.

---

## 10 — The 5% is a *structured-pruning tax*, not the model's redundancy limit ⭐⭐⭐ (downstream-anchored + perplexity)

**What we test.** Whether the ~5% ceiling (Finding 1) is a property of *the model* or of *the 32×32 tile*. We delete the **same weights** at the **finest possible granularity (1×1, unstructured)** with the **same SparseGPT-family repair**, matched weight-sparsity — granularity is the intended difference. (One disclosed asymmetry: 32×32 gets the *exact* joint-LS tile repair while 1×1 gets the *weaker* sequential sweep — which handicaps the 1×1 side, so the gap is conservative, not inflated.)

**Why we ran it.** Standard unstructured SparseGPT reaches ~50% on LLMs; we get ~5% at 32×32. If 1×1 also broke at ~5%, then 5% is a real redundancy limit. If 1×1 sailed on, 5% is the price of *block structure*. (The purity probe predicted the answer: the removable weights are **practically diffuse** — harvestable as pure tiles ≈0 at any block ≥2×2 — so only fully-unstructured pruning should reach the redundancy. Note the probe's *raw* verdict label reads "CLUSTERED" because there is a statistically significant 2×2 co-location (z≈7); it is the near-zero *harvestable mass* that makes it diffuse in practice, not the significance test.)

**Result.** 1×1 blows the 32×32 ceiling away — on **both** axes, so it is not perplexity-gaming:
- **Perplexity** (same repair, matched sparsity): 1×1 stays near dense where 32×32 collapses — **1×1 at 50% (15.69) ≈ 32×32 at 5% (15.62)**, a ~10× sparsity ratio at iso-perplexity.
- **Downstream capability** (matched calib 64, real tasks): 1×1 keeps **~100% of ability through 20%, 98% at 30%, 91% at 40%, 82% at 50%** — at the *same doses* 32×32 retains only **43% at 20%, 23% at 30%, collapsed at 50%** (and is already down to 79% by just 10%). 1×1 crosses the 90%-retained bar (Finding 1's definition of the 5% point) at **~41%** — now *measured*: p40 retains 0.913, p50 0.819, so the crossing sits inside the small 40→50% gap, not across the steep part. On the *same capability axis*, unstructured redundancy is **~8× larger** than the 32×32 ceiling.

So the model's redundancy is real and large (>20% removable at ~zero capability loss); it is just **diffuse** — scattered weight-by-weight, not packaged into removable blocks — so any tiling ≥2×2 can't reach it. The comparison is **conservative**: 32×32 actually gets the *stronger* exact joint-LS tile repair while 1×1 gets the weaker sequential sweep, so giving 1×1 the exact repair would only widen the gap.

**⚠️ The honest framing — say it exactly this way.** This is a **structured-pruning tax, not a free 10× win.** Unstructured 1×1 zeros give **no speedup on dense-GEMM hardware** (the matmul still runs every multiply); only *block/tile* sparsity maps to acceleration — and that is precisely the kind that caps at ~5%. So: *in principle removable* ≈ >20% (unstructured); *removable in a way hardware can exploit* ≈ 5% (structured); the large gap between them is the tax this model pays for its diffuse redundancy. Read 5% as "block pruning cashes only a small slice of a large, scattered redundancy," **not** "this model has little redundancy." (Secondary caveat: the 1×1 repair calibrates on WikiText and the downstream tasks are multiple-choice; a generative/OOD probe would further harden the capability half.) **Baseline note:** 1×1 unstructured pruning to ~50% is *standard* SparseGPT/Wanda — used here only as a known reference point to bound the tax, **not** as a new result.

**Numbers (Qwen3-4B, dense ppl 13.22 / retained 1.00):** 1×1+repair ppl 13.48 / 13.73 / 14.51 / 15.69 at 20/30/40/50% (32×32: 30.95 / 61.56 / 76.30 / 137.52). 1×1 retained-ability 1.00 / 1.00 / 0.98 / 0.91 / 0.82 at 10/20/30/40/50% (32×32: 0.79 / 0.43 / 0.23 / collapsed-at-50 *inferred* from ppl 137.5, no downstream run). Probe: harvestable-as-pure-tiles 5.0% → 0.02% → ~0 at T = 1 → 2 → ≥4 (practically diffuse).

**Figure:** `figures/f10_structured_tax.png` — two panels: (L) perplexity vs sparsity, 1×1 vs 32×32; (R) retained ability vs sparsity, 1×1 vs 32×32, 90% line + the ~5% → ~40% crossing.

---

# Two things we are careful about (for the talk)

1. **"Usable operating point" means the *structured/block* capability-*preserving* point, 5%.** Findings 4c/4d/8 say selection/magnitude lose "at the usable point" — that point is 5% (~90% ability retained) *for 32×32 block pruning* (unstructured 1×1 preserves capability to ~40%; Finding 10). Finding 1's 10% is *usable-but-degraded* (~79%), a different thing; above 5% the selection/magnitude comparisons flip or become races between broken models. We state the 5% scope on every affected finding.
2. **The o_proj / perplexity story is marginal, not a recommendation.** Finding 7's "o_proj is safe everywhere" is a *one-at-a-time, perplexity-only* observation (now tile-32); Finding 2b shows the joint, downstream-anchored reality (gaming) — and it uses the *same* perplexity signal 7 leans on. The two agree only once you separate marginal (7) from joint (2b) scope, so 7 is a sensitivity map, never a safety claim.

---

# Pending experiments

## ★ The tile-size experiment — RESOLVED (now Finding 10)

**Verdict: the 5% is a structured-pruning tax — not a tiling artifact, and not the model's redundancy limit.** The two fast instruments the plan called for both landed and agree:
- **Purity probe** → the removable 5% is **diffuse** (harvestable as pure tiles ≈0 at any block ≥4×4; only 1×1 reaches it). This *predicted* that no intermediate tile size would help.
- **1×1 endpoint (decisive)** → confirmed on perplexity **and** downstream (Finding 10): unstructured redundancy is ~8× the 32×32 ceiling, but is uncashable as hardware speedup.

**Settled — the intermediate *square* sweep {16, 8, 4}.** The probe predicts a null (harvestable ≈0 for every T≥2) and the 1×1 endpoint already brackets the curve, so multi-day GPU on intermediate square tiles has near-zero discriminating power. One T=4 anchor could be added later purely to *draw* the C(T) curve — it will not change the conclusion. **Still open — non-square 1×N strips.** The probe shows *strong row anisotropy* (ξ_row ≈ 2.9 vs ξ_col ≈ 0.7; A ≈ 0.24 = row-correlated, **not** low/isotropic and not columnar), so 1×N *row* strips could in principle harvest more than square tiles of the same area. This is the one granularity question the 1×1-vs-32×32 endpoints do **not** settle — worth a cheap check before closing shape.

## Other pending items
- ~~Close the 2b cross-tile gap~~ **✓ done** — tile-32 ppl re-run of the exact gamed model reproduces below-dense (12.21 / 11.45); both controls also re-run tile-32 (Finding 2b; `experiments/oproj_targeted_tile32/`).
- ~~Confirm Finding 6 at tile-32~~ **✓ done** — re-run refuted the N=24 optimum (it's monotonic at tile-32); Finding 7 also re-run clean at tile-32.
- **Recommended-method capability data:** wanda_recon / random_recon downstream ladder (finalising at tile-32 now).
- **Second model (Llama-3.2-3B)** — the generality gate; every capability finding is n=1 model.
- **Iterative / sequential calibration** — the one shot at genuinely raising the ~5% ceiling.

---

# Where things live

- Active data (all tile-32): `experiments/{downstream, wholemodel, screen, screen_wholelayer, layer_budget, depth, screen_cluster}/` — `depth` (F6) and `screen_cluster` (F7) were re-run clean at tile-32. **Finding 10:** `experiments/tilesize/{ppl, downstream, probe}/` (1×1 sweep + purity probe). **Finding 2b tile-32:** `experiments/oproj_targeted_tile32/` + `experiments/oproj_blanket_tile32/`.
- Archived tile-64 (superseded / closed threads): `experiments/archive/{full_scan, depth, oproj*, screen_cluster*}/` — the old tile-64 versions, kept for provenance.
- Figures: `figures/`
- Raw working log with full history: `FINDINGS.md`
