# How might redundancy depend on depth, component, family, size, or task?

**Group 5 — Neuron / FFN-unit redundancy**  
**Proposal question 5 (Weeks 3–4)**  
**Model:** `Qwen/Qwen3.5-4B` (4-bit, 8 GB GPU) · **Baseline:** WikiText-2 test PPL **17.06**

> **Errata (2026-09-09).** This is the Weeks 3–4 *plan*, written before any
> results. The hypothesis outcomes below are Qwen-only and several have since
> reversed: **H1 refuted** (the depth-tertile ablation never makes the middle
> tertile the most expensive), **H2 refuted as stated** (the clean GPT-2 size arm
> breaks the trend), **H3 supported on gated FFNs only**, **H5 not supported**
> with **H5′** (replacement) taking its place. Frozen verdicts:
> [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md). The scope reasoning and
> experiment design in this document still stand.

---

## Summary

At neuron granularity we treat each **post-activation FFN unit** (dimension of the SwiGLU intermediate after gating) as one neuron. Redundancy means multiple such units carry overlapping information: they activate on similar inputs, can be masked with little effect, or are interchangeable after short recovery.

We expect redundancy to be **non-uniform** across depth and task, **concentrated in low-importance FFN units**, and **scale with model size** within the Qwen3.5 line. Component comparisons stay inside the FFN block; other granularities (heads, blocks, weights) are out of scope.

---

## Scope table

| Dimension | In scope? | Rationale | Planned experiment (Weeks 5–8) |
|-----------|-----------|-----------|--------------------------------|
| **Depth** | **Yes** | FFN neurons at different layer indices may specialize (early syntax vs late semantics). Geva et al. show layer-dependent memory roles. | Per-layer **activation frequency** and **ablation ΔPPL** on WikiText; compare early / middle / late **tertiles** of layers. |
| **Component type** | **Partial** | Neurons live in **FFN (MLP) submodules** only. Qwen3.5 layers also contain Gated DeltaNet and attention — we do not claim neuron redundancy there. | Hook `gate_proj`, `up_proj`, and post-activation hidden states; define primary neuron index on **post-SwiGLU activation**. Optional: correlate gate vs up channel importance. |
| **Model family** | **Limited** | Hardware limits us to **Qwen lineage**. Cross-family (LLaMA, Mistral) needs more VRAM / mlsp. | **Qwen3.5-4B** (primary) vs **Qwen3-0.6B** (fallback baseline PPL 39.78): compare **fraction of rarely-firing FFN neurons** at matched calibration size. |
| **Model size** | **Yes** | Lazy-neuron and scaling literature suggests larger models may have more spares. | **Qwen3.5-4B** vs **Qwen3.5-9B** (#11): same WikiText calibration, compare activation sparsity and ablation curves at 10% / 25% removal. |
| **Task** | **Yes** | Downstream tasks may activate different neuron subsets than raw LM perplexity. | WikiText vs **PIQA** (and optionally ARC-Easy): **Jaccard overlap** of top-*k* active neurons per layer; task-conditioned ablation of “WikiText-redundant” neurons. |

**Explicitly out of scope:** attention-head redundancy (Group `attn_head`), full-block removal (Group `Block`), unstructured weights (Group `Weight`), tile-level patterns (Group `Tile`).

---

## Hypotheses

### H1 — Depth: middle FFN layers are *less* redundant than early/late layers

**Reasoning:** Early layers encode local features; late layers are task- and position-sensitive. Middle layers often hold the most distributed factual “memory” (Geva et al.) where individual neurons are more specialized and less interchangeable.

**Test:** For each layer ℓ, compute mean activation frequency *f*ℓ and ablation importance *I*ℓ (ΔPPL when masking the bottom 10% neurons by *f*). Compare tertiles {early, mid, late}.

**Prediction:** ΔPPL from low-frequency neuron removal is **smallest in early layers**, **largest in middle layers**.

### H2 — Model size: larger Qwen3.5 models have a higher fraction of redundant neurons at the same removal ratio

**Reasoning:** Larger capacity may allow more lazy or duplicate FFN units (Lazy Neuron phenomenon); redundancy-guided pruning should outperform random pruning more clearly at 4B than 0.6B for the same sparsity.

**Test:** On matched WikiText calibration (e.g. 512 sequences), measure % neurons with activation frequency &lt; ε. Then remove bottom 25% by frequency; compare ΔPPL vs random 25% for 0.6B, 4B, 9B.

**Prediction:** **Sparsity of firing** increases with size; **ΔPPL gap** (redundancy-guided vs random) increases with size.

### H3 — Task: neurons redundant under WikiText are not fully redundant under PIQA

**Reasoning:** LM perplexity stresses next-token prediction on encyclopedic text; PIQA stresses physical commonsense. Overlap of high-activation neuron sets should be incomplete.

**Test:** Build per-layer sets *S*WikiText, *S*PIQA of top 5% neurons by activation frequency on 1k prompts each. Report |*S*WikiText ∩ *S*PIQA| / |*S*WikiText ∪ *S*PIQA| per layer.

**Prediction:** **Low Jaccard overlap** in middle/late layers; ablating WikiText-redundant neurons hurts PIQA **more** than ablating WikiText-redundant neurons hurts WikiText PPL at the same sparsity.

**Result (2026-07-30, `recovery_week11-12.md`):** **Supported behaviourally, with a caveat on the overlap.** Ranking by importance (not frequency) on PIQA prompts vs WikiText-2 gives a Jaccard of 0.33 on the 10% masked set — clearly task-dependent, but 6.3× the chance floor of 0.05, and full-ranking Spearman is +0.58, so the redundant set is *mostly* corpus-independent. The behavioural test is the decisive one and shows a clean double dissociation: at 25% masking the WikiText ranking is better on perplexity (52.9 vs 94.8) while the PIQA ranking is better on PIQA accuracy (0.630 vs 0.594, +3.6 pp).

### H4 — Component: gate and up projections correlate, but post-activation magnitude is the right pruning target

**Reasoning:** SwiGLU combines gate and up; redundancy is most meaningful on the **effective activated unit**, not raw weight columns.

**Test:** Rank neurons by (a) |gate ⊙ up| post-activation vs (b) gate-only frequency. Measure rank correlation and which ranking better predicts ablation ΔPPL.

**Prediction:** Post-activation ranking **better predicts** removability than gate-only.

### H5 — Duplication: one twin of a near-duplicate pair is cheaper to remove than a matched-importance neuron

*(Added 2026-07-14 after review: the proposal defined and measured "duplicated" neurons but never tested whether duplication implies removability.)*

**Reasoning:** If two neurons in a layer have |ρ| ≥ 0.9 on the calibration set, their contributions to the residual stream are nearly collinear, so the survivor should be able to carry most of the pair's function.

**Test:** Mask one twin of each disjoint duplicate pair and compare ΔPPL against uncorrelated neurons of **matched importance** (plus random and lowest-importance controls) at the same neuron count. Also mask both twins to see how much the survivor was absorbing.

**Prediction:** `duplicate_one` costs less than `importance_matched`; if so, replacement/merging (Weeks 11–12) is worth trying over plain masking.

**Result (2026-07-26, `intervention_week9-10.md`):** **Not supported** — duplicates are no cheaper than matched-importance neurons. But the pairs are strongly *non-additive*: in Qwen3-0.6B layer 2, one twin costs +1.8 PPL and both cost +2898, so the survivor really does carry the function.

**Follow-up (2026-07-30, `recovery_week11-12.md`):** the prediction H5 generated **is** confirmed. Fitting `h_drop ≈ α·h_keep + β` and routing the dropped twin's contribution through its partner cuts the cost of removing it from **+1.84 to +0.14 PPL** in layer 2 (92.6% recovered; 87.3% across all pairs), with no training. So duplication does not make a neuron cheap to *delete*, but it does make it cheap to *replace* — and the fitted gain is far from 1 (mean |α| ≈ 0.36), which is why plain masking failed.

---

## Measurement protocol (ties to repo)

| Step | Tool | Output |
|------|------|--------|
| Register FFN hooks | `src/redundancy/hooks.py` → `iter_mlp_modules` | Per-layer activation tensors |
| Activation stats | `src/redundancy/metrics/` (Week 5) | Frequency, magnitude histograms |
| Layer bins | Config: `depth_bins: [0.33, 0.66]` | Early / mid / late aggregates |
| Ablation | `src/redundancy/pruning/` → `scripts/run_pruning.py` | ΔPPL per strategy and removal ratio |
| Duplicate-pair ablation | `scripts/run_pair_ablation.py` | ΔPPL per condition (H5) |
| Task prompts | WikiText loader + lm-eval PIQA | Task-specific neuron sets |
| Size compare | `qwen3.5-4b.yaml`, `qwen3-0.6b.yaml`, 9B config (#11) | Cross-size tables |

**Calibration size:** 256–512 sequences (VRAM-safe); full test eval only for final PPL reporting.

**Qwen3.5 caveat:** Hybrid stack mixes Gated DeltaNet, full attention, and FFN. We only attach hooks to modules matching `mlp` / `ffn` / `feed_forward` in `hooks.py`; layer-type metadata should be logged per index.

---

## Draft text for one-page proposal (#10)

> **Depth.** We split layers into early, middle, and late tertiles and measure per-layer FFN activation frequency and ablation impact. We hypothesize middle layers are least redundant because they host the most specialized key–value-like neurons (Geva et al.).
>
> **Component.** Our granularity is the post-SwiGLU FFN unit. We compare gate/up statistics but prune on post-activation importance. Attention and DeltaNet submodules are excluded.
>
> **Model family.** We stay within the Qwen lineage (Qwen3.5 primary, Qwen3-0.6B as a small baseline). Cross-family comparison is deferred unless mlsp GPU access is available.
>
> **Model size.** We compare Qwen3.5-4B and 9B (and 0.6B where useful) on the same calibration data, testing whether larger models exhibit more low-frequency neurons and a wider gap between redundancy-guided and random pruning.
>
> **Task.** We contrast WikiText-2 (LM) with PIQA (commonsense). We expect partially disjoint active-neuron sets and higher task sensitivity when removing neurons identified as redundant on WikiText alone.

---

## Checklist (issue #6)

- [x] Table: dimension → in/out of scope → planned experiment
- [x] At least two testable hypotheses (H1–H5 above)
- [ ] Team review / assign owners per dimension
- [x] Merged into `reports/group_5/proposal_week3-4.md` (#10) — Q5 section

## References

- Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021)
- Li et al., *The Lazy Neuron Phenomenon* (2022)
- Ma et al., *LLM-Pruner* (2023)
- Baseline: `reports/group_5/baseline_week1-2.md`
