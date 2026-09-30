# Group 5 — Neuron / FFN-unit Redundancy

**Weeks 3–4 one-page proposal · Ready for seminar (slides in `slides_week3-4.md`).**
**Primary model:** `Qwen/Qwen3.5-4B` (4-bit, 8 GB GPU) · **Larger model:** `Qwen3.5-9B` (#11) · **Small control:** `Qwen3-0.6B`
**WikiText-2 test PPL:** 4B = **17.06** · 9B = **12.90** (Henrik, #11) · 0.6B (fallback) = **39.78**

> **Status — deliverable complete for Weeks 3–4 writing.** Sibling reports (#2 datasets, #3 metrics, #6 dimensions, #9 limitations) and this proposal are in-repo. Slide deck: `reports/group_5/slides_week3-4.md` (Marp). Remaining experimental follow-ups live in #13–#16, not in this proposal.

---

## Q1 — What does redundancy mean at the neuron / FFN-unit granularity?

A **neuron** in this project is one **post-SwiGLU activation channel** inside an FFN block of a Qwen3.5 layer (i.e. one entry of the intermediate hidden state after the gate · up · activation). A neuron is **redundant** if any of the following holds on our calibration data:

1. **Lazy** — it almost never fires (\|activation\| > ε on a small fraction of token positions).
2. **Duplicated** — its activation pattern is highly correlated (|ρ| > 0.9) with another neuron in the same layer.
3. **Removable** — masking it (and its kind) to zero yields a small ΔPPL on WikiText-2 *and* a small accuracy drop on PIQA.
4. **Recoverable** *(stretch)* — any small remaining degradation can be undone by short fine-tuning, suggesting the neuron's role was already covered elsewhere in the network.

We commit to **masking**, not structural surgery: parameter shapes do not change; therefore, we make no claims regarding runtime or FLOP savings.

> **Result on senses 2 and 4 (2026-07-30, `recovery_week11-12.md`).** *Recoverable:* 200 LoRA steps undo **76–83%** of the perplexity cost of masking, measured against an unmasked model given the same adapter budget — the control matters, because that budget alone improves perplexity by 10.7 points on its own. *Duplicated:* correlation does not make a neuron cheap to delete (Weeks 9–10) but it does make it cheap to **replace** — routing a twin's contribution through its partner recovers 87–93% of the masking cost with no training.

## Q2 — Which papers / methods are most relevant?

(From the #4 reading list.)

- **Geva et al. 2021 — *Transformer FFN Layers Are Key-Value Memories*** — Reframes FFN neurons as key–value memory slots and shows that different depths play different roles (early = local features, middle = factual memory, late = distributional output). Motivates our depth-tertile split (H1).
- **Ma et al. 2023 — *LLM-Pruner*** — Demonstrates that *middle layers* tend to be the most prunable in LLMs once dependency graphs are respected; gives us the practical pruning recipe (importance score → structured mask) we mimic at neuron granularity.
- **Li et al. 2022 — *The Lazy Neuron Phenomenon*** *(optional)* — Empirical evidence that activation sparsity grows with scale; underwrites H2 (larger Qwen3.5 → more low-firing neurons).

## Q3 — Which model(s) can we realistically run?

| Model | Quantization | VRAM | WikiText-2 test PPL | Use |
|-------|--------------|------|---------------------|-----|
| **Qwen3.5-4B** | 4-bit NF4 | ~8 GB (RTX 4060) | **17.06** | Primary — all measurement, ablation, intervention |
| **Qwen3.5-9B** | 4-bit | 7.4 GiB / 16 GiB (Henrik, 2026-06-20) | **12.90** | Size-scaling comparison (H2) |
| **Qwen3-0.6B** | 4-bit | < 4 GB | 39.78 | Small-model control + sanity-check fp16 cross-validation |

Cross-family (LLaMA, Mistral) is **out of reach** without additional GPU access.

## Q4 — Which datasets and metrics?

(Drafted in `datasets_week3-4.md` (#2) and `metrics_week3-4.md` (#3).)

**Datasets:** WikiText-2 (`wikitext-2-raw-v1`, `test` for PPL, 256–512 sequences of `train` for activation calibration) and **PIQA** (`validation`, via `lm-evaluation-harness`). **ARC-Easy is a fallback only** — not in the primary plan. C4 and OCR/multimodal data are out of scope.

**Outcome metrics:** Overall model performance before and after neuron masking is measured with WikiText-2 test perplexity (sliding window, context 2048, stride 512) and PIQA accuracy (`acc`, `acc_norm` via `lm-evaluation-harness`).

**Internal redundancy metrics:** For each neuron *i* in layer ℓ: activation frequency *f*<sub>i,ℓ</sub>, mean/RMS activation magnitude, pairwise correlation (top-*k* per layer), and ablation sensitivity ΔPPL when masking set *S*. **ε threshold:** RMS-relative per layer (primary); fixed ε (e.g. 1e-3) as sanity-check alternative.

**Success criteria (first pass):** when masking 10 % of neurons per layer, ΔPPL < +1 on WikiText-2 and PIQA accuracy drop < 2 pp; the main quantity of interest is the gap between redundancy-guided and random pruning at matched sparsity.

---

## Q5 — How might redundancy depend on depth, component, family, size, or task?

(Long form in `redundancy_dependencies_week3-4.md` — closes #6.)

### Scope

| Dimension | In scope? | Plan (Weeks 5–8) |
|-----------|-----------|------------------|
| **Depth** | Yes | Per-layer activation frequency + ablation ΔPPL; compare early/mid/late tertiles. |
| **Component** | Partial (FFN only) | Post-SwiGLU activation as the neuron unit; gate/up correlation as a secondary check. Attention, Gated DeltaNet, full blocks, weights, tiles are out of scope (other groups). |
| **Model family** | Limited | Qwen lineage only — 3.5-4B primary, 3-0.6B small control. |
| **Model size** | Yes | 4B vs 9B vs 0.6B on matched calibration; activation sparsity + ablation curves at 10 % / 25 % removal. |
| **Task** | Yes | WikiText-2 vs PIQA: Jaccard overlap of top-*k* active neurons per layer + task-conditioned ablation. |

### Hypotheses

1. **H1 (depth).** Middle FFN layers are *less* redundant than early or late layers.
2. **H2 (size).** Larger Qwen3.5 models contain a higher fraction of low-firing neurons, and the redundancy-guided-vs-random pruning gap *grows* with size.
3. **H3 (task).** Neurons that look redundant under WikiText are *not* redundant under PIQA; PIQA degrades faster when those neurons are ablated.
4. **H4 (component).** Post-activation magnitude (|gate ⊙ up|) ranks neurons for ablation **better** than gate-only frequency.
5. **H5 (duplication).** *(Added 2026-07-14 after Haoyi's review: the proposal measured duplicate neurons but attached no hypothesis to them.)* For a near-duplicate pair (|ρ| ≥ 0.9), masking **one** twin costs less ΔPPL than masking an **uncorrelated neuron of matched importance**, because the two contributions are nearly collinear. If it holds, replacing rather than masking becomes worthwhile.

> **Scoreboard — superseded.** The verdicts recorded here on 2026-07-30 came
> from Qwen only, and four of the five have since changed. Do not cite this
> paragraph; the frozen table is `reports/group_5/hypothesis_scoreboard.md`.
> In short: **H1 refuted** (the depth-tertile ablation, once run, never makes
> the middle tertile the most expensive), **H2 refuted as stated** (the clean
> GPT-2 size arm breaks the trend), **H3 supported on gated FFNs only** (it
> fails on GPT-2), **H4 supported** with the frequency half split out as H7, and
> **H5 not supported** — superseded by **H5′**, the replacement experiment,
> which is the result that holds.

### Paragraph

> At neuron granularity we expect redundancy to be **non-uniform across depth and task**, **concentrated in low-importance post-SwiGLU units**, and to **scale with model size** within Qwen3.5. We measure per-layer activation frequency and ablation ΔPPL on WikiText-2, compare top-active neuron sets between WikiText-2 and PIQA, and contrast 4B against 9B and 0.6B. Attention, full blocks, unstructured weights, and tile-level patterns belong to sibling groups.

---

## Q6 — How do we plan to measure redundancy?

(Drafted in `metrics_week3-4.md` (#3); aligned with Marcel's #7 comment of 2026-06-17.)

**Method (Marcel's plan, adopted):**

1. Start from **activation frequency** — how often each post-SwiGLU neuron fires on WikiText-2 calibration (the *Lazy Neuron* probe). Plot a histogram of *f* per layer; tag "lazy" neurons as the low-frequency tail.
2. **Layer-first granularity:** average activation frequency per layer to pick the layers most likely to contain lazy neurons (LLM-Pruner observation: middle layers are typically the most prunable). Then drill into individual neurons inside those layers.
3. Add a **similarity / duplicate** probe (Key-Value-Memory inspired): inside the lazy layers, look for pairs of neurons with near-identical activation vectors across the calibration set.

**Granularity:** per-neuron (post-SwiGLU index) **and** per-layer aggregate. Per-block and per-component (gate vs up) only as secondary checks (H4).

**Plots we plan to produce:**

- Histogram of activation frequency, x = *f*, y = neuron count, one per layer.
- Heat-map of per-layer mean *f* across depth (early / mid / late tertiles).
- Correlation matrix (top-*k* off-diagonal pairs only) for two or three lazy layers.

**Hooks:** `iter_mlp_modules` (`src/redundancy/hooks.py`) already filters to FFN/MLP/feed_forward modules; the measurement pipeline will live in `scripts/run_measurement.py` + `src/redundancy/metrics/`.

## Q7 — What intervention will we try first?

(Marcel's plan, #7 comment 2026-06-17; revised after the Weeks 5–8 measurement and Haoyi's review.)

**Arm A — masking the least *important* neurons.**

1. Rank neurons within each layer by **importance** `RMS(h_i) · ‖W_down[:, i]‖`. *(Revised: the original plan ranked by activation frequency, which saturates on SwiGLU — see `measurement_week5-8.md`. Frequency is kept as a second arm so the original hypothesis is still tested.)*
2. Mask (set to zero in the forward pass) the bottom **5 / 10 / 25 / 50 %** per layer; re-evaluate WikiText-2 PPL and PIQA accuracy at each ratio.
3. **Random baseline:** at each ratio, mask the same number of neurons per layer chosen uniformly at random; plot the random curve on the same axes. The deliverable is the *gap* between the curves, not the absolute drop.

**Arm B — duplicate-pair ablation (tests H5).** For near-duplicate pairs, compare masking one twin, both twins, and matched-importance / random / lowest-importance controls at the same neuron count. This is what makes the duplication measurement actionable rather than descriptive.

Merging / replacing / fine-tuning recovery are tracked as later stages (Weeks 11–12); Arm B is what decides whether replacement is worth trying.

> **Result (2026-07-26 / polish 2026-07-29).** Both arms are implemented and run — see `intervention_week9-10.md`. Arm A: importance-guided masking beats random at every ratio on perplexity (4B at 25%: 24.98 vs 47.54 PPL) and on PIQA for both models; frequency is weaker on PPL and mixed on PIQA. Arm B: H5 as stated is **not** supported on 0.6B (and only noise-level on 4B), but near-duplicate pairs in the 0.6B massive-activation layer are jointly critical — one twin costs +1.8 PPL, both cost +2898.

## Q8 — Limitations

(Long form in `limitations_week3-4.md` (#9).)

- **Definition is a choice.** Redundancy is defined on the *post-SwiGLU activation channel*; gate-only or weight-column views would rank neurons differently. We commit to one definition and report the alternatives as secondary checks.
- **Calibration-set bias.** Lazy-on-WikiText ≠ lazy in general (this is exactly H3). We cross-check on PIQA prompts.
- **Masking ≠ removal.** Tensor shapes are unchanged; we make **no** wall-clock or FLOP speedup claim.
- **4-bit + 8 GB hardware.** Activation magnitudes are quantization-noisy and calibration sets are small (256–512 sequences). We require effect sizes to be large (e.g. *f* < threshold on **both** calibration halves before calling a neuron lazy) and bootstrap confidence intervals.
- **Hybrid Qwen3.5 stack.** Not every layer is a standard MLP-only layer. We log block type per layer and compare depth tertiles only across matching block types.
- **Single family.** All cross-size claims are explicitly *"within Qwen lineage"*.

---

## Deliverables checklist (issue #10)

- [x] One-page proposal at `reports/group_5/proposal_week3-4.md` answering all 8 questions
- [x] Q1 redundancy definition
- [x] Q2 starting papers (#4)
- [x] Q3 models + hardware (4B 17.06, 9B 12.90 from #11, 0.6B 39.78)
- [x] Q4 datasets + metrics (#2 / #3)
- [x] Q5 dimensions + H1–H5 (#6; H5 added after review)
- [x] Q6 measurement plan (#7, Marcel)
- [x] Q7 first intervention (#7, Marcel)
- [x] Q8 limitations (#9)
- [x] 5–10 min slide deck (`reports/group_5/slides_week3-4.md`, Marp)
- [ ] Present in seminar (date TBD)
- [x] Group writing sign-off on this draft (experimental follow-ups tracked in #13–#16)

## References

- Sibling deliverables (Group 5, Weeks 3–4):
  - `reports/group_5/baseline_week1-2.md`
  - `reports/group_5/datasets_week3-4.md` (#2)
  - `reports/group_5/metrics_week3-4.md` (#3)
  - `reports/group_5/redundancy_dependencies_week3-4.md` (#6)
  - `reports/group_5/limitations_week3-4.md` (#9)
  - `reports/group_5/slides_week3-4.md` (Marp deck for #10)
- Issue comments incorporated: #7 (Marcel, measurement & intervention), #11 (Henrik, Qwen3.5-9B baseline).
- Papers: Geva et al. 2021, Ma et al. 2023, Li et al. 2022.
