---
marp: true
title: Group 5 — Neuron / FFN Redundancy — Final
paginate: true
theme: default
style: |
  section { font-size: 27px; }
  h1 { font-size: 38px; }
  h2 { font-size: 30px; }
  table { font-size: 20px; }
  em { color: #555; }
---

# Neuron / FFN-unit Redundancy

## Group 5 — final analysis

**6 models · 3 families · 2 FFN designs**
GPT-2 124M/355M/774M · Qwen3-0.6B · Qwen3.5-4B · Llama-2-7B

WikiText-2 perplexity + PIQA · masking, not surgery

*Full report: `reports/group_5/analysis_week15-16.md`*

---

# The one-slide version

**Redundancy is real and rankable** — an activation-aware importance score finds
FFN neurons that mask far more cheaply than random ones, in every model.

**It transfers across families.** Llama-2-7B: half the FFN removed by importance
costs **2.27×** perplexity; at random, **299×**.

**Almost every structural intuition we had was wrong.** Neither depth nor model
size predicts redundancy once the confounds are separated.

**Duplicated neurons are cheap to *replace*, not cheap to *remove*** — and how
much replacement recovers is predicted by how well one twin predicts the other.

---

# What a neuron is

One post-activation channel of an FFN block — one entry of
`act(gate(x)) ⊙ up(x)`, or of `GELU(c_fc(x))` on GPT-2.

It reaches the residual stream through **one weight column**:

$$\text{FFN}(x) = \sum_i h_i \cdot W_{\text{down}}[:, i]$$

**Importance** $= \mathrm{RMS}(h_i)\cdot\lVert W_{\text{down}}[:,i]\rVert_2$
*(LLM-Pruner / Wanda style)*

We **mask** via a forward pre-hook on `down_proj`: the whole contribution
disappears, no parameter shape changes.

→ works on 4-bit models · → **no FLOP or speed claim anywhere**

---

# Four senses of "redundant" — they disagree

| Sense | Measured as | Verdict |
|---|---|---|
| **Lazy** | fires rarely | Nearly useless. Saturates on SwiGLU, *negative* info on GELU |
| **Low-contribution** | `RMS(h)·‖W_down[:,i]‖` | **The signal that works** |
| **Duplicated** | \|ρ\| ≥ 0.9 in-layer | Common — but ≠ cheap to delete |
| **Removable** | ΔPPL when masked | Ground truth the others proxy for |

The project is largely the story of the first three disagreeing with the fourth.

---

# Headline: importance vs random

Relative to each model's own unmasked baseline — **importance / random**

| masked/layer | gpt2 | Qwen3-0.6B | Qwen3.5-4B | Llama-2-7B |
|---:|---:|---:|---:|---:|
| 10% | **1.31×** / 1.47× | **1.13×** / 1.71× | **1.10×** / 1.40× | **1.09×** / 1.20× |
| 25% | **2.67×** / 3.75× | **1.57×** / 5.89× | **1.32×** / 2.52× | **1.27×** / 1.89× |
| 50% | **10.57×** / 27.91× | **2.96×** / 85.93× | **2.11×** / 17.71× | **2.27×** / 299.10× |

The deliverable is the **gap**, and it widens with the ratio.

**Llama-2-7B at 50%:** 2.27× vs 299× — a **132-fold** difference.
PIQA holds at 0.675 against a 0.765 baseline with half the FFN gone.

---

# ![w:1000](artifacts/cross_family_removal.png)

*Solid = importance · dash-dot = frequency · dashed = random · log axis*

---

# Reversal 1 — frequency does not transfer

Our **original** plan ranked by firing frequency (Lazy Neuron). It is
**worse than random on every GELU model at every ratio**:

| masked/layer | gpt2 | gpt2-medium | gpt2-large | Qwen3-0.6B | Qwen3.5-4B | Llama-2-7B |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 15.4× / 3.8× | 349.8× / 4.1× | 35.0× / 1.6× | **2.0×** / 5.9× | **1.9×** / 2.5× | **1.6×** / 1.9× |
| 50% | 125.7× / 27.9× | 2577.7× / 189.5× | 237.1× / 69.6× | **7.2×** / 85.9× | **9.7×** / 299.1× | **6.8×** / 17.7× |

**0 of 12 on GELU · 11 of 12 on SwiGLU** — the split follows FFN design exactly.

**Why:** at α = 0.25 the gated models have a real quiet subpopulation
(Llama-2 3.12%, Qwen3-0.6B 4.30%); GELU models have ~none (0.02–0.05%). GELU is
always on, so *how often* a neuron fires says nothing about whether it matters.

→ **Do not move frequency-based pruning heuristics across FFN designs.**

---

# Reversal 2 — depth (H1) refuted, and the proxy was the problem

Mask 10% of each depth tertile, measure the cost:

| model | early | middle | deep | worst |
|---|---:|---:|---:|---|
| gpt2 | **+5.24** | +4.08 | +2.31 | early |
| gpt2-medium | **+7.06** | +2.92 | +1.77 | early |
| gpt2-large | +1.06 | +0.98 | +1.01 | flat |
| Qwen3-0.6B | **+2.69** | +0.19 | +0.81 | early |
| Qwen3.5-4B | +0.34 | +0.37 | **+1.02** | deep |
| Llama-2-7B | +0.27 | +0.10 | **+0.42** | deep |

**The middle tertile is never the most expensive.** H1 refuted.

**The methodological point:** through Weeks 5–10 we scored H1 from the *static*
bottom-10% importance share, which on GPT-2 nominated the **middle** tertile —
the opposite answer. A static score ranks neurons well *within* a layer; it does
not predict what a layer's removal costs.

---

# Reversal 3 — size (H2) refuted as stated

Cross-family it looks like textbook H2 (relative cost at 25% falls with size:
2.67× → 1.57× → 1.32× → 1.27×) — but that varies size **and** architecture
**and** corpus together.

GPT-2 varies **only** size:

| model | params | @10% | @25% | @50% | bottom-10% share |
|---|---:|---:|---:|---:|---:|
| gpt2 | 124M | 1.307 | 2.670 | **10.565** | 6.45% |
| gpt2-medium | 355M | 1.374 | 2.470 | 16.153 | 6.59% |
| gpt2-large | 774M | **1.109** | **1.461** | 14.874 | 6.58% |

Holds at 25%; **fails at 10% and 50%**. The static profile barely moves across a
6× parameter range. The guided-vs-random *gap* **shrinks** with size within Qwen
(3.76× → 1.90×).

→ **FFN design predicts pruning tolerance better than parameter count does.**

---

# H3 — task-specificity, on gated models only

Re-ran the whole measurement on **PIQA-prompt** calibration. At 25% masking:

| model | WikiText ranking | PIQA ranking | random | H3 |
|---|---|---|---|---|
| Llama-2-7B | 11.99 PPL / 0.750 | 14.31 PPL / **0.785** | 17.89 / 0.695 | **supported** |
| Qwen3-0.6B | 52.86 PPL / 0.594 | 94.80 PPL / **0.630** | 198.74 / 0.594 | **supported** |
| gpt2 | **137.89 PPL / 0.562** | 311.68 PPL / 0.544 | 193.86 / 0.570 | not supported |

Clean **double dissociation** on both gated models (+3.5 / +3.6 pp PIQA).
On GPT-2 the PIQA ranking loses on *both* metrics — and to random on perplexity.

Set overlap at a 10% budget: 0.33 Jaccard (Qwen3-0.6B, 6.3× chance), 0.17
(gpt2), 0.14 (Llama-2) → **partial**, not total, task specificity.

*Same GELU/SwiGLU split as the frequency result.*

---

# The duplicate-neuron thread

**Origin:** Haoyi's review — *"you find duplicated neurons but there seems to be
no hypothesis based on it? Perhaps we need a neuron replacement experiment?"*

**H5 (drop):** masking one twin of a \|ρ\| ≥ 0.9 pair should cost less than
masking a matched-importance uncorrelated neuron.

| model | pairs | drop one | control | margin | coverage | verdict |
|---|---:|---:|---:|---:|---:|---|
| gpt2 | 22 | +1.164 | −0.688 | −1.852 | 0.57× | False |
| gpt2-medium | 15 | +0.004 | +0.267 | +0.263 | **1.01×** | True |
| gpt2-large | 16 | −0.119 | +0.034 | +0.153 | 0.65× | True |
| Qwen3-0.6B | 14 | +2.241 | −0.279 | −2.520 | 0.60× | False |
| Qwen3.5-4B | 4 | −0.013 | +0.033 | +0.046 | **1.00×** | True |
| Llama-2-7B | 17 | +0.044 | −0.204 | −0.248 | **0.94×** | False |

**3–3, no consistent sign → H5 is not a usable claim.**

---

# We were wrong about *why*, twice

**Excuse 1 (Weeks 9–10):** "the control was unmatched" — where duplicates are
massive-activation outliers, no uncorrelated neuron comes close (0.57–0.60×).

**Llama-2 killed it:** 11008 neurons/layer → control reaches **0.94×**, a fair
comparison. H5 **still fails**.

**Then medium/large flipped it the other way.** Restricting to fair controls
(≥0.94×) gives +0.263, +0.046, −0.248 PPL on 4–17 pairs — noise floor.

**Retired evidence:** masking the *second* twin costs far more than the first
(53,000× on gpt2-medium). We read that as "the survivor absorbed the job" — but
the procedure masks the **lower-importance** twin first, so the second twin is
more important *by construction*. Confounded.

---

# H5′ — replacement is the result that holds

Don't ask "cheap to delete?" Ask **"is the information available elsewhere?"**

Fit `h_drop ≈ α·h_keep + β` on calibration activations; route the dropped twin's
contribution through its partner via a hook. **Same neuron, two treatments** — no
importance confound.

| model | pairs | mean r² | mask only | after merge | **cost undone** |
|---|---:|---:|---:|---:|---:|
| gpt2 | 22 | 0.902 | +1.164 | **−0.023** | **102%** |
| Qwen3-0.6B | 14 | 0.892 | +2.241 | +0.284 | **87%** |
| Llama-2-7B | 17 | 0.618 | +0.044 | +0.023 | **47%** |

**Recovery tracks fit quality** — 0.902 → 102%, 0.892 → 87%, 0.618 → 47%.
Two fitted scalars, **no training**, ≈ what 200 LoRA steps buy.

---

# Why the gain has to be fitted

- **α matters, β does not.** Dropping the offset changes ≤0.02 PPL — but
  mean \|α\| ≈ 0.36–0.74, far from 1.
  → *this is why plain masking fails*: correlation fixes the **shape** of the
  relationship, not the **scale**.

- **Correlation alone does not predict merge value.** Qwen3-0.6B's deep layer has
  the *best* fits (r² 0.965, \|ρ\| 0.997) yet merging there makes perplexity
  **worse** — those neurons were already free to drop.
  → **merge where masking hurts; drop by importance where it does not**

- **Where masking was already free, there is nothing to recover** —
  gpt2-medium/large and Qwen3.5-4B cost ≤0.01 PPL to mask, so their "%
  recovered" is meaningless, not low.

---

# H8 — recovery, and why the control is everything

200 LoRA steps on the **unmasked** Qwen3-0.6B: 33.75 → 23.08 PPL.
So raw recovery reads **318%** and means nothing.

We report masked+LoRA against unmasked+LoRA at **identical budget**.

| model | ratio | budget | masked | +LoRA | control | **adj. rec.** |
|---|---:|---:|---:|---:|---:|---:|
| gpt2 124M | 10% / 25% | 819k | 67.50 / 137.89 | 36.86 / 44.52 | 33.66 | **80% / 87%** |
| gpt2-medium 355M | 10% / 25% | 819k | 51.08 / 91.80 | 26.39 / 30.16 | 24.12 | **84% / 89%** |
| gpt2-large 774M | 10% / 25% | 819k | 36.00 / 47.44 | 22.51 / 24.70 | 21.65 | **76% / 80%** |
| Qwen3-0.6B | 10% / 25% | 819k | 38.15 / 52.86 | 24.15 / 26.33 | 23.08 | **76% / 83%** |
| Qwen3.5-4B | 10% | **51k** | 20.71 | 27.59 | 25.80 | **2%** |

Four models, two families, 6× size range, **one budget** → **76–89%**.

**Recovery works. The size clause does not.** It predicted degradation with
size; across the GPT-2 ladder the trend is non-monotonic (80 → 84 → 76) and
spans ≤9 pp over 6× parameters.

**And the 4B row is not a matched comparison:** 51k training tokens against
819k, **16× less**. A ≤9 pp size effect cannot reach 2%.

**Open:** re-run 4B, or Llama-2-7B, at 819k. The cliff is unexplained — but
"recovery fails at scale" was never the parsimonious reading.

---

# Frozen scoreboard

| ID | Claim | Verdict |
|---|---|---|
| H1 | Middle layers less redundant | **Refuted** |
| H2 | Bigger ⇒ more redundant; gap grows with size | **Refuted as stated** |
| H3 | WikiText-redundant ≠ PIQA-redundant | **Supported on gated FFNs only** |
| H4 | Importance ranks better than frequency | **Supported** |
| H5 | One twin cheaper to **mask** than matched control | **Not supported** |
| **H5′** | Twins cheap to **replace**, ∝ fit r² | **Supported** |
| **H6** | Redundancy is not Qwen-specific | **Supported** |
| **H7** | Frequency works on SwiGLU, fails on GELU | **Supported** |
| **H8** | Short LoRA recovers masking cost, degrading with size | **Recovery yes (76–89%); size clause no** |

H5′ / H6 / H7 / H8 **name results we already ran** — not new experiments.

*`reports/group_5/hypothesis_scoreboard.md`*

---

# Limitations we want on the record

**Method** · masking ≠ removal, so **no speed/FLOP claim** · importance is a
separable proxy (§H1 shows where it breaks) · **single-seed** random baselines ·
pair counts 4–22, so H5/H5′ rest on tens of neurons · scalar substitution only

**Coverage** · **two tasks only** (WikiText-2, PIQA) — no visual benchmark
despite Qwen3.5-4B being an image-text hybrid · **Qwen3.5-9B: baseline only**
(needs ≥16 GB) · **Llama-2-7B recovery untested** (4-bit 7B QLoRA trips CUBLAS on
8 GB) · one LoRA budget

**Comparability** · GPT-2 context 1024 vs 2048 elsewhere → every cross-model
number normalised to its own baseline · Llama/Qwen 4-bit vs GPT-2 fp32 (works
*against* the Llama result) · architecture contrast rests entirely on GPT-2

---

# Conclusion

**Redundancy at neuron granularity is real, rankable, and transferable.** One
activation-aware score, six models, three families, two FFN designs — strongest
on the largest model.

**The cheap proxies do not survive a second architecture.** Firing frequency is
worse than random on every GELU model. Depth and size predict nothing once the
confounds are separated. The static proxy that supported our depth hypothesis
was contradicted by the ablation it stood in for.

**The sharper notion of redundancy is "available elsewhere", not "spare".**
Duplicated neurons are not cheap to delete — but replacement recovers the cost
in proportion to how linearly one twin predicts the other. The information was
never spare; it was recoverable, and only an intervention that preserves it is
cheap.

*Thank you — and thank you Haoyi for the two review comments that produced the
replacement experiment and the family expansion.*

---

# Reproducing

```powershell
uv sync
uv run pytest tests/ -q     # 76 offline tests, no GPU

uv run python scripts/run_measurement.py   --config configs/measurement/neuron_activations.yaml
uv run python scripts/run_pruning.py       --config configs/pruning/neuron_masking.yaml
uv run python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation.yaml
uv run python scripts/run_merge.py         --config configs/pruning/merge_qwen3_0.6b.yaml
uv run python scripts/run_recovery.py      --config configs/recovery/lora_qwen3_0.6b.yaml

# whole cross-model matrix; skips stages whose artifact exists
uv run python scripts/run_cross_family_followups.py

# regenerate every table from the JSONs on disk (no GPU, seconds)
uv run python scripts/summarize_h1_depth.py
uv run python scripts/summarize_hypotheses.py
uv run python scripts/compare_families.py --model gpt2 ... --logy
```

Every result JSON records model, dataset, seed, intervention, and the exact
command.
