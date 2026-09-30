# Group 5 — frozen hypothesis scoreboard

**This file is the single source of truth for every hypothesis verdict.** Where
an earlier report disagrees with the table below, this file wins; the earlier
reports carry an errata block pointing here.

Frozen **2026-09-09** for the Weeks 13–16 final analysis. Closes
[#20](https://github.com/geeeeenccc/Redunformer-Neuron/issues/20). Every number
is traceable to a JSON artifact in `experiments/results/`; the per-hypothesis
tables live in `artifacts/`.

**Evidence base:** 7 models, 3 families, 2 FFN designs.

| Model | Params | FFN | Baseline PPL (full WikiText-2 test) |
|---|---:|---|---:|
| `gpt2` | 124M | GELU, no gate | 50.07 |
| `gpt2-medium` | 355M | GELU, no gate | 36.65 |
| `gpt2-large` | 774M | GELU, no gate | 31.99 |
| `Qwen/Qwen3-0.6B` | 0.6B | SwiGLU | 39.78 |
| `Qwen/Qwen3.5-4B` (primary) | 4B | SwiGLU | 17.06 |
| `meta-llama/Llama-2-7b-hf` | 7B | SwiGLU | 10.58 |
| `Qwen/Qwen3.5-9B` | 9B | SwiGLU | 12.90 full split; intervention baseline 15.36 on 79 documents |

---

## The table

| ID | Claim | Verdict | One-line evidence |
|---|---|---|---|
| **H1** | Middle FFN layers are *less* redundant than early/late | **Refuted** | Tertile ablation on 7 models: the middle tertile is never the most expensive. Early is worst on 3, deep on 3 (4B, 9B, Llama-2), flat on 1. |
| **H2** | Larger models are more redundant, and the guided-vs-random gap *grows* with size | **Refuted as stated** | GPT-2 124M/355M/774M with architecture fixed: cost falls at 25% but rises at 50%; bottom-10% importance share is flat (6.45 / 6.59 / 6.58%). Within Qwen the guided-vs-random gap *shrinks* with size (3.76× → 1.90× → 1.61× at 9B). |
| **H3** | WikiText-redundant neurons are not PIQA-redundant | **Supported on gated FFNs only** | Clean double dissociation on Qwen3-0.6B (+3.6 pp PIQA at 25%), Qwen3.5-4B (+5.0 pp) and Llama-2-7B (+3.5 pp). On GPT-2 a PIQA-calibrated ranking is worse on *both* metrics — and worse than random on perplexity. |
| **H4** | Post-activation importance ranks neurons better than firing frequency | **Supported** | `RMS(h)·‖W_down[:,i]‖` beats frequency on perplexity in all 7 models that were swept. Frequency's own behaviour is split out as H7. |
| **H5** | One twin of a \|ρ\|≥0.9 pair is cheaper to **mask** than a matched-importance uncorrelated neuron | **Not supported** | 3 of 7 models supported, 4 not. No consistent sign. Where the control is genuinely matched (≥0.94×) the margins are ≤0.26 PPL — at the noise floor. Not a usable claim. |
| **H5′** | Duplicate twins are cheap to **replace**: routing `h_drop ≈ α·h_keep + β` through the partner recovers most of the masking cost, in proportion to fit r² | **Supported** | Where masking actually costs something: 102% recovered at r²=0.902 (GPT-2), 87% at r²=0.892 (Qwen3-0.6B), 47% at r²=0.618 (Llama-2). Recovery tracks fit quality. No training. Qwen3.5-9B (4 pairs, +0.012 PPL to mask) is in the "already free" bucket; its 28% is not a score. |
| **H6** | Importance-guided FFN redundancy is a property of trained decoder-only transformers, not of Qwen or of gated FFNs | **Supported** | 7 models, 3 families. Strongest on Llama-2-7B: half of every FFN masked by importance costs 2.27× perplexity, random costs 299×. |
| **H7** | Ranking by firing frequency beats random only in gated FFNs (SwiGLU), not in always-on GELU | **Supported** | 0 of 12 comparisons on GELU, 11 of 12 on the first three SwiGLU models, and 4 of 4 on Qwen3.5-9B. On `gpt2-medium` at 50%, frequency is 13× *worse* than random. |
| **H8** | Short LoRA recovers the masking cost, and recovery degrades with model size | **Recovery holds below ~1B; fails at 7B on this recipe** | 76–89% adjusted recovery at 819k tokens on `gpt2` 124M (80/87%), `gpt2-medium` (84/89%), `gpt2-large` (76/80%) and Qwen3-0.6B (76/83%). The GPT-2 ladder moves ≤9 pp. Llama-2-7B at the **same** 819k budget overfits: −6% at 10%, 48% at 25%. Qwen3.5-4B at 2% used 16× fewer tokens (51k) and is not a matched comparison; the 7B run is. |

**H5 vs H5′ is the distinction that matters:** duplicated neurons are **not**
cheap to *delete*, but they **are** cheap to *replace*. Naive masking throws the
shared information away; merging keeps it.

### Provenance of H5′, H6, H7

H1–H5 are the proposal's original hypotheses (`proposal_week3-4.md` Q5); H5 was
added 2026-07-14 after Haoyi's review pointed out that duplication was measured
but never tested. H5′, H6, H7, and H8 **name results we already ran** — they are
not new experiments. H5′ comes from the Weeks 11–12 replacement study, H6 and H7
from the Weeks 13–14 family expansion, H8 from the recovery arms.

---

## What changed, and why

Three Qwen-era verdicts reversed once the family expansion landed, and a fourth
reversed when the size arm was completed. These are the corrections to carry
into the final report.

| Hypothesis | Qwen-only reading (Weeks 5–12) | Verdict now | What forced the change |
|---|---|---|---|
| H1 | "supported" — from the static bottom-10% importance share | **refuted** | The *ablation* was never run per tertile until Weeks 13–16. It disagrees with the static proxy: on GPT-2 the proxy nominates the middle tertile as least redundant, but masking the early tertile costs more than twice as much. |
| H2 | "partial / larger degrades less" | **refuted as stated** | The Qwen comparison varied size *and* architecture *and* corpus together. GPT-2 small/medium/large varies only size, and the trend breaks at 10% and 50%. |
| H4 / frequency | "weaker than importance, mixed on PIQA" | split into **H4 supported** + **H7** | Frequency is not merely weaker on GELU models, it is *worse than random* — a qualitative difference that deserves its own hypothesis. |
| H5 | "rejected, but the control was unmatched, so the mechanism probably replicates" | **not supported**, and the excuse is gone | Llama-2-7B has 11008 neurons per layer, so its matched control reaches 0.94× the duplicates' importance — a fair comparison. H5 still fails there. Adding `gpt2-medium` and `gpt2-large` then flipped the verdict *the other way*, which is the real finding: the effect has no consistent sign. |

The within-pair asymmetry (masking the second twin costs far more than the
first) is **no longer offered as evidence** for H5. The procedure masks the
lower-importance twin first, so the second twin is the more important neuron by
construction, and that alone explains the gap. H5′ is the confound-free version
of the same question, because it compares two treatments of the *same* neuron.

---

## Per-hypothesis detail

| Hypothesis | Table | Underlying artifacts |
|---|---|---|
| H1 | [`artifacts/h1_depth_tertiles.md`](artifacts/h1_depth_tertiles.md) | `measurement_*.json`, `pruning_h1{early,middle,deep}_*.json` |
| H2, H4, H6, H7 | [`artifacts/cross_family_tables.md`](artifacts/cross_family_tables.md) | `measurement_*.json`, `pruning_*.json` |
| H3 | [`artifacts/h3_task_specificity.md`](artifacts/h3_task_specificity.md) | `task_overlap_*.json`, `pruning_piqaranked_*.json` |
| H5, H5′ | [`artifacts/h5_drop_vs_merge.md`](artifacts/h5_drop_vs_merge.md) | `pair_ablation_*.json`, `merge_*.json` |
| H8 | [`artifacts/h8_recovery.md`](artifacts/h8_recovery.md) | `recovery_*.json` |

Narrative reports, in order: `baseline_week1-2.md` → `proposal_week3-4.md` →
`measurement_week5-8.md` → `intervention_week9-10.md` →
`recovery_week11-12.md` → `cross_family_week13-14.md` →
`analysis_week15-16.md` (final).

## What we are explicitly *not* claiming

Stated once here so the final report does not have to relitigate it:

- **No speed, memory, or FLOP claim.** Every intervention is a forward hook;
  parameter shapes never change. "Removed" always means "masked to zero".
- **Qwen3.5-9B** has measurement, masking, H1, pair ablation, and merge. No LoRA
  arm; Llama-2-7B at the matched budget already answers that question. Quote 9B
  perplexity against its in-run baseline (15.36 on 79 documents), not the 12.90
  full-split number.
- **Llama-2-7B recovery** was run at 819k tokens. Adjusted recovery is −6% at
  10% and 48% at 25%; both arms overfit. One recipe, not a proof that 7B
  masking is permanent.
- **Two evaluation tasks only** (WikiText-2 perplexity, PIQA accuracy). No
  visual/multimodal benchmark was run, even though Qwen3.5-4B is an
  image-text hybrid — see #12.
- **Single random draw per ratio.** The random baselines are one seeded sample,
  not an average with error bars. The gaps are large enough that this is
  unlikely to flip a conclusion, but it is not a confidence interval.
- **Correlation pairs are a sample, not a census** — three drill-down layers per
  model, from a reservoir subsample. Pair counts are small (4–22).
