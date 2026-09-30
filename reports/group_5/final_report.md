# Neuron-level redundancy in feed-forward layers

**Group 5**
**Seminar report, due 30 September 2026**

Numbers below are relative to each model's own unmasked baseline unless a raw perplexity is given.

## Who did what

| Person | What they owned |
|---|---|
| Herai Hench | The experimental pipeline and the main 8 GB campaign: measurement, masking, duplicate-pair tests, replacement, and LoRA recovery, including the GPT-2 size arm that separated model size from architecture. Also the running synthesis of the results into the group reports. |
| Marcel Schlappner | The measurement and intervention design the later runs followed, and the Qwen3.5-4B LoRA recovery, which did not fit on the shared card and was run on Colab. |
| Sooraj Rathore | The cross-model reading of size and task specificity, and the offline check that the random baselines are not a single unlucky seed. That pass is what turned several early verdicts into the ones in this report. |
| Henrik Kunzelmann | Everything that needed a 16 GB card: the Qwen3.5-9B baseline, then its measurement, masking sweep, depth ablation, and duplicate-pair tests, plus the Llama-2-7B recovery at the same training budget as the smaller models. Those runs close the size and recovery questions. |

The split above is the work each person is answerable for. Runs and write-ups were shared as the results came in; the names mark who carried each piece.

## The problem

Large language models spend a large share of their parameters in the feed-forward block of each layer. A natural question is whether some of those units are spare: present in the network, but not doing work that the rest of the model cannot cover.

We studied that at the level of a single neuron. For us a neuron is one channel of the feed-forward activation, the input to the down-projection. On a gated model that is one entry of the element-wise product `act(gate(x)) * up(x)`. On GPT-2 it is one entry of `GELU(c_fc(x))`. The channel reaches the residual stream through one column of the down-projection, so its contribution is `h_i · W_down[:, i]`.

We did not remove neurons from the weight matrices. We masked them with a forward hook, which sets that channel to zero. Parameter shapes never change, so this report makes no claim about speed, memory, or FLOPs.

## Questions we iterated on

We started from a simple plan and had to drop most of it.

1. Are some neurons redundant because they rarely fire?
2. Are middle layers less redundant than early or late layers?
3. Are larger models more redundant, and does a good ranking pull further ahead of chance as models grow?
4. Are the neurons that look spare on ordinary text the same ones that look spare on a downstream task?
5. If two neurons are near-duplicates, is one of them cheap to delete? If not, can its contribution be routed through the other?
6. After masking, does a short fine-tune undo the damage, and does that get harder as the model gets larger?

## What we did

**Ranking.** The score we kept is `RMS(h_i) · ‖W_down[:, i]‖`: how large the activation is, times how strongly that column writes into the residual stream. This is the same idea as Wanda (Sun et al., 2024) and LLM-Pruner (Ma et al., 2023). We compared it with firing frequency and with a random mask that removes the same number of neurons in each layer.

**Masking.** We zeroed the lowest-scoring neurons at 5%, 10%, 25%, and 50% of each layer, then measured WikiText-2 perplexity and PIQA accuracy.

**Duplicates.** For pairs with correlation at least 0.9 inside a layer, we masked one twin and compared that with masking an uncorrelated neuron of similar importance. Separately, we fitted `h_drop ≈ α · h_keep + β` on calibration activations and added the dropped twin's contribution back through its partner. That needs two scalars and no training.

**Recovery.** After importance masking we trained a short LoRA on WikiText-2, and we trained the same adapter on the unmasked model. The second run is the control. A LoRA budget can change perplexity even when nothing is masked, which is why the control exists. On the smaller models that budget helps. On Qwen3.5-4B and Llama-2-7B the unmasked control got worse. We report how much of the masking gap remains after both models have had the same budget.

**Models.** GPT-2 at 124M, 355M, and 774M (GELU, no gate). Qwen3-0.6B, Qwen3.5-4B, and Qwen3.5-9B, plus Llama-2-7B (all SwiGLU). GPT-2 ran in full precision. The Qwen and Llama runs that did not fit in memory ran in 4-bit. Qwen3.5-4B was our main model. The 9B model and the Llama recovery needed a 16 GB card.

## Implementation

One calibration pass over WikiText-2 records, per neuron, firing rate, activation size, the importance score above, and pairwise correlation inside a few layers. A hook on the down-projection applies a mask or, for the duplicate experiment, the fitted replacement. Evaluation is WikiText-2 perplexity and PIQA. The same scripts retarget a model by swapping a config file. Nothing in the intervention changes the parameter tensors.

## What we found

### Importance finds spare neurons; firing rate does not

Importance-ranked masking is cheaper than random masking in every model, and the gap grows as we mask more. The figure below shows this on a log scale for six models. Solid lines with markers are importance, dashed lines are random, and dash-dot lines are firing frequency. Qwen3.5-9B is not on the plot; its 50% numbers are in the table. The heading drawn inside the figure says removal. The operation is masking.

![Cross-family removal curves](artifacts/cross_family_removal.png)

Relative perplexity at 50% of each layer masked, importance versus random:

| model | importance | random |
|---|---:|---:|
| GPT-2 124M | 10.6× | 27.9× |
| Qwen3-0.6B | 3.0× | 85.9× |
| Qwen3.5-4B | 2.1× | 17.7× |
| Llama-2-7B | 2.3× | 299× |
| Qwen3.5-9B | 2.1× | 13.0× |

**Take-away.** The ranking transfers across three families and both feed-forward designs. Llama-2-7B is the clearest case: half the feed-forward channels masked by importance costs about twice the perplexity, and the same count chosen at random costs about 300 times. PIQA on that model stays at 0.675 against a 0.765 baseline.

Firing frequency was our original plan, following the lazy-neuron phenomenon (Li et al., 2023): neurons that rarely fire were expected to be the spare ones. It is the wrong signal to move across architectures. On the GPT-2 models it is worse than random at every ratio we tried. On the gated models it usually beats random and still loses to importance. GPT-2 is effectively always on, so how often a neuron fires says little about whether it matters. Two of the gated models (Qwen3.5-4B and 9B) also have almost no silent neurons at our threshold, and frequency still beats random there, so the useful part is the ordering of firing rates, not a pile of dead units. We did not record that distribution, only the fraction below a cutoff.

### Model depth and size do not affect the results as expected

We expected middle layers to be less redundant than early or late layers. A static summary of importance suggested that. However, masking 10% of the channels in each depth band and measuring the resulting cost did not confirm this expectation:

| model | early | middle | deep | worst |
|---|---:|---:|---:|---|
| GPT-2 124M | **+5.24** | +4.08 | +2.31 | early |
| GPT-2 355M | **+7.06** | +2.92 | +1.77 | early |
| GPT-2 774M | +1.06 | +0.98 | +1.01 | flat |
| Qwen3-0.6B | **+2.69** | +0.19 | +0.81 | early |
| Qwen3.5-4B | +0.34 | +0.37 | **+1.02** | deep |
| Qwen3.5-9B | +0.16 | +0.14 | **+0.57** | deep |
| Llama-2-7B | +0.27 | +0.10 | **+0.42** | deep |

Entries are raw perplexity changes. Compare columns within a row. The baselines differ, so a large change on GPT-2 is not a larger effect than a small change on a 9B model.

**Take-away.** The middle band is never the expensive one. Early layers cost the most on the smaller models; deep layers cost the most on the three largest gated models. The static score ranks neurons inside a layer. It does not predict what removing a whole band costs.

Size fails in a similar way. Inside GPT-2, where architecture stays fixed, relative cost at 25% masking does fall with size (2.67×, 2.47×, 1.46×). At 10% and at 50% it does not, and the static profile barely moves from 124M to 774M. Inside Qwen the gap between importance and random shrinks as the model grows (3.8×, 1.9×, 1.6× at 25%), which is the opposite of what we predicted. Feed-forward design predicts how well a model tolerates masking better than parameter count does.

### The task matters on gated models

We repeated the measurement on PIQA prompts and masked by that ranking instead. At 25% of each layer:

| model | ranked on WikiText | ranked on PIQA | random |
|---|---|---|---|
| Llama-2-7B | 12.0 PPL / 0.750 | 14.3 PPL / **0.785** | 17.9 / 0.695 |
| Qwen3.5-4B | 25.0 PPL / 0.705 | 33.1 PPL / **0.755** | 47.5 / 0.675 |
| Qwen3-0.6B | 52.9 PPL / 0.594 | 94.8 PPL / **0.630** | 199 / 0.594 |
| GPT-2 124M | **138 PPL / 0.562** | 312 PPL / 0.544 | 194 / 0.570 |

**Take-away.** On the gated models, each ranking wins on the data it was fit to. On GPT-2, the PIQA ranking performs worse on both metrics, even compared to random masking for perplexity. The two corpora still share a core of cheap neurons. Calibrating on the wrong text costs accuracy. It does not make the ranking useless.

### Duplicates are cheap to replace, not cheap to delete

Near-duplicate pairs exist in every model. Masking one twin of a pair was not reliably cheaper than masking a matched unrelated neuron. Across seven models the comparison goes one way on three and the other way on four, and wherever the control neuron really matches the twin in importance the difference is a few tenths of a perplexity point or less.

What does work is keeping the information. When masking a twin degrades performance, routing it through its partner recovers most of the performance loss, with recovery depending on how well one activation predicts the other. No training is involved.

| model | pairs | fit r² | cost of masking | cost after replacement | recovered |
|---|---:|---:|---:|---:|---:|
| GPT-2 124M | 22 | 0.90 | +1.16 | −0.02 | 102% |
| Qwen3-0.6B | 14 | 0.89 | +2.24 | +0.28 | 87% |
| Llama-2-7B | 17 | 0.62 | +0.04 | +0.02 | 47% |

**Take-away.** Correlation says the two neurons move together. It does not say they have the same scale, so the gain α has to be fitted. On the models where masking a twin was already free, including Qwen3.5-4B and 9B, there is nothing for replacement to recover, and a percentage there is noise.

### Short fine-tuning recovers the damage only on the smaller models

Every recovery run below, except the 4B row, used the same budget: 200 steps, batch 2, gradient accumulation 4, sequence length 512, about 819 thousand training tokens. The 4B run used about 51 thousand.

| model | 10% masked | 25% masked |
|---|---:|---:|
| GPT-2 124M | 80% | 87% |
| GPT-2 355M | 84% | 89% |
| GPT-2 774M | 76% | 80% |
| Qwen3-0.6B | 76% | 83% |
| Llama-2-7B | **−6%** | **48%** |
| Qwen3.5-4B (smaller budget) | 2% | — |

**Take-away.** Up through GPT-2 774M and Qwen3-0.6B, most of the masking cost comes back, and model size inside GPT-2 barely matters. At Llama-2-7B the same budget overfits: training loss falls and WikiText perplexity rises on both the masked model and the unmasked control. The 4B result looked like the same failure, but that run was not comparable, because it saw 16 times less text. The 7B run was comparable at 819 thousand tokens, and it still failed. We did not try a shorter schedule, so this is one training recipe, not a proof that a 7B mask can never be repaired. Qwen3.5-9B was not fine-tuned. The 7B run already answers that question.

## Problems

**Solved, by changing the method.** Firing rate saturated on gated models, so we stopped using it as the main ranking. The depth hypothesis was supported by a static score and rejected by the actual masking experiment, so we stopped treating that score as a substitute for an ablation. The first duplicate experiment masked the less important twin first, which made the second twin look special for a reason that had nothing to do with duplication. Replacement compares two treatments of the same neuron and avoids that.

**Solved, by more compute.** The 8 GB card could not finish a 4-bit Llama-2-7B recovery (a library error during evaluation). We ran the experiment on a 16 GB GPU. The same card produced the 9B measurement and masking results.

**Still open.** We never evaluated the image-text model, Qwen3.5-4B, on images. A neuron that fires only on image tokens would look idle on WikiText and we would call it spare. Masking is not deletion, so we cannot say the network got faster. Random baselines are one seed each; we checked that another seed would not pick a very different amount of importance, but we do not have error bars on perplexity. The duplicate results rest on a few layers and, in the Qwen3.5 models, on four pairs. Recovery was tried at one setting.

## What we would stand behind

Spare feed-forward neurons are real, and an activation-aware score finds them in every model we tested. How often a neuron fires, which layer it sits in, and how large the model is do not tell you what we thought they would. A duplicated neuron is not a free deletion. Its activity is often available in its partner, and a two-scalar rewrite recovers it in proportion to how linear that relationship is. A short fine-tune repairs masking on the smaller models and, at the budget we could match, overfits at 7B.

## References

- Li et al. (2023). *The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers.* ICLR.
- Ma, Fang, and Wang (2023). *LLM-Pruner: On the Structural Pruning of Large Language Models.* NeurIPS.
- Sun, Liu, Bair, and Kolter (2024). *A Simple and Effective Pruning Approach for Large Language Models.* ICLR.
