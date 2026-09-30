# Weeks 13–14 — Does the redundancy result survive a change of model family?

**Group 5 deliverable: the generalisation check raised in review — Qwen3.5 is
one family, and every number in Weeks 5–12 came from it.**

> **Errata (2026-09-09).** Two things in this report were written before the
> `gpt2-medium` / `gpt2-large` duplicate-pair arms existed:
>
> - **§5 called H5 "refuted".** With all six models in, the verdict flips 3–3
>   and the margins are at the noise floor wherever the control is fair. The
>   correct verdict is **"not supported"** — no consistent sign — not "refuted".
>   The substantive point is unchanged: H5 is not a usable claim, and H5′
>   (replacement) is the result that holds.
> - **§7's scoreboard is superseded** by
>   [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md), which is the single
>   frozen source of truth. It also carries H1's reversal, which happened after
>   this report when the depth-tertile *ablation* was finally run.
>
> Everything else here still stands. Final report:
> [`analysis_week15-16.md`](analysis_week15-16.md).
**Status:** complete for six models across three families — GPT-2 small/medium/
large, Llama-2-7B, Qwen3-0.6B, Qwen3.5-4B (all run 2026-08-14 except the two
Qwen baselines). Tables regenerate with `scripts/compare_families.py`; the
generated version lives in
[`artifacts/cross_family_tables.md`](artifacts/cross_family_tables.md).

**Headline:** redundancy and its repair survive the change of family, and
Llama-2-7B is the strongest case we have — half its FFN neurons can be removed
by importance for 2.27× perplexity, where random removal costs 299×. Three
Qwen-era conclusions do *not* survive: the firing-frequency ranking is worse
than random on every GELU model, the "bigger is more redundant" trend disappears
once size is varied with architecture held fixed, and H5 turns out to be
genuinely refuted rather than mis-scored.

Until now "LLMs carry redundant FFN neurons" was not separable from "Qwen
carries redundant FFN neurons". This stage adds a family whose FFN is
architecturally the *opposite* of Qwen's, a second gated family from a different
vendor, and a size arm within one architecture.

| Contrast | Models | What it isolates | Verdict |
|---|---|---|---|
| architecture | GPT-2 (GELU, no gate) vs Qwen/Llama (SwiGLU) | is redundancy a gating artifact? | no — but frequency ranking is (§3) |
| family | Llama-2-7B vs Qwen3.5-4B (both SwiGLU) | is it a vendor artifact? | no — Llama is the strongest result (§2) |
| size | GPT-2 124M / 355M / 774M | H2 with architecture held fixed | not supported (§4) |

---

## 1. The pipeline did not actually support other families

Every neuron accessor assumed the `nn.Linear` weight layout
`[out_features, in_features]`. GPT-2's MLP is built from transformers'
`Conv1D`, which computes `x @ W` and therefore stores
`[in_features, out_features]`, with no `in_features` attribute. On GPT-2 this
went wrong in two places at once:

- `layer_width` fell through to `weight.shape[1]` and reported the **hidden**
  size 768 instead of the **intermediate** size 3072, so masking built a
  768-wide mask for a 3072-wide activation and raised a shape error.
- `input_column_norms` reduced over the wrong axis, returning 768 numbers
  instead of 3072 — the wrong length, but *silently* so, which would have
  produced plausible-looking nonsense importance scores.

The orientation is now normalised once, in `hooks._dense_weight`, and
`layer_width` knows about `Conv1D.nx`. Nothing downstream changed.
`tests/test_model_families.py` pins this on a toy GPT-2-shaped model; the
sharpest check is that merging an *exactly* duplicated neuron is lossless,
which fails on any surviving transpose slip.

Two things confirm the fix on the real models: every layer of all six models
reports `col_norm_available: true` — including Llama-2's 32 layers, which
exercises the 4-bit dequantization path — and masking a neuron is numerically
identical to that neuron never firing.

> Separately, `src/redundancy/hooks.py` on `main` had carried unresolved merge
> conflict markers since the June "Merge from origin to main" commit, so the
> whole package failed to import with a `SyntaxError`. Fixed in
> `2cd20fa` — that commit should be cherry-picked to `main`.

## 2. Redundancy is not a gating artifact

GPT-2 has no gate, no SwiGLU, a different tokenizer, and is 5× smaller than
our small control. It still shows the same qualitative profile.

| model | layers × width | max \|r\| | pairs ≥ 0.9 | bottom-10% importance share |
|---|---|---|---|---|
| GPT-2 124M | 12 × 3072 | **1.000** | 49/49 | 6.45% |
| GPT-2 355M | 24 × 4096 | 0.998 | 58/58 | 6.59% |
| GPT-2 774M | 36 × 5120 | 0.997 | 98/98 | 6.58% |
| Llama-2-7B | 32 × 11008 | 0.997 | 85/85 | 6.80% |
| Qwen3-0.6B | 28 × 3072 | 0.999 | 92/92 | 4.58% |
| Qwen3.5-4B | 32 × 9216 | 0.946 | 4/4 | 6.39% |

Near-duplicate neurons are present in GPT-2 at least as strongly as in Qwen —
the top pair in `transformer.h.2.mlp.c_proj` reaches ρ = 0.9999 — and in
Llama-2 as strongly again. And the importance ranking beats matched-sparsity
random masking at every ratio in four of six models, and at the ratios that
matter for pruning (≥25%) in all six:

| masked/layer | GPT-2 imp / rand | Llama-2-7B imp / rand | Qwen3-0.6B imp / rand | Qwen3.5-4B imp / rand |
|---:|---:|---:|---:|---:|
| 5% | **1.11×** / 1.20× | **1.05×** / 1.12× | **1.04×** / 1.25× | **1.03×** / 1.08× |
| 10% | **1.31×** / 1.47× | **1.09×** / 1.20× | **1.13×** / 1.71× | **1.10×** / 1.40× |
| 25% | **2.67×** / 3.75× | **1.27×** / 1.89× | **1.57×** / 5.89× | **1.32×** / 2.52× |
| 50% | **10.57×** / 27.91× | **2.27×** / 299.10× | **2.96×** / 85.93× | **2.11×** / 17.71× |

(perplexity relative to each model's own unmasked baseline: GPT-2 51.65,
Llama-2-7B 9.46, Qwen3-0.6B 33.75, Qwen3.5-4B 18.87. The two larger GPT-2
models are the exception at 5–10%, where random is marginally cheaper — §4b.)

**Llama-2-7B is the clearest demonstration in the project.** Removing half of
every FFN by importance costs 2.27× perplexity; removing the same number of
neurons at random costs 299×, a 132-fold difference. PIQA holds at 0.675 against
a 0.765 baseline with half the FFN gone. Whatever the importance score is
picking up on, it is not weak, and it is not Qwen-specific.

**So H1 generalises.** Importance-guided redundancy is a property of trained
decoder-only transformers, not of gated FFNs and not of one vendor.

## 3. The *frequency* ranking does not generalise

This is the sharpest new result, and it is a negative one. Our originally
planned Lazy-Neuron-style ranking by firing frequency is **worse than random on
every GELU model at every ratio** — 12 out of 12 comparisons — while beating
random on the gated models in 11 of 12:

| masked/layer | gpt2 | gpt2-medium | gpt2-large | Llama-2-7B | Qwen3-0.6B | Qwen3.5-4B |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 2.45× / 1.20× | 29.01× / 1.07× | 1.27× / 1.03× | **1.09×** / 1.12× | **1.07×** / 1.25× | 1.10× / 1.08× |
| 10% | 4.85× / 1.47× | 51.29× / 1.28× | 1.66× / 1.09× | **1.15×** / 1.20× | **1.18×** / 1.71× | **1.23×** / 1.40× |
| 25% | 15.36× / 3.75× | 349.77× / 4.10× | 35.03× / 1.55× | **1.60×** / 1.89× | **1.97×** / 5.89× | **1.87×** / 2.52× |
| 50% | 125.72× / 27.91× | 2577.74× / 189.49× | 237.12× / 69.60× | **9.71×** / 299.10× | **7.18×** / 85.93× | **6.81×** / 17.71× |

(frequency / random, each relative to that model's own baseline; bold = frequency wins)

The split follows FFN design exactly — 0 for 12 on GELU, 11 for 12 on SwiGLU —
and it is not subtle: on gpt2-medium at 50%, ranking by frequency is **13×
worse** than picking neurons at random. Three GPT-2 sizes rule out "GPT-2 small
is just an odd little model", and two vendors on the gated side rule out a Qwen
quirk.

The lazy-neuron measurement mostly explains it. At the strictest RMS-relative
threshold (α = 0.25) the gated models have a real quiet subpopulation to find
and the GELU models have none:

| model | FFN | lazy neurons @ α = 0.25 | frequency beats random |
|---|---|---:|---|
| gpt2 / medium / large | GELU | 0.05% / 0.02% / 0.02% | 0 of 12 |
| Llama-2-7B | SwiGLU | **3.12%** | 4 of 4 |
| Qwen3-0.6B | SwiGLU | **4.30%** | 4 of 4 |
| Qwen3.5-4B | SwiGLU | 0.00% | 3 of 4 |

GELU is smooth and effectively always on, so firing frequency carries almost no
information about whether a neuron matters, and ranking by it selects neurons
that turn out to be load-bearing. The multiplicative gate in SwiGLU produces
genuinely quiet channels — 3–4% of them in both Llama-2 and Qwen3-0.6B.

**Qwen3.5-4B remains unexplained**: 0.00% lazy neurons, yet frequency still
beats random at three of four ratios. So the ranking cannot depend *only* on a
dead subpopulation; the *ordering* of firing rates must carry signal in gated
FFNs even when no neuron is quiet enough to cross our threshold. Our measurement
records only how many neurons fall below the threshold, not how the rates are
distributed, so it cannot settle this. Recording the per-neuron frequency
distribution would.

**Practical consequence stands regardless:** frequency-based pruning heuristics
should not be transferred across FFN designs without re-validation. Importance
(`RMS(h) · ‖W_down[:, i]‖`) transferred; frequency did not.

![Cross-family removal curves](artifacts/cross_family_removal.png)

*(Log axis. Solid = importance, dash-dot = frequency, dashed = random; colour is
the model. Committed copy of `experiments/results/figures/cross_family_removal.png`,
which is gitignored like every other generated figure.)*

## 4. Size (H2) — the clean test does not support it

The cross-family ordering looks like textbook H2: at 25% masking the relative
cost falls almost monotonically with size (GPT-2 124M 2.670× → Qwen3-0.6B
1.566× → Qwen3.5-4B 1.323× → Llama-2-7B 1.267×), with the 7B model cheapest of
all. But that comparison varies size *and* architecture *and* training corpus
together. GPT-2 small/medium/large vary only size, and they tell a different
story:

| model | params | layers × width | rel PPL @ 10% | @ 25% | @ 50% |
|---|---:|---|---:|---:|---:|
| gpt2 | 124M | 12 × 3072 | 1.307 | 2.670 | **10.565** |
| gpt2-medium | 355M | 24 × 4096 | 1.374 | 2.470 | 16.153 |
| gpt2-large | 774M | 36 × 5120 | **1.109** | **1.461** | 14.874 |

H2 holds at 25%, where cost falls by 45% from small to large. It fails at 50%,
where **both larger models are worse than the smallest**, and it fails at 10%,
where medium is worse than small. A hypothesis that only holds in the middle of
the sweep is not the clean monotone trend the Qwen-only evidence suggested.

The measurement side says the same thing more directly: the static redundancy
profile barely moves across a 6× parameter range. The share of importance held
by the bottom 10% of neurons is 6.45% / 6.59% / 6.58% for small / medium /
large, and the strongest correlated pair is ≈1.0 in all three. **Within a family,
scaling does not make neurons more redundant by these measures** — it changes
how gracefully the model absorbs their removal, and only in part of the range.

Two caveats worth keeping. Larger GPT-2 models are better models to begin with
(PIQA 0.628 → 0.680 → 0.714, baseline PPL 51.65 → 37.17 → 32.46), so at 50%
they also have further to fall. And the 50% arm is deep in the regime where all
three are destroyed (10–16× baseline), so its ordering is the least meaningful
of the three.

**Revised reading of H2:** the cross-family trend is real but is at least partly
an architecture-and-quality effect, not a pure size effect. Held to one family,
size buys tolerance to *moderate* pruning only.

The gated models make the same point from the other side. Llama-2-7B is 1.8×
the size of Qwen3.5-4B and only slightly cheaper to prune at 25% (1.267× vs
1.323×), while the 12× jump from Qwen3-0.6B to Qwen3.5-4B buys more (1.566× →
1.323×). The strongest predictor of how well a model absorbs pruning in our data
is its FFN design, not its parameter count.

### 4b. The importance ranking needs sparsity to pay off

The size arm surfaced a second thing worth flagging. On the two larger GPT-2
models, importance-guided masking is **slightly worse than random** at 5% and
10%:

| model | 5% imp / rand | 10% imp / rand | 25% imp / rand |
|---|---:|---:|---:|
| gpt2 | 1.112 / 1.199 | 1.307 / 1.472 | 2.670 / 3.754 |
| gpt2-medium | 1.156 / **1.067** | 1.374 / **1.278** | 2.470 / 4.103 |
| gpt2-large | 1.053 / **1.031** | 1.109 / **1.085** | 1.461 / 1.552 |

The differences are small (≤0.09× baseline) and the ranking wins decisively
once ≥25% is removed, which is where it matters. The likely reason is that
importance is heavy-tailed: at 5% a random draw almost never hits a critical
neuron either, so the two arms are effectively tied, while the ranking removes a
*coherent* block of low-magnitude neurons rather than a scattered one. This does
not undermine H1 — but "importance beats random at every ratio in every model",
true of the Qwen runs and of GPT-2 small, is no longer accurate as stated, and
§2's table should be read as covering the sparsities that matter for pruning.

## 5. H5 as stated does not hold — and Llama-2 removes our excuse for it

Full six-model table (the two larger GPT-2 arms were added 2026-08-23; the
generated version is [`artifacts/h5_drop_vs_merge.md`](artifacts/h5_drop_vs_merge.md)):

| model | pairs | 1st twin | matched control | margin | control coverage | verdict |
|---|---:|---:|---:|---:|---:|---|
| GPT-2 124M | 22 | +1.164 | −0.688 | −1.852 | 0.57× | False |
| GPT-2 355M | 15 | +0.004 | +0.267 | +0.263 | **1.01×** | True |
| GPT-2 774M | 16 | −0.119 | +0.034 | +0.153 | 0.65× | True |
| Llama-2-7B | 17 | +0.044 | −0.204 | −0.248 | **0.94×** | False |
| Qwen3-0.6B | 14 | +2.241 | −0.279 | −2.520 | 0.60× | False |
| Qwen3.5-4B | 4 | −0.013 | +0.033 | +0.046 | **1.00×** | True |

The Weeks 9–10 write-up explained away GPT-2's and Qwen3-0.6B's False verdicts
as a broken control: the matched-importance arm reached only 0.57–0.60× the
duplicates' mean importance, so it was a materially easier ablation. Llama-2
tests that excuse directly, because with 11008 neurons per layer there is a much
larger pool to draw a matched control from — and indeed **its control reaches
0.94×**, a genuinely matched comparison. H5 still fails there
(+0.044 for the duplicate twin against −0.204 for the control).

**So the excuse is dead, but so is the claim in either direction.** The verdict
splits 3–3, and restricting to the three models where the control is genuinely
matched (≥0.94×) does not settle it either: `gpt2-medium` and Qwen3.5-4B come
out True with margins of +0.263 and +0.046 PPL, Llama-2-7B comes out False with
−0.248. Those are noise-floor effects on 4–17 pairs, and the margins that *are*
large (−1.85 on GPT-2 124M, −2.52 on Qwen3-0.6B) both come from runs whose
control was unmatched.

**The honest conclusion: H5 as stated is not supported, because the effect has
no consistent sign.** Being a duplicate does not reliably make a neuron cheaper
to mask than being unimportant does. That is enough to stop making the claim —
and §5b is the version of the question that does have a stable answer.

The within-pair asymmetry is also weaker evidence than we treated it as. Masking
the second twin costs more than the first in three of four models — dramatically
so on Llama-2 (29×) and Qwen3-0.6B (1286×) — but the procedure masks the
*lower-importance* twin first, so the second twin is the more important neuron by
construction. On Llama-2 it is about 49× more important, which is more than
enough to explain a 29× cost on its own. Only GPT-2 shows excess beyond that (the
second twin is 1.4× more important but 1.9× more costly), and that margin is too
small to carry the claim.

**The asymmetry is therefore consistent with "the survivor absorbed its
partner's job", but it does not distinguish that from "important neurons cost
more to remove".** The replacement experiment does.

### 5b. Replacement is the result that holds up

Ablation asks "is this neuron cheap to delete?". Replacement asks the question we
actually care about — "is its information available elsewhere?" — by handing the
dropped twin's contribution to its partner via the fitted map
`h_drop ≈ α·h_keep + β`. Crucially, this compares two treatments of the *same*
neuron, so it is free of the importance confound above. Same pairs and same eval
slice as §5.

| model | fit quality (mean / min r²) | mask only | after merge | masking cost undone |
|---|---:|---:|---:|---:|
| GPT-2 124M | 0.902 / 0.833 | +1.164 | **−0.023** | **102%** |
| Qwen3-0.6B | 0.892 / 0.451 | +2.241 | +0.284 | 87% |
| Llama-2-7B | 0.618 / 0.082 | +0.044 | +0.023 | 47% |

Merging beats masking in all three families, and **how much it recovers tracks
how linearly predictable the twin is** — 0.902 → 102%, 0.892 → 87%, 0.618 → 47%,
in order. That relationship is the mechanistic claim: the repair works exactly to
the extent that one twin is a linear function of the other, which is what
"redundant" should mean. Dropping the constant `β` barely changes any of the
three, so the recovery comes from rescaling rather than a bias fudge.

Llama-2 recovers least because its pairs are the loosest: mean r² 0.618, and its
worst pair is 0.082 — selected on correlation of activations over a short
calibration slice, some pairs simply do not hold up. Its absolute stakes are also
tiny (+0.044 PPL to mask 17 of 352,256 neurons), so this arm is the least
informative of the three despite being the largest model.

**So the claim to make is "duplicated neurons are cheap to *replace*, in
proportion to how well one predicts the other"** — not "duplicated neurons are
cheap to remove", which §5 shows is false. Naive masking throws the shared
information away; merging keeps it.

**Recommendation for the write-up:** lead with the replacement result and its r²
relationship, report H5-as-stated as refuted rather than inconclusive, and drop
the within-pair asymmetry as evidence unless it is re-run with the twin choice
randomised so importance is not confounded with masking order.

## 6. Reproducing this

```bash
for m in gpt2 gpt2-medium gpt2-large llama2-7b; do
  python scripts/run_measurement.py --config configs/measurement/neuron_activations_$m.yaml
  python scripts/run_pruning.py     --config configs/pruning/neuron_masking_$m.yaml
done
python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation_gpt2.yaml
python scripts/run_merge.py         --config configs/pruning/merge_gpt2.yaml
python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation_llama2-7b.yaml
python scripts/run_merge.py         --config configs/pruning/merge_llama2-7b.yaml

python scripts/compare_families.py --model gpt2 --model gpt2-medium \
    --model gpt2-large --model meta-llama/Llama-2-7b-hf \
    --model Qwen/Qwen3-0.6B --model Qwen/Qwen3.5-4B \
    --out reports/group_5/artifacts/cross_family_tables.md --logy
```

Llama-2-7B runs in 4-bit NF4 inside 3.9 GB of VRAM. Its measurement config
halves the reservoir to 2048 tokens (32 × 2048 × 11008 × 2 B ≈ 1.4 GB) to fit a
16 GB host, and its masking config lowers the lm-eval batch size and caps PIQA
at 200 examples — so its accuracy column is coarser than the others' and small
differences in it should not be read.

### Two infrastructure notes for whoever runs this next

Both cost hours to diagnose and are fixed in the branch, but they are the kind
of failure that looks like a broken experiment rather than a broken tool.

**Downloading.** `huggingface_hub.snapshot_download` stalled at 0 bytes on the
large shards on this machine, with and without `HF_HUB_DISABLE_XET=1`, while
small files downloaded fine. Use `scripts/download_model.py`, which is plain
ranged HTTP. It now retries with resume when the connection drops mid-shard
(a 1.5 GB file died at 1.27 GB) and no longer descends into `onnx/`, which both
crashed the download and would have pulled a second copy of the weights.

**Loading on Windows.** Llama-2 is the first *sharded* checkpoint in the
project, and loading it killed the interpreter with an access violation and no
traceback. It is not corruption — both shards match their SHA256 on the Hub.
`safetensors.safe_open` maps the checkpoint into memory, so reading a weight is
an ordinary memory access, and when Windows cannot service the page fault
(13.5 GB of mappings against ~4 GB of free RAM) the process dies rather than
raising. `src/redundancy/safetensors_pread.py` reads each tensor's byte range
with ordinary file reads instead; it is enabled automatically on Windows, and
`REDUNDANCY_DISABLE_PREAD_SAFETENSORS=1` turns it off. Transformers' own
`disable_mmap` is not usable here because it reads whole shards into RAM.

## 7. What this changes in the scoreboard

> Superseded by [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md), which is
> the frozen version and also covers H1's later reversal. Kept here as the
> record of what *this stage* changed.

| Hypothesis | Before (Qwen only) | After (6 models, 3 families) |
|---|---|---|
| H6 redundancy exists and is rankable across families | supported (Qwen) | **supported in every model**; strongest on Llama-2-7B (2.27× vs 299× at 50%) |
| H2 larger models are more redundant | supported within Qwen | **not supported** when size is varied alone (GPT-2 arm); the cross-family trend is confounded with architecture |
| H5 duplicates are cheap to drop singly | "not supported", blamed on a bad control | **not supported** — the bad-control excuse fails on Llama-2 (0.94×), and with all six models the verdict has no consistent sign (3–3) |
| H5′ duplicates are cheap to **replace** | 87% repaired (Qwen) | supported in all three families (102% / 87% / 47%), and recovery **tracks fit r²** |
| H7 *(new)* frequency ranking transfers | assumed | **refuted** — 0 of 12 on GELU, 11 of 12 on SwiGLU |
| H4 *(narrowed)* importance ranking helps at any sparsity | assumed | holds ≥25%; at 5–10% it ties or slightly loses on larger GPT-2 |

## 8. Limitations

- The architecture contrast rests entirely on GPT-2, which is small and old. A
  modern non-gated model would be a better control, but there are few, so
  "GELU" and "GPT-2" are not fully separable in the frequency result.
- GPT-2's perplexity window is 1024 (its context limit) versus 2048 for the
  Qwen and Llama runs, so absolute PPL is not comparable across the two — which
  is why every cross-model number here is normalised to each model's own
  baseline.
- Llama-2-7B is evaluated in 4-bit NF4 while the GPT-2 models are fp32, so its
  baseline carries quantization error the others do not. This works *against*
  the Llama result rather than inflating it — a quantized model has less
  headroom, not more.
- The GPT-2 size arm confounds size with model quality: larger GPT-2 models
  start from a much better baseline (PPL 51.65 → 32.46), so they have further
  to fall at high sparsity.
- Correlation drill-down covers 3 layers per model, so "pairs ≥ 0.9" counts
  are a sample, not a census.
- Duplicate-pair counts are small and uneven: 22 pairs on GPT-2, 17 on Llama-2,
  14 on Qwen3-0.6B, and only 4 on Qwen3.5-4B.
- PIQA is capped at 200 examples for Llama-2 (VRAM), so ±0.03 swings in its
  accuracy column are noise.
