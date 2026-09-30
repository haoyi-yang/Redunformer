# Weeks 5–8 — Neuron / FFN-unit Redundancy Measurement

**Group 5 deliverable: first implementation of the redundancy measurement + plots.**
**Status:** implementation complete; GPU runs done on 2026-06-29 (Qwen3-0.6B + Qwen3.5-4B). See `experiments/results/measurement_*.json` and `experiments/results/figures/`.

This stage implements the *measurement* described in the proposal (Q4/Q6). No
compression or masking yet — that is Weeks 9–10. The goal is to quantify and
visualize **where** redundant FFN neurons appear.

> **Key revision (2026-06-29).** The first pass used RMS-relative firing
> frequency as the headline "lazy neuron" signal. On SwiGLU models that signal
> **saturates** — post-SwiGLU activations are almost never near zero, so >99% of
> neurons fire on ~100% of tokens and the frequency plots are nearly empty. We
> kept frequency (as a sweep, it is a valid negative result) and added the
> **activation-aware importance** metric (`RMS × ||W_down[:, i]||`, LLM-Pruner /
> Wanda style) as the primary, non-saturated redundancy signal that will drive
> the Weeks 9–10 ablation.

---

## 1. What a "neuron" is here

A neuron is one **post-SwiGLU activation channel** inside an FFN block, i.e. one
entry of `act_fn(gate_proj(x)) * up_proj(x)`. That vector is exactly the **input
to `down_proj`**, so we capture it with a forward **pre-hook** on the down
projection (`src/redundancy/hooks.py::iter_ffn_down_projections` /
`NeuronActivationCollector`). GPT-2 style MLPs (`mlp.c_proj`, single GELU) are
also supported for cross-checks.

## 2. Metrics implemented

All in `src/redundancy/metrics/` and computed in a **single forward sweep** over
the calibration set, with memory bounded by a per-layer reservoir (not the
calibration size):

| Metric | Definition | Exact / estimated |
|--------|------------|-------------------|
| **Neuron importance** `imp_i,ℓ` *(primary)* | `RMS(h_i) · ‖W_down[:, i]‖₂` — expected L2 contribution of neuron `i` to the residual (LLM-Pruner / Wanda style) | RMS exact; column norm from (dequantized) weights |
| **Importance share of bottom-k%** | share of total layer importance held by the least-important `k%` of neurons; small ⇒ those neurons are cheap to remove | derived |
| **Activation frequency (RMS-relative)** `f_i,ℓ` | fraction of token positions with `|a_i| > α·RMS_ℓ`, swept over α ∈ {1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1} | estimated on the reservoir subsample |
| **Activation frequency (fixed-ε)** | fraction with `|a_i| > ε` (default 1e-3) | exact streaming |
| **Mean magnitude** `E[|a_i|]` and **RMS** | per-neuron, over the full calibration set | exact streaming |
| **Layer RMS** `RMS_ℓ` | `sqrt(mean over neurons & tokens of a²)` | exact streaming |
| **Pairwise correlation** | top-k `|ρ|` neuron pairs per selected layer (Pearson on the reservoir); `|ρ| ≥ abs_threshold` flags duplicate candidates | estimated on the reservoir |

**Importance** is the primary redundancy signal: the post-SwiGLU activation
`h_i` contributes `h_i · W_down[:, i]` to the FFN output, so a neuron with small
`RMS(h_i) · ‖W_down[:, i]‖` contributes little regardless of how *often* it
fires. For 4-bit (bitsandbytes) models the down-projection weight is dequantized
on the fly to read the column norms; if that fails the importance falls back to
activation RMS alone (`col_norm_available` records which path was used).

The RMS-relative frequency was the originally-planned **"lazy neuron" probe**
(Li et al. 2022). It saturates here (see the revision note), so we report it as a
**threshold sweep** over α: even as α grows the lazy fraction stays near zero for
most layers, which is itself the finding. The fixed-ε frequency is the
sanity-check alternative agreed in `metrics_week3-4.md`.

### Why a reservoir

The layer-RMS-relative threshold is not known until the whole sweep finishes, and
caching every activation row is infeasible on 8 GB. We therefore keep an exact
streaming aggregate for everything that can be counted against a *fixed*
threshold, plus a bounded **reservoir** (Algorithm-R, default 4096 token rows per
layer, stored in float16) for the RMS-relative frequency and correlation. This
keeps peak memory at `max_reservoir_tokens × width × 2 bytes` per layer.

## 3. Outputs

`scripts/run_measurement.py` writes, per run, to `experiments/results/`:

- **`measurement_<model>_<ts>.json`** — per-layer summary (mean `f`, lazy
  fraction, layer RMS, mean magnitude), depth-tertile aggregates (early / middle
  / deep), and the top correlated pairs for the selected layers. Includes
  model id, seed, config path, and full command for reproducibility.
- **`measurement_<model>_<ts>.npz`** — per-neuron arrays (`importance`,
  `col_norm`, `freq_rms`, `freq_fixed`, `mean_abs`, `rms`) keyed by layer,
  consumed by the Weeks 9–10 masking stage.
- **Figures** under `experiments/results/figures/`:
  - `imp_hist_*` — per-layer log-log histograms of neuron importance (heavy-tailed).
  - `imp_depth_*` — removability proxy: importance share of the bottom-10%
    neurons vs depth (the H1 depth test).
  - `freq_hist_*` — per-layer histograms of activation frequency (log y).
  - `freq_depth_*` — heatmap of the frequency distribution across depth.
  - `lazy_sweep_*` — lazy-neuron fraction vs depth, one line per α threshold.
  - `corr_layer<i>_*` — correlation heatmap of a real neighborhood (top
    duplicate pairs padded with high-variance neurons) in each selected layer.

## 4. How to run

Primary (Qwen3.5-4B, 4-bit, ~8 GB GPU):

```powershell
uv run python scripts/run_measurement.py --config configs/measurement/neuron_activations.yaml
```

Smoke (small control model, CPU-friendly):

```powershell
uv run python scripts/run_measurement.py --config configs/measurement/neuron_activations_smoke.yaml --use-fallback-model
```

Key config knobs (`configs/measurement/neuron_activations.yaml`):
`calibration.seq_len` / `max_blocks`, `measurement.fixed_eps` / `rms_alpha` /
`max_reservoir_tokens`, and the `correlation` block (`layers: auto | tertiles |
[explicit]`, `k`, `max_neurons`, `abs_threshold`).

## 5. Findings (2026-06-29 runs)

Runs: **Qwen3-0.6B** (28 FFN layers, width 3072, 128×512 calib tokens) and
**Qwen3.5-4B** 4-bit (32 LM FFN layers, width 9216, 256×512 calib tokens). The
4B is an ImageTextToText hybrid: 56 down-projections are hooked but only the 32
`language_model` FFN layers fire on text input (the vision-tower projections are
correctly never measured).

1. **Firing frequency is saturated → use importance.** Even sweeping α up to
   0.25, the lazy-neuron fraction stays ≈0 across depth for the 4B
   (`lazy_sweep_*`). Post-SwiGLU activations are essentially never near zero, so
   the *Lazy Neuron* probe (designed for ReLU MLPs) does not transfer. The
   activation-aware **importance** distribution is heavy-tailed and
   discriminative (`imp_hist_*`).
2. **Depth structure appears in importance, not frequency (H1).**
   *(Superseded — see the note at the end of this section.)* The bottom-10%
   importance share (cheap-to-remove mass) is highest in early/mid layers
   (~7.5% at 4B layers 5–9) and lowest at the **deepest layer** (3.4% at 4B
   L31; 2.3% at 0.6B L27) — i.e. the least-important neurons in deep layers are
   the cheapest to prune. Deep 4B layers also show a second high-importance bump
   (massive-activation neurons).
3. **Massive-activation layer (0.6B L2).** Layer 2 of 0.6B has a layer RMS ~40×
   its neighbours; under the α-sweep ~84% of its neurons fall below α·RMS at
   α=0.1. This is the known massive-activation / attention-related outlier
   phenomenon and is now surfaced by the sweep.
4. **Duplicate neurons exist (correlation).** Layers are auto-selected by a
   duplication score; the drill-down finds near-duplicate pairs (e.g. 0.6B L27
   ρ≈0.997; 4B early layers ρ≈0.91–0.95), visualized as real correlation
   neighborhoods in `corr_layer*`.

**Implication for Weeks 9–10:** rank neurons for masking by **importance** (and
correlation for merging), not by firing frequency. Compare against a
random-at-matched-ratio baseline as planned.

> **Follow-up (2026-07-26).** Weeks 9–10 confirmed this: importance-guided
> masking beats both frequency and random on perplexity at every ratio, and the
> duplicate pairs found in finding 4 turned out to be *jointly critical* rather
> than spare capacity (masking one twin in 0.6B layer 2 costs +1.8 PPL, masking
> both costs +2898). See `intervention_week9-10.md`.

> **Correction to finding 2 (2026-09-09).** The bottom-10% importance share is a
> *static proxy* for removability, and the proxy turned out to be a poor one. We
> finally ran the ablation it was standing in for — mask 10% of the neurons in
> each depth tertile and measure the cost — on all six models, and the two
> disagree. On GPT-2 the proxy nominates the middle tertile as the least
> redundant, but masking the **early** tertile costs more than twice as much
> (+5.24 vs +4.08 PPL). Across all six models the middle tertile is *never* the
> most expensive, so **H1 is refuted**, not supported. The useful lesson is
> methodological: a static redundancy score ranks neurons well *within* a layer
> but does not predict how much a layer's removal costs. Table:
> [`artifacts/h1_depth_tertiles.md`](artifacts/h1_depth_tertiles.md).

## 6. Tests

`tests/test_metrics.py` (10 tests) runs fully offline (synthetic tensors + a toy
SwiGLU module — no model/dataset download):

- streaming aggregates (mean, RMS, fixed/RMS frequency) on known inputs;
- multi-α frequency sweep is monotone in α and the lazy fraction is non-decreasing;
- importance = `RMS × col_norm`, bottom-share math, and fallback when weights absent;
- `input_column_norms` on a known Linear weight;
- a dead neuron is detected; a planted near-duplicate pair is recovered;
- correlation neighborhood / dense matrix / duplication score behave;
- the collector discovers FFN `down_proj` layers, captures token counts, and
  returns usable weight column norms;
- all plotting helpers and the calibration block builder produce output.

```powershell
uv run pytest tests/ -q
```

## 7. Mapping to the proposal

| Proposal item | Where |
|---------------|-------|
| Q6 activation-frequency probe + per-layer histogram | `freq_hist_*`, `lazy_sweep_*`, `NeuronActivationStats` |
| Q6 layer-first depth view | `imp_depth_*`, `freq_depth_*`, `depth_tertiles` in JSON |
| Q6 duplicate / similarity probe | `top_k_correlated_pairs`, `corr_layer*` |
| Q4 importance (primary) + RMS-relative ε sweep + fixed ε (sanity) | `importance`, `rms_alphas` / `fixed_eps` |
| H1 (depth) | importance-share-by-depth + tertile aggregates |
| LLM-Pruner importance ranking | `RMS × ‖W_down[:, i]‖` in `LayerActivationResult.set_weight_col_norm` |

## 8. Limitations carried into this stage

- Importance uses `RMS(h_i) · ‖W_down[:, i]‖`, a separable proxy for the true
  (non-separable) L2 contribution of a neuron across tokens; good for ranking,
  not an exact ablation ΔPPL (that comes in Weeks 9–10).
- RMS-relative frequency and correlation are estimated on a reservoir subsample;
  importance, fixed-ε frequency and magnitudes are exact. Increase
  `max_reservoir_tokens` if memory allows for tighter estimates.
- For 4-bit models the down-projection weight is dequantized to read column
  norms; if dequantization is unavailable importance falls back to RMS
  (`col_norm_available` flags this).
- Masking ≠ removal still applies later; this stage only *measures*.
- Single calibration corpus (WikiText-2 train); the WikiText-vs-PIQA comparison
  (H3) reuses this pipeline with a PIQA-prompt calibration set. **Done in Weeks
  11–12** (`configs/measurement/neuron_activations_piqa_qwen3_0.6b.yaml`): the two
  corpora agree on only a third of the 10% masked set, and each ranking wins on
  its own metric — see `recovery_week11-12.md` §4.

## References

- Li et al. 2022 — *The Lazy Neuron Phenomenon*
- Geva et al. 2021 — *FFN Layers Are Key-Value Memories*
- Ma et al. 2023 — *LLM-Pruner*
- `proposal_week3-4.md` (Q4/Q6/Q7), `metrics_week3-4.md`
