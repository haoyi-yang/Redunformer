# Metrics

This project uses two distinct kinds of "metric", and keeping them separate is essential.

- A **pruning metric** decides *which* tiles to remove.
- An **evaluation metric** decides *whether* the resulting pruned model is still good.

SparseGPT is a *pruning* metric, not an evaluation metric: its reconstruction error is a
local, per-layer signal used while pruning and does not by itself tell you whether the whole
model still works.

---

## Pruning metrics (which tiles to remove)

All operate on a weight matrix `W` (shape `[out, in]`) split into `tile_size × tile_size` tiles.
Lower score = better pruning candidate; tiles are pruned by sorting ascending and removing the
lowest fraction. Code: `src/redundancy/scoring.py`, `src/redundancy/recovery.py`, and the
`prune_*` functions in `scripts/run_pruning.py`.

| Method (`--method`) | What it scores | Needs calibration | Why we added it |
|---|---|---|---|
| `random` | nothing — random tile selection (per `--seeds`) | no | the **floor** control; every other method must beat it |
| `magnitude` | tile Frobenius norm (weight size only) | no | naive **weight-only** reference |
| `wanda` | activation-aware importance | yes | data-aware screening (Wanda) |
| `sparsegpt` | masked-tile output error | yes | how much removing a tile changes the layer output |
| `sparsegpt_recon` | reconstructed-tile error + weight compensation | yes | tests whether other weights can *absorb* a tile |

**Wanda** (report eq. 12) — mean over the tile of `|W_ij| · ‖X_:,j‖₂`, where `‖X_:,j‖₂` is the
per-input-column activation norm from calibration data (`= sqrt(diag(H))`, `H = XᵀX`).

**SparseGPT masked error** (report eq. 22) — relative change in the layer output when a tile is
zeroed: `‖XWᵀ − XW₋ᵀᵀ‖²_F / ‖XWᵀ‖²_F`. With `H = XᵀX` this reduces to a ratio of quadratic forms
over the tile's columns. No weight updates.

**SparseGPT reconstructed error** (report eq. 23) — after removing a tile, the surviving weights
of the affected output rows are re-optimised to reconstruct the original output. Using the damped
inverse Gram matrix, the minimum error per tile is `trace(W_T · S · W_Tᵀ)` with `S = (Hinv[P,P])⁻¹`,
and the compensating weight update is applied so the pruned model realises the benefit. Always
`≤` the eq. 22 masked error. **Note:** its advantage is largest at low/mid sparsity and shrinks at
high sparsity (fewer surviving weights to reconstruct from).

*Deferred:* the **combined Wanda→SparseGPT** method (Wanda selects tiles, SparseGPT reconstructs
the survivors) and the **magnitude/random baselines** at matched settings.

---

## Evaluation metrics (is the pruned model still good?)

Perplexity alone can hide task-level degradation, so we evaluate along three complementary axes.
Code: `src/redundancy/eval.py`.

### 1. Perplexity
Standard sliding-window perplexity on WikiText-2 test. `--eval-frac` evaluates a fixed subset of
the corpus (measured ~5.1× faster at 20%) for fast **screening**, where only the *ranking* of
layer×matrix combinations matters; full evaluation is used for headline/final results. The same
subset is applied to every comparable run so rankings stay valid.

### 2. Output-distribution divergence (dense vs. pruned)
Recorded on a small fixed probe, on every experiment:

- **KL divergence** of the output distributions:
  `D_KL(P_dense ‖ P_pruned) = Σ_i P_dense(i) · log( P_dense(i) / P_pruned(i) )`
- **Top-1 agreement** — fraction of positions where both models predict the same next token.
- **Hidden-state cosine similarity** — `⟨h_dense, h_pruned⟩ / (‖h_dense‖ · ‖h_pruned‖)`.

Why: these are cheap and highly sensitive, measuring *behavioural drift* token by token —
ideal for the screening stage where full downstream evaluation is too slow.

### 3. Downstream task accuracy
For final whole-model variants, accuracy on HellaSwag, PIQA and ARC-Easy via the
lm-evaluation-harness (already configured and baselined for the dense model). This tests whether
the pruned model retains actual task ability, which perplexity can fail to reveal.

---

## What is saved

Every experiment writes JSON (per layer, incrementally) containing: perplexity, the divergence
block (KL / top-1 / cosine), tile counts, the exact pruned-tile mask (coordinates), method,
sparsity, layer, matrix, and the calibration/eval configuration. Results are therefore complete
and reproducible from the stored record; visualization runs offline from the JSON
(`scripts/plot_screening.py`).
