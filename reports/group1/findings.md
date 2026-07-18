# Weeks 5–8 Findings: Block Influence Measurement (GPT-2 small × WikiText-2)

Group 1 — Block / Layer-level Redundancy. Measurement and visualization only (no pruning yet;
that is Weeks 9–10). Figures live in `figures/`; regenerate with
`uv run python scripts/run_measurement.py --config configs/measurement.json`.

## Indexing convention (read this before the figures)

Two different indexings appear across the three figures, and they must not be lined up 1:1:

- **Hidden state `k`** = `hidden_states[k]` from GPT-2's forward pass. `k = 0` is the **embedding
  output** (before any transformer block); `k = 12` is the **final block's output**. There are
  `L + 1 = 13` hidden states.
- **Block `i`** is the **transition** from hidden state `i` to hidden state `i+1`. There are
  `L = 12` blocks (0–11). BI block 0 is the 0→1 transition; BI block 11 is the 11→12 transition.

So: the **BI bar chart** and the **residual chart** are indexed by **block (0–11)**; the
**cosine heatmap** is indexed by **hidden state (0–12)**. Heatmap "hidden state 11" is *not* the
same object as BI "block 11".

## Figure 1 — Block Influence per block (`bi_bar.png`)

Finding: **the first and last blocks carry the most influence** (BI: block 0 ≈ 0.82,
block 11 ≈ 0.68), while the middle blocks are near-flat (≈ 0.03–0.08, minimum at block 4 ≈ 0.03).
This reproduces ShortGPT's "first/last critical, middle redundant" pattern — and confirms it
holds even at GPT-2-small's 124M scale, which was an open question in our proposal.

Caveat: block 0's BI is measured against the **embeddings** (hidden state 0), so part of its
height is the embeddings→contextualized jump. Do **not** read this as "block 0 works harder than
block 11" — both are simply the two most influential blocks.

## Figure 2 — Hidden-state cosine similarity, centered (`cosine_heatmap.png`)

Finding: two bright clusters — roughly hidden states **1–7** and **8–11** — with a shift around
**7→8**. Early and late representations occupy distinct subspaces. Hidden state 12 sits apart from
the rest, consistent with block 11's high BI (the final block moves the representation somewhere
new).

Caveat: we center (subtract the across-layer mean) before taking the cosine, because raw cosine on
the GPT-2 residual stream saturates near 1 everywhere (a shared common direction washes the map
out). Centering exposes the cluster structure, but it **forces some pairs negative by
construction** — so the dark corners are **partly a centering artifact**. Describe the result as
*two clusters with a mid-network shift*, **not** as early/late blocks "doing opposite work" or
"anti-correlating." Proper follow-up: linear CKA (deferred), which centers activations as part of
its definition.

## Figure 3 — Relative residual update per block (`residual_norms.png`)

Metric: `mean_t ( ||X_{i+1,t} − X_i,t|| / ||X_i,t|| )` — the size of each block's residual update
**relative** to the incoming stream (the raw, un-normalized norm just grows with depth and isn't
comparable block-to-block). Plotted on a **log y-axis** so block 0 doesn't blow out the scale.

Finding: block 0 ≈ 10.9, block 11 ≈ 1.3, middle blocks ≈ 0.3, with a gentle rise toward block 11.
The block-11 elevation corroborates the BI ranking.

Caveat: block 0's relative update is **inflated** because its denominator is the **embedding**
norm (smaller than deeper residual-stream norms) — the same embedding-regime caveat as its BI. Do
not read "10.9" as "block 0 does ~30× the work."

## Reconciling BI and the residual update (pre-registered before Weeks 9–10 pruning)

The two metrics **disagree about the middle blocks**, and the disagreement is informative:

- BI says middle blocks ≈ 0.03 ⇒ cosine ≈ 0.97 ⇒ almost no **directional** change.
- Relative residual says those same blocks update the stream by ≈ 0.3 ⇒ ~30% of its norm ⇒
  **not nothing**.

Both are correct. Middle blocks make sizable-**magnitude** updates that are largely **aligned**
with the existing residual stream: they barely rotate its direction (low BI) while still adding
real magnitude. So **"redundant under BI" means "does not rotate the representation," not "does
nothing."**

Pre-registered consequence: when we prune low-BI middle blocks in Weeks 9–10, perplexity may
degrade **more** than BI ≈ 0.03 suggests, because we also remove that ~30% magnitude contribution.
Framing for the report: **BI is a reliable *ranking* signal, but its tiny absolute values overstate
removability.** Our own residual plot is the internal evidence for this, and it protects the
report if the pruning curve looks worse than the BI chart implies.
