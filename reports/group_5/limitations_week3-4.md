# Limitations — Group 5 (Neuron / FFN-unit)

**Weeks 3–4 deliverable for issue #9 — Proposal question 8.**
**Status:** draft — assignee did not provide an update in time, so this is filled by the rest of the group and is **open for review**.

---

## Key facts: what neuron redundancy can and cannot tell us

1. We measure redundancy of **one specific object**: the post-SwiGLU activation channel inside an FFN block. Claims do **not** generalize to attention heads, full transformer blocks, individual weight columns, or hardware tiles — other groups own those.
2. Activation-based redundancy is **relative to the calibration data**. A neuron that looks lazy on WikiText-2 may be busy on PIQA — that is the whole point of H3, but it also bounds how strong our conclusions can be.
3. **Masking ≠ removing.** All our Weeks 5–10 numbers come from zero-masking; we never resize tensors. Any speedup claim would require structural surgery and we are not making one.
4. We work in **4-bit** on an 8 GB GPU. Activation magnitudes are quantization-noisy, calibration sets are small (256–512 sequences), and we cannot run full multi-seed sweeps. Effect sizes have to be large to be trustworthy.
5. Qwen3.5 is a **hybrid stack** (Gated DeltaNet + full attention + FFN). "Layer ℓ" is not the same kind of layer across the depth — we have to log per-layer block type and only compare like with like for the depth dimension (H1).

---

## Risks (what could invalidate our results) and mitigations

### Definition & measurement

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Wrong "neuron" definition (gate column vs up column vs post-activation) | Different definitions give different redundancy rankings | Fix the unit to **post-SwiGLU activation channel** for all primary results; report gate-only as a *secondary* check (H4) |
| Calibration-set bias (WikiText-only) | Lazy-on-WikiText ≠ lazy-overall | Use a second calibration corpus (PIQA prompts) for cross-task overlap (#6 H3); never report task-conditional claims without a same-budget PIQA pass |
| Activation threshold ε for "fires / doesn't fire" arbitrary | Frequency histogram is sensitive to ε | Use per-layer ε proportional to that layer's activation RMS; report results at two thresholds |
| Small calibration size (256–512 sequences) | Rare neurons may look lazy by chance | Bootstrap confidence intervals on activation frequency; require *f* < threshold on **both halves** of the calibration set before calling a neuron "lazy" |

### Architecture & hardware

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Qwen3.5 hybrid layers (DeltaNet + attention + FFN) | Naïve "layer index" comparisons mix non-comparable blocks | `iter_mlp_modules` (`src/redundancy/hooks.py`) already filters to MLP/FFN/feed_forward names; **log layer-type metadata per index** so depth-tertile aggregates only contain real FFNs |
| 4-bit quantization noise on activations | Tiny activations get rounded to zero, inflating "lazy" counts | (a) collect activations in fp16/bf16 *after* dequant in the hook, (b) cross-check a subset of layers with fp16 inference on Qwen3-0.6B (smaller, fits) |
| 8 GB VRAM cap | Cannot store full activation tensors for all layers × all tokens | Stream statistics (running mean/var/freq counters) instead of buffering tensors; only buffer for the few layers we visualize |
| Masking ≠ real speedup | Parameter count drops on paper, runtime does not | Explicitly state this in the proposal; flop / runtime numbers are **not** a deliverable |

### Recovery, fairness, scope

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Short fine-tuning gives "free" recovery | Could mask the fact that our pruning destroyed something real | Always report both *pre-recovery* and *post-recovery* numbers; never report only post-recovery |
| Random baseline mismatch | An unfair random baseline (e.g. global vs per-layer sparsity) makes our method look better than it is | Match sparsity **per layer**, not globally, and match exact neuron count, not approximate fraction |
| Single model family (Qwen) | Cannot claim generality | Report Qwen3.5-4B vs Qwen3.5-9B vs Qwen3-0.6B; explicitly say "within Qwen lineage" in every cross-size claim |
| Overlap with sibling groups (heads / blocks / weights / tiles) | Risk of duplicating their claims or making contradictory ones | Stay **inside FFN blocks**; treat attention, full blocks, unstructured weights, and tile-level patterns as out of scope |

---

## Things we will **not** claim

- Wall-clock or FLOP speedups from neuron removal.
- Cross-family generalization (LLaMA, Mistral, etc.) — out of hardware reach.
- That our 4-bit activation statistics are identical to full-precision ones.
- Anything about attention heads, full blocks, individual weights, or tile patterns.
- That short fine-tuning recovery proves the pruned neurons were redundant; it only proves they were *recoverable* on the same calibration data.

---

## Deliverable checklist (issue #9)

- [x] Key facts: what neuron redundancy can / cannot tell us (5 bullets above)
- [x] Risks list (definition, hardware, recovery, scope)
- [x] Mitigations for each risk
- [x] Section merged into one-page proposal (#10) — see `proposal_week3-4.md` → Q8
- [ ] Reviewed by assignee / group

## References

- Seminar §6.5; general limitations in §1 and §7
- `reports/group_5/baseline_week1-2.md` (hardware + 4-bit baseline)
- `reports/group_5/redundancy_dependencies_week3-4.md` (#6 — defines what is in / out of scope)
- `src/redundancy/hooks.py::iter_mlp_modules` (FFN filter)
