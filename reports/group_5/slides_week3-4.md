---
marp: true
title: Group 5 — Neuron / FFN Redundancy
paginate: true
theme: default
style: |
  section { font-size: 28px; }
  h1 { font-size: 40px; }
  h2 { font-size: 32px; }
  table { font-size: 22px; }
---

# Group 5 — Neuron / FFN-unit Redundancy

**Weeks 3–4 proposal + results through Weeks 11–12**

Qwen3.5 family · WikiText-2 + PIQA · masking (not surgery)

---

# Q1 — What is a redundant neuron?

A **neuron** = one **post-SwiGLU** activation channel in an FFN block.

Redundant if, on calibration data, it is:

1. **Lazy** — almost never fires
2. **Duplicated** — |ρ| ≥ 0.9 with another unit in the same layer
3. **Removable** — small ΔPPL / PIQA drop when masked
4. **Recoverable** — short fine-tuning undoes most of the damage

We **mask** (zero activations). Shapes unchanged → **no FLOP / speed claim**.

---

# Hypotheses H1–H5

| | Claim | Status |
|---|---|---|
| **H1** depth | Middle layers less redundant | ~~Supported~~ → **Refuted** |
| **H2** size | Larger models → more redundancy / bigger guided–random gap | ~~Partial~~ → **Refuted as stated** |
| **H3** task | WikiText-redundant ≠ PIQA-redundant | Supported on **gated FFNs**, not GELU |
| **H4** component | Post-activation importance > frequency | Supported (frequency split out as **H7**) |
| **H5** duplication | One twin cheaper than matched-importance control | **Not supported** — see **H5′** instead |

> **These verdicts are from 2026-07-30 and four of five have since changed.**
> This deck is the Weeks 3–4 proposal talk; it is kept as a historical record.
> Current verdicts, including the added H5′ / H6 / H7 / H8, are frozen in
> `reports/group_5/hypothesis_scoreboard.md`.

---

# Weeks 5–8 — Measurement

- Frequency saturates on SwiGLU → **importance** `RMS(h)·‖W_down[:,i]‖` is the ranking
- Near-duplicate pairs found (esp. 0.6B massive-activation layer)
- Artifacts: 0.6B + 4B measurement JSONs

**Implication:** prune by importance, not lazy frequency.

---

# Weeks 9–10 — Intervention

**Arm A — removal-ratio sweep** (importance / frequency / random)

- Importance beats random on PPL at every ratio (both models)
- Q4 @ 10% **not** met without recovery (4B +1.82 PPL)

**Arm B — H5 pair ablation**

- Duplicates not cheaper than matched-importance controls
- But jointly critical: one twin +1.8 PPL, both +2898 (0.6B L2)

---

# Weeks 11–12 — Recovery & replace

**LoRA recovery (0.6B):** 76–83% of masking damage undone vs unmasked control  
*(raw % is misleading — adapters alone buy ~10 PPL on WikiText)*

**4B QLoRA (Colab):** same 10% mask (+1.83 PPL) but adapters **overfit**; adj. recovery **~2%**

**Neuron replacement:** fit `h_drop ≈ α·h_keep + β` → **87–93%** of mask cost recovered, **no training**

**H3:** WikiText vs PIQA rankings disagree on ~⅔ of the 10% set; each wins on its own metric

**Still open:** 9B pipeline (#13–#15)

---

# Models & baselines

| Model | WikiText-2 PPL | Role |
|---|---:|---|
| Qwen3-0.6B | 39.78 | Small control |
| Qwen3.5-4B | **17.06** | Primary |
| Qwen3.5-9B | **12.90** | Size (Henrik #11) |

Configs: `qwen3.5-4b.yaml`, `qwen3.5-9b.yaml`, …

---

# Limitations (honest)

- Masking ≠ removal → no speed/memory claim
- Single seed; eval subsets; PIQA limits 200–500
- 4B QLoRA overfits this schedule (adj. recovery ~2%); laptop 8 GB cannot train it
- 9B measurement / prune / recover still TODO
- Merge is a hook, not a structural fold

---

# What to run / where to read

| Stage | Report | Entry point |
|---|---|---|
| Baseline | `baseline_week1-2.md` | `scripts/run_baseline.py` |
| Measure | `measurement_week5-8.md` | `scripts/run_measurement.py` |
| Intervene | `intervention_week9-10.md` | `run_pruning.py` / `run_pair_ablation.py` |
| Recover | `recovery_week11-12.md` | `run_recovery.py` / `run_merge.py` |

**Proposal siblings:** datasets / metrics / dependencies / limitations under `reports/group_5/`.

---

# Thanks / questions

Group 5 · Redunformer Neuron

Open follow-ups: **#13–#16** (9B + 4B Colab)
