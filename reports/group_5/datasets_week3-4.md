# Datasets — Group 5 (Neuron / FFN-unit)

**Weeks 3–4 deliverable for issue #2 — Proposal question 4 (datasets half).**
**Status:** reviewed (Sooraj Rathore, 2026-06-21) — datasets section approved; ARC-Easy kept as **fallback only**.

---

## Summary

We commit to **WikiText-2** as the primary LM calibration / evaluation dataset and **PIQA** as the primary downstream task. **ARC-Easy** is a **fallback** only (not in the primary Week 5 plan) — used if PIQA gives a degenerate signal for task-conditioned analysis (H3). Anything OCR / multimodal (e.g. OlmOCR from the suggestion list) is **out of scope** for Group 5.

Everything below already runs (or will run) on our existing pipeline (`scripts/run_baseline.py` + `configs/datasets/wikitext2.yaml`) on an 8 GB RTX 4060.

---

## Primary and optional datasets

| Purpose | Dataset | Role | Split used | Approx. size | Status |
|---------|---------|------|------------|--------------|--------|
| **LM calibration + baseline** | WikiText-2 (`wikitext`, config `wikitext-2-raw-v1`) | Primary | `test` (2,891 docs / ~297 k tokens) | small (~12 MB) | **Running** — `configs/datasets/wikitext2.yaml`, baseline PPL 17.06 |
| **Activation-statistics calibration** | WikiText-2 train subset | Primary (Weeks 5–8) | `train`, 256–512 sequences | bounded by `max_samples` | Planned — same loader |
| **Downstream commonsense** | PIQA (via `lm-evaluation-harness`) | Primary | `validation` | 1,838 items | Planned (Week 5+) |
| **Downstream science QA** | ARC-Easy (via lm-eval) | Fallback only | `test` | ~2,376 items | Not in primary plan — run only if PIQA cross-check is needed |
| **OCR / multimodal (OlmOCR etc.)** | — | **Out of scope** | — | — | Tracked in #12, not pursued by Group 5 |
| **Larger LM corpus (C4 subset)** | — | Out of scope for now | — | — | Deferred (would require streaming + more disk) |

**Why these choices:**

- WikiText-2 is the seminar's default and what every other group uses for LM PPL — keeps our numbers comparable. It is small enough to run end-to-end on 8 GB VRAM in 4-bit.
- PIQA is short, multiple-choice, and supported by `lm-evaluation-harness` out of the box, so it adds **task-level** signal (proposal Q5: task dimension) with almost no extra engineering. It is also one of the two examples explicitly named in the seminar.
- ARC-Easy is a **fallback** only — not part of the primary evaluation plan; we run it only if PIQA results need a second downstream task for H3.
- C4 / OlmOCR are dropped because they add data engineering or modality work that does not improve any of our four hypotheses.

---

## Hardware fit / pipeline confirmation

| Dataset | Loader | Config | Hardware-checked? |
|---------|--------|--------|-------------------|
| WikiText-2 | `src/redundancy/data.py` (HF `datasets`) | `configs/datasets/wikitext2.yaml` | **Yes** (4B baseline = 17.06, 9B baseline = 12.90 per #11) |
| PIQA | `lm-evaluation-harness` (`HFLM` wrapper in `src/redundancy/eval.py::run_lm_eval`) | new: `configs/eval/lmeval_piqa.yaml` (to be added in #3 / #7 follow-up) | Pending first run |
| ARC-Easy | `lm-evaluation-harness` | optional config | Pending |

`run_lm_eval` already exists, so PIQA only needs a small config wrapper and a single command invocation.

---

## Splits we will actually report on

- **WikiText-2 `test`** for all PPL numbers (matches existing 4B / 9B baseline reports — apples to apples).
- **WikiText-2 `train` (256–512-sequence subset)** for neuron activation calibration. This is *only* used to produce statistics; we do not report metrics on it.
- **PIQA `validation`** for downstream accuracy (standard practice in lm-eval; PIQA test labels are hidden).

---

## Deliverable checklist (issue #2)

- [x] Short list of primary and optional datasets with rationale
- [x] Split / size noted per dataset
- [x] Pipeline / hardware confirmation (WikiText-2 running; PIQA wires into existing `run_lm_eval`)
- [x] Fed into one-page proposal (#10) — see `reports/group_5/proposal_week3-4.md` → Q4
- [x] Reviewed by assignee / group (Sooraj Rathore, 2026-06-21)

## References

- Seminar §2.3 (suggested eval tasks)
- `configs/datasets/wikitext2.yaml`, `configs/eval/baseline_full.yaml`
- Sibling deliverables: `metrics_week3-4.md` (#3), `redundancy_dependencies_week3-4.md` (#6)
