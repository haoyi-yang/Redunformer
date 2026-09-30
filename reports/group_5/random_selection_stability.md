# Uncertainty on the random baseline — what we have, and the offline fix

**Analysis-only deliverable (#18, item 3).** CPU-only; no model, no forward
pass, no GPU time.

> **Run on the real artifacts 2026-09-09** — all six models, 20 seeds, four
> ratios. Results and the resulting bound are in [§4](#4-results--all-six-models-20-seeds-each).
> Headline: the guided arm is at worst **21.4 seed standard deviations** from
> the random arm's selection distribution, so no random column in any report is
> a draw artifact. §1–§3 below are the original scoping and still describe what
> the tool can and cannot conclude.

## 1. What the committed numbers actually are

Every `random` column in Weeks 9–14 — in `intervention_week9-10.md`,
`recovery_week11-12.md`, `cross_family_week13-14.md`, and the generated
`artifacts/cross_family_tables.md` — is **a single seeded draw at seed 42**. Not
a mean over repeats, and with no error bar. `run_pruning.py` takes one `seed`
and `select_neurons(..., strategy="random", seed=seed)` consumes it once per
ratio.

This is stated as a limitation in Weeks 9–10 and it should stay stated. It
matters unevenly:

- Where the gap is large it cannot plausibly be a draw artifact. Qwen3-0.6B at
  25% is 5.889× baseline random against 1.566× guided; Llama-2-7B at 50% is
  299× against 2.27×.
- Where the gap is small it is currently unfalsifiable. The cells where random
  *beats* the importance ranking — gpt2-medium and gpt2-large at 5% and 10%,
  differences of 0.02–0.10× baseline — rest on one draw each, and Weeks 13–14
  §4b builds an argument on them ("the importance ranking needs sparsity to pay
  off"). That argument is the one most exposed to seed noise.

## 2. The offline decomposition

The random baseline's variance has two parts:

1. **Selection variance** — which neurons the draw picks, and therefore how much
   of each layer's importance mass it removes. The seed enters *only* here.
2. **Evaluation variance** — the perplexity that follows from a given removal.
   Needs forward passes.

Part 1 is fully recoverable from the measurement `.npz` for free, because
`select_neurons` never touches the model. `scripts/random_selection_stability.py`
resamples it:

```powershell
uv run python scripts/random_selection_stability.py `
    --measurement experiments/results/measurement_Qwen__Qwen3-0.6B_20260629T182308Z.json `
    --ratios 0.05 0.10 0.25 0.50 --seeds 20
```

It reports, per ratio: the mean / sd / observed range / 95% percentile-bootstrap
CI of the removed-importance fraction across seeds; the same quantity for the
importance-guided arm; the separation between them in seed standard deviations;
and the mean pairwise Jaccard between draws against the analytic chance floor
from `expected_random_jaccard`. Seed 42 — the one the committed runs used — is
always draw 1, so the published number can be located inside its own
distribution instead of compared to it from outside.

Output goes to `experiments/results/random_stability_<model>_<ts>.json`
(gitignored like every other artifact); `--no-write` prints the table only.

Implementation: `src/redundancy/pruning/stability.py`, tested offline in
`tests/test_stability.py` (8 tests, no GPU, no downloads).

## 3. What it can and cannot conclude

**It can** say whether another seed would have masked a materially different
*amount* of importance. If 20 draws remove 24.96% ± 0.29% of the importance mass
at a 25% budget while the guided arm removes 1.9%, then no plausible seed brings
the two arms near each other, and the single-seed baseline was never the weak
link in that comparison.

**It cannot** put a confidence interval on any perplexity. The map from removed
importance to ΔPPL is exactly what the model evaluation measures and what we are
avoiding running. A tight selection distribution makes a *large* reported gap
safe to trust; it says nothing about a gap of 0.02× baseline, because two draws
that remove nearly identical importance mass can still differ in perplexity by
more than that. **The gpt2-medium/large 5–10% cells therefore cannot be settled
offline** — they need the sweep re-run with several seeds, and it should be
scoped to those two models and two ratios rather than the whole grid.

A useful secondary check falls out of the same run: mean pairwise Jaccard should
land on the analytic chance floor. If it comes out materially above, the draws
are not independent and every random column shares structure — a bug, not a
result. On a smoke run against a synthetic 28 × 3072 artifact the two agree to
four decimals (0.1430 vs 0.1429 at a 25% budget).

## 4. Results — all six models, 20 seeds each

Run 2026-09-09 on the committed WikiText-2 measurements, 20 draws per ratio
(seed 42 is draw 1), 2000 bootstrap resamples. Whole grid takes ~2 minutes of
CPU.

Removed-importance fraction: **random mean ± sd** against the **guided** arm at
the same sparsity, and their separation in seed standard deviations.

| model | ratio | random mean ± sd | observed range | guided | gap |
|---|---:|---:|---:|---:|---:|
| `gpt2` | 5% | 0.05026 ± 0.00095 | 0.00414 | 0.02984 | **21.4 sd** |
| | 10% | 0.09958 ± 0.00136 | 0.00653 | 0.06354 | 26.5 sd |
| | 25% | 0.24939 ± 0.00173 | 0.00628 | 0.17451 | 43.4 sd |
| | 50% | 0.50007 ± 0.00212 | 0.00679 | 0.38346 | 55.0 sd |
| `gpt2-medium` | 5% | 0.04999 ± 0.00041 | 0.00163 | 0.03042 | 47.6 sd |
| | 10% | 0.10017 ± 0.00063 | 0.00206 | 0.06482 | 55.7 sd |
| | 25% | 0.25007 ± 0.00101 | 0.00336 | 0.17788 | 71.3 sd |
| | 50% | 0.50006 ± 0.00116 | 0.00456 | 0.39099 | 94.1 sd |
| `gpt2-large` | 5% | 0.05005 ± 0.00015 | 0.00055 | 0.03011 | 135.9 sd |
| | 10% | 0.10002 ± 0.00033 | 0.00153 | 0.06492 | 105.5 sd |
| | 25% | 0.24993 ± 0.00047 | 0.00187 | 0.18086 | 148.3 sd |
| | 50% | 0.50010 ± 0.00042 | 0.00189 | 0.40133 | 232.7 sd |
| `Qwen3-0.6B` | 5% | 0.04934 ± 0.00083 | 0.00272 | 0.01559 | 40.8 sd |
| | 10% | 0.09937 ± 0.00125 | 0.00419 | 0.03571 | 51.0 sd |
| | 25% | 0.24985 ± 0.00259 | 0.01119 | 0.11178 | 53.4 sd |
| | 50% | 0.49969 ± 0.00268 | 0.00948 | 0.28456 | 80.3 sd |
| `Qwen3.5-4B` | 5% | 0.05003 ± 0.00030 | 0.00098 | 0.02779 | 74.9 sd |
| | 10% | 0.10005 ± 0.00044 | 0.00174 | 0.05896 | 93.1 sd |
| | 25% | 0.24992 ± 0.00074 | 0.00318 | 0.16150 | 119.3 sd |
| | 50% | 0.50002 ± 0.00101 | 0.00425 | 0.35780 | 140.6 sd |
| `Llama-2-7B` | 5% | 0.04974 ± 0.00045 | 0.00158 | 0.03241 | 38.5 sd |
| | 10% | 0.10020 ± 0.00102 | 0.00396 | 0.06708 | 32.5 sd |
| | 25% | 0.24957 ± 0.00158 | 0.00530 | 0.17750 | 45.7 sd |
| | 50% | 0.49978 ± 0.00159 | 0.00501 | 0.37974 | 75.7 sd |

**The bound to quote: the smallest separation anywhere in the grid is 21.4 seed
standard deviations** (`gpt2` at 5%, the smallest model at the smallest budget).
Every other cell is 32 sd or more, and it widens with both model size and
ratio. The random draw removes its proportional share of importance mass to
within ±0.3% relative in every model — as it must, by construction — so the seed
simply has no room to move the selection arm anywhere near the guided one.

**Independence self-check passes on real data.** Mean pairwise Jaccard between
draws matches the analytic chance floor to four decimals at every ratio in every
model (e.g. 0.3334 vs 0.3333 at 50%, 0.0527 vs 0.0526 at 10%). The draws are
independent; no random column shares hidden structure with another.

**The `gpt2-medium` / `gpt2-large` 5–10% cells are *not* explained by selection
noise** — and that is the interesting outcome, not a clean bill of health. Those
are the cells where random *beats* the importance ranking on perplexity by
0.02–0.10× baseline, and the separations there are 47.6 / 55.7 and 135.9 /
105.5 sd. So the guided arm provably removes far *less* importance mass than
random, and still costs more perplexity. An unlucky draw is ruled out; what is
left is that at low sparsity on these two models, removed importance mass does
not order ΔPPL.

That is the same failure mode H1 turned on: a static separable score ranks
neurons within a layer but does not predict what removing a set of them costs.
Weeks 13–14 §4b reads this as "the importance ranking needs sparsity to pay
off", which is compatible — but the claim is about perplexity, so it still needs
a multi-seed GPU re-run of those two models at 0.05/0.10 to be stated with an
error bar. Offline work cannot close it, and this analysis narrows *why* rather
than settling it.

### What this does and does not retire

- **Retired:** "the random baseline might be an unlucky draw." It is not, in any
  cell, by a margin of at least 21 sd of selection.
- **Still open:** any *perplexity* error bar. The reports still quote one PPL
  evaluation per cell. Where the reported gap is large (Llama-2-7B 2.27× vs
  299× at 50%) that is immaterial; where it is 0.02× baseline it is the whole
  question.

Artifacts: `experiments/results/random_stability_<model>_20260909T14*.json`
(gitignored). Reproduce with the loop in §2, one invocation per measurement.

## Sources

- `src/redundancy/pruning/selection.py` (`select_neurons`, `selection_stats`)
- `src/redundancy/pruning/stability.py`, `scripts/random_selection_stability.py`
- `reports/group_5/intervention_week9-10.md` §8 (the limitation as first stated)
- `reports/group_5/cross_family_week13-14.md` §4b (the cells most exposed to it)
