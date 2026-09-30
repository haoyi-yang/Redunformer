# H2 (size) — cross-size comparison table

**Analysis-only deliverable (#18).** Every number here is read off artifacts
that already exist; nothing new was run on a GPU. Its job is to put the size
evidence in one place for the final report, because it is currently spread over
three write-ups that reach different conclusions.

**H2 as stated in the proposal:** larger models carry *more* redundancy, in two
senses.

* **(a) Absolute tolerance** — at the same per-layer removal ratio, a larger
  model degrades less relative to its own baseline.
* **(b) Ranking headroom** — the gap between guided and matched-sparsity random
  masking *grows* with size, i.e. there is more spare capacity for a ranking to
  find.

Both clauses are scored below. They do not agree.

---

## 1. Qwen lineage — relative perplexity at 10% and 25%

`rel PPL` = masked perplexity ÷ that model's own unmasked baseline in the same
run, so the columns are comparable across models even though the absolute
baselines are not (see §6). `gap` is `random − guided` in units of the baseline;
`ratio` is `random ÷ guided`, which is the quantity clause (b) is about.

| model | params | baseline PPL | 10% guided | 10% random | 10% gap | 10% ratio | 25% guided | 25% random | 25% gap | 25% ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.6B | 33.75 | **1.130** | 1.708 | +0.578 | **1.51×** | **1.566** | 5.889 | +4.322 | **3.76×** |
| Qwen3.5-4B (4-bit) | 4B | 18.87 | **1.097** | 1.397 | +0.300 | **1.27×** | **1.324** | 2.519 | +1.196 | **1.90×** |
| Qwen3.5-9B (4-bit) | 5.7B¹ | 12.90² | — | — | — | — | — | — | — | — |

¹ `reported_parameter_count` 5.72 B from the Weeks 1–2 baseline JSON, i.e. the
4-bit checkpoint as loaded, not the nominal 9 B.
² Full WikiText-2 test split, Weeks 1–2. **Not** the in-run baseline a masking
sweep would produce, so it cannot be mixed with the columns to its right — it is
listed only so the row is not empty.

**The 9B row stays blank until [#13](https://github.com/geeeeenccc/Redunformer-Neuron/issues/13)
(measurement) and [#14](https://github.com/geeeeenccc/Redunformer-Neuron/issues/14)
(masking sweep) land.** Both are open. A masking sweep on 9B produces its own
in-run baseline, and that is the number that belongs in the `baseline PPL`
column when the row is filled.

Same runs, in absolute ΔPPL, for anyone who wants the raw magnitudes:

| model | 10% guided | 10% random | 25% guided | 25% random |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | +4.40 | +23.91 | +19.11 | +164.99 |
| Qwen3.5-4B | +1.83 | +7.49 | +6.11 | +28.67 |

### What this pair says

1. **Clause (a) is supported.** The 4B degrades less than the 0.6B at both
   ratios and under both rankings — 1.324× vs 1.566× guided at 25%, and 2.519×
   vs 5.889× random. The larger model absorbs the same proportional damage more
   gracefully.
2. **Clause (b) is contradicted.** The guided-vs-random ratio is *larger* for
   the **small** model at both ratios (3.76× vs 1.90× at 25%; 1.51× vs 1.27× at
   10%). Whatever extra tolerance the 4B has is spread across all its neurons
   rather than concentrated in a bigger pool of spare ones — random masking
   benefits from scale at least as much as the ranking does.

So on the Qwen lineage H2 is **half supported**, and the half that fails is the
one the intervention was designed to exploit.

---

## 2. The clean within-family test does not support H2 either

The Qwen pair varies size, but it also varies training corpus and generation
(Qwen3 → Qwen3.5). The GPT-2 size arm from Weeks 13–14 varies *only* size, with
architecture and tokenizer held fixed, and it is the honest test of clause (a):

| model | params | rel PPL @ 10% | @ 25% | @ 50% | gap @ 10% | gap @ 25% |
|---|---:|---:|---:|---:|---:|---:|
| gpt2 | 124M | 1.307 | 2.670 | **10.565** | +0.165 | +1.084 |
| gpt2-medium | 355M | 1.374 | 2.470 | 16.153 | −0.097 | +1.633 |
| gpt2-large | 774M | **1.109** | **1.461** | 14.874 | −0.024 | +0.091 |

Clause (a) holds at 25% (cost falls 45% from small to large), fails at 10%
(medium is worse than small), and fails at 50% (**both** larger models are worse
than the smallest). Clause (b) fails outright: the gap is *negative* for medium
and large at 10%, i.e. random masking is marginally cheaper there than the
importance ranking.

The measurement side agrees: the share of layer importance held by the bottom
10% of neurons is 6.45% / 6.59% / 6.58% across a 6× parameter range. Static
redundancy barely moves with size within a family.

For context, the full cross-family ordering at 25% looks like textbook H2 —
GPT-2 124M 2.670 → Qwen3-0.6B 1.566 → Qwen3.5-4B 1.324 → Llama-2-7B 1.267 — but
that comparison varies size, architecture, vendor, and training corpus at once.
Llama-2-7B is 1.8× the size of Qwen3.5-4B and only 0.06× baseline cheaper at
25%, while the 12× jump from 0.6B to 4B buys four times that. **FFN design
predicts pruning tolerance better than parameter count does in our data.**

---

## 3. Recovery does not scale with size either

The size dimension shows up a third time, in Weeks 11–12, and it points the same
way: what worked on the small model did not transfer.

| model | masking @ 10% | after LoRA, vs unmasked control | **adjusted recovery** |
|---|---:|---:|---:|
| Qwen3-0.6B (200 steps, rank 16) | +4.40 PPL | +1.07 PPL | **75.8%** |
| Qwen3.5-4B (100 steps, rank 16, Colab) | +1.83 PPL | +1.79 PPL | **2.3%** |

The 4B run overfit — eval perplexity rose ~7 points on *both* the masked and the
unmasked arm — so this is a statement about the budget we could actually run,
not a proof that 4B masking damage is permanent. It is still the only recovery
evidence we have at that size, and it is negative.

---

## 4. Verdict for the final report

| Clause | Qwen lineage (0.6B → 4B) | Clean size arm (GPT-2 124M → 774M) | Overall |
|---|---|---|---|
| (a) larger degrades less | supported | mixed — holds at 25%, fails at 10% and 50% | **partial** |
| (b) guided-vs-random gap grows with size | contradicted (3.76× → 1.90×) | contradicted (gap turns negative at 5–10%) | **not supported** |
| recovery transfers to larger models | not supported (76% → 2%), but at 16× less budget | **supported** (80/87% → 84/89% → 76/80%, non-monotonic, ≤9 pp spread) | **no usable size trend** |

**Recommended wording:** *"H2 is not supported. Larger models absorb moderate
pruning somewhat better, but the effect is confounded with architecture and
model quality, it does not survive a within-family size sweep at every ratio,
and the guided-vs-random headroom that H2 predicted would grow with size in fact
shrinks."*

> **Update 2026-09-09 — the recovery clause changed sign.** It was `untested` on
> the GPT-2 arm when this was written. The full ladder has since run at an
> identical 819k-token budget:
>
> | | 124M | 355M | 774M | Qwen3-0.6B |
> |---|---:|---:|---:|---:|
> | 10% masked | 80% | **84%** | 76% | 76% |
> | 25% masked | 87% | **89%** | 80% | 83% |
>
> There is no usable size trend. The ladder is non-monotonic — 355M is the best
> of the three — and the whole 6× parameter range spans at most 9 pp. 774M does
> sit below 124M, so "recovery is completely size-independent" would be too
> strong, but nothing here resembles degradation.
>
> **Two things reframe the 4B outlier.** First, a ≤9 pp size effect cannot reach
> 2%; the 4B cliff is not this trend extrapolated. Second, and more decisively,
> the 4B run was **not budget-matched** — 100 steps at batch 1 over 128 blocks,
> about **51k training tokens against 819k, 16× less**. Its adjusted recovery is
> internally valid, since both arms shared that budget, but it cannot be
> compared to the rows above.
>
> So "recovery stops working at scale" was never the parsimonious reading, and
> the alternative — that the 4B overfit a small pool, which it demonstrably did,
> losing ~7 PPL on the masked *and* unmasked arms while train loss fell — is
> better supported. The cliff remains unexplained; what is clear is that the
> experiment as run cannot separate size from budget. This is what
> [#23](https://github.com/geeeeenccc/Redunformer-Neuron/issues/23) is meant to
> decide, at a matched budget.

This supersedes the Weeks 9–10 reading ("H2, partial"). Verdicts are now frozen
in [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md), which records H2 as
**refuted as stated** — same evidence as the "not supported" wording above, and
the scoreboard's phrasing is the one to quote. Issue
[#20](https://github.com/geeeeenccc/Redunformer-Neuron/issues/20) is closed.

---

## 5. What a 9B row would still settle

Filling the blank row is worth doing, but it should be framed narrowly. It
extends the Qwen lineage to a third size point with architecture and vendor held
fixed — the only such series we have on a modern gated model — and would say
whether the shrinking guided-vs-random ratio (3.76× → 1.90× → ?) is monotone or
whether the 0.6B is simply an unusually redundant small model. It would **not**
rescue clause (b) unless the ratio goes back up.

Required for the row: `run_measurement.py` (#13) then `run_pruning.py` (#14) with
the same 128 × 512-token calibration budget and the same ratio sweep, so the
`rel PPL` columns stay comparable.

---

## 6. Why every column is relative

Absolute baselines in this table are **not** comparable across rows:

- The masking sweeps evaluate on per-run subsets of WikiText-2 test — 157
  documents for the 0.6B, 79 for the 4B — not the full split. The 4B's Weeks 1–2
  full-split baseline is 17.06, versus 18.87 in the sweep.
- The 9B baseline (12.90) is a full-split number from a different stage, on a
  different machine, run by a different person.
- GPT-2's perplexity window is 1024 tokens (its context limit) against 2048 for
  the Qwen and Llama runs.
- Llama-2-7B and the two Qwen models are evaluated in 4-bit; the GPT-2 models are
  fp32.

Every comparison above is therefore within-run and normalised to each model's own
baseline. The gaps are unaffected by any of this; the absolute perplexities are
not comparable and should not be quoted side by side.

Two further caveats carry into the final report:

- **Masking ≠ removal.** Parameter shapes are unchanged throughout, so no speed,
  memory, or FLOP claim follows from any row here.
- **One random draw per ratio.** Every `random` column is a single seeded sample,
  not a mean over repeats with error bars. The gaps at 25% are far too large for
  that to flip a conclusion, but the 5–10% GPT-2 cells where random *wins* are
  small enough to be within draw-to-draw noise. See
  [`random_selection_stability.md`](random_selection_stability.md) for the
  offline analysis that bounds this without any GPU time.

---

## Sources

- `reports/group_5/intervention_week9-10.md` §2 (0.6B and 4B sweeps)
- `reports/group_5/recovery_week11-12.md` §5 (4B QLoRA recovery)
- `reports/group_5/cross_family_week13-14.md` §4 (GPT-2 size arm)
- `reports/group_5/artifacts/cross_family_tables.md` (generated relative table)
- `reports/group_5/artifacts/baseline_Qwen__Qwen3.5-9B_20260620T075843Z.json` (9B baseline)
- Underlying artifacts, all gitignored:
  `experiments/results/pruning_Qwen__Qwen3-0.6B_20260726T093709Z.json`,
  `experiments/results/pruning_Qwen__Qwen3.5-4B_20260729T230350Z.json`,
  `experiments/results/pruning_gpt2*_20260814T*.json`
