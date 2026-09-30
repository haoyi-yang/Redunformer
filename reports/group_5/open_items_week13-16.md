# Open items going into Weeks 13–16 (final analysis)

**Analysis-only deliverable (#18, item 4).** Collected by re-reading every
Weeks 1–14 report and checking each open question against the repo's current
state, so a few things listed as pending elsewhere are already closed here.

Three tiers: experiments that would change a conclusion, bookkeeping that has to
happen before the write-up is frozen, and framing caveats the final report must
state rather than fix.

> **Status 2026-09-09.** Most of tier B and two of tier A are now closed;
> per-row updates are inline below. Net movement:
>
> - **Done:** A4 (H3 on the 4B), B1 (scoreboard frozen), B3 (stability run on
>   real artifacts), B4 (task board refreshed). Plus a `gpt2-medium`/`gpt2-large`
>   recovery arm that was not on this list.
> - **Still open, and now the top priority:** A3 — reframed. The 4B recovery run
>   used **16× fewer training tokens** than the runs it is compared against, so
>   "recovery does not transfer to 4B" confounds size with budget. Tracked as
>   [#23](https://github.com/geeeeenccc/Redunformer-Neuron/issues/23).
> - **Downgraded:** A2. The offline stability run (B3) shows the guided arm is
>   ≥21 seed sd from the random selection distribution *everywhere*, including
>   the two cells A2 was about, so a seed cannot flip which neurons get picked.
>   What A2 would still buy is a perplexity error bar, which is a smaller claim
>   than "could overturn a result".
> - **Not doable:** B2 — the 4B PIQA accuracies were never recorded.

---

## A. Experiments that would change a conclusion

| # | Item | Why it matters | Issue |
|---|---|---|---|
| A1 | **Qwen3.5-9B arm** — measurement, then masking sweep, then (optionally) recovery | The only blank row in [`size_h2_summary.md`](size_h2_summary.md). Extends the Qwen lineage to a third size point with architecture and vendor held fixed — the only such series we have on a modern gated model. A 9B baseline exists (12.90, full split) but no in-run baseline, so nothing else can be filled in. | [#13](https://github.com/geeeeenccc/Redunformer-Neuron/issues/13), [#14](https://github.com/geeeeenccc/Redunformer-Neuron/issues/14), [#15](https://github.com/geeeeenccc/Redunformer-Neuron/issues/15) |
| A2 | ~~Multi-seed random baseline for gpt2-medium / gpt2-large at 5% and 10%~~ **— downgraded** | Weeks 13–14 §4b argues "the importance ranking needs sparsity to pay off" from cells where random wins by 0.02–0.10× baseline — each a single draw. **B3 has since bounded the selection half offline:** at those exact cells the guided arm is 48–136 seed sd from the random removed-importance distribution, so no seed picks a materially different amount of importance mass and the claim is not seed-fragile in that sense. What remains is that the guided arm provably removes *less* importance mass and still costs more perplexity — a real effect needing a PPL error bar, not a draw artifact. Still worth doing; no longer "could overturn a result". | — |
| A3 | **4B recovery at a matched budget** — *reframed, now the top open item* | Not the hyper-parameter question we thought. Auditing the artifacts on 2026-09-09 showed the Colab 4B run used `steps: 100`, `batch_size: 1`, no `grad_accum`, 128 blocks = **51k training tokens**, against 200 / 2 / 4 / 256 = **819k** for all three local runs. **16× less.** Its 2% adjusted recovery is internally valid but not comparable to the 76–89% band, so "recovery does not transfer to 4B" conflates model size with training budget. Compounding this, the GPT-2 ladder at the standard budget gives 80/87% → 84/89% → 76/80% across 124M → 355M → 774M: non-monotonic and ≤9 pp over 6× parameters, far too small to reach 2%. Either a 4B re-run at 819k or a Llama-2-7B run at 819k closes it. | [#23](https://github.com/geeeeenccc/Redunformer-Neuron/issues/23) |
| A4 | ~~**H3 on the 4B** — PIQA-calibrated measurement plus one masking sweep~~ | **done** 2026-09-09 — `measurement_piqacalib_Qwen__Qwen3.5-4B_*`, `task_overlap_Qwen__Qwen3.5-4B_*`, `pruning_piqaranked_Qwen__Qwen3.5-4B_*`. Folded into [`artifacts/h3_task_specificity.md`](artifacts/h3_task_specificity.md). |
| A5 | **Record the per-neuron firing-rate distribution, not just the lazy count** | Qwen3.5-4B has 0.00% lazy neurons at α = 0.25 yet frequency still beats random at three of four ratios. Our measurement records only how many neurons fall below the threshold, so it cannot explain this — and it is the one loose end in the otherwise clean "frequency does not transfer across FFN designs" result. | — |
| A6 | **PIQA-calibrated merge coefficients** | `α, β` are fitted and evaluated on WikiText-2. A pair whose gain shifts across corpora would merge worse than reported, and H5′ (replacement) is now the project's headline positive result — it should not rest on a single corpus. | — |

---

## B. Bookkeeping before the write-up is frozen

| # | Item | State |
|---|---|---|
| B1 | ~~**Freeze the hypothesis scoreboard.**~~ | **done** — [`hypothesis_scoreboard.md`](hypothesis_scoreboard.md) is the single source of truth; every earlier report carries an errata block pointing to it rather than a silent edit. H1 also reversed in the process (the depth ablation contradicted the static proxy). [#20](https://github.com/geeeeenccc/Redunformer-Neuron/issues/20) closed. |
| B2 | ~~Copy the 4B recovery PIQA numbers out of the Colab JSON.~~ **— not possible** | **The numbers do not exist.** `recovery_colab_Qwen__Qwen3.5-4B_20250812T115950Z.json` has `lm_eval_limit: 200` in its config but no `lm_eval` block on any arm, and none appears in [#16](https://github.com/geeeeenccc/Redunformer-Neuron/issues/16) either — the transcription from the gitignored Colab original captured perplexity only. Recorded as a gap in [`artifacts/h8_recovery.md`](artifacts/h8_recovery.md) rather than left as a pending task. Recoverable only if the original Colab JSON resurfaces. |
| B3 | ~~**Run the offline random-baseline stability analysis on the real artifacts.**~~ | **done** 2026-09-09 — all six models, 20 seeds, four ratios, ~2 min of CPU. Worst-case separation **21.4 seed sd**; pairwise Jaccard matches the analytic chance floor to four decimals. Table in [`random_selection_stability.md`](random_selection_stability.md) §4; the caveat in the final report is now a measured bound. |
| B4 | ~~**Refresh `docs/tasks/README.md`.**~~ | **done** — board covers #11–#23 with current states. |
| B5 | ~~Cherry-pick the `hooks.py` conflict-marker fix to `main`.~~ | **done** — `2cd20fa` is on `main` and the file imports cleanly; the note in Weeks 13–14 §1 is stale. |
| B6 | ~~Finish the 4B QLoRA recovery on Colab.~~ | **done** — [#16](https://github.com/geeeeenccc/Redunformer-Neuron/issues/16), 2026-08-12; the *result* is A3. |

---

## C. Framing caveats the final report must state

These are not fixable at this point in the seminar. They are limits on what the
results mean, and the report is stronger for stating them plainly.

1. **Masking ≠ removal, so there is no efficiency claim anywhere.** Every
   intervention is a forward hook; parameter shapes never change. No speed,
   memory, or FLOP number follows from any table in this project. The structural
   version is well-defined — fold `α·W_down[:, i]` into the survivor's column and
   delete column `i`, on an unquantized model — and is the obvious follow-up, but
   it was not run, so "50% of Llama-2's FFN neurons can be removed for 2.27×
   perplexity" must be written as *masked*, not *removed*.
2. **Absolute perplexities are not comparable across models or stages.**
   Different evaluation subsets (157 / 79 / 256 / 332 / 512 rows), different
   context windows (1024 for GPT-2 vs 2048 elsewhere), and 4-bit versus fp32.
   Every cross-model statement in the final report should be normalised to each
   model's own in-run baseline, as `size_h2_summary.md` and
   `artifacts/cross_family_tables.md` already are.
3. **PIQA is a coarse instrument here.** 500 examples for the 0.6B runs, 200 for
   the 4B and Llama-2 runs. A 3.6 pp difference at limit 500 is ~18 questions;
   swings of ±0.03 at limit 200 are noise and should not be interpreted.
4. **Recovery is measured on its own fine-tuning domain.** Adapters trained on
   WikiText-2 train and evaluated on WikiText-2 test flatter recovery. The
   unmasked control bounds this and PIQA is the out-of-domain check, but a third
   corpus was never run.
5. **The duplicate-pair inventory is a sample, not a census.** Correlation
   drill-down covers three layers per model and a reservoir subsample, and the
   counts are small and uneven (22 / 17 / 14 / 4 pairs). H5 and H5′ both rest on
   that sample.
6. **The within-pair asymmetry should be dropped as evidence.** Masking the
   second twin costs more than the first, but the procedure masks the
   *lower-importance* twin first, so the second twin is more important by
   construction. Weeks 13–14 recommends dropping it unless re-run with the twin
   choice randomised. The replacement experiment (H5′) carries the claim instead,
   and it is free of that confound because it compares two treatments of the
   *same* neuron.
7. **One architecture contrast, and it is an old model.** GPT-2 carries the
   entire "redundancy is not a gating artifact" argument, so "GELU" and "GPT-2"
   are not fully separable in the frequency result.

---

## Suggested priority

**Superseded — this was written before tier B closed.** For the record, it read:
*A1 (9B row) and B1 (freeze the scoreboard) first, A2 third as the only item
that could retract a published claim.*

**Current order, all GPU-bound and scoped in
[`docs/tasks/henrik_gpu_handoff.md`](../../docs/tasks/henrik_gpu_handoff.md):**

1. **A3 at a matched budget** ([#23](https://github.com/geeeeenccc/Redunformer-Neuron/issues/23))
   — Llama-2-7B, or a 4B re-run at 819k tokens. It replaced A2 as the item most
   likely to retract a claim, because the claim it targets ("recovery does not
   transfer at scale") is currently confounded rather than merely noisy.
2. **A1, the 9B row** ([#13](https://github.com/geeeeenccc/Redunformer-Neuron/issues/13)
   → [#14](https://github.com/geeeeenccc/Redunformer-Neuron/issues/14) →
   [#15](https://github.com/geeeeenccc/Redunformer-Neuron/issues/15)) — now a
   confirmation point rather than the H2 decider, since the GPT-2 size arm
   settled H2 with architecture held fixed.
3. **A visual benchmark on the 4B** ([#12](https://github.com/geeeeenccc/Redunformer-Neuron/issues/12))
   — not on the original list, and arguably a missing *experiment* rather than
   missing coverage: our primary model is an image-text hybrid measured and
   scored on text alone, and H3 says redundancy is task-dependent on exactly
   this kind of FFN.
4. **A2**, for the perplexity error bar, now that selection is bounded.

A5 and A6 remain open and unranked; both are single-run extensions that would
tighten H7 and H5′ respectively without changing either verdict.
