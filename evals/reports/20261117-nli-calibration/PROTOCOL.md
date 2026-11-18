# F2 + word swaps under a calibrated word gate: protocol (pre-registered November 17, 2026)

Committed before any F2 + words output exists. Git history is the evidence.
Dev split only. The locked test split is not generated, scored or opened; a
new locked-test run needs Joshua's approval. No LLM is used anywhere.

## What had been read before this file

- **Calibration of NLI against human substitute judgments** (README, part 1):
  - The best balanced threshold for the veto model (min of both directions, per
    sentence) against SWORDS human acceptability is **0.97** (Youden, 0.971).
  - For the evaluation model's mean it is 0.976.
  - Lowering either threshold rejects fewer human-acceptable swaps but accepts
    more human-rejected ones. It does not agree better with humans.
- **Re-gating of cached word outputs on top of the incumbent** (README, part 2):
  dev screen (34 passages), the old 90-passage panel and the paper run.
  - A per-swap sentence veto removes most of the passage-NLI loss and keeps
    most of the FK gain.
  - The remaining gate failure on dev is gate 4, on the entity check: two swaps
    replace sentence-initial words ("Discovering", "Decreases") that the
    heuristic counts as entities.
- **Not yet seen:** any F2 + words output or score.

## Arms (design-F2-words.json)

- **F2**: the locked-test winner, unchanged (cached on dev).
- **F2+v7**, primary: F2 with `words=v7` at its defaults (one edit per passage).
- **F2+v7 (threshold 0.7)**, secondary: the one v7 setting that survived the
  sweep screen. It is reported, but it cannot override the primary's result.

Generation uses the frozen dev plan of the parameter sweeps (same generation
and scoring signatures). It runs in a temporary worktree at the plan's path,
whose 90 pinned sources match `main` byte for byte.

## Word gates (applied per swap, after generation)

The word stage is last and makes at most one edit per passage. With no refill,
vetoing a swap in the pipeline gives exactly the F2 output, so a gate applied
afterwards is equivalent. The gate reads the sentence that holds the swap
(`preservation.split_sentences`), before and after the swap.

- **A1, primary.** Two conditions:
  - veto-model entailment (`cross-encoder/nli-deberta-v3-small`, min of both
    directions, as `EntailmentVeto`) ≥ **0.97**;
  - the evaluator's explicit checks keep every number, negation and entity in
    that sentence.

  A1 uses a model different from the evaluator, so passage NLI stays an
  independent check.
- **A2, sensitivity.** Evaluation NLI mean, both directions, ≥ **0.90**, plus
  the same checks. It is not eligible for adoption, because it filters on the
  metric that the gates then score.
- **none.** Every swap kept (reported for reference).

The passage gates are unchanged: gates 1–4 of the frozen rule, with the fixed
entity check, plus the v3 over-edit gate on the 16 simple controls.

## Judging (`python -m evals.nli_calibration judge`)

F2+v7 under A1 **beats F2 on dev** only if, over all 100 dev passages:

- (a) it passes gates 1–4;
- (b) the paired ΔFK vs F2 is ≤ −0.10;
- (c) the 95% CI of ΔFK lies entirely below 0;
- (d) the paired Δ passage NLI vs F2 is ≥ −0.010;
- (e) it passes the over-edit gate.

These are the sweep's adoption rules (a)–(e), with F2 as the reference. The
deltas use the v3 domain-stratified bootstrap (10,000 resamples, seed
20261116). ΔS (Kea-Roy) is reported alongside.

A dev win would not be adoption. These dev passages already chose F2, and
confirming the result would need a new locked-test run, which only Joshua can
approve.

## Budget

At most 8 queue chunks of ≤ 25 minutes, at `later` priority, one job at a
time, on Yoga with 1 torch thread. Output and decision caches are checkpointed
per cell. If generation is not complete within the budget, the report says
**incomplete** and makes no claim.
