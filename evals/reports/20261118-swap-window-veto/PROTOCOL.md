# Window veto for word swaps: protocol (pre-registered November 18, 2026)

Committed before any window score on the gold sets and before any output of
[design-dev.json](design-dev.json) exists. Git history is the evidence. Dev
split only: the locked test split is not generated, scored or opened, and a
locked-test run needs Joshua's approval. No LLM is used anywhere. Raw text
stays local; this folder holds aggregates only.

## What had been read before this file

- The [November 17 NLI calibration](../20261117-nli-calibration/README.md):
  sentence-level NLI rejects 25% of human-gold substitutes at 0.95, and one
  rejected swap costs about 0.46 passage NLI.
- The orchestrator's November 17 look at 40 pipeline swaps (dev and old
  panel, local only): scoring a ±6-word window instead of the sentence
  rescued about 13% of rejected swaps while bad ones stayed low. That
  motivated the window; it was read on dev swaps, so the window size and
  threshold are chosen below on the gold sets only, never on dev.
- Not seen: any window score on the gold sets, any output of this design.

## The veto (`cleartext.swap_veto.WindowSwapVeto`, opt-in)

For each candidate swap, the word stage passes the sentence before and after
that one swap. The veto cuts the same window of W whitespace words on each
side of the changed words from both sentences and scores the evaluation NLI
(DeBERTa-v3-base-mnli-fever-anli, the evaluation model; float32 weights, as
in the November 17 study) in both directions. The swap passes only if:

- the mean of the two directions is at least T, and
- the evaluator's number, negation and entity checks
  (`preservation.explicit_checks`) all keep 1.0 on the whole sentence.

A vetoed swap is skipped on its own; the pipeline's existing refill lets the
next-ranked word take the slot (`veto_refill`, default true). Defaults do not
change: the veto runs only when passed as `word_veto=`.

## Choosing W and T (gold sets only)

- **Pairs**: nli_calibration's `gold.jsonl` (SWORDS test, TSAR-2022 test,
  MultiLS trial), already built on November 17.
  - Human-good: SWORDS substitutes accepted by ≥ 50% of annotators, TSAR and
    MultiLS substitutes named by ≥ 2 annotators (3,687).
  - Human-bad: SWORDS substitutes no annotator accepted, and the random
    same-POS, frequency-matched words for TSAR and MultiLS (3,640).
  - Borderline SWORDS items and single-annotator substitutes are left out.
- **Grid**: W ∈ {2, 3, 4, 6, 8, 10, whole sentence}; T from 0.500 to 0.995
  in steps of 0.005. The decision includes the checks.
- **Rule**: the (W, T) with the highest pooled balanced accuracy
  ((recall + specificity) / 2, human-good = positive) among settings whose
  pooled precision is ≥ 0.90. If no setting reaches 0.90 precision, the
  highest pooled balanced accuracy overall, and the report says so. Ties go
  to higher precision, then the smaller window, then the higher threshold.
- **Reported**: precision, recall and balanced accuracy pooled and per data
  set, with 95% CIs from a bootstrap over targets (2,000 resamples).
- The choice is written to `veto-choice.json` and committed before the dev
  plan is frozen; `swap_veto freeze` refuses an uncommitted or modified
  choice.

## Dev run (design-dev.json)

All arms run on the 100-passage dev split (84 standard passages plus the
16 simple controls), on top of F2, at the word stage's defaults (one word
edit per passage):

| Arm | Role |
|---|---|
| F2 | reference (the locked-test winner, unchanged) |
| F2+v7 (window veto) | judged |
| F2+contextual (window veto) | judged |
| F2+v7, F2+contextual (no veto) | reported for reference only |

Generation reuses the parameter-sweep dev state's caches (T5 beams, word
decisions, small NLI, sentence NLI, metrics) only after `swap_veto freeze`
checks that every source and artifact they depend on is byte-identical; the
only new source is `models/src/cleartext/swap_veto.py`. F2's regenerated
outputs must equal the sweep state's.

## Lexical simplicity metric (new, reported next to FK and Kea-Roy S)

Over content words (alphabetic, not a spaCy stopword) of each output passage:

- **Word complexity (primary)**: mean predicted complexity from Kea-Roy's
  GBM word model (`evals/word_models.py`), with each word's sentence as its
  context. Lower is simpler.
- **Rare-word share**: share of content words with Zipf frequency < 3.5
  (wordfreq; the threshold of `readability.technical_term_count`).
- **Changed-word Zipf**: mean Zipf of content words inserted minus deleted
  against the source, per changed passage; and over applied word swaps, the
  Zipf of the replacement minus that of the replaced word.

## Success rule (per judged arm, against F2, paired over dev passages)

F2 + words **beats F2** only if all of these hold:

- (a) the 95% CI of Δ passage NLI has its lower end ≥ −0.005;
- (b) it passes gates 1–4 (`testset_v3.gates`: mean NLI ≥ 0.95, ≤ 5% of
  passages below 0.80, no number or negation failure, entities ≥
  rule_based on the same passages);
- (c) it passes the over-edit gate on the 16 simple controls;
- (d) the 95% CI of Δ word complexity lies entirely below 0;
- (e) ΔFK (mean) ≤ 0: FK is not worse.

Deltas use the v3 domain-stratified bootstrap (10,000 resamples, seed
20261116). ΔS (Kea-Roy), Δ rare-word share and swaps applied are reported
alongside. A dev win is not adoption: these dev passages already chose F2,
and a confirming locked-test run needs Joshua's approval.

## Budget

At most 12 queue chunks of ≤ 30 minutes, at `later` priority, one job at a
time, on Yoga with 1 torch thread for generation and scoring. Outputs are
checkpointed per (passage, config). If an arm is not complete within the
budget, it is judged on the passages done and reported as **incomplete**.

## Choice (added after the gold scores, before any dev output)

`swap_veto calibrate` on the 7,327 gold pairs gave **W = 2 words, T =
0.945** ([veto-choice.json](veto-choice.json), [calibration.json](calibration.json)).
No setting in the grid reaches pooled precision 0.90 (the highest is
0.889), so the rule's fallback applied: the highest pooled balanced
accuracy overall. At that setting, pooled precision is 0.748, recall 0.775
and balanced accuracy 0.755. Whole-sentence scoring reaches 0.721 at best.
Nothing else in this file changed.
