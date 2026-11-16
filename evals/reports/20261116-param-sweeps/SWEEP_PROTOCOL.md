# Parameter-sweep protocol (pre-registered November 16, 2026, before any scoring)

This file and the design files beside it are committed before the sweep
produces any score, smoke runs included. Git history is the evidence. Later
changes appear below under "Amendments", each dated and labeled, and say
whether any score had been read.

## Question

The November 10 winner, the **incumbent**, is `words=off, phrases=authored,
structure=t5+compound_tagger`, with every stage at its current setting. Do
tuned settings of any stage (parameters, not just on/off levels) beat it under
the frozen selection rule? No LLM is used: the stages are rules, CRFs,
gradient boosting, small BERT-family encoders, a T5-small splitter and small
NLI cross-encoders, as before.

## Data

- **Sweep panel: the dev split of test set v3** (`josh/testset-v3-20261116`,
  `evals/reports/20261116-testset-v3/`): about a third of ~300 sourced
  passages (60–120 words), split by document. **The locked test split is never
  generated, scored or opened by this study.** `param_sweeps.freeze` refuses
  files named `*LOCKED*` and any passage with `split == "test"` or an ID in a
  committed test ID list.
- **Smoke panel:** the old 90 sourced passages
  (`evals/data/week7_sourced_passages.json`). It was inspected in earlier
  studies, so its numbers test the harness and are labeled **smoke only**. They
  do not select anything.
- **ASSET test** (359 sentences, already used on November 9–10, not fresh) and
  OneStopEnglish (if the test-set worker adds it) are scored for the finalists
  only.

## Metrics

Same evaluator, models and settings as the frozen rule: FK grade and readability from
`evals/readability.py`; passage-level bidirectional NLI (`nli_mean`), numbers,
negations and entities from `evals/preservation.py` with
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, CPU, no truncation. Added:

- **Human-calibrated simplicity S**: `evals/evaluate_base.evaluate_record`
  with the trained word weights and G/H/I sentence weights from
  `evals/trained_weights.json` (fitted to Lens SimpEval human simplicity;
  held-out r ≈ 0.55–0.63). It is the only human-calibrated readability score
  in the repository today. If the test-set worker publishes a newer one before
  phase A is scored, it is reported alongside S; it does not replace S here.
- Per-domain FK and NLI on the dev split.

The per-split NLI floor and the word veto use Chhoeng's
`cross-encoder/nli-deberta-v3-small`, a different model from the evaluation
NLI, so the gate model is not optimized directly.

## Selection rule

1. **Gates:** the four frozen gates of
   [SELECTION_RULE.md](../20261109-ensemble-study/SELECTION_RULE.md), unchanged,
   applied over the dev panel. Gate 4 uses `rule_based`'s entity mean on the
   same panel.
2. **Winner:** lowest mean FK among configs that pass every gate. Configs
   within 0.1 grade of it are tied, and the tie goes to the higher
   human-calibrated S. *Deviation from the frozen rule:* its tie-break was ASSET SARI.
   ASSET test is no longer fresh, and its 0/0 convention flips the November 10
   tie. SARI is still reported for the finalists, under both conventions.
3. **Adoption (no-change default):** the winner replaces the incumbent only
   if, against the incumbent on dev, (a) it passes all gates, (b) the mean
   paired ΔFK ≤ −0.10 (outside the tie band), (c) the 95% bootstrap CI of ΔFK
   is entirely below 0, and (d) the mean paired Δ passage NLI ≥ −0.010.
   Otherwise the recommendation is **no change**.
4. **Bootstrap:** paired per-passage deltas against the incumbent, percentile
   95% CI, 10,000 resamples, seed 20261110 (`ensemble_study.paired_delta`).
5. **Finalists for the locked test (at most 3):** the adopted config (or the
   incumbent if none is adopted); the incumbent; and the lowest-FK config that
   passes every gate, is not one of those two and has a ΔFK CI entirely below 0.
   If no such config exists, there are two finalists.

## Per-component goals

| Component | Goal (answered yes/no in the report) |
|---|---|
| T5 split acceptance | Find a looser setting (waived soft checks, passage-aware check, or loose checks plus a per-split NLI floor) that recovers part of `t5_unchecked`'s −1.2 FK and still passes every gate. |
| T5 decoding | Does trying lower beams (`candidates`) or another beam width add passing splits? |
| T5 minimum length | Does skipping short sentences raise NLI enough to be worth its FK cost? |
| Compound tagger | Does a lower tagger threshold add passing splits? |
| Authored phrases | Do more edits or a lower gain threshold lower FK safely? |
| Phrase finder | Find a CRF threshold, replacer floor or SimplePPDB cut-off that passes the gates and lowers FK by more than v2's `crf` (−0.03). |
| Passive | Does a fixed placeholder pass gate 2 where `auto` failed? |
| Word swaps | Find **any** word setting (high threshold, strict detector, margins, meaning floor, strict or non-refilling veto) that passes the gates and lowers FK. |

## Search space and budget

Designs written by [derive_designs.py](derive_designs.py):

| Phase | File | Configs | Content |
|---|---|---:|---|
| 0 | design-0-references.json | 13 | incumbent; the November 10 levels re-measured on dev |
| A | design-A-structure.json | 39 | 1-D: 14 single waived checks and all soft checks waived, passage check (2), safe split = all soft checks waived + passage-aware check × NLI floor {none, 0.5, 0.7, 0.8, 0.9, 0.95}, min words {8, 12, 16, 20}, decoding (beams 1, beams 4, try 2 of 2, try 4 of 4), tagger threshold {0.2, 0.3, 0.5, 0.6}, compound min sentence words {10, 15}, min clause words {3, 5} |
| A | design-A-phrases.json | 18 | authored gain {0, 0.05, 0.1}, edits {2, 3}; CRF threshold {0.2, 0.4, 0.5, 0.7}, replacer NLI floor {0.75, 0.9}, sources manual+verified; SimplePPDB cut-off {4.3, 4.5, 4.8, 5.0}; passive placeholder {someone, people} |
| A | design-A-words.json | 27 | decision threshold {0.6, 0.7, 0.8, 0.9} for v7 and contextual; v7 meaning floor {0.3, 0.5}, detector {0.5, 0.6, 0.7}, noun 0.9, gain margin {0.1, 0.2}, zipf margin {0.5, 1.0}, alpha {0, 2}, 2 edits; veto {0.575, 0.75, 0.9, 0.97} with/without refill |
| B | design-B-combined.json | ≤ 17 | derived mechanically (below) |
| C | finalists on ASSET (and OneStopEnglish) | ≤ 3 | — |

Each sweep changes one parameter from the incumbent. The phrase-finder,
SimplePPDB, passive and word sweeps change one level as well (for example
`phrases=crf` plus a threshold), since their parameters exist only at that
level. Phase 0 gives each level's own default.

**Order and pruning (cheap first):** phase 0 and structure, then phrases, then
words. Word configs are first generated and scored on shard 0/3 of dev (every
third passage). A word config goes on to the full panel only if, on that
third, it passes gates 1–3 (NLI mean ≥ 0.95, at most 5% of passages below 0.80,
and no number or negation failure). Screened-out configs are reported with
their screen numbers and are not eligible to win. No other pruning is used.

**Phase B rule:** for each component (T5 acceptance, T5 decoding, compound,
authored phrases, finder, passive, words), the candidate is the phase-0/A
setting with the lowest dev FK that passes every gate and has FK below the
incumbent's. The four candidates with the lowest FK form a full factorial
(2^4 = 16 configs, each on or off). If more than four components have
candidates, one extra config switches on all of them.
`python derive_designs.py combined aggregate-dev.json` writes it. The design
is committed before it runs.

**Budget:** at most 40 queue chunks of ≤ 30 minutes (`--priority later`, one
job at a time, Yoga, 1 torch thread, `PYTHONHASHSEED=0`, offline models),
checkpointed per (passage, config) cell, with stage and metric caches. If the
budget runs out, the report covers the complete configs, lists the pending
ones and makes no recommendation that depends on them.

## Reporting

Everything goes in `README.md` in this folder, aggregates only:

- FK, NLI, gates, human S, changed passages and paired deltas with CIs for every config.
- Per-parameter response curves (FK vs NLI) and the Pareto frontier.
- An answer to each component goal.
- The recommendation (or "no change") and the finalists.
- Raw passage text, outputs and checkpoints stay in
  `~/.local/state/cleartext/param-sweeps-20261116/`.

## Amendments

(none yet)
