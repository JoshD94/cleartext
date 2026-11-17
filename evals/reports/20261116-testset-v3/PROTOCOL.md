# Test set v3 protocol (pre-registered before any score)

Written 2026-11-16 on branch `josh/testset-v3-20261116`, after the panel and
its split were committed (`7d3afcc`, OneStopEnglish split `9e65067`) and
before any system ran on any v3 passage. Git history for this file is the
evidence. Do not edit it after scoring starts; a revision goes into the
report as a separately labelled alternative.

## Why

The `words=off, phrases=authored, structure=t5+compound_tagger` winner was
selected on the 90 week 7 sourced passages that also produced its headline
numbers (FK 14.115 → 13.506, passage NLI 0.984), so those numbers are
optimistic. Test set v3 is a new, larger panel with a held-out split that no
system has seen.

## Data

- **Panel** (`evals/data/testset_v3/`): 300 passages of 60–120 words,
  selected by `evals/build_testset_v3.py` (seed 20261116, rule-defined
  document pools, no LLM). Six domains of 42 standard passages: computer
  science (IETF RFCs), corporate communication (SEC 10-Ks), political science
  (presidential addresses), health/medical (openFDA drug labels),
  legal/government (Federal Register significant rules), science (CC BY 4.0
  arXiv abstracts). Plus 48 `simple_control` passages from Simple English
  Wikipedia, where the right output is a near no-op.
- **Split**: by document, stratified by domain, one third dev.
  **Dev = 100** (14 per domain + 16 controls), **test = 200** (28 per domain
  + 32 controls). Text stays local; the manifest holds hashes.
- **OneStopEnglish** (CC BY-SA 4.0, commit `37f8db3`): 544 advanced sentences
  aligned to both an intermediate and an elementary version; seeded split,
  **181 dev**, **363 test** (locked). Sentence-level split; the corpus has
  no article IDs in its aligned files, so dev and test sentences can share
  an article.
- **ASSET test** was already used on November 9 and is not rescored.

## Systems (fixed now, not tuned on v3)

| Short name | Config |
|---|---|
| `original` | identity |
| `rule_based` | dictionary baseline |
| `v7_words` | `words=v7, phrases=authored, structure=rules` (the v7 baseline) |
| `nov9_t5` | `words=off, phrases=authored, structure=t5` (November 9 winner) |
| `v2_winner` | `words=off, phrases=authored, structure=t5+compound_tagger` (November 10 winner) |

Same frozen artifacts as the November 10 study (T5 checkpoint, compound
tagger, v7 run), CPU, one thread, `PYTHONHASHSEED=0`, offline models, through
`evals/ensemble_study.py` with a new frozen plan whose panels are the v3 dev
split and OneStopEnglish dev only.

## Metrics

Computed per passage, by `evals/testset_v3.py` on top of the cached Week 7
metrics:

- **Co-primary readability**: Flesch–Kincaid grade (`readability.py`,
  lower is simpler) and **Kea-Roy's human-calibrated simplicity S**
  (`human_score.py`: trained word weights A–E, fitted sentence weights G/H/I
  from `trained_weights.json`; higher is simpler; S is relative to the
  source, so an unchanged passage scores 0.25).
- **Meaning**: passage-level bidirectional NLI mean and sentence-level aligned
  NLI (DeBERTa-v3-base MNLI/FEVER/ANLI, no truncation; abstentions count as
  < 0.80 and are excluded from means), numbers and negations preserved,
  entities preserved with the **fixed** check (sentence-initial words that
  are capitalized only by position are not entities; the old capitalized-span
  value is reported as `entities_capitalized`).
- **Over-edit** (simple controls): changed-token rate (edited source tokens
  over source tokens, difflib word alignment, case-insensitive) and FK delta
  (output FK minus source FK).
- **OneStopEnglish**: corpus SARI with two references, Xu 0/0 = 0 convention
  as primary and `evaluate`'s 0/0 = 1 alongside; FK of outputs.
- **Uncertainty**: percentile 95% CIs from a bootstrap stratified by domain
  (10,000 resamples, seed 20261116); differences are paired per passage.
  OneStopEnglish SARI CIs resample sentences (1,000 resamples, same seed).

Report denominators and changed-only counts with every mean. These are
automatic signals, not percentages of correct simplifications.

## Gates (frozen November 9 rule, entity check fixed)

Over all passages of the split being judged:

1. Mean passage NLI ≥ 0.95.
2. At most 5% of passages have passage NLI < 0.80.
3. Numbers and negations preserved on every passage.
4. Mean entities preserved (fixed check) ≥ `rule_based`'s on the same passages.

**Over-edit gate** (simple controls only): mean changed-token rate ≤ 0.10
and no control passage with passage NLI < 0.80. FK delta is reported, not
gated.

## Dev run (this task)

Descriptive only. No system, threshold or metric is chosen or tuned on dev
results; the five systems above are already fixed. The dev report gives every
metric above with CIs, the gates for each system, and the winner's paired
deltas against each other system.

## Final locked-test run (one shot, triggered later by Joshua)

- Freeze a plan with panels `test` (the 200 test passages) and
  `onestop_test` (363 sentences), using `freeze --unlock-test`. The
  generation and scoring sources and artifacts must hash the same as the dev
  run's plan; otherwise the test result is reported with the differences
  listed and is not called a confirmation.
- Run the five systems once. No rerun after reading scores, except to
  complete missing cells of the same plan.
- Judge `v2_winner`:
  - **R (readability)**, on the 168 standard test passages: paired ΔFK vs
    `original` with 95% CI entirely below 0, **and** paired ΔS vs `original`
    with 95% CI entirely above 0.
  - **M (meaning)**: gates 1–4 on all 200 test passages.
  - **O (over-edit)**: the over-edit gate on the 32 controls.
  - **Verdict**: *confirmed* if R, M and O all pass; otherwise *not
    confirmed*, naming each failed part. No other config may be substituted
    on test.
- Comparisons with `v7_words` and `nov9_t5` are reported as paired deltas
  with CIs. "Simpler than X" needs the ΔFK CI below 0; "no worse in meaning
  than X" needs the passage-NLI delta CI lower bound ≥ −0.005.
- OneStopEnglish test SARI and FK are reported for every system with CIs,
  without a gate.

## Amendment 1 (2026-11-16, from the PR #18 agent review)

Added after dev scoring had started on the plan frozen at `422109b`. The only
scores looked at were from a dry run of the report code on the first 27 dev
passages. No system, metric definition, gate or judgement rule changes.

- **Entity check fix.** The common-word rule now applies only to
  single-word spans. As first written it cut multi-word names to their last
  word ("United States" → "States"). The `-ly` rule became a fixed list of
  sentence adverbs, because it dropped names such as "Italy" and "Lilly". This
  is what "capitalized only by position" was meant to mean.
- **Dev run vs main.** The in-flight dev run keeps its frozen `422109b`
  sources, so its scoring signature hashes the pre-fix `preservation.py`.
  Every entity number in the dev report is recomputed from the texts at
  report time with the fixed check. The cached Week 7 metrics it reuses
  (readability, NLI, numbers, negations) do not depend on the entity code.
- **Reference for the final run.** In place of "hash the same as the dev
  run's plan": generation sources and artifacts must hash the same as the dev
  plan. Scoring sources must hash the same as `main` after PR #18 merges,
  where the only difference from the dev plan is the entity code in
  `preservation.py`. The report-time files (`testset_v3.py`, `human_score.py`,
  `evaluate_base.py`, `trained_weights.json`, `preservation.py`,
  `readability.py`, `sari.py`) must also match the hashes the dev report
  records in `report_sources_sha256`. Any difference is listed, and the
  result is not called a confirmation.
- **Reporting without v7.** If v7 words is not complete when the other four
  systems are, the dev report may omit it (`--omit v7_words`) and say so.
- **Wording.** The changed-token rate counts the longer side of each edited
  run, so pure insertions count too. The study branch is
  `josh/testset-v3-20261116`; the protocol merged through
  `josh/testset-v3-split-20261116`.
