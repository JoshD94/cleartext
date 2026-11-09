# Compound splitting: scoped guards and a WikiSplit tagger (November 9)

Raises compound-split coverage without an LLM. No text is generated: a
logistic-regression tagger decides which "and"/"but"/"so"/"yet" to split at,
and the existing parser rules make the edit and run the preservation checks.

## Held-out WikiSplit test set

`models/scripts/eval_compound_wikisplit.py` on WikiSplit `test.tsv`: 3,569
sentences containing a connector. 431 are **labelled positives**: the person's
first sentence is exactly the text before a connector and the second ends with
exactly the text after it. The other 3,138 are unlabelled: a person split them
elsewhere, so a split there is reported but not scored right or wrong.

| Setting | Positives split at the person's connector (recall) | Of positive splits, at that connector | Exact match with the reference | Unlabelled sentences split |
|---|---:|---:|---:|---:|
| `guards="strict"` (PR #9 behaviour) | 33/431 (7.7%) | 97.1% | 25 | 70 (2.2%) |
| `guards="scoped"` | 51/431 (11.8%) | 94.4% | 38 | 115 (3.7%) |
| scoped + tagger (threshold 0.4) | **82/431 (19.0%)** | 95.3% | 57 | 218 (6.9%) |

Exact matches are lower than boundary matches because WikiSplit authors often
reword the second half (pronouns, "However,"). Tune-set figures for the tagger
setting: 21.1% recall, 92.3% boundary precision.

**Manual check of unlabelled splits.** One reviewer (agent, not a human study)
read 40 random unlabelled test splits before the last fixes (35 good, 3 wrong,
2 awkward) and 40 fresh ones after them (39 good, 1 wrong, then fixed). The
four failure patterns became guards and regression tests. The test set has
therefore been inspected; treat these as development-quality numbers.

## Week 7 panels (coverage only)

| Setting | 18 authored passages changed | 90 sourced passages changed |
|---|---:|---:|
| strict | 0 | 1 |
| scoped | 1 | 3 |
| scoped + tagger | 3 | 6 |

Eight of the nine tagger-setting edits read correctly. In one sourced finance
passage a trailing "as of December 31, 2023 and 2024" that scoped over both
clauses stays only on the second; trailing shared modifiers are a known limit.
Automatic readability/NLI scoring of these panels was not rerun.

## Tagger

Trained on every 10th line of WikiSplit train: 71,395 sentences, 109,683
connectors, 8,992 positives. Unchosen connectors in labelled sentences are
negatives (weight 1); connectors in unlabelled sentences are weight 0.3. On
tune connectors in labelled sentences, threshold 0.4 gives 94.9% precision and
61.8% recall (`tagger-report.json`). The splitter uses only 19% of positives
because the copy-subject and scope checks still abstain on the rest.

Files: per-setting summaries (`test-*.json`, `tune-tagger.json`), tagger report,
panel coverage and `provenance.json` (commit, hashes, packages, job IDs).
Per-sentence outputs and the model file stay local.
