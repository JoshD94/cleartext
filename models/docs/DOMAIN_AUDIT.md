# Local lexical audit

The audit runs the frozen v7 lexical pipeline on attributed sentences. It saves the exact input, output, chosen edit, candidate scores and rejection reasons. It also runs identity and dictionary baselines when requested. A target probe inspects a word even when detection skips it. Probes do not change the sentence output.

The local checkout has CompLex, CWI, SWORDS, BenchLS and TSAR data. It has no arXiv, SEC 10-K or GovInfo corpus, preprocessing module or teammate evaluation module. The included `diagnostics/domain-audit.jsonl` contains authored diagnostics and examples from the preserved owner conversation. These are ungraded and separate from train, development and test data. They are not excerpts from the proposed technical corpus.

Run from `models/` with the existing environment. Use a new output directory each time.

```sh
.venv/bin/python scripts/audit_domains.py --baselines --output outputs/domain-audit-v7-20261002
.venv/bin/python scripts/audit_domains.py --baselines --preserve-detail --output outputs/domain-audit-detail-20261002
```

Long runs belong in a named Yoga tmux session, with four CPU threads and a suitable memory cap. The script sets the thread limits and requires locally cached Hugging Face files. It never downloads models or fits on audit input.

To use actual corpus sentences, pass `--input path/to/sample.jsonl`. Each line needs `id`, `domain`, `text` and a `source` object. Corpus sources need `kind: "corpus"`, `attribution` and a `url` or local `path`. An optional `target` names one parser token. A repeated target needs an exact character `start`, so the audit cannot silently probe the wrong occurrence. The first 30 records run by default; `--limit` allows up to 100. Sampling follows input order.

The output directory contains:

- `manifest.json`, with the selected records, input checksum, frozen model and source checksums, dependency versions and local BERT cache hashes.
- `source/`, with a copy of the Python code used for the run.
- `records.jsonl`, with every candidate and model score. Each record is flushed before the next sentence starts.
- `summary.json`, with change coverage by domain and baseline, timing and an unset human precision field.
- `report.md`, with the original and rewritten sentences and the highest scoring candidates.

Candidates that the lazy safety check never reaches have `guard_status: "not_evaluated"`. A lack of rejection reasons does not mean they passed the safety checks. `selected_for_target` also differs from `applied_to_output`, because the pipeline chooses at most one word edit across all detected targets.

## Conservative detail experiment

`--preserve-detail` loads an experimental subclass. After the learned model scores the complete candidate list, the subclass rejects proposals reached through a direct WordNet hypernym edge. The original features and probabilities stay the same, and selection can fall back to another candidate. The check allows same-synset synonyms, similar adjectives and verb-group links that are not hypernyms. It uses relations rather than a list of banned words.

This experiment addresses broader rewrites such as `database -> information`, `encodes -> converts` and `filibuster -> delay`. It cannot detect detail loss inside a synset, a wrong contextual sense, or technical terms that WordNet handles poorly. Its name describes a conservative policy, not a guarantee of preserved meaning. The v7 configuration, artifacts, demo and prediction command keep their current defaults.

The cached comparison runs separately:

```sh
.venv/bin/python scripts/experiments/audit_detail_policy.py --output outputs/detail-policy-dev-20261002.json
```

It uses all BenchLS as development, including the historical holdout. It repeats the existing target-word-grouped 10-fold comparison three times with the fixed v7 threshold and meaning floor. It reports correct edits, wrong edits, precision, change coverage and net correct edits for each repeat. It does not read TSAR gold or search for a new threshold. BenchLS substitute labels do not measure full-sentence detail preservation, and this reused development protocol is not a fresh test.

## First run

The first 12-sentence diagnostic run changed 10 sentences and scored 333 candidates across 23 detected targets. Median lexical analyze time was 2.13 seconds; the maximum was 10.86 seconds. Model loading, baseline calls and extra target probes are excluded. The first sentence includes lazy BERT loading, and Yoga was shared with other jobs. These times are a sample observation, not a controlled speed benchmark.

The run reproduced `database -> information`, `encodes -> converts`, `filibuster -> delay` and `dividend -> profit`. `convoluted -> complex` and `commence -> begin` also appeared. The short `She deposited the money.` input abstained; the longer input from the saved v7 examples is included in subsequent runs to check the role of context.

For the database and encoding examples, `sense_fit` was 1.0. That signal counts the probability of the original sense from which the candidate was reached. A hypernym reached from the right sense can therefore get a high meaning score while dropping detail. The decision model already has specificity features, but those features did not block these outputs. Expanding the candidate list is a lower priority than measuring this failure on actual technical text.

All automatic rule checks and probabilities remain diagnostic evidence. No human accuracy or technical-domain benchmark score is claimed.

## Fixed development comparison

The direct-hypernym policy is not adopted as the default. On 929 BenchLS development cases, it reduced net correct edits in every repeat, by 66, 63 and 54. The mean loss of 61 is far beyond the existing net gain requirement of 5.

| Metric | v7 | Experimental detail policy |
|---|---:|---:|
| Mean correct edits | 425.3 | 336.3 |
| Mean wrong edits | 189.3 | 161.3 |
| Mean net correct edits | 236.0 | 175.0 |
| Change coverage | 66.2% | 53.6% |
| Substitute precision | 69.2% | 67.6% |

Counts and net are means across three repeats. Precision pools their predictions. Threshold 0.45 and meaning floor 0.10 stayed fixed. The earlier v7 result of 243.7 optimized a threshold within each repeat before reporting the mean. This experiment deliberately uses the deployed threshold in every repeat, so its baseline is 236.0. Both are reused development results.

On the identical 13 diagnostic inputs, v7 changed 11 sentences and the experimental policy changed 8. The policy blocked the inspected broader candidates for database, encoding, deposits, dividends, filibusters and constitutions. It kept `convoluted -> complex` and `commence -> begin`. It also selected weaker fallback edits, including `encrypted -> coded` and `delayed -> retarded`. Blocking one candidate therefore does not establish that the final sentence is better.

The reviewable outputs are `outputs/domain-audit-v7-review-20261002/`, `outputs/domain-audit-detail-review-20261002/` and `outputs/detail-policy-dev-20261002.json`. The two audit folders include copied source code. The development file records input and source checksums. Earlier runs and the failed compact-table attempt remain in the logs for inspection.

The next useful input is a fixed, attributed sample of the existing team corpus. Run it through this audit before selecting a new preservation rule. The model needs to distinguish contextual fit from retained detail and to inspect the replacement it chooses after rejecting a broader candidate. Adding masked-LM candidates can wait until those checks work on technical text. No new human labels or synthetic gold were made during this work, and TSAR gold was not used in the comparison.

The continuation implemented a learned alternative to this blanket policy. [The relation experiment](RELATION_DECISION.md) improved fixed-threshold BenchLS development precision and net correct edits while retaining coverage. Its diagnostic failures keep it opt-in; the original v7 demo remains current.
