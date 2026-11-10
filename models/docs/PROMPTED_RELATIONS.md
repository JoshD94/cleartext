# BERT support for candidate ranking

The opt-in run `runs/ensemble-prompted-relations-20261002` uses BERT suggestions to rank the existing WordNet candidates. It keeps the original word visible in the first sentence and masks the target in a second copy. This gives BERT both the intended word and its context. The local cached encoder supplies the predictions; no API or new training data is involved.

The decision model adds three features to the prior 48: whether the inflected candidate appears among BERT's top 20 tokens, its reciprocal rank, and its token probability. The probability belongs to the predicted token before inflection. It is a learned ranking signal, not a calibrated probability that an edit preserves meaning. WordNet provenance, the meaning floor, and the existing guards still apply.

## Development result

All 929 BenchLS cases are reused development data, including the historical holdout. The comparison uses three target-word-grouped 10-fold repeats, the same candidates, threshold 0.45 and meaning floor 0.10. The first 48 feature columns match exactly within the comparison. No TSAR gold was read and no threshold was selected from these results.

| Mean across repeats | Prior relation model | BERT support model |
| --- | ---: | ---: |
| Edits matching a listed substitute | 438.7 | 510.0 |
| Edits absent from the substitute list | 175.0 | 136.0 |
| Listed-substitute edit precision | 71.5% | 78.9% |
| Edit coverage | 66.1% | 69.5% |
| Correct minus incorrect edits | 263.7 | 374.0 |
| Candidate log loss | 0.237885 | 0.221769 |

Net gains were 105, 109 and 117 edits across the three repeats. These are dependent development comparisons, not independent test results. An absent substitute is counted as wrong by this benchmark, but may still be reasonable. Full-sentence detail preservation needs separate assessment.

Results and fold predictions are in `outputs/prompted-relations-dev-20261002`. `error-report.json` records transitions and remaining errors across repeats. It identifies 124 cases with an unlisted selected substitute in every repeat. This report does not create new semantic labels.

## Candidate coverage experiment

The separate BERT proposal audit raises the number of cases with at least one listed substitute from 807 to 895 of 929, or 86.9% to 96.3%. It recovers 88 of the 122 cases missed by the current candidate pool. It adds 12,326 candidate rows, including 1,638 listed substitutes.

Those new rows lack the WordNet provenance required by the current meaning features. They are research proposals only and are not routed into the saved ranking model. The coverage number is an upper bound from candidate generation, not achieved edit accuracy. Single-token proposals also leave multiword substitutions unresolved.

## Diagnostic limits

The 13 existing ungraded examples remain mixed. The new model preserves `database` and `harsh sentence`, and keeps the useful `convoluted` to `complex` edit. It still changes `encodes` to `converts`, `classify` to `separate`, and `filibuster` to `delay`. New choices include `dividend` to `profit` and `constitution` to `law`, which lose distinctions. `deposited` becomes `put` in the bank example.

This run remains opt-in. The v7 files, default configuration and running demo are unchanged. Benchmark gains do not justify promoting these technical failures.

## Validation and reproduction

The full suite passes 74 tests. A native inference check compares 12 fixed, evenly spaced cache records against the live single-target path. Candidate words and ranks match exactly; the largest probability difference is 2.17e-7. This checks batched-cache versus live inference consistency, not prediction quality.

The jobs ran on Yoga in `cleartext-bert-candidates-20261002`, `cleartext-prompted-relations-20261002`, and `cleartext-prompted-parity-20261002`, with four CPU threads and a 3 GB memory cap. Each exited successfully. Yoga had over 11 GB available before the ranking and parity jobs.

Run from `models`, with new output paths:

```sh
.venv/bin/python scripts/experiments/audit_bert_candidates.py --help
.venv/bin/python scripts/experiments/refine_prompted_relations.py --predictions outputs/bert-candidate-coverage-20261002 --output outputs/prompted-relations-review --save-run runs/ensemble-prompted-relations-review
.venv/bin/python scripts/experiments/check_prompted_support.py --records outputs/bert-candidate-coverage-20261002/records.jsonl --output outputs/prompted-support-review.json
.venv/bin/python scripts/audit_domains.py --run runs/ensemble-prompted-relations-20261002 --baselines --output outputs/domain-audit-prompted-review
```

Long jobs belong in named Yoga tmux sessions after resource checks. Next work should examine the persistent errors and test whether BERT support helps distinguish broader replacements from same-sense edits. Retain the current candidate pool until novel proposals have a defensible meaning score. The actual team technical corpus is still absent locally.

## Follow-up: relation interactions rejected

A second fixed comparison added four explicit interactions: reciprocal BERT rank times same-sense fraction, rank times hypernym fraction, unsupported hypernym fraction, and rank times hypernym breadth. It kept all 51 prior features and the same candidates, folds, threshold and meaning floor.

The extra features reduced mean net correct edits by 7.3. Repeat deltas were -5, -6 and -11. Log loss improved by only 0.000283, below the predeclared 0.001 bar. This variant failed both acceptance criteria and was not fitted into a saved run or registered in the pipeline. The experiment and two passing feature tests remain for reproducibility, with results in `outputs/prompted-specificity-dev-20261002`.

The best candidate from this round remains the 51-feature BERT support model. Further hand-built interactions on the same data are unlikely to settle the technical detail failures. A useful next experiment is a meaning score for novel BERT candidates that does not invent WordNet senses. Keep the existing constraints and evaluate on the existing development labels before considering a new candidate pool.

To regenerate the error report with verified cache inputs:

```sh
.venv/bin/python scripts/experiments/report_prompted_errors.py --experiment outputs/prompted-relations-dev-20261002 --output outputs/prompted-errors-review.json
```
