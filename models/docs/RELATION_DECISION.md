# Learned relation experiment

The new local decision model adds WordNet relation features to the frozen v7 candidate scores. It improves the reused BenchLS development comparison at the existing threshold. The v7 artifacts, demo and default model stay in place. The candidate model is opt-in.

## Change

The generator groups direct hypernyms and verb-group links under the same source flag. Its provenance can also contain several relations for one candidate. For example, `commence -> begin` shares some senses and has other verb-group links. `encode -> convert` is reached through a direct hypernym. The previous specificity feature takes the minimum breadth change across matching relations. That minimum does not describe this mixture.

The new features expose the fractions of same-sense, direct-hypernym, verb-group and similar-to links. They also include mean and maximum hypernym breadth change, candidate polysemy, the number of original senses that generated the candidate, and an interaction between breadth change and the original sense probability. Breadth uses the existing capped WordNet descendant count. These are signals for the learned ranker, not rules that certify meaning preservation.

The 39 v7 features remain the prefix of each row. Nine appended features produce 48 columns. Candidate generation, the BERT scores, the fit stacker, detector, guardrails and edit threshold remain fixed. The decision model has the same logistic and gradient-boosting members as v7. Pair features use resolved lemmas in both cached and live scoring. Missing provenance raises an error; cases with no candidates remain abstentions.

## Development result

The comparison uses 929 BenchLS cases and 14,617 candidate rows. All BenchLS is reused development, including its historical holdout. Three target-word-grouped 10-fold repeats use seeds 0, 1 and 2, threshold 0.45 and meaning floor 0.10. There is one feature variant and no threshold search. TSAR gold is not read. Substitute labels do not judge retained detail in a complete sentence.

| Metric | v7 | Relation features |
|---|---:|---:|
| Mean correct edits | 425.3 | 438.3 |
| Mean wrong edits | 189.3 | 175.0 |
| Mean net correct edits | 236.0 | 263.3 |
| Pooled substitute precision | 69.2% | 71.5% |
| Change coverage | 66.2% | 66.0% |
| Mean candidate log loss | 0.240508 | 0.237887 |

Net means correct edits minus wrong edits. Net gains were 22, 24 and 36 in the three repeats. The mean gain of 27.3 and mean log-loss reduction of 0.002621 both exceed the existing noise rule, net gain 5 or log-loss reduction 0.001. These repeats share cases and are not independent test sets or confidence intervals.

Selection changes include regressions. Across the three repeats, 7, 6 and 8 previously correct choices became wrong choices. The corresponding wrong-to-correct counts were 17, 14 and 18. The saved paired selections expose those cases rather than reporting only aggregate gains.

The fixed-threshold v7 baseline is 236.0 net. The earlier saved 243.7 result selected a threshold within each repeat. Those numbers use different protocols.

## Diagnostic result

Both models ran on the same 13 saved, ungraded inputs with identity and dictionary baselines. The fresh v7 outputs matched the earlier v7 audit on every input. The candidate changed 12 sentences; v7 changed 11. Four sentence outputs differed.

| Input | v7 edit | Candidate edit |
|---|---|---|
| Company will commence the audit | commence to begin | commence to start |
| Firm may not distribute a dividend | dividend to profit | distribute to give |
| Constitution guarantees due process | constitution to law | guarantees to assures |
| Judge handed down a harsh sentence | unchanged | harsh to rough |

The candidate retained `dividend` and `constitution` in those examples, but the fallback edits still need review. `harsh sentence -> rough sentence` is a weak new edit. It kept `convoluted -> complex` and continued to produce `database -> information`, `encodes -> converts`, `deposited -> gave` and `filibuster -> delay`. The development gain therefore does not justify claiming that detail loss is fixed, or changing the demo default.

Median lexical analyze time was 2.35 seconds for v7 and 2.91 seconds for the candidate. Maxima were 12.56 and 18.28 seconds. Yoga's concurrent workload changed during the serial runs, so these observations are not a controlled latency comparison. No human precision was computed.

Review the outputs in `outputs/relation-features-dev-review-20261002/`, `outputs/domain-audit-v7-relations-check-20261002/` and `outputs/domain-audit-relations-20261002/`. The first failed preflight log remains in `outputs/relation-features-dev-20261002.log`. It exposed cached cases with no WordNet candidates; the corrected feature code handles them without inventing provenance.

The next validation input is still a fixed, attributed sample of the existing team corpus. The checkout has no arXiv, SEC 10-K or GovInfo data. The audit can ingest that sample when it becomes available. In particular, inspect both broader proposals and the words chosen after a rejection. BenchLS substitute labels alone cannot settle whether technical detail survives.

## Reproduce

Run from `models/` with the existing environment and new output paths. Sustained experiments belong in a named Yoga tmux session with four threads and a suitable memory cap.

```sh
.venv/bin/python scripts/experiments/refine_relations.py \
  --output outputs/relation-features-dev-another-run \
  --fit-run runs/ensemble-relations-another-run
```

The script reads only the two BenchLS cached tables and v7 configuration. It saves source copies, input and source hashes, dependency versions, per-repeat candidate probabilities and fold assignments, paired sentence selections, and `summary.json`. It fits an optional separate run only if the noise rule passes. Existing outputs and run directories are never overwritten.

The saved candidate from this comparison is `runs/ensemble-relations-20261002`. Its configuration points at the unchanged v7 components and its own decision model. Use it explicitly:

```sh
.venv/bin/python scripts/audit_domains.py --baselines \
  --run runs/ensemble-relations-20261002 \
  --output outputs/domain-audit-relations-another-run

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/predict_ensemble.py \
  --run runs/ensemble-relations-20261002 \
  'Although this sentence is convoluted, the main idea is simple.'
```

Omitting `--run` keeps the current v7 default. Audit inputs remain ungraded diagnostics, separate from development labels. No human labels, synthetic gold, paid services, pushes or PR updates were used.

The complete local suite passed 54 tests. New tests cover relation mixtures, positive synonym and adjective controls, cached/live feature parity, the unchanged v7 feature prefix, breadth interaction, missing provenance and empty-candidate abstentions. Fatal and undefined-name lint passed. All v7 artifact hashes match the takeover fingerprints. The only integration changes to existing source are the opt-in feature registration and prediction CLI run argument; the earlier uncommitted implementation remains in place.

The subsequent [context-window fix](CONTEXT_WINDOW.md) provides a separate configuration for targets beyond BERT's original input cutoff. It reuses this ranker's weights. BenchLS contexts are too short to exercise that bug, and its 13 short diagnostic outputs match this run.
