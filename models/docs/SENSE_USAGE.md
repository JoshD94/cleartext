# Candidate sense usage experiment

The new learned ranker did not improve the existing contextual model. Its mean net score fell by 1.33 edits across three grouped development repeats, and candidate log loss increased by 0.000527. No fitted run was retained. The experiment code remains available for reproduction.

## What we trained

WordNet can put two words in the same sense even when that reading is uncommon for one of them. For example, `teach.v.01` includes `learn`, an uncommon reading that can reverse the apparent roles in an ordinary replacement. A shared dictionary sense alone cannot establish a safe substitution.

We appended nine features to the frozen 52-column contextual model. They describe candidate sense usage priors, related sense mass, ambiguity, the difference between source and candidate priors, and overlap between the verbs' sentence patterns. The priors use WordNet lemma counts with add-one smoothing. Every contextual target sense contributes according to its cached probability. These signals contain no new correctness labels or word-specific bans.

WordNet expands morphological ambiguity when looking up senses. `found` includes readings of both `found` and `find`; `bound` includes readings of `bound` and `bind`. The feature code follows those same analyses when resolving usage counts and verb frames. The first attempt stopped before fitting because it required literal lemma membership. Its manifest and source snapshot remain preserved locally. Nine feature tests cover the correction, rare readings, canonical sense names, invalid distributions and the unchanged 52-column prefix.

## Comparison

All 929 BenchLS cases, including the historical holdout, are reused development data. The fixed pool contains 14,617 candidates and 511 target-word groups. We fitted one variant using three target-word-grouped 10-fold repeats, threshold 0.45 and meaning floor 0.10. Both models used the same learner settings and folds. Refitting the baseline reproduced every stored fold probability within 1e-12. The prior 52 feature columns match exactly.

| Mean across three repeats | Contextual baseline | With usage features |
| --- | ---: | ---: |
| Edits matching a listed substitute | 512.0 | 511.0 |
| Edits absent from the substitute list | 129.7 | 130.0 |
| Listed-substitute edit precision | 79.79% | 79.72% |
| Edit coverage | 69.07% | 69.00% |
| Correct minus incorrect edits | 382.33 | 381.00 |
| Candidate log loss | 0.213591 | 0.214118 |

Net changes were 0, -1 and -3. Log loss increased in every repeat. The frozen adoption bar required a mean net gain of five edits or a log-loss decrease of 0.001. Neither condition passed. No threshold search, TSAR labels, reserved domain predictions, new human ratings or generated gold entered this comparison.

These dependent development results do not measure full-sentence accuracy. An unlisted substitute counts as incorrect here but is not necessarily a semantic error. The result supports rejecting this particular feature addition, not a claim that sense usage can never help.

## Candidate availability

A separate descriptive audit counted the cases where a listed substitute remains available:

| Stage | Cases out of 929 |
| --- | ---: |
| Any candidate generated | 918 |
| A listed substitute generated | 807 |
| A listed substitute passes the candidate guard | 772 |
| A listed substitute also passes the meaning floor | 690 |

The pool lacks a listed substitute for 122 cases. The guard removes the last listed option for another 35, and the meaning floor removes it for another 82. Consequently, even an ideal selector cannot choose a listed substitute on more than 74.3% of these cases with the current pool and eligibility rules. This bounds coverage of listed matches, not attainable precision among a smaller set of edits.

The next useful investigation is a paired audit of those 35 guard losses and 82 meaning-floor losses. It should distinguish defective inflection or sense scoring from legitimate rejection before relaxing any checks. Missing candidates also remain a separate bottleneck. This experiment gives no evidence for adding more dictionary columns to the current ranker.

## Execution and reproduction

The successful fit ran on Yoga in `cleartext-sense-usage-r2-20261028`, capped at 2 GB and two CPU threads. It took 78.8 seconds and peaked at 636 MiB resident memory. The Mac was checked; Yoga fit the placement rules. The related 29 tests and fatal/undefined-name lint passed. The demo still responds with HTTP 200.

Aggregate results, protocol and fingerprints are in [the evidence directory](../evidence/sense-usage-20261028/). Raw cached cases, probabilities, feature tables and source snapshots remain outside Git in `/home/joshuadirga/.local/state/cleartext/sense-usage-20261028`. The failed first attempt is `development`; the successful comparison is `development-r2`.

Run from the repository root after resource checks, in a named tmux job. Use new output and run paths:

```sh
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/refine_sense_usage.py \
  --protocol models/evidence/sense-usage-20261028/protocol.json \
  --output /path/to/new-comparison \
  --save-run models/runs/new-sense-usage-run
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/audit_candidate_ceiling.py \
  --output /path/to/new-candidate-ceiling.json
```

The comparison verifies its frozen cache hashes before fitting. It retains a run only if the adoption bar passes. There is no production loader registration for this rejected variant.
