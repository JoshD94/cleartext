# Eligibility checks and entity alignment

This follow-up traces the 117 development cases that lost their last listed substitute to a candidate guard or the meaning floor. It fixes an entity-role comparison bug and tests a separate proposal to remove the meaning floor. The floor proposal failed its frozen precision requirement and was rejected.

## Tracing the losses

The earlier candidate availability audit counted 35 cases with a generated listed substitute but no guarded listed substitute, and another 82 with a guarded listed substitute below the 0.10 sense-fit floor. All 929 BenchLS cases are reused development data, including the historical holdout. Listed substitutes are the benchmark's existing annotations, not new judgments about semantic correctness.

The native audit traces every listed candidate in those 117 cases. All 82 meaning-case scores reproduce exactly from their cached full sense distributions and WordNet provenance. Their low scores are real outputs of the current scorer. They are not a row alignment or cache disagreement. This does not establish that every rejection is justified.

The guard losses involve changes in dependency parses and named-entity recognition. One historical guard case already passes the current legacy guard because of an earlier copula alignment change. The paired comparison therefore evaluates both implementations freshly in the same runtime and reports historical cache differences separately.

## Entity-role bug

The old entity-role check looked up each original name in the candidate's newly recognized entities. An unchanged organization could become a facility label or disappear from recognition after a nearby word replacement. The check then rejected it even when its words, subject or object role, and predicate attachment were identical.

The old dictionary also keyed entities by their text, collapsing repeated occurrences of the same name. An earlier occurrence's changed role could escape the check if the final occurrence kept its role.

For equal token counts, the corrected check follows each original PERSON, ORG and GPE occurrence at its token span. It requires the same words, dependency label and head position. New entity recognition does not determine whether that original occurrence kept its role. Different token counts retain the previous conservative path. Argument roles, entity text retention, numbers, quantities, negation, modality and technical phrase checks remain separate.

Seven deterministic tests cover changed or missing entity labels, repeated names, role swaps, changed predicate attachment, internal name heads and the guard's use of original entity anchors. These are mechanism fixtures, not semantic training labels. An older detail-policy test now exercises its enabled guard path and explicitly checks the existing no-guard ablation behavior. All 101 related checks pass on the final comparison source, including the seven entity checks. The suite covers entity alignment, invariants, structural changes and detail preservation.

## Paired guard results

The complete comparison covers 929 cases and all 14,617 candidates. The correction restores 46 candidates across seven cases and blocks no previously passing candidate in this pool. All 46 previously failed only `entity_roles`. Every other guard check agrees exactly. This verifies the change's scope, not the semantic correctness of all restored words.

Fresh legacy guards disagree with 50 historical cached candidate pass flags. Those differences are excluded from the correction's attribution by comparing both implementations freshly on identical parses. The fresh listed-candidate ceiling rises from 773 to 776 cases after guards, and from 691 to 694 after the unchanged meaning floor.

| Mean across three repeats | Fresh legacy guard | Corrected entity guard |
| --- | ---: | ---: |
| Default listed matches | 512.0 | 514.0 |
| Default unlisted edits | 129.7 | 129.0 |
| Default listed-substitute precision | 79.79% | 79.94% |
| Default edit coverage | 69.07% | 69.21% |
| Default correct minus incorrect edits | 382.33 | 385.00 |
| Strict listed matches | 271.0 | 272.0 |
| Strict unlisted edits | 28.0 | 28.0 |
| Strict listed-substitute precision | 90.63% | 90.67% |
| Strict edit coverage | 32.19% | 32.29% |

Default net gains are 3, 2 and 3. Each strict repeat gains one listed edit without adding an unlisted edit; corrected strict precisions are 90.88%, 90.78% and 90.33%. Only two distinct cases change their selected word. None changes from a listed word to an unlisted word. These are dependent development repeats with the target supplied, not fresh tests or full-pipeline accuracy estimates.

We retain the entity comparison correction for its tested mechanism and modest paired benefit. No new model is fitted. The missing-candidate bottleneck remains larger: 122 cases have no generated listed substitute. After this correction, guards remove the last listed option for 31 cases and the meaning floor removes it for another 82. The next candidate-generation experiment should keep these stages separate and use independent evaluation before claiming broader gains.

## Meaning-floor ablation

We tested one fixed alternative: remove the 0.10 sense-fit floor while preserving the candidate pool, candidate guards and contextual ranker's saved out-of-fold probabilities. We used the two existing thresholds, 0.45 and 0.80, with all three target-word-grouped 10-fold repeats. No model was retrained and no threshold grid was searched.

| Mean across three repeats | Existing default policy | Default with no floor |
| --- | ---: | ---: |
| Listed matches | 512.0 | 539.0 |
| Unlisted edits | 129.7 | 132.3 |
| Listed-substitute edit precision | 79.79% | 80.29% |
| Edit coverage | 69.07% | 72.26% |
| Correct minus incorrect edits | 382.33 | 406.67 |

The default net gains were 25, 22 and 26 edits. However, at threshold 0.80, no-floor precision was 90.28%, 90.20% and 89.81%. The frozen retention rule required at least 90% precision and 20% coverage in every strict repeat, as well as a default mean net gain of five edits. The third repeat failed. The average increase in listed matches does not justify retaining this policy under that rule.

No floor setting or model configuration changed as a result. These development comparisons do not measure full-sentence accuracy. They also do not replace the earlier reused TSAR result for the existing strict policy.

## Execution and reproduction

The 117-case audit completed on the Mac. Two later Mac comparison attempts stopped when unrelated simulator work pushed memory pressure to warn. The first saved 65 cases; the second stopped before saving a case. Both attempts remain outside Git. Neither partial comparison enters the final metrics.

The complete comparison ran on Yoga in `cleartext-entity-queued-20261028`, after another heavy job finished. Its scope capped memory at 1.5 GiB and CPU at one core. Sampled available host memory stayed above 4.89 GiB. It evaluated freshly parsed old and corrected guards against identical documents, with an assertion that every other guard check agrees. Saved contextual probabilities, the 0.10 floor and the two existing decision thresholds remained fixed. All 101 related tests passed in 19.14 seconds; fatal/undefined-name lint passed. The active demo was preserved.

Aggregate evidence and input/source fingerprints are in [the evidence directory](../evidence/eligibility-audit-20261028/). Raw cases, cached probabilities, chosen words, parse traces and failed-attempt logs remain outside Git in `/home/joshuadirga/.local/state/cleartext/eligibility-audit-20261028`.

Run from the repository root with the existing local data caches. Choose new output paths and use a named tmux job after checking resources:

```sh
~/.local/bin/personal git show d12087a:models/src/cleartext/lexical.py > /path/to/legacy-lexical.py
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/compare_entity_guards.py \
  --export-only --output /path/to/candidates.json
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/compare_entity_guards.py \
  --input /path/to/candidates.json --legacy-source /path/to/legacy-lexical.py \
  --output /path/to/guard-comparison
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/evaluate_entity_guards.py \
  --comparison /path/to/guard-comparison --output /path/to/guard-evaluation
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/ablate_meaning_floor.py \
  --protocol models/evidence/eligibility-audit-20261028/floor-protocol.json \
  --output /path/to/floor-ablation
```

The native scan needs the installed spaCy English model. The loss audit also needs the existing WordNet corpus and sense-distribution cache. `audit_eligibility_losses.py --export-only` exports the 117 historical losses; native mode accepts that file with `--input` and a new directory with `--output`.
