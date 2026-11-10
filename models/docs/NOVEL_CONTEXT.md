# Novel BERT context validation

The context validator is implemented as a separate prototype. Both fitted versions failed the acceptance requirement, so the saved safe configurations disable novel replacements. The demo and the saved best ensemble remain unchanged.

## Training data and signals

`train_novel_context.py` reuses existing SWORDS votes for single-word candidates outside the cached WordNet generator. A positive label means at least half of the existing votes are `TRUE`. These labels concern contextual appropriateness. They do not measure simplicity or full detail retention.

The selection contains 15,270 pairs, including 1,145 positives, across 370 contexts. A fixed context-group split assigns 12,625 pairs to training and 2,645 to calibration. Exact normalized BenchLS sentence overlap is excluded. The existing SWORDS test cache contributes no retained non-generated pairs, so these training pairs all come from SWORDS dev. Both historical SWORDS splits were previously used elsewhere in fitting. This is reused development data.

The seven inputs are context embedding similarity, argument fit, bigram fit, BERT slot score, BERT slot score relative to the original, source-candidate embedding similarity and missing candidate embedding. The validator excludes the old learned SWORDS checker and WordNet membership features. Three native checks reproduce the cached training features within 0.00001.

## Acceptance requirement

Before fitting, the policy required at least 20 calibration acceptances with at least 90% observed precision. Threshold selection scans 0.50 through 0.975 in steps of 0.025 and chooses the qualifying threshold with most acceptances. This is a development selection rule, not a confidence guarantee.

| Validator | Calibration AUC | Log loss | Acceptances at 0.50 | Precision at 0.50 | Safe threshold |
| --- | ---: | ---: | ---: | ---: | --- |
| Scaled logistic regression | 0.6983 | 0.2760 | 2 | 50% | Disabled |
| One boosted model | 0.7610 | 0.2589 | 8 | 12.5% | Disabled |

The boosted model uses the same saved features and split. Its settings are fixed at 150 iterations, learning rate 0.05, 15 leaves, minimum 20 samples per leaf and regularization 10. No additional search followed this attempt.

## Experimental fallback

`NovelCandidateClearText` preserves any edit chosen by the existing pipeline. It considers novel candidates only after the whole sentence receives no edit. The fallback also requires a supported dictionary lemma and inflection, positive predicted complexity reduction, passing mechanical checks and less than 0.10 source-posterior antonym support. These conditions can miss valid edits and cannot certify meaning.

The paired comparison uses a fixed 0.50 shadow threshold declared before looking at BenchLS results. This threshold failed the calibration requirement. A shadow run requires explicit `allow_shadow=True`; its configuration forbids deployment. The safe run retains a null threshold. Both paths consider the first three distinct novel proposals by raw BERT rank before dictionary validation. A rejected proposal does not refill the shortlist.

The comparison reuses all 929 BenchLS development cases and the saved probabilities from three target-word-grouped ten-fold repeats. It retains existing selections exactly and tests the fallback only on abstentions. Listed-substitute membership is joined after feature calculation and selection. An unlisted edit is not necessarily wrong, and a listed edit does not establish full detail preservation.

| Repeat | Base listed/edits | Shadow listed/edits | Base net | Shadow net | Net change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 513/642 | 527/672 | 384 | 382 | -2 |
| 1 | 509/637 | 526/672 | 381 | 380 | -1 |
| 2 | 514/646 | 528/677 | 382 | 379 | -3 |

The shadow adds 30, 35 and 31 edits, respectively. Their listed-match precision is 46.7%, 48.6% and 45.2%. Mean net change is -2.0, and mean total precision falls by 1.56 percentage points. Three native batch checks reproduce the novel features within 0.00001. The disabled safe gate matches the base selections in all repeats. Neither the calibration requirement nor the paired adoption requirement passes. No model is promoted.

## Outputs

- Logistic training: `outputs/novel-context-training-20261003`
- Boosted training: `outputs/novel-context-boosted-20261003`
- Paired comparison: `outputs/novel-fallback-comparison-20261003`
- Complete native domain audit: `outputs/novel-fallback-domain-review-20261003`
- First audit attempt: `outputs/novel-fallback-domain-20261003`

The native audit uses the 13 existing authored and owner diagnostics plus 12 existing Europarl sentences. The Europarl text comes from reused CompLex component test data. Actual arXiv, SEC 10-K and GovInfo samples remain missing. All outputs are ungraded; no new human ratings or synthetic semantic labels were created. TSAR was not used.

The completed audit matches all 25 base outputs with the disabled safe gate. The base changes 17 sentences; the shadow changes 18. Every existing edit is retained. The only additional shadow edit is the database case below. All three variants pass the automatic output, grammar and consistency checks, which leaves the observed meaning risks unresolved. Maximum retained-score roundoff is 0.000000000000000111.

The shadow changes `database` to `library` in the first diagnostic. Its context probability is 0.5639, predicted complexity gain is 0.0254, and its dictionary, mechanical and antonym checks pass. This changes the technical reference despite those checks. It supports rejecting the fallback, without turning the inspection into a new training label.

Existing edits also raise detail concerns, including `encodes -> converts`, `classify -> separate`, `dividend -> profit`, `filibuster -> delay` and `constitution -> law`. The base also chooses `reduces -> decreases` with predicted complexity gain -0.0698. These outputs remain available in the trace rather than being presented as verified simplifications.

The first audit stopped after six saved records because an unchanged edit had utility values differing by 0.000000000000000111. A separate native replay confirmed identical output and edit choices. The completed audit requires exact text and edit identity and permits at most 0.000000000001 score roundoff. Regression tests reject changed text, changed replacements and larger score drift.

The next bounded experiment should add the existing full-span BERT replacement similarity signal to this validator. Keep the grouped SWORDS split, acceptance requirement and frozen BenchLS comparison fixed. The current seven signals fail to distinguish enough acceptable novel candidates; lowering the gate would admit the observed detail loss.

## Full-span similarity experiment

Joshua authorized this experiment after recovery. The prototype now accepts an explicitly configured eighth input, `replacement_similarity`. It averages the existing BERT encoder's last four layers over every piece of the original and replacement spans. The original seven-feature configuration still works. Loading rejects reordered columns and a model whose width differs from its configuration.

`train_novel_context_similarity.py` reuses the exact 15,270 saved pairs, labels, context groups, training indices and calibration indices. It appends the new signal and fits one boosted model with the same settings as the previous attempt. The cache saves each completed context with substitute IDs, offsets and scores; `--resume` requires matching source and input fingerprints. It uses the already downloaded encoder offline.

The acceptance requirement stays at 90% observed calibration precision with at least 20 acceptances. The 0.50 shadow threshold remains fixed and diagnostic only. The paired comparison reuses the seven saved raw columns and all three frozen baseline prediction arrays. Adoption checks the calibrated safe variant for mean net gain of at least five and precision at least equal to the baseline. The domain audit reuses verified baseline outputs and runs the two new variants on the same 25 ungraded cases. Meaning correctness remains unmeasured.

The new outputs are `outputs/novel-context-similarity-20261003`, `outputs/novel-similarity-comparison-20261003` and `outputs/novel-similarity-domain-20261003`. The new run is `runs/novel-context-similarity-20261003`. Saved prior runs and the default demo remain intact. The private job directory records tests, settings, stage exits and final verification. No automatic model promotion is enabled.

The completed fit raises calibration AUC from 0.7610 to 0.7928 and lowers log loss from 0.2589 to 0.2526. At 0.50 it accepts 44 calibration pairs with 54.5% observed precision. No threshold meets the unchanged requirement, so the safe gate stays disabled.

| Repeat | Base listed/edits | Similarity shadow listed/edits | Net change |
| --- | ---: | ---: | ---: |
| 0 | 513/642 | 574/749 | +15 |
| 1 | 509/637 | 573/750 | +15 |
| 2 | 514/646 | 574/752 | +14 |

Mean shadow net gain is 14.67, while total listed-match precision falls by 3.34 percentage points. Safe selections exactly match the frozen baseline. The shadow trades precision for more edits and fails the adoption requirement. These remain reused development results.

The native audit stopped after 22 saved cases on a score assertion. The failing case had passed exact text and edit-identity checks, but the numeric difference was not saved. A fresh replay of that case reproduced every saved baseline score exactly with both new variants. The failed attempt remains intact; its numeric cause is unresolved. The continuation preserves the 22 completed records, reuses that strict native replay, and computes only the final two cases. Score tolerance remains 0.000000000001. New audits save native values before assertions to make future failures inspectable.

The complete audit is `outputs/novel-similarity-domain-continuation-20261003`. The safe gate matches all 25 baseline outputs and retains all 17 existing changed sentences. The shadow changes 19 sentences, adding `unfortunately -> sadly` and `duly -> properly`. It retains `database` in the diagnostic that the earlier shadow changed to `library`. All variants pass automatic checks; their meaning correctness remains ungraded. Existing base detail-loss edits remain unresolved. No model is promoted. All 136 tests pass, including strict recovery tests that reject changed edits and score drift.

## Validation and jobs

All 132 tests pass. Ruff checks for undefined names and syntax errors pass, as does `git diff --check`. The final verification replays the existing SWORDS vote labels and cached feature columns, checks disjoint context groups, reproduces calibration probabilities, replays all three frozen baseline selections and checks native output retention. The original source archive, v7 artifacts and saved best artifacts match their preservation fingerprints. The demo responds with HTTP 200.

Jobs ran on Yoga with four CPU threads. Training, paired comparison and domain audit used a 3 GB memory cap; the cached boosted fit used 2 GB. The completed sessions are `cleartext-novel-context-20261003`, `cleartext-novel-boosted-20261003`, `cleartext-novel-fallback-20261003` and `cleartext-novel-domain-review-20261003`. The domain audit took 511 seconds for three native variants across 25 cases. This is shared-host batch time, not a controlled inference benchmark.

The private handoff directory contains `novel-context-verification.json`, job logs, checkpoints and `verify-novel-round.py`. No push, PR update, agent launch or model promotion occurred.
