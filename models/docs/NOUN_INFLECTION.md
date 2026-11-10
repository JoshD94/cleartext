# Noun-phrase candidate inflection

The existing generator inflects the first word of every WordNet phrase. That works for many verbs, but it can pluralize a noun modifier and leave the head singular. For example, `musical_composition` becomes `musicals composition` and `computer_scientist` becomes `computers scientist` when replacing a plural noun. Singular inflection can also damage a modifier such as `arms` in `arms race`.

The opt-in generator inflects the noun head for common-noun tags `NN` and `NNS`. It uses the existing parser on the phrase, WordNet's noun exceptions and the installed inflection lexicon. It preserves modifiers, capitalization, candidate provenance and target senses. Single words and phrases with other tags keep the existing inflection behavior.

WordNet's exceptions handle compounds such as `court martial -> courts martial`. When the parser cannot identify one noun head, or an isolated parse leaves a postpositive adjective ambiguous, the generator drops the proposal. It also drops plural proposals when the head meets the existing frequency-based mass-noun heuristic. That heuristic can reject rare count nouns; it is not a grammatical authority. A parser can still choose the wrong head, so this change does not guarantee correct grammar for every compound.

Set `noun_phrase_heads` to `true` in a separate run configuration to enable the generator. The default generator and current v7 configuration keep their existing behavior. Existing model files and the demo are preserved.

Run the bounded comparison from `models/` with new output and run paths. Long model work belongs in a named Yoga tmux job after checking resources.

```sh
.venv/bin/python scripts/experiments/check_noun_inflection.py \
  --output outputs/noun-head-another-check \
  --base-run runs/ensemble-windowed-20261002 \
  --save-run runs/ensemble-noun-heads-another-run

.venv/bin/python scripts/audit_domains.py --baselines \
  --run runs/ensemble-noun-heads-another-run \
  --output outputs/domain-audit-noun-heads-another-run
```

The comparison uses all 929 BenchLS cases as reused development, including the historical holdout. It reuses frozen candidate scores when the proposal and resolved target match the cache. It scores changed proposals with the same frozen components and checks five cached rows against fresh scores. Both policies use the same target-word folds in three ten-fold repeats, the deployed threshold of 0.45 and meaning floor of 0.10. Candidate order follows the cached lemma and provenance order, including renamed forms. This avoids changing tied rank features because WordNet returned its edges in a different order. The decision model is fitted separately for each policy and repeat. There is no threshold search, new gold or TSAR evaluation.

Outputs include proposal changes, candidate tables, fold assignments, held-out probabilities, selected words, source snapshots and hashes. A separate experimental run contains the new decision model. Saving that run does not promote it to the default. Candidate populations differ, so their log-loss values are descriptive. Paired edit outcomes are the useful benchmark comparison.

BenchLS substitute labels do not grade phrase grammar or full-sentence meaning. The original cached development tables contain 934 multiword noun proposals across 185 cases, including 242 plural proposals. None of the 934 match an existing gold substitute. Those counts explain why a grammar fix may have little effect on this benchmark. They do not establish that every phrase is wrong.

The tests cover modifier preservation, noun number, capitalization, prepositional heads, a WordNet compound exception, uncertain heads, mass-noun rejection, unchanged verbal and single-word behavior, configuration routing and candidate provenance. A comparison regression test demonstrates that different enumeration order changes tied rank features, then verifies that alignment removes that change.

## Development result

The completed comparison reused 14,307 candidate scores and scored 203 new forms. It changed proposals in 103 cases and required no refresh of baseline scores. The five fresh cache checks agreed within 0.000005 in fit features. Candidate rows fell from 14,617 to 14,510. Both variants retained a listed gold candidate in 807 cases, so this change added no gold coverage on BenchLS.

| Measure | Existing relation model | Noun heads, decision refitted |
| --- | ---: | ---: |
| Net correct minus wrong, seed 0 | 269 | 263 |
| Net correct minus wrong, seed 1 | 264 | 255 |
| Net correct minus wrong, seed 2 | 258 | 256 |
| Mean net | 263.7 | 258.0 |
| Pooled precision | 71.5% | 71.1% |
| Edit coverage | 66.1% | 65.7% |
| Mean candidate log-loss | 0.237885 | 0.240065 |

The mean net loss is 5.7 edits per repeat. The change does not clear the existing adoption bar of net +5 or log-loss -0.001. Candidate populations differ, so the log-loss comparison does not establish a calibration change. The baseline is within one net edit of the earlier relation experiment across all three repeats. This is reused development, not a fresh test result.

The 13-input domain audit also gives a reason to reject this refitted model. Its output changes `She deposited the money.` to `She gave the money.`, where the previous windowed relation run kept the sentence. The generator's verbal inflection is unchanged; refitting the decision model changes decisions outside the noun phrases. The other 12 outputs match the previous run, including known detail-loss failures. The audit uses ungraded diagnostics, and 13 changed sentences does not mean 13 correct simplifications.

Keep this generator and refitted run opt-in. The code corrects tested grammatical forms, but this evaluation supports no model-accuracy claim or default promotion.

Review `outputs/noun-head-dev-review-20261002/` and `outputs/domain-audit-noun-heads-review-20261002/`. The experimental saved run is `runs/ensemble-noun-heads-review-20261002`. The earlier run remains separate. It reordered 110 baseline cases without changing their scores, which affected tied rank features; the completed comparison above corrects that ordering issue.

## Frozen decision check

The follow-up isolates the generator. In each development fold, it trains the decision model only on baseline candidates and uses that same model to predict both candidate tables. It verifies the prior table inputs and sources before reading them. The saved run shares the previous relation decision weights and fit stacker; it fits no new deployed weights.

The baseline nets are 269, 264 and 258. The noun-head nets are 268, 264 and 257, a mean change of -0.7. Pooled precision is 71.5% for both variants, and coverage changes from 66.1% to 66.0%. This also fails the adoption threshold. It is a nearly neutral development result for the isolated generator, not evidence of higher accuracy.

All 13 diagnostic outputs match the previous windowed relation run. The `deposited -> gave` regression introduced by refitting is absent. Known detail-loss failures still remain. Prefer this frozen-weight configuration when inspecting the grammar fix, and keep the current v7 default unchanged.

```sh
.venv/bin/python scripts/experiments/check_frozen_noun_heads.py \
  --tables outputs/noun-head-dev-review-20261002 \
  --output outputs/noun-head-frozen-another-check \
  --save-run runs/ensemble-noun-heads-frozen-another-run
```

The completed configuration is `runs/ensemble-noun-heads-frozen-20261002`. Evidence is in `outputs/noun-head-frozen-dev-20261002/` and `outputs/domain-audit-noun-heads-frozen-20261002/`. All comparison and audit jobs exited successfully. The full test suite passes with 65 tests, and fatal/undefined-name lint passes.
