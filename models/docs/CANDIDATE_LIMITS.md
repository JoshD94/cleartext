# Where the lexical ensemble still has room

The best saved ranker selects about 512 listed substitutes per repeat on 929 reused BenchLS development cases. The current candidate pool and filters permit a listed substitute in 690 cases. This measures agreement with a benchmark list, not retained sentence meaning.

The new audit separates candidate generation, cached safety guards, the meaning floor, acceptance probability and ranking. It replays saved predictions instead of fitting another model. Each target word stays in one fold, and selected words must match the earlier saved selections exactly.

| First blocking stage | Repeat 0 | Repeat 1 | Repeat 2 |
| --- | ---: | ---: | ---: |
| No listed candidate generated | 122 | 122 | 122 |
| All listed candidates fail cached guards | 35 | 35 | 35 |
| Remaining listed candidates fail the meaning floor | 82 | 82 | 82 |
| Remaining listed candidates fall below acceptance probability | 152 | 156 | 152 |
| An unlisted candidate outranks an eligible listed candidate | 25 | 25 | 24 |
| A listed candidate is selected | 513 | 509 | 514 |

The stages partition the cases in policy order. The probability stage can contain an abstention or an unlisted edit. Records keep that outcome separate. A listed substitute blocked by a guard is not proof of a bad guard; an unlisted edit is not proof of a bad replacement.

WordNet alone contains a listed option in 807 cases. Cached guards reduce that to 772, and the meaning floor reduces it to 690. These are conditional ceilings with a perfect selector. They do not predict what a trained model will achieve.

## Bounded candidate review

The existing masked-BERT cache contains 12,326 novel proposals. Its top-20 predictions add a listed option to 88 of the 122 generation misses, raising the unrestricted union ceiling to 895 cases. All novel proposals lack the target-sense links used by the current acceptance policy.

The review chooses the first three unique novel surfaces by cached BERT rank, before looking at benchmark membership. That shortlist still covers 76 of the generation misses. It never accepts proposals or changes a threshold.

For each shortlisted proposal it records:

- The exact rewritten sentence, using the existing article adjustment.
- Predicted CompLex difficulty before and after the replacement.
- Full-span contextual BERT cosine, with a 256-piece window.
- Existing safety checks, differential grammar checks and protected-term overlap.
- Exact WordNet candidate senses and direct same, broader, related, narrower or antonym connections to the source sense distribution.
- The current origin-based meaning-floor result, without inventing target senses.

WordNet relation support counts source-posterior mass for which any candidate sense has that relation. It is not a candidate posterior. Relation totals can overlap. Dictionary glosses are inventory entries, not context-confirmed explanations. Cosine, lower predicted complexity and passing rules do not establish retained meaning.

The evidence records contain no benchmark labels. A separate file joins existing listed-substitute membership after scoring. It creates no new gold labels or human ratings. Three fixed positions compare batched cosine with single-candidate inference.

The first review scored 2,752 proposals in 228 seconds. Of these, 1,787 passed the supported mechanical checks and had lower predicted complexity. They included a listed option for 60 generation misses. There were also 79 proposals with direct WordNet antonym links. High cosine can coexist with an antonym link, as with external to internal. Passing these checks is insufficient for acceptance.

## Dictionary validation result

The audit exposed a morphology bug. WordNet automatically expands `found` to senses of `find`, even when a proposal's exact lemma is `found` and its surface is `founded`. That creates a false shared-sense signal. The proposal generator can also treat an already inflected token as a lemma and produce `consisteds`.

`candidate_validation.py` adds an opt-in `DictionaryValidatedBertGenerator`. It requires an exact WordNet lemma for the target part of speech and a surface form supported by the existing inflection dictionary. It keeps each proposal's original provenance and leaves novel target-origin senses empty. A supported dictionary form is not a semantic judgment. The dictionaries can omit valid words or inflections, so this check can abstain on them.

Replaying all 929 cached cases removed 83 of 12,326 novel proposals. The number of generation misses covered by a listed option stayed at 88. On the earlier top-three shortlist, it removed 25 of 2,752 proposals and kept all 716 listed matches, including the same 76 covered generation misses. The number with a valid, mechanically passing, lower-complexity listed option stayed at 60.

Exact lemma inventories eliminated all apparent same, broader and related sense connections among those novel shortlisted proposals. The earlier signals came from morphological alternatives. The corrected inventory still finds 188 narrower connections and 79 antonym connections. These are dictionary relations, not context-confirmed meaning labels.

The corrected shortlist has 1,780 dictionary-valid proposals that pass the supported mechanical checks and have lower predicted complexity. None is accepted. The current meaning floor still has no target-origin evidence for them. Results and corrected evidence are in `outputs/proposal-validation-20261003`. The earlier review is preserved with its original source snapshot.

## Reproduce

From `models/`, use new output directories:

```sh
.venv/bin/python scripts/experiments/audit_candidate_limits.py --output outputs/candidate-limits-20261003
.venv/bin/python scripts/experiments/inspect_bert_proposals.py --output outputs/proposal-review-20261003
.venv/bin/python scripts/experiments/validate_bert_candidates.py --output outputs/proposal-validation-20261003
```

The second command belongs in a named Yoga tmux job with a 3 GB memory cap and four CPU threads. The validation replay ran in `cleartext-proposal-validation-20261003` with a 2 GB cap and no encoder inference. It reuses the earlier cosine and mechanical checks and recomputes dictionary evidence. All commands save input hashes, source snapshots and records. They use existing local data and models, with no downloads, TSAR tuning or model promotion. The v7 demo remains unchanged.

The cached cosine measures insertion into the original slot before the article adjustment. The recorded output and mechanical checks use the adjusted sentence. This difference matters for article-changing edits. The validation manifest records that scope explicitly.

The next model change needs a defensible way to score novel candidates' meaning. Assigning them invented WordNet origins or lowering the meaning floor would hide the missing evidence. The probability bottleneck also deserves investigation, but maximizing BenchLS list matches alone could worsen the detail loss already seen on technical text. Existing local data has no independent labels for those complete edits.
