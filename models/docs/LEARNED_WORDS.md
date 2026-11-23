# Learned word stage

A trained word simplifier (arm p50 is on by default since November 23; see [DEFAULT_PIPELINE.md](DEFAULT_PIPELINE.md)) with four stages: eligibility
(complex word and term protection), candidates, ranker with a no-swap gate,
and per-edit checkers for swaps and splits. It uses no LLM. Code is in
`src/cleartext/learned_words/`; the study driver is `evals/learned_words.py`,
run through `scripts/learned-words`:

```sh
scripts/learned-words featurize --max-seconds 1500   # repeat until COMPLETE
scripts/learned-words nli train                     # NLI for checker rows
scripts/learned-words train                         # writes train.json; first run nli train-top if refused
scripts/learned-words nli test && scripts/learned-words evaluate
scripts/learned-words splits featurize && scripts/learned-words splits fit
scripts/learned-words apply && scripts/learned-words page
```

Results, protocol, data licenses and caveats are in
[evals/reports/20261118-learned-words/](../../evals/reports/20261118-learned-words/README.md).
Downloads are parsed as data only (`python -I`) from `CLEARTEXT_LEXSIMP_RAW`.
They are not committed: several have no license or a non-commercial one.
