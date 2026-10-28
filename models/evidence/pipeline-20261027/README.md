# Pipeline evaluation evidence

The [experiment report](../../docs/PIPELINE_BENCHMARK.md) describes the shared
inputs, stage variants and precision/coverage target.

`data_manifest.json` freezes the 190 input IDs and source-file hashes without
copying corpus text. The v7 precision files record the predeclared protocol,
three grouped development repeats, frozen threshold and reused TSAR comparison.
`contextual_precision_policy.json` records the stronger ranker's separately
selected development threshold and reproduction of its historical selections.
`contextual_precision_test.json` records all 373 reused TSAR cases and the
three fixed native-ranking checks. Its freeze and runtime files identify the
policy, model, cache and package versions. The strict policy matches a published
substitute in 80 of 84 accepted edits, with 22.5% coverage. This is not a
full-sentence accuracy result.

The `bounded-*` files record numeric equivalence for masked-position vocabulary
scoring, agreement with the 81 saved partial outputs, and agreement between the
experimental function and production code. Initial model-load timing is kept
explicit and is not a speedup estimate.

`early_qualitative_review.json` contains coding-assistant diagnostic notes by ID.
They are not independent human labels or an accuracy sample.

Raw datasets, predictions, per-candidate features, model files and logs remain
under `~/.local/state/cleartext/pipeline-eval-20261027/` on Yoga. Aggregate evidence
does not establish 90% full-pipeline accuracy.
