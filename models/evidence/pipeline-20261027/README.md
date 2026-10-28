# Pipeline evaluation evidence

The [experiment report](../../docs/PIPELINE_BENCHMARK.md) describes the shared
inputs, stage variants and precision/coverage target.

`data_manifest.json` freezes the 190 input IDs and source-file hashes without
copying corpus text. The v7 precision files record the predeclared protocol,
three grouped development repeats, frozen threshold and reused TSAR comparison.
`contextual_precision_policy.json` records the stronger ranker's separately
selected development threshold and reproduction of its historical selections.

`early_qualitative_review.json` contains coding-assistant diagnostic notes by ID.
They are not independent human labels or an accuracy sample.

Raw datasets, predictions, per-candidate features, model files and logs remain
under `~/.local/state/cleartext/pipeline-eval-20261027/` on Yoga. Aggregate evidence
does not establish 90% full-pipeline accuracy.
