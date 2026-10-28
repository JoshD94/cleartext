# Structural quality evidence, October 27

[The experiment report](../../docs/STRUCTURE_QUALITY.md) explains the input/output
audit, shared discourse checks, training-filter revision and mixed result.

`summary.json` records the result and decision. `data_manifest.json`,
`training_freeze.json`, `protocol.json` and `source_snapshot.json` fix the inputs,
source code and training choices. `training.json` and `comparison.json` contain
aggregate scores on the complete, unchanged development panels.

`baseline_review.json` and `posttraining_review.json` preserve the coding
assistant's qualitative judgments by id, without corpus text. These are not
independent human ratings. `baseline_reproduction.json` isolates the three
check-only changes. `saved_weight_verification.json` records exact reproduction
of all 96 broader raw and returned outputs after reloading the selected weights.

`filter_revision.json` explains why the first attempt was stopped after 80 updates.
`reference_compatibility.json` records disagreement between published references
and the stricter shared checks. `resources.json` describes the completed Yoga job.
`bundle.json` identifies a local weights bundle, which has not been uploaded.

No raw public corpus rows, decoded outputs or model weights are stored here.
Those remain in the local experiment directories named in the report. Keep all
197 development inputs in future comparisons and exclude this lineage's data
and reserved sets before claiming new inputs.
