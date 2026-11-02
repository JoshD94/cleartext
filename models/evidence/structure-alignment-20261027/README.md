# Structural training alignment evidence

See [the experiment report](../../docs/STRUCTURE_ALIGNMENT.md) for the findings
and reproduction commands.

- `protocol.json` records the policy fixed before preparation and training.
- `preparation-audit.json` records the failed assertion and normalized-copy fix.
- `manifest.json` records target eligibility counts and data fingerprints.
- `seq2seq_config.json` records the training configuration and initializer files.
- `training_freeze.json` links the configuration to the prepared data.
- `environment.json` records software versions, host choice and validation.
- `seq2seq_training.json` records completed exposure, loss and decoded selection.
- `development-comparison.json` records per-epoch reused-development deltas.
- `provenance.jsonl` records source hashes at preparation, training and comparison.
- `resource_summary.json` records sampled capacity and training peak RSS.
- `holdout-use-policy.json` preserves the reserved set if no new weights win.
- `pretest-decision.json` freezes the chosen checkpoint before reserved decoding.
- `dev-comparison/` records the saved-weight verification on reused development.
- `test-comparison/` records the single reserved comparison and its freeze.
- `completion.json` records the experimental-only decision and future exclusions.
- `bundle.json` identifies the local weights bundle, which has not been uploaded.

Only aggregate evidence and fingerprints belong here. Published raw pairs,
decoded text and model binaries stay in the local experiment store. Parser
agreement and reference overlap do not establish grammaticality, preserved
meaning or easier reading.
