# Eligibility audit evidence

The entity-role correction restores 46 candidates across seven cases. With frozen contextual scores, default selections gain two listed matches per repeat and strict selections gain one. The separate proposal to remove the meaning floor failed its frozen strict precision requirement and was rejected. See [the report](../../docs/ELIGIBILITY_AUDIT.md).

- `loss-audit-summary.json` and `loss-audit-manifest.json` record the 117-case native audit and exact reproduction of all 82 meaning scores.
- `floor-protocol.json`, `floor-summary.json` and `floor-manifest.json` record the fixed no-floor experiment, all six policy comparisons and its failed retention bar.
- `guard-protocol.json`, `guard-summary.json` and `guard-manifest.json` record the complete paired guard scan, historical cache disagreements and identical nonentity checks.
- `policy-summary.json` and `policy-manifest.json` record the six unchanged-score policy comparisons and candidate ceilings.
- `source-freeze.json` fingerprints the production code, experiments and related tests before execution.
- `verification.json` records completion, resource limits, all 101 tests and the preserved failed or stopped attempts.

All 929 BenchLS cases are reused development data with the target supplied. Listed matches use existing benchmark annotations. These results do not measure full-sentence accuracy. Raw sentences, annotations, chosen words, probabilities and model weights remain outside Git. No new model, decision threshold or meaning-floor setting was retained from this audit.
