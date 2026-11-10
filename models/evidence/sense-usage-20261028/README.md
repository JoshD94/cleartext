# Sense usage development evidence

The candidate usage feature experiment failed its frozen adoption bar. See [the report](../../docs/SENSE_USAGE.md) for the comparison and candidate availability diagnosis.

- `protocol.json` records the one-variant protocol and the morphology correction made before fitting.
- `summary.json` contains all three grouped development results, paired transition counts and runtime.
- `manifest.json` fingerprints the fitting inputs, code, packages and WordNet archive.
- `verification.json` records baseline replay, unchanged feature columns, resource limits, tests and the preserved failed attempt.
- `candidate-ceiling.json` counts available listed substitutions before and after eligibility checks, with source and input hashes.

These files contain aggregate results and fingerprints. Raw sentences, candidate choices, labels, probabilities, feature tables and model weights stay outside Git. No fitted usage model was retained.
