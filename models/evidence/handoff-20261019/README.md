# Saved experiment evidence

These files preserve local settings and aggregate results captured on October 19, 2026. They are historical receipts from the existing runs, not results from a new experiment.

- `runs/` contains copies of run settings, training summaries, selection comparisons and validation results.
- `outputs/` contains copies of completed experiment summaries.
- `sources.json` records the original relative path and SHA-256, the copied file's SHA-256, and fields excluded from each copy.
- `local-artifacts.json` records paths, sizes and SHA-256 hashes for all 179 original local run files, including model binaries. It does not provide remote storage for those binaries.

Per-example benchmark material is excluded from these copies. Excluded fields can include predictions, source text, sample outputs and diagnostic case records. Scalar counts remain. The original files are unchanged on Yoga under `models/runs/` and `models/outputs/`.

Use [../../docs/HANDOFF-20261019.md](../../docs/HANDOFF-20261019.md) for branch history, outstanding Google Doc tasks and evaluation limits. The live loader still reads the original local `runs/` tree; these copies are not loader inputs.
