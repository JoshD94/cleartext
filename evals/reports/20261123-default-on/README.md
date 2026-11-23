# Default pipeline switch, November 23, 2026

Joshua approved turning on what passed (01:19 EST, Nov 23): F2 → structure
v3 (`KEPT_V3`, 0.5 floor) → learned words p50. Description, results and
caveats: [models/docs/DEFAULT_PIPELINE.md](../../../models/docs/DEFAULT_PIPELINE.md).
Aggregate dev results: [judge-dev.json](judge-dev.json).

- Queue jobs (Yoga, `later`): check + dev `553071c6928e`, `a3675385e88b`;
  score + judge `6c198edca340`.
- State (raw text, outputs, memos): `~/.local/state/cleartext/default-20261123/`
  (`env.sh`, `check_dev.sh`, `score.sh`). Structure v3 dev outputs and their
  metrics were copied from `structure-20261122/` (arm `combo-v3`).
- Nothing was tuned on these results; no rated item was generated.
