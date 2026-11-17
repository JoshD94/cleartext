# Test set v3 (manifest only)

300 sourced passages, 60-120 words, selected by `evals/build_testset_v3.py`
(seed 20261116, no LLM anywhere). Six content domains of 42 passages plus 48
`simple_control` passages from Simple English Wikipedia, where the right
output is a near no-op.

- `manifest.json`: per passage ID, domain, kind, split, source, licence,
  sentence span, word count and the SHA-256 of its text. No passage text.
- `dev_ids.txt` (100) and `test_ids.txt` (200): document-level split,
  stratified by domain, one third dev. Committed before any system was
  scored on this panel.
- The **test split is locked**: no system may run on it until the one-shot
  final run in the pre-registration
  (`evals/reports/20261116-testset-v3/PROTOCOL.md`). `ensemble_study freeze`
  refuses a panel containing test IDs unless `--unlock-test` is passed.

Rebuild the local text (outside Git) and check it against the manifest:

```bash
python -m evals.build_testset_v3 build --state ~/.local/state/cleartext/testset-v3-20261116
python -m evals.build_testset_v3 verify --state ~/.local/state/cleartext/testset-v3-20261116
```

The build caches every download under `<state>/raw/http`; `--offline` rebuilds
from that cache. Live sources can change (openFDA, arXiv OAI, EDGAR ticker
order, Wikipedia category lists), so a fresh fetch may select different text;
`verify` then fails rather than silently scoring another panel.
