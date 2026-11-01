# Cleartext

CS4701 text simplification project. Joshua's teammates consume the demo, model
outputs, evaluation reports and reproducible experiments.

## Layout

- `models/src/cleartext/`: lexical pipeline and optional sentence experiments.
- `models/demo/`: existing local demo; preserve its process and settings.
- `models/scripts/`: experiment and prediction entrypoints.
- `models/tests/`: model, parser and experiment contract checks.
- `models/docs/`: decisions and limitations, with links to aggregate evidence.
- `models/evidence/`: committed aggregate results and manifests.
- `evals/`: Kea-Roy's readability/preservation suite and pipeline adapters.
- `evals/reports/20261021-v7/`: the six-passage submitted-report baseline.

## Environment and accounts

Use Joshua's personal account and `~/.local/bin/personal` for GitHub/service
commands. Never use Alora credentials for this project. Read the personal
account and host workflow before remote changes or substantial compute.

Yoga is the default development host. Native jobs need no GUI or browser.
Use `models/.venv/bin/python` and `PYTHONPATH=models/src:.`. Environments,
weights, caches and raw experiment inputs are deliberately excluded from Git.
Do not infer that a missing artifact calls for retraining or downloading data.
Consult `models/README.md` and the relevant experiment manifest first.

## Verification

Run `scripts/verify` for the focused offline checks on the model/eval branch.
It loads the local parser and mocked tiny tensors; it does not train or decode
a production transformer. Estimate up to 2 GB RAM and follow Joshua's placement
and resource rules before running it.

`CLEARTEXT_PYTHON` overrides the Python interpreter. `CLEARTEXT_EVAL_DEPS`
can point to a separate directory containing the optional evaluation packages.
`CLEARTEXT_VERIFY_OUT` chooses the JUnit/log destination, outside Git by default.
This check does not replace native benchmark execution or certify quality.

For sustained inference, estimate peak RAM, check processes/resources on both
hosts, choose with `pick-host`, and run in a named Yoga tmux session with a
systemd memory cap. Preserve unrelated jobs. Stop or pause your newest job if
resources cross the personal profile's limits. Record host, command, versions,
input hashes, model hashes, source hashes and completion/failure receipts.

## Benchmark landmines

- The submitted October 21 report uses six authored passages, v7 lexical only,
  one word edit, with phrase/structure/grammar stages off. The name
  `full_pipeline` in that old report does not describe the later trained stack.
- The current `run_evals.py` full-pipeline flag enables rule stages. It does not
  by itself select the separately trained T5 checkpoint.
- `evals/pipeline_benchmark.py` explicitly loads T5 and supports the larger
  frozen development panel. Require an independent input file and complete
  expected system list when scoring; never silently drop missing predictions.
- `evals/status_report_benchmark.py` exposes explicit replay settings and can
  import a historical source archive against unchanged local model artifacts.
- Report automatic metrics with denominators, abstentions and changed-only
  results. Embedding/NLI scores are not percentages of correct outputs.
- Previously inspected development panels cannot become independent test sets.
  Preserve reserved-domain restrictions and keep larger-panel raw text and
  predictions local. The six historical authored examples have public evidence.
- Keep experimental morphology, grammar preprocessing, stricter thresholds and
  source-anchored split selection opt-in unless separately authorized and shown
  to meet their recorded adoption criteria.

Do not commit credentials, environment files, model binaries, downloaded
corpora or raw client data. Keep failed attempts and provenance honest. Before
handback, close only resources created for the task and leave a short handoff.
