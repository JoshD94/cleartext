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
- `demo-web/`: public demo of the default pipeline (see its README).

## Environment and accounts

Use Joshua's personal account and `~/.local/bin/personal` for GitHub/service
commands. Never use Alora credentials for this project. Read the personal
account and host workflow before remote changes or substantial compute.

Yoga is the default development host. Native jobs need no GUI or browser.
Use `models/.venv/bin/python` and `PYTHONPATH=models/src:.`. Environments,
weights, caches and raw experiment inputs are deliberately excluded from Git.
Do not infer that a missing artifact calls for retraining or downloading data. On a new host, set NLTK_DATA and HF_HUB_CACHE for isolated
offline resources; Hugging Face refs/main must contain the exact revision with
no trailing newline.
Consult `models/README.md` and the relevant experiment manifest first.

## Comments and lint

No code comments. The one exception explains a weird, unconventional business
decision and starts with `Why:`. `scripts/lint` (needs Node and `uvx`) runs the
vendored comment check (`scripts/comment-check/`) and ruff (`ruff.toml`).
Hash-bound sources keep their old comments and findings: they are `legacy` in
`.comment-check.json` and per-file ignores in `ruff.toml`. Never edit their
bytes to satisfy lint.

## Verification

Run `scripts/verify` for lint plus the focused offline checks on the model/eval branch.
It loads the local parser and mocked tiny tensors; it does not train or decode
a production transformer. Estimate up to 2 GB RAM and follow Joshua's placement
and resource rules before running it.

`CLEARTEXT_PYTHON` overrides the Python interpreter. `CLEARTEXT_EVAL_DEPS`
can point to a separate directory containing the optional evaluation packages.
`CLEARTEXT_VERIFY_OUT` chooses the JUnit/log destination, outside Git by default.
This check does not replace native benchmark execution or certify quality.

For sustained inference and the full verify suite, submit through Joshua's
`~/.local/bin/job` queue with this thread as owner; record the ID and end the
turn. Do not run heavy work inline, poll, sleep or `job wait`. Small edits,
Git and a single focused test file may run inline. Use `--account personal`
for agent jobs. Check required paths before submission. Record host, command,
versions, input/model/source hashes and completion/failure receipts.

## Benchmark landmines

- The submitted October 21 report uses six authored passages, v7 lexical only,
  one word edit, with phrase/structure/grammar stages off. The name
  `full_pipeline` in that old report does not describe the later trained stack.
- The current `run_evals.py` full-pipeline flag enables rule stages. It does not
  by itself select the separately trained T5 checkpoint.
- `evals/pipeline_benchmark.py` explicitly loads T5 and supports the larger
  frozen development panel. Require an independent input file and complete
  expected system list when scoring; never silently drop missing predictions.
- Report automatic metrics with denominators, abstentions and changed-only
  results. Embedding/NLI scores are not percentages of correct outputs.
- `evals/ensemble_study.py` binds cached outputs to the source of
  `StudyGenerator` and `models/src`; editing either needs a new frozen plan
  and state dir. Older states' signatures come from their batch receipts;
  `freeze --reuse-state` reuses memos/metrics only after its compatibility check. SARI there uses the 0/0=0 convention (`evaluate`'s 0/0=1
  rewards copying). The November 9 study scored ASSET test outputs for
  finalists; treat ASSET test as used, not fresh.
- Previously inspected development panels cannot become independent test sets.
  Preserve reserved-domain restrictions and keep larger-panel raw text and
  predictions local. The six historical authored examples have public evidence.
- Keep other experimental stages (morphology, grammar preprocessing, stricter
  thresholds) opt-in unless authorized and shown to meet adoption criteria.
- `compound_only` and `compound_full_guarded` select the opt-in dependency
  compound splitter. Week 7 `structure_only` retains the old rules; neither name
  selects T5. Passive conversion and negation rewriting belong to teammates.
  Chhoeng's `passive/` and `pos_tag/` code runs unchanged through
  `cleartext/teammate_stages.py` (opt-in); never put those folders on
  `sys.path`: `pos_tag/evaluate.py` shadows Hugging Face `evaluate`.
  The WikiSplit tagger model is a local artifact
  (`evals/reports/20261109-compound-tagger/`).
- The 90-passage native ladder exceeded a 75-minute Yoga queue limit while
  computing lexical-only. Whole-system logging/checkpoints lose partial work.
  Use per-passage checkpoints and bounded batches; bind resume data to frozen
  source, input, model and runtime signatures. Automatic scores remain pending
  until the complete expected matrix is present. Raw checkpoints stay local.
- Test set v3 (`evals/data/testset_v3/`, protocol in
  `evals/reports/20261116-testset-v3/PROTOCOL.md`): 300 passages + 544
  OneStopEnglish sentences, text local only. The **test split is locked**
  for one pre-registered final run; `ensemble_study freeze`/`run` refuse test
  IDs unless the plan was frozen with `--unlock-test`. Never pass it otherwise.
- `evals/param_sweeps.py` (stage settings as config parameters; Nov 16 study)
  unlocks test only via `freeze --unlock-test <committed finalists.json>
  --dev-plan <dev plan>`. Editing `SweepGenerator` or `models/src` changes
  its generation signature and invalidates cached outputs.
- A Git worktree lacks the ignored `models/runs/` and `models/data/` (the v7
  bigram cache); symlink both from the primary checkout before running the
  pipeline there.
- The Nov 16 sweep's dev plan pins its sources by absolute path in
  `cleartext-wt-sweeps`. To add cells to it, check out a detached worktree of
  `main` at that path (sources still match). NLI rejects about 25% of
  human-gold word swaps at 0.95; do not lower the bound for that
  (`evals/reports/20261117-nli-calibration/`).
- `cleartext.swap_veto.WindowSwapVeto` is an opt-in per-swap veto (window NLI
  + checks, `word_veto=`). `evals/swap_veto.py freeze` reuses the Nov 16 sweep
  dev state's memos only while every source stays byte-identical; on dev it
  did not make words beat F2 (`evals/reports/20261118-swap-window-veto/`).
- **Default pipeline (on since Nov 23): F2 → structure v4 (`KEPT_V4`, 0.5
  entailment floor) → learned words arm p50** (`evals/default_pipeline.py`,
  `models/docs/DEFAULT_PIPELINE.md`). F2 output comes from `ensemble_study`/
  `param_sweeps`; `default_pipeline run` applies the later stages. The v7 demo
  is a separate, unchanged process. Each stage passed blind ratings on fresh
  text; the stages were not rated jointly (dev gates only).
- `cleartext.learned_words` (`scripts/learned-words`, runs `python -I`): rows
  and models in `~/.local/state/cleartext/learned-words-20261118/`; p50 is
  round 19's frozen `r19/model.pkl`, SHA-pinned. Editing `features.py`/`data.py`
  invalidates the feature state. Joshua's 60 marked swaps, the round-2 page and
  `stacker-joshua60.json` are held out: never train, tune or choose on them.
  Its split checker (rejects 38/73 F2 splits) stays opt-in.
- `cleartext.structure_v2`: new ops get a new version and preset (`KEPT_V3`
  fixed, `KEPT_V4` opt-in); rated items cannot validate rules written from
  them: use a fresh window (`pool-c` excludes 8-grams of all earlier pools).
- Public demo (cleartext-ai2.pages.dev): backend runs on Yoga as `cleartext-demo` and
  `cleartext-tunnel` user services; the Pages Function finds the quick-tunnel URL in KV
  (re-registered on restart). Submitted text is never written to disk or logged.

Do not commit credentials, environment files, model binaries, downloaded
corpora or raw client data. Keep failed attempts and provenance honest. Before
handback, close only resources created for the task and leave a short handoff.
