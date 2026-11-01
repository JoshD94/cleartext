# Runbook for Joshua: model arms on the 18 Week 7 passages

Kea-Roy's Week 7 ladder harness is ready; the model arms need a machine
with the trained artifacts (Yoga). This is the one-command version of
the remaining gap: phrase_only unexecuted, and no model numbers on the
18 authored passages.

## Prerequisite

Kea-Roy's branch must be pushed first (`kea-roy/week7-evals`, from the
Week 7 working tree). It contains the ladder in
`evals/model_adapter.py`, the 18 passages in
`evals/data/week7_passages.json`, and the reports in
`evals/reports/week7/`.

## Run (local import; demo-server variant below)

From the repo root, on Yoga, with the models environment:

```bash
git fetch origin
git checkout kea-roy/week7-evals
models/.venv/bin/python evals/run_evals.py \
  --passages data/week7_passages.json \
  --systems original,rule_based,lexical_only,phrase_only,structure_only,lexical_structural,full_guarded,full_no_guardrails \
  --out results/week7-ladder \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

Notes:

- The local transport loads `models/runs/ensemble-v7-20260928` by
  default; `--model-run <path>` selects another run.
- `full_no_guardrails` needs a pipeline whose `analyze()` accepts a
  `guardrails` switch (the structural branch,
  `josh/structural-models-20261026`). On the main-branch pipeline it
  raises a clear error instead of running guarded; that is the harness
  refusing to fake the arm, not a crash to work around.
- `structure_only` via this harness runs the main-branch old-rules
  `structural()` (comma+and splits, explicit-agent passives). The T5
  structure arm is the benchmark's `structural_only`, already executed
  Oct 27-28 on 190 inputs; do not conflate the two.
- First run downloads the two eval models (~1-2 GB) if not cached.
  Expect a few seconds per passage after that.

## Demo-server variant

Start `models/demo/server.py` on the artifact machine, then add
`--model-url http://127.0.0.1:8770`. `full_no_guardrails` will raise
over HTTP (the server exposes no guardrail switch); run that one arm
via the local import above.

## Paste back

Into the PR thread (or the team doc), from Yoga:

- `evals/results/week7-ladder/eval_results.md` (the aggregate table)
- `evals/results/week7-ladder/eval_results.json` (per-passage metrics
  AND outputs: the runner now stores each output, which unblocks the
  text review in `yoga_review_packet.md`)

## If phrase_only changes nothing

That is a result, not a failure: it means `phrase_model` was absent
(`phrase_model.pkl` in `runs/context-20260924/`) or no inventory
phrase fired on these 18 passages. Report which one: check whether
`runs/context-20260924/phrase_model.pkl` exists, and whether any of
the 10 inventory phrases in `models/src/cleartext/rewrites.py`
(`PHRASES`) occur in the passages.
