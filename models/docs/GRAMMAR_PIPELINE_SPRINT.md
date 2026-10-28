# Grammar and evaluation reliability, October 28

The opt-in input corrector now permits a second local LanguageTool check after
adding a missing negative-contraction apostrophe. Only tense-preserving agreement
edits inside those newly repaired contractions can apply in that check. There
is no general rewrite loop and no language-model input fixer.

The saved October 27 trial contains the specific failure this addresses. Its
authored `He dont know the result.` input became `He don't know the result.`.
The saved follow-up checker response proposed `does` and `did` for `do`. The
new policy permits `does` and rejects the tense change to `did`. This example
demonstrates a rule mechanism, not a general correction-accuracy estimate.

The policy is `local-grammar-v3`. `GrammarCorrector(contraction_followup=False)`
provides a one-pass control with the same response validation. The default
correction budget remains eight operations across both checks. A second check
runs only when the first applied a contraction repair and budget remains.
The correction stage remains opt-in wherever it was opt-in before.

Flat `edits` still refer to the original input. A follow-up repair composes into
the original contraction's replacement and records `first_replacement` and
`followup_rules`. The `passes` array keeps each intermediate input, output,
suggestion and edit in that pass's coordinates. Top-level `suggestions` always
describe the original input, preserving the screening manifest's meaning.

Every response is validated before parsing or applying suggestions. Boolean,
fractional, negative and out-of-range offsets, malformed rules and non-string
replacements cause abstention. A contraction suggestion must insert exactly one
apostrophe in the expected position and cover a whole word. Failure of either
service check or protection parsing returns the unchanged original, no applied
flat edits, and `available=False`. Failed second checks mark `rolled_back=True`;
their `passes` entries are an attempted history. Missing checker matches still
do not certify grammar.

Code protection uses the parent's shared `protected_literals.literal_spans`
helper, including multi-backtick spans and unfinished fenced blocks. The
pre-existing natural-language quote checks remain in place.

## What has run

The lightweight tests use fake parses and authored rule responses. They exercise
agreement, tense restrictions, edit composition after other length changes,
protection, failure rollback, the shared edit budget and complete evaluation
matrices. They load no model and establish no semantic accuracy.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 models/.venv/bin/python -m pytest -q \
  models/tests/test_grammar_sprint.py \
  models/tests/test_grammar_sprint_experiment.py \
  models/tests/test_grammar_screening.py
```

The existing ten-system comparison was checked against its independent frozen
passage file. All 1,900 saved predictions match the 190 expected inputs, their
references and panel assignments. This verifies completeness and alignment of
that saved comparison. Its Kea-Roy readability and preservation scores remain
automatic development diagnostics, as described in
[the pipeline report](PIPELINE_BENCHMARK.md).

The scorer previously compared systems' input sets with one another. Equal missing
inputs or a missing whole system could pass that check. It now calls
`cleartext.grammar_pipeline_alignment` before importing or computing metrics and requires the expected
passage file and explicit expected systems. It rejects missing, duplicate,
unexpected and misaligned rows. It never removes a row or scores output quality.
The scorer's cache signature includes both the passage file and expected system list.
Integration regressions reject shared missing inputs and a missing system before scoring.

## Prepared native trial

`cleartext.grammar_sprint_experiment` separates correction from model inference.
`correct` requires all 96 unique saved grammar-panel inputs in original order,
runs fresh one-pass and follow-up checks, and retains every row, including failed
checks. It fingerprints the panel and code before checking. It also compares
fresh one-pass outputs with archived one-pass corrections. A separate
`diagnostics` mode runs both policies on all 17 original authored diagnostics.
It retains the 14 original mechanism expectations and all three report-only
cases, with every output, edit, suggestion and failure. A development panel
with no follow-up changes cannot replace these mechanism checks.

`predict` refuses missing, reordered, edited or failed correction records and
one-pass drift. It fingerprints model files and source code, then runs the same
frozen native sentence generator on original, one-pass and follow-up inputs for
all 96 rows. Exact duplicate input strings share predictions. Every row and all
three conditions remain in the output. This trial does not rerun the ten-system
full ensemble comparison, tune weights or create correctness labels.

The prepared job and exact local paths are in
`~/.local/state/cleartext/improvement-sprint-20261028/grammar-pipeline/`.
The parent must schedule it after the host resource checks. The correction
phase is estimated at 2 GiB peak including LanguageTool and parser dependencies.
The native generator phase is estimated at 2.5 GiB. The job uses one CPU thread
and a 3 GiB memory cap. `run-corrections.sh` starts a private LanguageTool server,
runs the 96-input panel and all 17 diagnostics, and stops that server on exit.
It does no model loading. Native generation is a separate command for the
parent to schedule only if correction results justify it. These are planning
estimates, not measured new trial usage.

The implementation worker ran lightweight checks only. The parent then ran the
native correction trial below. No benchmark input was excluded for grammar.

## Parent native correction result

The private LanguageTool 6.6 trial completed on Yoga under a 3 GiB scope and one
CPU quota. All 96 development inputs remain. Fresh one-pass outputs reproduce
all 96 archived corrections, and follow-up produces the same 96 preprocessing
outputs. Three requests timed out, two in one-pass and one in follow-up. They
remain recorded as unavailable with unchanged text. This is not a fully
successful checker run. No native model comparison followed identical inputs
and unresolved checker failures.

Both conditions completed all 17 separate authored diagnostics and retained all 14
existing expected mechanism outputs. The report-only `He dont know the result.`
case changes from `He don't know the result.` to `He doesn't know the result.`
with the bounded follow-up. The other 16 outputs match. This confirms the intended
native rule behavior without estimating general correction accuracy.

The successful server used port 8788 and the existing JDK 17 by absolute path.
An unrelated service owned 8787; it was preserved. The private server stopped
on completion. [Aggregate evidence](../evidence/grammar-sprint-20261028/) retains
all denominators, failures, input hashes and the 1900-row integrity audit.
