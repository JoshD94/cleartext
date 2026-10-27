# Aligning training targets and runtime checks

October 27, 2026. This experiment fixes the disagreement found in the
[bottleneck audit](STRUCTURE_BOTTLENECKS.md), then continues the existing T5-small
checkpoint. The generator remains experimental. The v7 demo is unchanged.

## What changed

Continued-training eligibility, ordinary generation and candidate search now call the
same `generation_checks` function. It checks the existing content, grammar,
tense and attribution signatures. Training still adds its structural and
discourse filters. Search still adds its boundary filters. Passing these checks
does not certify grammar or meaning.

The attribution repair handles a narrow parser error. In "Alice thinks that
Bob works here and Alice hopes that Carol will join him", the parser can attach
"hopes" to "works". A repeated matrix subject, a distinct embedded subject and
an independently finite complement now allow the second report to stand alone.
The tests cover repeated names and pronouns, plus embedded claims that must
retain their original attribution.

An explicit subject by itself does not permit a split. "Alice said that Bob
left and Carol stayed" must still keep both claims under "said". Unresolved
name-to-pronoun links, such as "Alice ... she", remain conservative. This is a
parser heuristic, not a coreference or semantic model.

The final data audit also caught two normalized copy labels with spacing or
capitalization differences that changed the parser result. Copy labels now
pass the shared checks too. The failed preparation is preserved locally.

## Data and selection

Of 3,986 prior training pairs, 3,618 remain eligible: 2,620 published rewrites
and 998 published copy annotations. The other 368 are quarantined with their
original labels. None become fabricated copy targets. Every retained target
passes the runtime checks, including an independent assertion after filtering.
The aligned development panel contains 101 pairs.

A second development panel contains 96 previously unused DeSSE pairs, sampled
by a fixed source hash with basic length and markup filters. Their reference
quality does not affect sampling. Only 12 references pass the runtime checks;
all 96 remain in the decoded report. Those 12 comprise six rewrite references
and six copy references. This large disagreement remains a limit of
the published decomposition task and our conservative checks.

The reserved comparison contains another 96 previously unused official DeSSE
test pairs. Every source and full target from eight earlier prepared datasets
is excluded before sampling. All official DeSSE test keys are excluded from
continued training. Exact exclusions do not rule out near duplicates, overlap
between original corpora or exposure during base-model pretraining.

Checkpoint selection evaluates the initializer and each complete epoch. It
averages scores from the aligned panel and the compatible references in the
broader panel, with equal panel weight. For changed references, outputs that do
not add a shorter sentence receive the identity output's score. Published copy
references use the actual output's score. This custom development score reduces
rewards for punctuation-only edits. It is not standard SARI or a quality rating.
The small compatible subset of the broader panel limits selection reliability.
All-reference overlap is reported separately, including incompatible references.

## Training and comparison

The frozen protocol uses two complete epochs, 906 optimizer updates and 7,236
training examples seen. All 3,618 pairs fit the token limits. Initialization is
the earlier continued T5-small checkpoint, with a new Adafactor optimizer,
learning rate 0.0001, effective batch size eight and seed 4701. This is continued
weight training, not restoration of the earlier optimizer state.

The run uses Yoga, two CPU threads and a 4 GB memory cap, in tmux session
`cleartext-alignment-20261027`. The source checkpoint is preserved. No new paid
services, synthetic labels or human review are used.

Training completed all 906 updates and 7,236 examples in 34.3 minutes. Peak
training RSS was 1.36 GiB; sampled available RAM stayed above 5.1 GiB. The
selected checkpoint is step 906.

| Checkpoint | Development loss | Selection score | Aligned shorter splits, of 101 | Broader shorter splits, of 96 | Broader all-reference overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initializer | 0.18906 | 51.04 | 52 | 11 | 32.50 |
| One epoch, step 453 | 0.19034 | 50.15 | 52 | 12 | 33.25 |
| Two epochs, step 906 | 0.17240 | 52.31 | 57 | 13 | 33.81 |

All 32 aligned copy annotations and all six broader copy annotations remain
copies at every checkpoint. The broader panel's compatible-reference selection
score is unchanged at 41.49; the winning selection score comes from improvement
on the easier aligned panel. Overall broader reference overlap rises by 1.31,
but only four of 96 returned outputs differ from the initializer. These are
modest development gains, not evidence that the model reliably preserves meaning.
Both panels were used for selection, so their bootstrap intervals are development
diagnostics rather than independent confirmation.

The saved selected weights reproduce every in-memory broader development
output. The [pre-test decision](../evidence/structure-alignment-20261027/pretest-decision.json)
freezes the selected weights, baseline, reserved inputs and policy before the
first reserved prediction.

## Reserved result and decision

The single reserved comparison uses 96 examples, including 15 copy references.
Both models run the repaired checks and identical two-beam decoding.

| Returned behavior | Initializer | Continued step 906 |
| --- | ---: | ---: |
| Changed outputs | 12 | 15 |
| Shorter sentence splits | 11 | 14 |
| Copy references kept | 15 of 15 | 15 of 15 |
| All-reference overlap | 33.93 | 35.12 |
| Compatible-reference structural score, 26 examples | 41.58 | 43.45 |

Only three returned outputs differ. The mean overlap gain is 1.19 points, with
a 95% paired-bootstrap interval of 0.00 to 2.71 over 1,000 resamples. The interval
touches zero. This is a small automatic-metric gain on one bounded seed, with
limited evidence about broader reliability. Fourteen splits are outputs, not
fourteen independently verified correct simplifications. The checks and the
published references leave naturalness and subtle meaning uncertain.

Keep step 906 as an experimental checkpoint. The demo still reports rules and
ranker backends, with rules preferred. No model, threshold or selection policy
changed after reserved evaluation. The separate project-domain test remains
unpredicted.

The [aggregate evidence](../evidence/structure-alignment-20261027/README.md)
contains configuration, history, fingerprints, frozen comparisons and resource
measurements. This 96-example reserved set is now inspected reporting material.
Future experiments must exclude both this run's `data.json` and its separate
`reserved_test.json` when claiming unused inputs.

## Saved weights

The selected weights remain outside Git at
`~/.local/state/cleartext/structure-alignment-20261027/run-v2/seq2seq`.
Their SHA-256 is
`667acaaef2867a6e7141ac6b99a20e9d82df276e7d95bd95af01633a4fd7fcff`.

A local 244.5 MB bundle, `cleartext-structure-aligned-20261027.tar`, sits in the
experiment's parent directory. It includes the tokenizer, selected weights,
loading instructions and aggregate metadata. It contains no raw pairs or decoded
public text and has not been uploaded. The [bundle receipt](../evidence/structure-alignment-20261027/bundle.json)
records its fingerprint. Teammates can load that bundle without repeating
training.

## Reproduce

From the repository root, with the earlier local experiment store available:

```sh
export PYTHONPATH=models/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
state=/path/to/cleartext-experiment-store
run="$state/structure-alignment-20261027/run-v2"
initial="$state/structure-expanded-20261026/run/seq2seq"
models/.venv/bin/python -m cleartext.structure.alignment prepare --state "$state" --run "$run"
models/.venv/bin/python -m cleartext.structure.alignment train --run "$run" --initial "$initial"
models/.venv/bin/python -m cleartext.structure.alignment compare --run "$run" --initial "$initial" --output "$run/dev-comparison" --split dev
```

Freeze the selected weights and decision after development comparison before
running the separate `--split test` comparison once. This reproduces the
historical reporting set; it does not make those inputs unused again. Use a fresh experiment
directory. Run long commands under the host's resource controls and named tmux
session. Local data, decoded outputs and weights stay outside Git.

The focused structural, preservation and demo-adjacent regression suite passes
196 tests. Ruff reports no undefined names or syntax errors in the changed
modules. Tests verify both permitted independent reports and rejected detached
claims, normalized-copy eligibility, and decoded checkpoint selection using
real gradient updates in a tiny local T5 fixture.
