# Structural simplification follow-up, October 26

The ranker has been refitted with stricter clause checks. The demo offers it as
an experimental option, with structure off initially and grammar rules preferred.
The CRF and T5 remain comparison models. Their trained weights are unchanged.
The original v7 word models and the [first pilot record](../structure-20261026/README.md)
are preserved.

## Fresh comparison

| System | Raw SARI | Final SARI | Final changes | Mean longest sentence, words |
|---|---:|---:|---:|---:|
| Unchanged input | 30.41 | 30.41 | 0/256 | 27.8 |
| Grammar rules | 31.61 | 31.61 | 8/256 | 27.5 |
| Refitted ranker | 51.75 | 42.11 | 104/256 | 23.2 |
| Reused CRF | 44.51 | 37.91 | 47/256 | 25.9 |
| Reused T5-small | 59.75 | 45.53 | 102/256 | 23.3 |

| Learned system | SARI difference vs rules | Paired bootstrap 95% interval | Raw checks passed |
|---|---:|---:|---:|
| Ranker | 10.49 | 8.27 to 12.83 | 39.5% |
| CRF | 6.30 | 4.26 to 8.56 | 68.4% |
| T5-small | 13.91 | 11.44 to 16.48 | 49.6% |

Intervals use 1,000 paired resamples and seed 4701. They describe sampling
variation of this overlap metric on the filtered sample, not correctness.

All systems use the same fresh 256-pair sample of the official WikiSplit
validation file. SARI is the documented single-reference set-based implementation.
The comparison evaluates structural models alone, without v7 word substitutions.
Sentence length counts alphabetic spaCy tokens.
Raw proposals precede filtering and abstention. Checks depend on a parser and
cannot certify meaning or fluency. An unchanged noisy source can fail them too.
The rules row uses the existing implementation unchanged. Shared checks score
its outputs; the abstention policy applies to the learned-model rows.
The first pilot's 128 test rows informed the grammar changes and now count as
development evidence. Scores across those different samples do not measure a
controlled before/after gain.

T5 has the highest final SARI on this sample. Its point score exceeds the
ranker by 3.42; that is a reference-overlap observation, not a fluency ranking.

Agent inspection covered the first 20 changed outputs from each learned model
in deterministic test order. The ranker still duplicates subjects, splits inside
noun coordination and accepts fragments or wrongly attached participants. The
CRF can introduce invalid punctuation and detach a causal modifier. T5 produces
more plausible local splits in this inspected subset, but also changes tense
and drops a discourse contrast. Some awkward wording and citation debris are
inherited from the noisy sources. These are diagnostic observations, without
human ratings or a representative quality estimate. Per-row notes are in
`qualitative_review.json`.

The new checks close the specific finite-predicate and relative-fragment gaps
covered by the regressions. They do not solve general grammaticality. Even the
unchanged baseline fails 22 parser completeness checks. The learned ranker stays
an explicitly experimental demo option, with structure off and rules preferred.
T5 remains a CLI comparison model. Its CPU/MPS serving costs and residual errors
have not been validated for replacing the live default.

## Technical transfer

| System | Changed technical inputs |
|---|---:|
| Grammar rules | 0/12 |
| Ranker | 0/12 |
| CRF | 0/12 |
| T5-small | 4/12 |

The audit uses the first six eligible 20..70-word inputs from section 2,
including subsections, of each HTTP and TLS RFC. Selection precedes the final
model and code freeze. The earlier section-1 inputs informed development and
are excluded from this fresh audit. The fixed sample includes a TLS protocol
list item without a complete main clause. The fresh passages share documents
with the development inputs; this is not a document-held-out evaluation.
The list item remains in the denominator of 12.
There are no reference simplifications or human ratings. The annotations are
agent inspection and the technical change counts are coverage observations.
HTTP and TLS standards cover one technical genre. This audit supplies no
structural-simplification result for the proposed corporate or political domains.

The ranker's preview leaves all 12 inputs unchanged. Its raw proposals include
detached list items, participial fragments and a broken conditional. Checks
reject those proposals. They also reject a plausible TLS semicolon split because
token spacing changes the literal entity match for Diffie-Hellman. Abstaining
avoids these edits, but supplies no simplification on this small technical sample.
This supports keeping the learned demo option experimental.

Models, candidate generation, shared checks and thresholds remained frozen
through scoring and inspection. Demo labels now state the observed failure modes.
Technical-domain improvement still needs a separate development corpus,
better candidate generation and independent human meaning/fluency judgments.

## Training, data and freeze

Training and development rows are unchanged from the first pilot, with 2,000
training pairs and 128 development pairs. The fresh validation sampler excludes
normalized sources and full targets from every previously selected train,
development and original test row. The validation file has 5,000 rows, of which
4,371 meet the fixed filters; three within-file duplicates and 626 other rows
are excluded. SHA256 sampling with seed 4701 selects 256 pairs.
Exact matching does not remove paraphrased duplicates or pretraining overlap.

The ranker trains on 10,293 candidates with equal total weight per source.
Its gradient boosting settings are unchanged, with 150 iterations, 15 leaves,
learning rate 0.05 and L2 regularization 5. Selection uses the unchanged
development references. Thresholds 0, 1 and 2 tie at development SARI 40.4571;
the existing selection policy chooses 2. This threshold is fixed before fresh
evaluation. The ranker must be refitted because clause-check results are input
features.

The CRF reuses its original 526 representable training pairs and 131 copy
examples. T5-small reuses the selected checkpoint at update 300 from the original
400-update CPU run, with effective batch size four. No new CRF or T5 training
occurs in this follow-up. The original pilot record has their full budgets.

The final fitting and evaluation source is
`842abe3549a213cff2cd71535a266ba689f07f7b`. `provenance.json` records
module, data, input and model hashes written before fresh predictions.
Two earlier attempts were cancelled during development evaluation, at rows 33
and 1. Neither produced fresh general-test predictions. The preserved attempt
receipts explain the detached-relative fixes and the new technical reservation.

## Implementation and demo verification

The completeness check requires a finite main predicate and its own subject.
It rejects detached relative phrases through the main argument dependencies
while skipping embedded clause boundaries. Tests cover valid copulas,
auxiliaries and embedded relative clauses, plus fragments with participles or
relative verbs. Imperatives and detached relative fragments abstain conservatively.

The optional demo adapter accepts a trusted local ranker, rechecks its chosen
output and preserves multiple-sentence or over-70-word inputs. The pipeline's
technical-term protection can still roll back a stage. Structural traces report
the exact applied stage, including abstention. The strictness slider controls
word substitutions and does not alter structural selection.

The focused suite passed 83 tests. The demo API passed authored split,
detached-relative and modal/negation cases. A browser interaction selected the
experimental method, enabled structure, set zero word edits and produced the
expected server-sentence split with no visible error or horizontal overflow.
These are regression and integration checks, separate from the benchmark.
Standalone CLI prediction also passed with only the archived ranker extracted
into a new folder, without training data or the other model artifacts.

## Resources and artifacts

The comparison completed on the Mac in tmux session
`cleartext-followup-20261026`. T5 used MPS generation and the classic models used
CPU. A Yoga session of the same name monitored it and stored copied outputs.
Yoga's free disk was below its required reserve, so heavy work stayed on the Mac.
The estimated evaluation peak was 3 GB RAM. A watcher suspended this experiment
when Mac memory pressure reached warning and resumed on normal pressure.
The completed experiment sessions were closed; the demo remains running.
Unrelated jobs were left running. Timings include pauses on a shared host and
are not an isolated serving benchmark.

The Yoga demo runs in session `cleartext-demo-20261026` on port 8770.
Use the commands in the [experiment guide](../../docs/STRUCTURE_EXPERIMENTS.md)
to load the model. The fitted
ranker bundle contains no data rows and requires the existing v7 artifacts
for the full demo. Its local filename is
`cleartext-structure-ranker-20261026.tar.gz`, 100,538 bytes, with SHA256
`8fb81b3748a6c04f150d15ee4ac3cf7b328aed5ec3a5967a2a2eff594e7cb37f`.
The archived ranker matches the frozen weight hash. The original three-model
bundle is preserved for the unchanged CRF and T5 weights.
Use the follow-up bundle's ranker with this branch, since its check features
require the refitted weights.

Weights, input rows and per-sentence benchmark outputs remain outside Git.
This directory contains aggregates, provenance, authored diagnostics and
review annotations identified by source row IDs.
