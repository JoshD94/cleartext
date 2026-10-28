# Input errors, lost connections and screened-data training

October 27, 2026. The sentence generator sometimes inherits malformed input and
sometimes introduces a meaning error during a split. Those require different
fixes. This pass audits the fixed development examples, repairs two shared
checks, and continues the best experimental checkpoint on a separately screened
training pool. The running v7 demo remains unchanged.

## What the input/output audit found

The existing 96-example broad development panel has 14 normalized output
changes, plus one spacing-only change. I inspected all 15 byte-different outputs
in context. Six looked like useful splits without an obvious new error, four
carried over input problems, three lost meaning, and one only deleted a word.
The remaining change added a missing space. These are coding-assistant judgments
on inspected development examples, not independent human labels or an accuracy
estimate.

The meaning errors were two omissions of causal `so` and one split that left
the second statement outside the original `maybe`. The input grammar checker
reported no new selected warnings on any returned output. It therefore failed
to identify these meaning problems. Two outputs resolved a selected warning,
including the spacing-only edit. Fifteen still shared selected warnings with
their original input. Grammar-warning differences are candidates for inspection,
not verified repairs or regressions.

Twenty inputs copied back by the generator fail the parser's clause-completeness
diagnostic. Those are not twenty model-introduced grammar errors. Returning a
malformed original does not make it grammatical, but that source error must be
kept separate from damage caused by simplification.

## Shared checks for connections and uncertainty

The existing training filter counted discourse markers such as `so`, `or`,
`otherwise` and `therefore`. Generation did not enforce all of them. The shared
`generation_checks` contract now applies those counts to both published training
targets and model proposals.

A second check tracks `maybe`, `perhaps`, `probably` and `possibly` on predicates
in shared-subject conjunctions. For example, splitting an authored fixture into
`Maybe Alice saw the error. Alice did not understand it.` loses uncertainty from
the second statement. Repeating `Maybe` there preserves the diagnostic signature.
An explicitly different subject does not automatically inherit uncertainty.

The initializer reproduces every earlier raw prediction on all 96 broad inputs.
Only the three identified meaning-loss cases change their returned output after
the check repair, falling back to the original input. This isolates the guard
effect from weight training. The guard abstains rather than repairing a rejected
proposal.

These checks remain conservative. Exact marker counts reject valid synonyms and
some valid repetitions. Lemma sets and parser attachments do not represent every
scope relationship, resolve pronouns or certify meaning.

## Training data and frozen comparison

The previous aligned pool contains 3,618 published pairs. Local LanguageTool 6.6
checks cover both sides of each pair. The cached scan includes 6,633 unique texts
across training, development references and saved development outputs. It uses
the previously evaluated local rules without cloud or language-model services.

An initial preparation quarantined every review-only warning. Inspection exposed
valid names, dialect and style preferences among those exclusions. That attempt
was stopped at update 80 of 363 before its first trained-checkpoint validation.
The original preparation, logs and partial progress are preserved locally. Its
decoded performance did not motivate the filter revision.

The revised `training-grammar-v2` policy retains spelling-only and style-only
review cases. It quarantines the screen's selected grammar/punctuation/spacing
warnings, failed checks, four additional identified grammar or orthography rule
families, and targets that fail the shared generation checks. Those extra rule
families concern possessive apostrophes, missing `I've` apostrophes, sentence
capitalization and `everyday` versus `every day`.

The revised pool retains 3,523 pairs, including 958 copies, and quarantines 95.
It keeps 620 pairs with review-only warnings. Retained sources and targets are
byte-identical to their published labels. No suspect rewrite becomes a fabricated
copy target, and no grammar suggestion rewrites a training label. Retained pairs
can still contain errors the rules missed.

All 101 aligned and 96 broad development examples remain unchanged, including
their original reference-compatibility annotations. The earlier 96 reserved
examples are used only for overlap exclusion. No new reserved or project-domain
test prediction is part of this experiment.

The bounded continuation starts from aligned step 906, with a new Adafactor
optimizer, learning rate 0.0001, effective batch size eight and seed 4701. It
runs one complete epoch, 441 updates. Checkpoint selection compares the
initializer and completed epoch using the existing equal-panel decoded structural
overlap score, then development loss. It can retain the initializer. Final trained
outputs are also reported if the initializer wins.

This comparison uses inspected development material that also selects the
checkpoint. The overlap score is a proxy, with some published references still
incompatible with the new checks. Screening and additional training are combined
in one experiment, so a gain cannot be attributed to screening alone.

The reference diagnostic makes that limitation concrete. All 101 aligned
references still pass the shared checks. Six of the original twelve compatible
broad references now fail marker preservation, leaving six current-compatible
references, all copies. The historical twelve-example scoring group remains
fixed for comparison. Its score must not be presented as independent evidence
of better broad-domain simplification. Marker disagreement is not proof that
every published target is wrong; some omissions may be valid in context.

## Completed result and decision

The revised run completed all 441 updates and saw all 3,523 retained pairs once
in 15.8 minutes. The selection proxy rose from 48.78195 to 48.80797, so it selected
step 441. This 0.026-point difference does not establish a general improvement.
Development loss worsened from 0.17240 to 0.18079.

With the repaired checks identical for both models, aligned shorter splits fell
from 56 to 54 of 101. Broader shorter splits rose from 10 to 11 of 96. All 32
aligned and six broader copy references remained copies. Full-reference overlap
fell from 62.86490 to 61.72557 on the aligned panel and from 32.32418 to 31.84776
on the broader panel. The historical twelve-example broader selection score
rose from 34.69900 to 35.89038. Its small denominator and target disagreement are
material limits of the selected score.

Only four returned outputs differ between the same-check initializer and trained
model. I inspected all four:

- One previously rejected broader proposal now fixes a modal verb form and
  preserves its causal connection. This recovers one useful split.
- One aligned output previously dropped the object of a preposition. The trained
  model copies the grammatical original, avoiding the broken fragment. Both
  existing parser checks and the selected grammar warnings missed that fragment.
- One broader output stops making a minor word deletion without a split.
- One previously useful aligned split is lost. Its new proposal fails checks,
  so the returned text is the original sentence.

The checker found no new selected warning candidates in either model's returned
outputs across the 197 examples. The four-case review shows why that is not an
accuracy measurement. The trained model still inherits malformed inputs, and
the tests do not establish meaning preservation outside the inspected cases.

Keep the shared discourse repair. Save the trained candidate as experimental,
without promoting it over the previous checkpoint or enabling it in the demo.
The weight-training result is mixed; its tiny proxy gain is insufficient for a
quality claim. The separately verified guard-only comparison prevents the three
known meaning-loss outputs under the unchanged initializer.

The selected saved weights reproduce all 96 broader raw and returned predictions
exactly after reload. Their SHA-256 is
`3de186eee031e8812b019a9040775884798d01baabb072728bd03ac89f1503ac`.
The local `cleartext-structure-quality-20261027.tar` bundle contains those weights,
the tokenizer, loading instructions and aggregate metadata. It has no corpus rows
or decoded outputs and has not been uploaded.

There are 229 passing focused regression tests across the relevant suites,
including 13 quality-policy tests after the filter revision. Source and artifact
hashes, unchanged evaluation panels, original training labels and complete
quarantine accounting were also verified. The observed combined scope peak was
2.13 GiB under the 4 GiB cap; sampled available RAM stayed above 6.65 GiB. The
trial server and verification job stopped after completion.

## Local artifacts and reproduction

The revised job ran on Yoga in `cleartext-quality-v2-20261027`, with two CPU
threads and a combined 4 GiB memory cap for the local grammar server and model
work. Yoga fit the resource estimate; the Mac was reachable but unnecessary.
The resource monitor was configured to stop only this job below 2 GiB available.

Raw pairs, grammar suggestions, decoded outputs, quarantined labels and weights
stay outside Git under `~/.local/state/cleartext/structure-quality-v2-20261027/`.
The first attempt remains under `structure-quality-20261027/`. Aggregate evidence
belongs in `models/evidence/structure-quality-20261027/`.

From the repository root, use a fresh local run directory and the existing
LanguageTool server:

```sh
export PYTHONPATH=models/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
models/.venv/bin/python -m cleartext.structure.quality scan \
  --prior /path/to/aligned/data.json \
  --saved-outputs /path/to/aligned/dev-comparison/outputs.json --run /path/to/new-run
models/.venv/bin/python -m cleartext.structure.quality prepare \
  --prior /path/to/aligned/data.json --reserved /path/to/aligned/reserved_test.json \
  --run /path/to/new-run
models/.venv/bin/python -m cleartext.structure.quality train \
  --initial /path/to/aligned/seq2seq --run /path/to/new-run
models/.venv/bin/python -m cleartext.structure.quality compare --run /path/to/new-run
```

Use the host's memory controls and a named tmux session for long commands. These
commands reproduce an inspected development experiment, not a fresh test.
