# Learned meaning experiment, October 27

We trained a small paraphrase classifier on 2,048 published PAWS training pairs,
using frozen bidirectional MiniLM NLI scores as inputs. It improved PAWS test
classification, but did not establish better Cleartext simplification. Keep it
experimental. No demo defaults changed.

## Separate test results

The classifier and threshold were frozen before scoring 512 official test
pairs. Validation used 256 official validation pairs. Sampling ignored labels
and excluded exact sentence overlap across selected splits. Training also
excluded every released validation and test sentence string.

| Measure | NLI entailment threshold | Trained classifier |
| --- | ---: | ---: |
| Accuracy | 54.69% | 67.38% |
| Balanced accuracy | 58.02% | 68.83% |
| Precision among accepted pairs | 49.76% | 59.87% |
| Recall of equivalent pairs | 90.87% | 83.04% |
| Non-equivalent pairs incorrectly accepted | 211/282 | 128/282 |
| Equivalent pairs incorrectly rejected | 21/230 | 39/230 |

Always predicting the training majority class scores 55.08% accuracy and 50%
balanced accuracy. The learned accuracy gain over the fixed NLI threshold is
12.70 percentage points, with a paired 1,000-resample row-bootstrap 95% interval
of [8.98, 16.41] points. This supports a gain on this PAWS sample, not a claim
about better simplification. No pairs exceeded the token budget.

The fixed NLI baseline requires entailment probability at least 0.5 in both
directions. The learned classifier uses six symmetric probability features,
balanced logistic regression with C=1, and a validation-selected threshold of
0.55. Validation accuracy was 63.67%, versus 52.34% for the NLI baseline.
Its validation false-accept rate was 51.39%. The saved pretest decision already
ruled out using this as a trusted default gate.

## Cleartext diagnostics

On 16 authored diagnostic pairs, the classifier matched 14 authored expectations.
It rejected all eight authored meaning changes, including reversed roles,
removed negation, promoted uncertainty, dropped attribution and changed numbers.
It also rejected two valid splits, the Alice/Bob example and a split preserving
both people's record counts. These are agent diagnostics, not human ratings or
an independent benchmark. They were never training labels.

The two logged search failures still pass with about 0.99 entailment probability
in both directions. The checker misses the knowledge/how scope change in
`desse-test-257` and the incomplete prepositional phrase in `desse-test-359`.
An NLI score is not a grammar check or a semantic guarantee.

Applying the frozen classifier to previously inspected Cleartext outputs changes
only one selected search output across the saved development, former test and
domain development samples. The earlier test is now development material.
Its outputs and SARI stay unchanged. On the earlier development sample, search
SARI falls from 54.20 to 53.92 after one rollback. No simplification gain was
established. The reserved domain test inputs were not predicted.

## Reproduction and records

See the [protocol and commands](../../docs/STRUCTURE_MEANING.md). Core scoring
and fitting source is commit `538baa1`. Training, validation and test have
identical inference-module hashes and NLI weights. The test used the same
classifier hash recorded in the pretest decision. There were no test-driven
parameter or inference changes.

- [Preparation and pinned sources](preparation.json)
- [Training freeze](train_freeze.json), [validation freeze](dev_freeze.json) and [test freeze](test_freeze.json)
- [Validation threshold selection](development.json), [pretest decision](pretest_decision.json) and [test metrics](test_metrics.json)
- [Portable classifier](classifier.json), [environment](environment.json) and [resources](resources.json)
- [Development diagnostics](diagnostics_summary.json), [authored results](authored_results.json) and [logged failures](logged_failures.json)
- [Validation checks](validation.json)

The NLI model was pretrained on SNLI and MultiNLI. PAWS-Wiki contains existing
published human annotations. Google is the PAWS data source. Pinned public
downloads came from the publisher's Hugging Face repository after the original
archive returned HTTP 403. Sources and licenses are linked in the protocol.
Near duplicates, document overlap and pretrained exposure are not ruled out by
exact sentence exclusions. This experiment collected no new human ratings and
used no paid inference or synthetic training labels.

Only aggregate records, authored fixtures and the tiny classifier JSON are in
Git. Full pretrained weights, public raw rows and prediction traces remain in
the local experiment store. The original and continued T5 weights are preserved.
