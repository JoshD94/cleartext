# Candidate search experiment, October 27

This pass implemented and tested four-candidate T5 search with additional
sentence-boundary checks and bounded comma repair. It did **not** establish an
overall quality gain. Keep it opt-in. The demo and current decoder retain their
behavior. The experiment reused saved continued weights, with no new training,
human ratings, paid services or predictions on the reserved domain test inputs.

## Matched results

All variants use the same checkpoint and paired rows. SARI here is set-based,
single-reference token overlap on published DeSSE crowd decompositions. It is
not a human quality score.

| Variant | Dev SARI, 128 pairs | Dev changed | Fresh test SARI, 128 pairs | Test changed | Test annotated copies kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original text | 30.58 | 0 | 29.86 | 0 | 31/31 |
| Current two-beam decoder | 54.08 | 52 | 37.03 | 19 | 31/31 |
| Same raw output, stricter checks | 52.55 | 48 | 35.60 | 15 | 31/31 |
| Four-candidate search and primary-beam repair | 54.20 | 53 | 36.36 | 19 | 31/31 |

Search minus current decoder on the fresh test is -0.675 SARI, with paired
1,000-resample row bootstrap 95% interval [-2.049, 0.601]. The interval spans a
tie. Search minus stricter checks alone is +0.753, interval [-0.092, 2.157].
Neither comparison supports an overall improvement claim. These are a different
sample and a later pipeline than the earlier 256-pair continuation evaluation.

The new checks flag four changed baseline outputs and zero changed search
outputs on both dev and test. That is a selection diagnostic, not independent
evidence of grammar or meaning. Copied malformed sources still fail checks.
Both current decoder and search average 18.02 words in their longest output
sentence on the fresh test, versus 19.26 for the original text.

Observed mean prediction time on Yoga CPU with two threads was 0.264 seconds
for the current decoder and 0.305 seconds for search. This is about 16% more.
Source parsing was shared and excluded from both timings. Predictions ran in a
fixed order, so these measurements are not a general performance benchmark.

## What inspection found

The 24 domain development inputs have no approved references or quality scores.
Search changes two technical inputs. It repairs the observed QUIC comma splice
into four sentences and chooses a lower beam for another technical example.
The current decoder changes one technical and one political input. Search
returns the original for the observed detached political fragment. Across six
authored diagnostics, both change only the explicit separate-subject example.
Shared-subject and relative-clause rewriting remain limited.

Inspecting the seven differing fresh-test outputs found false rejections and
missed errors. The source-root gate can reject a complete conditional plus main
clause or a sentence starting with "As a result". Comma repair can still expose
an incomplete prepositional phrase. A lower beam can turn a knowledge/how clause
into a separate assertion, losing its scope. The existing verbal-complement
check misses that noun-complement case. See [agent diagnostics](agent_diagnostics.json).
These are agent notes, not human ratings. No settings changed after this test.

## Protocol and saved evidence

Inference source is commit `1fce73e`. All inference-module hashes and model
hashes match across final dev and test. The selected continued weight SHA256 is
`36787b852ab88f1d61673465d63046b293b4bf53a9929fa0afc077bb592b7756`.
The [guide](../../docs/STRUCTURE_SEARCH.md) explains prediction and reproduction.

An initial development run allowed repairs on every beam. Domain development
inspection found a lower-beam repair exposing a noun-list fragment. The first
test job was interrupted before its outputs or scores were read. Every one of
its 128 selected pairs was excluded from the final sample. Repairs were then
restricted to the primary beam and development rerun before the final test.
The underlying parser ambiguity remains. See [development decision](development_decision.json).

Final test sampling used previously unscored official DeSSE test pairs, fixed
source hashes, length/markup filters and exact source/full-target exclusions.
It used no reference grammar/content gates or model scores. All official DeSSE
test keys were excluded from the earlier training run. Exact matching cannot
rule out near duplicates, document overlap or pretrained-model exposure.
Beam width, alternative selection and repair change together, so the comparison
does not isolate their individual effects.

- [Preparation and fixed settings](preparation.json)
- [Development freeze](dev_freeze.json) and [metrics](dev_metrics.json)
- [Fresh test freeze](test_freeze.json) and [metrics](test_metrics.json)
- [Environment](environment.json), [resources](resources.json) and [validation](validation.json)

Only aggregate results, hashes and diagnostic IDs are committed here. Raw rows,
candidate traces and the unchanged model remain in the local experiment store.
