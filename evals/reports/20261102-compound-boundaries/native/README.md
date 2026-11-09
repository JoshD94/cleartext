# Completed compound-splitter replay

All **900/900** expected cells completed: 90 sourced development passages,
30 each from computer science, corporate communication and political science,
across Kea-Roy's eight-system ladder plus two opt-in compound arms. Completed
November 2 local time / November 3 UTC. No training or T5 decoding ran.

## Result

The new splitter changes **1/90 passages**, compared with 0/90 for the old
structure rules. Its guarded pipeline differs from `full_guarded` on exactly
that same one passage; both pipelines change 85/90 inputs overall, mostly
through the existing lexical stage. These lexical changes are not a gain from
compound splitting. The separate parser replay still accepts 0/18 authored
passages; it is not an additional 18-passage native scoring run.

| Metric (all 90 passages) | Existing guarded pipeline | With compound splitter | After minus before |
| --- | ---: | ---: | ---: |
| Flesch reading ease (higher is easier) | 39.485784 | 39.471898 | -0.013886 |
| Flesch–Kincaid grade (lower is easier) | 13.947921 | 13.949305 | +0.001385 |
| Mean sentence length, words | 24.941111 | 24.938889 | -0.002222 |
| Embedding similarity | 0.991188 | 0.991128 | -0.000060 |
| Bidirectional NLI mean | 0.837272 | 0.837191 | -0.000081 |

There is **no measured average readability improvement**. The changed passage
has a slightly shorter measured sentence length but worse Flesch scores. The
single standalone split has embedding similarity 0.994644 and NLI mean
0.978463; this is one inspected example, not an accuracy estimate. With lexical
edits included, NLI on that passage is only 0.165327 before and 0.158038 after,
despite embedding similarity above 0.99. This discrepancy warrants investigating
lexical substitutions and the metric's behavior; similarity alone does not
certify meaning preservation. The compound guard has no live NLI acceptance gate.

The splitter stays **opt-in**. Useful delivered changes are bounded relative-clause
support, a joint-subject safety check, preservation of exact untouched fragments,
optional length eligibility and inspectable rejection reasons. These are tested
mechanisms, with limited real-data coverage and no demonstrated general quality gain.

## Evidence and reproducibility

- [Exact eight-arm ladder](ladder.md) and [five-arm comparison](comparison.md).
- [Aggregate results](aggregate-results.json): all/changed-only scores, abstentions,
  domain counts and domain means. Changed-only means describe each system's own
  changed subset; they are not paired improvements.
- [Paired comparison](paired-comparison.json): both all-90 and the one identical
  incrementally changed subset, including before/after values for every metric.
- [Provenance](provenance.json): source/input/model/resource hashes, runtime,
  source commits, freeze signature, seed checks, per-attempt job receipts and
  raw-artifact checksums. Paths use `repo/` and `HOME/` aliases.

The complete replay used implementation `68b4fe6` and checkpoint driver
`d341687`, on Yoga CPU with one thread. `scripts/verify` passed **490 tests and
13 subtests**. Export validation independently checked every expected cell,
finite numeric metrics, all system means and exact paired output differences.

The earlier whole-system job reached its queue limit and lost unfinished work.
Recovery verified and reused its 180 completed original/rule-based cells, then
saved each generated output and scored cell in nine bounded batches. Batch four
was stopped and automatically requeued once. No duplicate inference job ran.
The local state is `$HOME/.local/state/cleartext/week7-compound-checkpoints-20261102`;
its frozen `plan.json` and `run-batch.sh` record exact commands. All 900 cells,
predictions and Kea-Roy's generated review checklist remain there, outside Git.
The checklist was generated, not human reviewed.

These passages were inspected during development. Automatic readability, NLI,
entity and literal checks are not percentages of correct simplifications or
independent test accuracy. The earlier Mac results remain a separate historical
run; the table above compares both arms within this completed Yoga replay.
