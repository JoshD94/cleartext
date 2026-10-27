# Structural repairs and domain review, October 26

The experimental splitter now rejects the known duplicated-subject and noun-list
splits. Its candidate generator copies a subject only across eligible main-clause
verb coordination. Character slices preserve technical spelling. The demo uses
the refitted ranker when the experimental option is selected. Structure starts
off, and grammar rules remain preferred.

These repairs reduce coverage. They do not establish better overall quality.

| Same training/development setup | Previous ranker | Repaired ranker |
|---|---:|---:|
| Training pairs | 2,000 | 2,000 |
| Development pairs | 128 | 128 |
| Training candidates, including identities | 10,293 | 2,153 |
| Development SARI | 40.46 | 34.98 |

The refit uses source `c6ffeaa7bc986bee246fa1ad3c4c4f2af2fc32c0`, ranker feature
version 2 and threshold 10. All five threshold candidates tied on development
SARI. The existing tie policy selects 10. Candidate labels are single-reference
SARI gain over identity, not human judgments. Earlier ranker weights require
frozen earlier source. Later source keeps these numerical feature meanings.

Seven authored diagnostics cover shared and separate subjects, a pronoun subject,
a noun list, passive ellipsis, a shared modal and Diffie-Hellman spelling. They
check specific behaviors and are not a quality benchmark. No fresh benchmark
test score was computed. The [previous comparison](../structure-followup-20261026/README.md)
is a different evaluation and must not be used as a before/after comparison here.

## Domain review material

Nine pinned official documents supply 72 unreviewed inputs, with 24 training,
24 development and 24 reserved test inputs. Each split has eight inputs per
domain. Documents differ across splits. Sources cover internet standards,
company reports and US inaugural speeches. The configuration, selection policy,
download hashes and exact-overlap exclusions are in `source_manifest.json` and
[the review guide](../../docs/STRUCTURE_REVIEW.md).

| System | Changed development inputs |
|---|---:|
| Grammar rules | 0/24 |
| Repaired ranker | 0/24 |
| Existing T5 | 1/24 |

Changed counts measure coverage. No human quality ratings or references exist.
The one T5 rewrite still contains a comma splice on inspection. This collection
shows that useful rewriting across these domains remains unresolved.

The first extraction included filing forms and fragments. It remains a local
diagnostic. Corporate section anchors and parser eligibility were revised before
reserved testing. The committed preparation code reproduced the selected input
files byte for byte. PDF headings and parsing errors can remain. Reviewers must
reject unsuitable inputs. This small convenience sample does not establish
broad domain performance or eliminate paraphrases and pretraining overlap.

The local author/reviewer sheet has 48 train/development rows. A separate blind
sheet groups 72 development system outputs into 25 unique variants. All human
fields remain blank. Its system key stays separate. No reserved test predictions
were generated. Raw source text, review rows and weights stay outside Git.

## Training and validation

The trainer can initialize from a local checkpoint, record its hashes and start
a fresh optimizer in a separate output directory. It cannot overwrite the input
checkpoint. Tiny-model regression tests verified an actual gradient step and
checkpoint reload without a Hub lookup. No further project T5 training ran.
The original T5 weights, CRF, v7 and older bundles remain intact.

The focused suite passed 124 tests in 10.17 seconds. Ruff and whitespace checks
passed. Seven API cases passed on both the temporary preview and live demo, with
outputs matching their displayed traces. A browser interaction rendered the
expected learned split. Archive hashes and a standalone CLI prediction passed.
The temporary preview server and browser tab were stopped; the live demo remains
running on Yoga in `cleartext-demo-20261026`.

The refit and development inference ran on the Mac because Yoga had only about
4.6 GB disk free, below the heavy-job reserve. The refit estimate was 2 GB RAM
and under 100 MB disk. Mac memory pressure was normal. Named sessions were
`cleartext-repair-20261026`, `cleartext-review-20261026` and
`cleartext-review2-20261026`. Initial and revised extraction audits remain local.
The Mac used the frozen repair commit; the review workflow commit is recorded
in `review_preparation.json`.

## Fitted artifact and next step

The local ranker bundle is `cleartext-structure-ranker-repaired-20261026.tar.gz`,
50,595 bytes. Its SHA256 is:

```
fde430d0866f360e60092bc34231698627612a6e2f5a45ffc891f828162eb0ac
```

`bundle.json` records the model hash too. The bundle contains no training rows.
It remains a local artifact, not an uploaded release. Reproduction needs the
recorded source, original training/development material and compatible Python
model dependencies. Receipts alone cannot recreate an identical pickle.

People must now approve reference simplifications and judge model outputs.
Then prepare a larger reviewed training mixture, continue T5 from the saved
checkpoint and choose settings on development data. Freeze settings before
reserved testing. Human correctness judgments and serving latency measurements
are still needed before changing the demo default.
