# What is holding back sentence simplification

October 27, 2026. The main problem is disagreement between the task our data
teaches, the task our validation rewards, and the changes our runtime permits.
The model also copies too often. Adding another verifier or more training steps
before resolving those disagreements is unlikely to be a good use of the next
experiment.

The subsequent [training-alignment experiment](STRUCTURE_ALIGNMENT.md) records
the shared-check repair and continued training.

This diagnosis uses previously inspected data. It is development analysis, not
a new held-out result. The [aggregate audit](../evidence/structure-audit-20261027/summary.json)
and [frozen inputs and source hashes](../evidence/structure-audit-20261027/freeze.json)
make the counts reproducible. No weights, runtime defaults or live demo behavior
changed. The audit includes model forward passes, not another training run.

## Where changes disappear

The latest two-beam run contains 128 official DeSSE test examples. They have
already been inspected in previous experiments and are now development material.

| Reference annotation | Inputs | Model copies | Proposed changes rejected | Changes returned |
| --- | ---: | ---: | ---: | ---: |
| Reference changes the sentence | 97 | 46 | 32 | 19 |
| Reference copies the sentence | 31 | 30 | 1 | 0 |
| Total | 128 | 76 | 33 | 19 |

These are normalized text changes, including punctuation. They are not counts
of correct simplifications. The annotations can be unsuitable, and copying is
sometimes the right answer. Still, the table separates two distinct limits.
Removing every rejection check would leave the model's 76 preferred copies.
Improving generation alone would still encounter the rejection checks.

Among the 52 proposed changes, clause completeness, argument roles and complement
scope each fail on 12 inputs. Failures overlap. Complement scope is the only
failure on nine inputs, argument roles on eight, and clause completeness on five.
Removing just the complement check would therefore return 28 changes instead of
19, without fixing the other bottlenecks or establishing correctness.

The earlier 256-example run, before the complement check, has the same broad
pattern: 156 model copies, 100 proposed changes, 41 rejected changes and 59
returned changes. It is a different sample, so its score is not a controlled
comparison with the newer 128-example run.

## Training targets and safety checks disagree

Of 2,986 changed training targets, 366 fail the current baseline checks. The
complement signature disagrees on 361, split across 255 DeSSE and 106 WikiSplit
targets. Six fail the tense signature, with one overlap. These are conflicts
with the checks, not 366 proven bad labels.

The selected checkpoint actually encountered 231 of those conflicting targets
during continuation, including 227 scope conflicts. This is not just an issue
in the unused portion of the prepared data. The [supporting counts](../evidence/structure-audit-20261027/supporting_records.json)
record the seeded exposure calculation and domain-output hash.

Some targets detach an embedded claim. An authored example of the pattern is
"Alice thinks that Bob works here" becoming "Alice thinks. Bob works here."
Matching that target can improve training loss and reference overlap while
losing the relationship between Alice's belief and the claim. The current scope
check correctly rejects this authored example. Prepared training failures include
the same pattern under think, agree, realize and say. Raw rows remain local.

The check also rejects good changes. Our authored diagnostic keeps both
attributions in "Alice thinks that Bob works here and she hopes that Carol will
join him" by splitting before "She hopes". The parser attaches hopes as a
conjunct of works. `complement_scopes` walks that conjunction back to thinks,
assigning an extra embedded relationship that disappears after the valid split.
Only `complement_scope` fails. All other checks pass.

An explicit subject alone cannot solve this. "Alice said that Bob left and
Carol stayed" also has a separate subject in the second embedded claim. A rule
that always stops scope inheritance at a new subject would allow some real
attribution loss. The next fix must distinguish an independent reporting clause
from another claim inside the same report.

There is a controlled demonstration of this gate's effect. All 128 development
sources and raw model outputs are identical between the older continuation run
and the newer baseline. Adding the complement check changes returned edits from
67 to 52 and the overlap score from 59.63 to 54.08. The 15 additional rollbacks
are caused by the check, not new weights or a slower generator. That score drop
does not establish that the guard made quality worse. Both useful rejections and
false rejections occur.

## Validation asks an easier, different question

`expanded.reference_quality` filters training and validation references through
our own grammar, content, tense and connective checks. The official test samples
use length, markup and exact-overlap filters without reference-quality filtering.
This makes validation unusually compatible with our rules.

| Prepared split | Changed references | Pass current baseline checks | Pass rate |
| --- | ---: | ---: | ---: |
| Training | 2,986 | 2,620 | 87.7% |
| Validation | 96 | 68 | 70.8% |
| Earlier test | 197 | 33 | 16.8% |

All 28 current validation conflicts are complement-scope conflicts. The later
128-example sample has 97 changed references and only 18 that pass the current
baseline checks. Some test targets remove connective words, content, modality or
attribution. Others encounter parser errors. Treating every reference as a safe
rewrite would reward behavior the runtime is explicitly trying to prevent.

The same parser helps select data and judge outputs. That creates agreement with
its own preferences rather than independent evidence of grammaticality. The
current set-based, single-reference SARI implementation also rewards punctuation
overlap. It does not tell us whether a person finds the sentence easier or
whether every claim is preserved. Neither high overlap nor passing every parser
check is sufficient for promotion.

## Copying and candidate selection both matter

Actual forward passes through the selected checkpoint show lower mean token loss
for copying than for the published changed reference on 53 of 96 validation
examples and 82 of 97 examples in the later sample. This supports a strong copy
preference. Teacher-forced loss is not decoding probability or a quality rating,
and the targets differ in length.

Four-beam search already generates alternatives. Rechecking every saved beam,
including those skipped after a valid copy, finds:

| Later 128-example sample | Inputs |
| --- | ---: |
| At least one passing changed candidate | 60 |
| Search actually returns a change | 19 |
| A passing change exists but the returned text is a copy | 41 |
| Preferred copy hides a passing changed candidate | 39 |
| A skipped passing candidate adds a sentence | 12 |

Seventeen of the 41 skipped opportunities have copy annotations. Many candidates
only change punctuation. The 41 should not be reported as missed good splits.

An offline policy that chooses the first passing change instead of stopping at a
copy raises the overlap score from 36.36 to 40.75. It returns 60 changes instead
of 19. This policy has not been adopted. It makes unnecessary edits on copy
annotations and uses the same checks that already miss known errors.

An oracle allowed to see the reference scores 41.71 among checked candidates
plus identity, versus 64.32 among all raw candidates plus identity. This is a
diagnostic ceiling for this saved candidate pool and this overlap metric. It is
not deployable or a semantic-quality ceiling. Better ranking can recover some
overlap, but cannot solve the entire data/check conflict.

## Training coverage and domain coverage are limited

The continuation prepared 3,986 rows. Its selected checkpoint is step 300 with
effective batch size eight. Reproducing the seeded shuffle gives 2,400 distinct
rows seen in that continuation, or 60.2%. It includes 615 annotated copies,
1,106 DeSSE rows and 1,294 WikiSplit rows. The initializer may already have seen
some replay examples, so the other 1,586 rows are not necessarily new to the
whole model lineage.

The complete run processed 3,200 rows over 400 steps. Step 400 had slightly worse
validation loss than step 300, 0.18478 versus 0.18351. Limited exposure is real;
it does not prove that more steps would improve generation. Checkpoint selection
currently uses teacher-forced loss on the filtered validation sample, including
targets the new scope check rejects. That objective needs attention first.

The continuation labels describe student essays and general Wikipedia text.
They provide no explicitly labeled project-domain training pairs for technical,
corporate or political text. In the saved 24-input domain development run, the
baseline proposes 13 changes and returns two. Nine proposed changes fail clause
completeness. Those inputs have no trusted simplification references, so two
returned changes is coverage, not accuracy. The reserved domain test was not
predicted during this audit.

## The verifier and demo do not close these gaps

The [previous learned meaning experiment](../evidence/structure-meaning-20261027/README.md)
improved PAWS classification but still accepted 128 of 282 non-equivalent test
pairs. It also accepted the two logged Cleartext scope/fragment failures. Another
threshold adjustment cannot be treated as a solution to structural grammar or
meaning preservation.

The live `/api/config` endpoint reports rules and the learned ranker, with rules
preferred. `models/demo/server.py` loads v7 word models and optionally the ranker.
It has no T5 backend. Training the structural generator therefore does not change
the demo. This is a separate integration gap, deliberately left while structural
quality is uncertain.

## What to do next

1. Make the data, checks and evaluation agree on attribution and discourse
   preservation. Keep target/check conflicts as a separate diagnostic group.
   Correct the conjunction-scope mechanism with positive and negative contrasts;
   do not blindly disable the scope check or discard every conflict as bad data.
2. Build a broader development partition using input-only sampling, with separate
   copy, coordination, relative-clause, attribution and domain breakdowns. Keep
   parser-selected validation as an easier diagnostic subset. Use new reserved
   material for the next final comparison. Do not retune on these former tests.
3. Compare the model's copy with passing alternatives using a bounded selector
   that rewards an actual shorter sentence split and penalizes unnecessary edits.
   The saved pool demonstrates headroom, but punctuation-only overlap gains and
   known grammar failures must be reported separately.
4. Retrain on published targets compatible with the clarified task. Test fuller
   training exposure and checkpoint selection based on decoded development
   behavior, alongside loss. Only then compare more data or a larger generator.
5. Add an opt-in T5 demo backend after a candidate clears that comparison, with
   visible backend identification and a reproducible local weights bundle.

These steps require no new human review round. They can use existing published
labels, public domain inputs and explicit authored diagnostics. Automatic checks
still leave uncertainty about naturalness and subtle meaning.

## Reproduce the audit

From the repository root, with the existing local experiment directories:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  models/.venv/bin/python models/scripts/audit_structure_bottlenecks.py \
  --state /path/to/cleartext-experiment-store \
  --output /path/to/new-audit-output \
  --probes models/evidence/structure-audit-20261027/probes.json \
  --teacher-forcing
```

The command refuses an existing output directory and reads local weights only.
It writes aggregate counts and local reference/candidate traces. Only aggregates,
hashes and authored probes are committed. The [environment](../evidence/structure-audit-20261027/environment.json),
[protocol](../evidence/structure-audit-20261027/protocol-v2.json) and
[resources](../evidence/structure-audit-20261027/resources.json) accompany the run.
Yoga session `cleartext-audit-20261027` ran with two CPU threads and a 4 GB cap.
The final audit took 70.4 seconds and peaked at about 880 MiB RSS. Both original
and continued checkpoint hashes remained unchanged.
