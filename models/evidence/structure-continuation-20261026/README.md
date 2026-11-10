# More data and continued T5 training, October 26

Continued T5 training improved the frozen held-out reference score from 36.17 to
39.38 on 256 published DeSSE pairs. It produced more accepted rewrites and copied
more of the references annotated as unchanged. This is a general sentence
splitting result. Project domain outputs still contain grammar errors, and a
later attribution repair has development evidence only. The demo default stays
unchanged.

## Data and training

[DeSSE](https://github.com/serenayj/DeSSE) provides published crowdsourced
decompositions. [WikiSplit](https://github.com/google-research-datasets/wiki-split)
provides mined Wikipedia edits. The [preparation guide](../../docs/STRUCTURE_CONTINUATION.md)
and `data_manifest.json` describe pinned downloads, exclusions and filters.
There are no agent-authored training references or fabricated human approvals.

| Split | DeSSE | WikiSplit | Total | Annotated copies |
|---|---:|---:|---:|---:|
| Training | 1,775 | 2,211 | 3,986 | 1,000 |
| Development | 128 | 0 | 128 | 32 |
| Test | 256 | 0 | 256 | 59 |

WikiSplit includes 211 filtered replay examples and 2,000 additional examples
from a later training scan range. Source/full-target exact overlaps with released
evaluation data, earlier inspected evaluations and all 72 project domain inputs
are excluded. The test sample uses basic length/markup/overlap filters, without
reference correctness gates. Published decompositions can omit facts or
discourse relations. These references are not independent model quality ratings.

The run initialized from saved T5 weights with SHA256
`43024edcf6d47e0019aaa8e90852007faf7485b95d7cd67f972569adbe141cec`.
It completed 400 optimizer steps with Adafactor, learning rate 0.0001, batch size
2, accumulation 4 and seed 4701. It processed 3,200 examples. No prepared pair
was removed by the tokenizer length check. The optimizer was new.

Token-weighted development loss selected step 300, with loss 0.18351, versus
0.24410 at initialization. Step 400 had loss 0.18478. Initialization was eligible
for selection. The selected checkpoint had processed 2,400 examples.
`training.json` records the full history and configuration.
The original checkpoint remains unchanged.

## Frozen comparison

Training and matched inference used source commit
`24a7e0589206161ce8547fd33e58042866d19335`. Model, data and source hashes were
recorded before scoring. Both models used identical deterministic two-beam
generation and the same checks, including predicate tense. Test scores did not
select the checkpoint or alter settings.

| Test metric, 256 pairs | Original T5 | Continued T5 |
|---|---:|---:|
| Raw single-reference SARI | 45.70 | 46.38 |
| Filtered single-reference SARI | 36.17 | 39.38 |
| Changed after checks | 41 | 59 |
| Annotated copies kept | 57/59 | 58/59 |
| Raw outputs passing all inference checks | 166 | 185 |
| Mean longest sentence, alphabetic words | 18.18 | 17.45 |
| Median inference seconds | 0.233 | 0.224 |
| p95 inference seconds | 0.482 | 0.469 |

The paired SARI gain is +3.22, with a 95% bootstrap interval of +1.63 to +4.75
over these sampled rows. Identity scores 29.35 and the existing rules score
29.59. This set-based, single-reference SARI implementation measures reference
overlap. It does not establish meaning preservation or fluency. The bootstrap
does not account for unknown essay grouping.

Without the new tense check, original and continued T5 score 36.26 and 39.47.
The matched training gain therefore persists when that filter is removed.
Development scores under the frozen checks are 53.44 and 59.63. Development
references are filtered and test references are not, so their absolute scores
should not be compared as evidence of overfitting.

Some copied input fragments fail the parser's complete-clause check. Both final
systems have 38 such test failures. A copied input is not a newly introduced
grammar error, and a passed check is not a correctness certificate. Latency is
single-process Mac MPS inference for this sample, not demo latency or CPU serving
throughput. Full aggregates are in `dev_comparison.json` and
`test_comparison.json`.

## Attribution repair after development inspection

Agent inspection covered the first 20 changed filtered development outputs per
model in fixed row order, all changed annotated copies and every changed domain
output. `agent_diagnostics.json` records concerns without publishing raw rows.
This was not an independent human evaluation.

Some outputs turn a statement under "agree", "think" or reported speech into an
independent assertion. The new forward complement scope check rejects missing
dependency/lemma relationships, including coordinated embedded claims. It allows
repeating the reporting predicate across a split. Parser attachments remain
ambiguous, so it can also reject valid rewrites. Relative clauses, speaker
identity, backward complements and coreference remain unresolved.

The guard was replayed on the same saved development raw outputs only. It
additionally rejected 9 original-model and 15 continued-model rewrites. Both
models then kept all 32 annotated copies. Development SARI became 49.57 and
54.08, with 44 and 52 changed outputs. The new guard has no fresh held-out score.
The frozen test result above was not recomputed under it.

## Project domain check

The 24 previously inspected development inputs include eight technical, eight
corporate and eight political sentences. No references or human ratings exist.

| Changed development inputs | Original T5 | Continued T5 |
|---|---:|---:|
| Technical | 1/8 | 1/8 |
| Corporate | 0/8 | 0/8 |
| Political | 0/8 | 1/8 |

Both technical outputs retain comma splices. The additional political output
detaches a subordinate fragment, despite passing parser checks. The scope guard
does not change these counts. Coverage therefore does not establish a domain
quality gain. The six existing authored diagnostics produce identical outputs
between models; both change only the explicit two-subject coordination example.
Simple shared-subject and relative-clause examples remain unchanged.

A new blind sheet groups the frozen and scope-guarded model outputs into 26
variants. Human fields remain blank. The original 48-row author/reviewer sheet
also remains blank. No predictions were generated for the reserved 24 domain
test inputs. Raw inputs, outputs and review sheets remain local.

## Artifacts and validation

The fitted bundle is `cleartext-t5-continued-20261026.tar.gz`, 225,721,722 bytes.
Its SHA256 is:

```
f831fbe6368d87b19daca590bf8121ef57f0aa528beadabcf9dbea58f82c6674
```

The selected weights have SHA256
`36787b852ab88f1d61673465d63046b293b4bf53a9929fa0afc077bb592b7756`.
Archive and extracted weights were verified after copying to Yoga. The local
bundle contains the checkpoint, tokenizer and training/data receipts. It contains
no training rows and is not an uploaded release. An absolute initializer path in
the original training receipt is historical metadata, not a runtime dependency.
The aggregate receipts in this directory omit personal paths and hostnames.

Extract the bundle into a new directory and use the repository dependencies:

```sh
python models/scripts/structure_experiment.py predict \
  --run /path/to/extracted-bundle --model seq2seq --device auto \
  --text "The analyst reviewed the report, and the manager checked the totals."
```

Current source applies both tense and complement scope checks. Reproducing the
frozen scores requires commit `24a7e05`, the recorded local data and the original
checkpoint too. Checkpoint receipts alone do not recreate an identical run.

Training and inference ran on the Mac because Yoga had about 4 GB disk free,
below its heavy-job reserve. The estimate was 3 GB RAM and 1 GB new disk use.
The named Yoga coordinator was `cleartext-continuation-20261026`; the Mac worker
was `cleartext-continue-20261026`. Training paused briefly at warning pressure
and resumed at normal pressure. The domain check used Mac session
`cleartext-domain-20261026` under a second Yoga coordinator. Both jobs completed
and their outputs were copied back. No unrelated process was stopped.

The focused suite passed 155 tests in 8.69 seconds. Ruff and whitespace checks
passed. New regressions cover changed tense, lost complement scope, repeated
attribution, comma-splice parser attachments, step-zero checkpoint retention,
source-group separation and published holdout exclusion. The live demo still
returns HTTP 200 at its configured Tailscale address.

The next data need is reviewed domain simplifications, including examples that
should remain unchanged. The existing sheets provide the material for that
review. More mined general text alone has not solved domain rewriting, and
automatic reference scores do not justify changing the demo default.
