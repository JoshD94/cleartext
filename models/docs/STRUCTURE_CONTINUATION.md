# More data and continued T5 training

This round uses published sentence pairs while the project's domain review
sheets remain unfinished. It does not invent human approvals or train on agent
rewrites. The existing v7 demo and experimental ranker remain available.

The [completed October 26 run](../evidence/structure-continuation-20261026/README.md)
records 400 training steps and a held-out SARI increase from 36.17 to 39.38 under
the frozen comparison checks. Domain development outputs still contain grammar
errors. A later attribution guard has development evidence only.

## Prepared data

[DeSSE](https://github.com/serenayj/DeSSE) supplies 13,200 released training
pairs of student essay sentences and crowdsourced decompositions. The
[ABCD paper](https://aclanthology.org/2021.acl-long.303/) describes the annotation
process, which intentionally removes discourse connectives. Published references
can therefore conflict with our requirement to preserve conditions and contrast.
The pinned repository includes an MIT license.

Training/development selection requires a complete single-sentence source and
either an annotated copy or two to four shorter complete output clauses. Shared
checks must preserve content, participants, quantities, negation and modality.
Additional counts preserve discourse words and finite verb tense/mood. These
parser checks narrow the sample and cannot certify correctness.

Extra [WikiSplit](https://github.com/google-research-datasets/wiki-split) examples
come from training lines 50,000 through 69,999, after the original pilot's scan
range. All released test/tune/validation sources and full targets are excluded
from this additional sample. The same conservative reference filters apply.
WikiSplit contains mined Wikipedia edits, without human correctness labels.

| Prepared split | DeSSE | WikiSplit | Total | Annotated copies |
|---|---:|---:|---:|---:|
| Training | 1,775 | 2,211 | 3,986 | 1,000 |
| Development | 128 | 0 | 128 | 32 |
| Test | 256 | 0 | 256 | 59 |

The WikiSplit contribution includes 211 filtered original training examples
and 2,000 additional examples. Copies come from the released annotations.

Development uses official DeSSE validation examples plus a source-hash 10%
partition of official training data. All annotations of the same normalized
source share an assignment. Every released test and development source/full
target is excluded before training selection. Prior inspected evaluation rows
and all 72 project domain inputs stay excluded.

The test sample uses only source/reference length, markup and exact-overlap
filters. It does not require reference content/grammar checks to pass. Test
references can contain omissions. Results need both raw and filtered model
outputs, copy accuracy and the reference-overlap score. DeSSE has no released
essay IDs for document grouping. Exact deduplication cannot exclude paraphrases,
shared authors or foundation-model pretraining overlap.

The [source configuration](../structure-expanded-sources.json) pins commits,
download hashes and URLs. Raw documents, pairs and weights remain outside Git.

```sh
PYTHONPATH=models/src /path/to/venv/bin/python -m cleartext.structure.expanded \
  --cache /path/to/desse-cache --wikisplit-cache /path/to/wikisplit-cache \
  --run /path/to/new-run --previous /path/to/original/data.json \
  --exclude /path/to/prior-evaluation-and-domain-inputs.json
```

## Training and comparison

The bounded continuation starts from our saved T5 weights. It uses 400 optimizer
steps, batch size 2, four accumulation steps, Adafactor, learning rate 0.0001 and
seed 4701. That processes 3,200 examples, rather than every prepared pair.
The optimizer is new. Token-weighted development loss selects the checkpoint,
including step zero. A continuation that never improves this loss retains its
initializer. Existing output checkpoints cannot be overwritten.

```sh
python models/scripts/structure_experiment.py train-seq2seq \
  --run /path/to/new-run --initial-checkpoint /path/to/original/seq2seq \
  --learning-rate 0.0001 --max-steps 400 --batch-size 2 \
  --accumulation-steps 4 --device mps
```

T5 generation now checks predicate tense/mood against the source. It rejects a
rewrite that changes "reviewed" to "reviews", while allowing a shared "was" to
be repeated across a split. This check is parser-dependent and does not certify
which participant has which tense. Ranker feature meanings are unchanged.

Generation also preserves forward complement scope. For example, splitting
"The company said that revenue increased and costs fell" must keep both claims
under "said". Repeating "The company said that" can preserve the relationship.
Matching dependency/lemma counts can still miss speaker changes, relative-clause
meaning and parser errors. Backward complements are excluded because comma
splices can produce spurious backward dependency attachments.

The comparison uses identical deterministic generation and checks for the
initializer and continuation. It also computes results without the added T5
checks, using the same raw outputs, to separate the effect of filtering from
changed weights. Checkpoints, data and source hashes are frozen before scoring.
The completed October 26 comparison predates the complement scope guard and
includes the tense check only. A separate development replay records the scope
guard's cost in coverage. The held-out score was not recomputed after that repair.

```sh
PYTHONPATH=models/src /path/to/venv/bin/python -m cleartext.structure.continuation \
  --run /path/to/new-run --baseline /path/to/original/seq2seq \
  --split dev --device mps
```

Use `--split test` only after model/settings selection is complete. Each
evaluation writes to a fresh directory. The primary automatic comparison is
continued versus initial T5 SARI under identical current checks on the reserved
256-pair sample. No settings may change in response to that test score.

Independent model ratings and approved references for technical, corporate and
political writing are still missing. This training round cannot establish
improvement on those domains or justify changing the demo default by itself.
