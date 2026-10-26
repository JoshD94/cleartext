# Learned sentence structure experiments

These experiments compare three approaches on the same bounded WikiSplit sample.
Training and comparison run through a separate CLI. The demo can load a checked
ranker as an optional structural stage before the existing v7 word models.

The [October 26 pilot record](../evidence/structure-20261026/README.md) contains
the actual training budgets, aggregate results, environment and model hashes.
The [follow-up record](../evidence/structure-followup-20261026/README.md) tracks
the revised checks, fresh validation slice, technical audit and demo decision.

The rewrite ranker is a gradient boosting regressor. Its candidates include the
existing structural rules and broader punctuation/conjunction split proposals,
with optional subject copying. It learns candidate preferences from the change
in single-reference SARI relative to leaving the source unchanged. Each source
has equal total training weight. This is reference-derived supervision, not a
human judgment of correctness. The generator still limits what it can choose.

The structural editor is a linear-chain CRF over source token features, including
POS, dependencies and neighboring tokens. It predicts keep/delete actions and
insertions from a vocabulary derived only from training references. A subject
copy slot can insert the current source subject rather than memorize its name.
Sequence alignment supplies weak labels. Unsupported insertions, content-word
deletions and failed alignment round trips are excluded. The saved training
summary reports how much of the data this restriction covers. It supports
monotonic edits, not arbitrary clause reordering. Training reference components
also supply identity examples as copy regularization. No synthetic simplification
gold or new human ratings are created.

The encoder-decoder is the 60-million-parameter T5-small checkpoint, fine-tuned
locally with a sentence-splitting prefix. It can rewrite more freely. Its best
checkpoint is selected by development teacher-forced loss. This generative model
is a new experimental branch of the architecture; it does not use a hosted API.

## Data and evaluation

The downloader pins WikiSplit to commit
`80f5bddf42d1a68201b7b107ee4b8d86d82e9fde` and records downloaded file hashes.
The initial pilot uses official train, tune as development, and test files.
It leaves the separate validation file unused for the follow-up. All eligible development/test sources and full targets
are excluded from the bounded training pool by normalized exact match. Development
also excludes eligible test matches. Sampling uses deterministic source hashes.
The default training pool is the first 50,000 released training rows, with 2,000
selected pairs, 128 development pairs and 128 test pairs. Filters require two
target sentences, bounded lengths and at least 80% source token retention.
The manifest reports filtering and overlap counts.

The dataset is automatically mined and has noisy editorial changes. Exact-match
filtering does not rule out paraphrased duplicates or overlap in T5 pretraining.
This is a filtered general-domain pilot, not a technical-domain benchmark.

Ranker and editor abstention thresholds are chosen on development SARI. Test
references do not select model settings. The final comparison refuses to overwrite
an existing result. It includes identity and the current structural rules, every
model's raw proposals and its filtered outputs. It reports single-reference,
set-based SARI with deletion precision, change rate, split rate, maximum sentence
length, content retention, diagnostic check failures, latency and paired bootstrap
intervals for SARI differences. The SARI implementation is documented in code;
it is not assumed numerically interchangeable with other tokenizations or versions.

Shared checks cover numbers, negation, modal verbs, names, protected terms,
content lemma retention and additions, clause completeness, and position-independent
predicate/argument signatures. They work across token count changes, unlike the
existing lexical substitution guard. Parser errors can reject valid edits. Passing
these checks does not establish semantic equivalence or fluency. Conservative
content checks can reject useful paraphrases, so raw results must be retained.
Modal and negation scope are compared by predicate, including shared operators
on coordinated verbs. Counting the words alone would miss a rewrite that changes
"may check and send" into "may check. Sends." Logical connective counts are also
preserved conservatively.
No automatic metric result promotes a model into the demo.

The revised completeness check requires a finite main predicate and its own
subject. An embedded relative-clause verb cannot complete a noun phrase, and a
participle needs a finite auxiliary. For example, "The detector emitting a
signal" fails, while "The detector is emitting a signal" passes this check.
Standalone subordinate clauses and imperatives abstain conservatively. The
relative-word check follows the main subject and argument phrases, so "some
of whom" cannot pass as a complete subject in a detached relative. It skips
embedded clause boundaries to preserve valid relative clauses inside a sentence.

## Follow-up evaluation

After inspecting the first pilot's test failures, that test set becomes
development evidence. Use `prepare-followup` for a separate evaluation slice
from the previously unused official `validation.tsv`. It preserves the original
training and development rows and excludes normalized sources and full targets
from every previously selected split. It records the previous data hash and
the new file hash. Near duplicates and pretraining overlap remain possible.

```sh
python models/scripts/structure_experiment.py prepare-followup \
  --run /path/to/followup --previous-run /path/to/pilot \
  --cache /path/to/cache --test-size 256
python models/scripts/structure_experiment.py train-ranker --run /path/to/followup
```

The ranker must be refit because clause-check results are inference features.
Reuse the unchanged editor and selected T5 checkpoint by copying or linking
`editor.pkl` and `seq2seq/` into the new run. Freeze model, code, data and input
hashes before running `evaluate`. Keep the original pilot immutable.

The initial technical development audit selects the first six eligible
20..70-word sentences from each HTTP and TLS RFC introduction. Once those
inputs inform grammar fixes, reserve a fresh audit from section 2 of each RFC
with `--section 2`. Selection precedes predictions.
It records source URLs and document/input hashes. There are no reference
simplifications or invented quality ratings. Review every changed output for
fluency, meaning, conditions, quantities and terminology, and identify that
review as agent inspection unless people actually supplied ratings.

```sh
PYTHONPATH=models/src python -m cleartext.structure.technical prepare \
  --cache /path/to/cache --inputs /path/to/technical_inputs.json --section 2
PYTHONPATH=models/src python -m cleartext.structure.technical audit \
  --run /path/to/followup --inputs /path/to/technical_inputs.json --device mps
```

## Demo integration

Use the existing demo environment with the structural dependencies installed.
Load only a trusted local ranker artifact, with the same sklearn version used
for fitting. The October 26 bundle uses scikit-learn 1.9.1 and spaCy 3.8.16,
as recorded in `models/requirements-lock.txt`, with `en_core_web_sm` 3.8.0.
Weights stay outside Git. To start from `models/`:

```sh
CLEARTEXT_STRUCTURE_RANKER=/path/to/followup/ranker.pkl \
  .venv/bin/python demo/server.py --host 127.0.0.1 --port 8770
```

Turn on "Simplify structure" and choose "Learned splitter (experimental)".
Structure starts off, and grammar rules remain the preferred method. Without
an artifact, only grammar rules are available.
The word strictness slider does not change the structural threshold selected
on development data. Inputs exceeding one sentence or 70 words are preserved
by the learned stage. The adapter rechecks the selected output before the
pipeline applies phrase and word edits. Detected technical terms retain the
pipeline's existing protection. The structural trace uses the actual pipeline
result, including abstention, rather than running a second prediction.

## Run

Use Python 3.12 and a separate environment:

```sh
python3.12 -m venv /path/to/structure-venv
/path/to/structure-venv/bin/pip install -r models/requirements-structure.txt
/path/to/structure-venv/bin/pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /path/to/structure-venv/bin/python models/scripts/structure_experiment.py all \
  --run /path/to/new-run --train-size 2000 --dev-size 128 --test-size 128 \
  --max-steps 400 --batch-size 4
```

Follow the host placement and named tmux instructions before running training.
Estimated peak RAM is 5 GB, with under 3 GB additional disk for the environment,
download cache and one saved T5 checkpoint. Check actual usage during the run.
The script checks capacity during training and limits MPS allocation. It uses four
CPU threads. CUDA, MPS or CPU is selected automatically, or use `--device`.
Mac training stops if the system reports warning or critical memory pressure.
For an attended or monitored tmux job, `CLEARTEXT_WAIT_FOR_CAPACITY=1` pauses
training on warning pressure and checks every 30 seconds until normal pressure
returns. Optimizer and training state stay in memory. Critical pressure still
stops the job. `capacity_status.json` records this separate from training progress.
The default optimizer is Adafactor with a fixed target learning rate of 0.001,
10% warm-up and internal update clipping. `--optimizer adamw` uses 0.0003 with
gradient clipping. A lower-memory CPU configuration uses `--device cpu
--batch-size 2 --accumulation-steps 2`, with an estimated peak RAM of 3 GB.
Accumulation keeps an effective batch of four examples. The training summary
records actual steps and examples seen, which may be less than a full epoch.

Stages can run separately: `prepare`, `train-classic`, `train-seq2seq`, `evaluate`.
`all --resume` reuses completed data and model stages after an interruption. An
incomplete T5 stage restarts from the pinned pretrained model; optimizer state is
not saved. Preserve original run arguments when resuming. Partial results are
recorded in `progress.json`. Models are local trusted pickle files and a T5
checkpoint. Data, weights and per-sentence outputs stay in ignored directories.
Publish only aggregate evidence and reviewed examples permitted by the source.

Run any of the three models explicitly:

```sh
/path/to/structure-venv/bin/python models/scripts/structure_experiment.py predict \
  --run /path/to/run --model editor \
  --text 'The server checks each request and sends a response to the client.'
```

## Sources

- [WikiSplit data and noise disclosure](https://github.com/google-research-datasets/wiki-split)
- [LaserTagger, the editing approach that informed this prototype](https://aclanthology.org/D19-1510/)
- [CRFsuite Python wrapper](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html)
- [T5-small model card](https://huggingface.co/google-t5/t5-small)
- [Adafactor configuration and memory use](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor)
- [BiSECT, a separate dataset for a later transfer evaluation](https://aclanthology.org/2021.emnlp-main.500/)
- [RFC 9110, HTTP semantics, technical audit input](https://www.rfc-editor.org/rfc/rfc9110.html)
- [RFC 8446, TLS 1.3, technical audit input](https://www.rfc-editor.org/rfc/rfc8446.html)
