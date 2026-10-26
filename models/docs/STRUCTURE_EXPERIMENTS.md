# Learned sentence structure experiments

These experiments compare three approaches on the same bounded WikiSplit sample.
They run through a separate CLI and do not change the v7 demo or its artifacts.

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

The encoder–decoder is the 60-million-parameter T5-small checkpoint, fine-tuned
locally with a sentence-splitting prefix. It can rewrite more freely. Its best
checkpoint is selected by development teacher-forced loss. This generative model
is a new experimental branch of the architecture; it does not use a hosted API.

## Data and evaluation

The downloader pins WikiSplit to commit
`80f5bddf42d1a68201b7b107ee4b8d86d82e9fde` and records downloaded file hashes.
It uses official train, tune as development, and test files. The separate
validation file is unused. All eligible development/test sources and full targets
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
