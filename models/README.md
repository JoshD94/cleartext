# ClearText models

Lexical simplification for CS 4701 using frozen BERT and small trained models. Given a sentence, the pipeline finds hard words, proposes simpler replacements, scores them with the ensemble, and applies edits that pass its acceptance and safety checks. It also reports predicted reading level before and after.

The current default is v7. The stronger 52-feature ranker and six inspection blocks are separate, opt-in runs. The demo still uses v7. Novel BERT replacements failed their acceptance checks and remain disabled in safe configurations.

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) indexes the results and limits of each experiment. [docs/cleartext-steps.html](docs/cleartext-steps.html) and [docs/report.txt](docs/report.txt) describe the earlier pipeline; use the experiment documents for subsequent changes.

## Current stack

1. Parse the input with spaCy. Optional structure and phrase rules run before word selection.
2. Skip function words, names and protected spans. Score eligible content words with CompLex difficulty models and the CWI hard-word detector.
3. Generate WordNet candidates and inflect them for the source word's grammatical form.
4. Score meaning with the SemCor/gloss ensemble plus frozen BERT sense vectors. Score candidate fit with 15 members and a learned SWORDS stacker.
5. Use an averaged logistic and gradient-boosting decision model to score each candidate. Apply the acceptance threshold, meaning floor and mechanical safety checks.
6. Rank acceptable edits across words, apply at most one word edit by default, and report predicted reading level before and after.

Independent scoring members read the same candidate and context. The implementation evaluates those members sequentially; BERT batches its inputs. This is one pass over the input, without repeatedly simplifying the edited sentence until convergence.

| Run | Added behavior | Status |
| --- | --- | --- |
| `ensemble-v7-20260928` | BERT meaning and slot-fit members, meaning floor, candidate and role fixes | Default library and demo |
| `ensemble-context-similarity-20261002` | Nine WordNet relation features, three BERT proposal-support features, full-span replacement similarity, target-centered meaning windows | Best saved development ranker, 52 decision features; opt-in |
| `ensemble-building-blocks-20261002` | Detail evidence, term detection, grammar validation, jargon explanations, document consistency and edit-quality reports | Opt-in wrapper around the 52-feature ranker |
| `novel-context-similarity-20261003` | Separate validator for candidates outside WordNet, considered only after the base pipeline abstains | Safe gate disabled; shadow comparisons are experimental |

Run names refer to local directories under `runs/`. Weights and caches are excluded from Git. The six blocks report model predictions and rule checks, with no calibrated overall quality score or guarantee that meaning is preserved.

## Building blocks

Each block is a small model or rule set with a shared interface, so members can be swapped or combined (`src/cleartext/ensemble.py`).

| Block | Data | Model | Code |
|---|---|---|---|
| Word difficulty | CompLex | Gradient boosting on frequency, syllables, length | `features.py`, `models.py` |
| Hard-word detector | CWI 2018 | Gradient boosting over both difficulty scores and word features | `detection.py` |
| Sentence difficulty | CEFR-SP | Gradient boosting on word, grammar and detector summaries | `features.py` |
| Candidates | WordNet, lemminflect | Synonyms, broader words, similar-to and multiword synonyms | `generation.py` |
| Safety checks | rules | Numbers, negation, modality, names, roles, a/an, mass nouns | `lexical.py`, `ensemble_pipeline.py` |
| Context | SWORDS | Gradient-boosted fit checker | `semantics.py`, `ensemble.py` |
| Phrases and structure | CompLex multiword, rules | Phrase difficulty model, 10 phrase pairs, 2 grammar rules | `phrase_complexity.py`, `rewrites.py` |
| Meaning | SemCor, WordNet glosses, BERT | Sense model with gloss backoff, mixed with BERT's nearest sense vector | `wsd.py`, `ensemble.py`, `contextual.py` |
| Ensemble decision | SWORDS, BenchLS | Stacked fit model, then an averaged logistic + boosting decision | `ensemble.py`, `refine.py`, `ensemble_pipeline.py` |

## Layout

- `src/cleartext/`: the library. `ensemble_pipeline.py` is the current pipeline; `pipeline.py`, `context_pipeline.py` and `sense_pipeline.py` are the earlier versions it grew from.
- `scripts/`: every step needed to rebuild the current model, listed in order below, plus `predict_ensemble.py`.
- `scripts/experiments/`: comparison rounds that only write result files, and the rejected v6 build.
- `demo/`: the v7 sentence demo and its step-by-step display.
- `diagnostics/`: attributed authored inputs and owner examples, without correctness labels.
- `scripts/legacy/`: report, verification and prediction scripts for the three earlier pipelines.
- `tests/`: unit tests (`pytest tests -q`).
- `docs/`: the step-by-step report and the experiment log.

Datasets, caches, trained models and generated reports are not committed (`data/`, `runs/`, `outputs/`, `logs/`, `work/`). Several datasets have noncommercial or unstated licenses, so rebuild them locally with the scripts below.

## Setup

Linux, Python 3.12, CPU only.

```sh
bash scripts/setup.sh    # uv, virtual environment, spaCy models, NLTK data; writes requirements-lock.txt
```

`requirements-lock.txt` records the exact versions used. Scripts set four CPU threads; `scripts/run_job.sh NAME CMD...` runs a long job with its log in `logs/NAME.log`.

## Rebuild the current model (v7)

Run with `.venv/bin/python` from this folder, in order. Later steps read earlier artifacts from `runs/` and caches from `data/cache/`.

1. `scripts/train.py`: downloads CompLex, CEFR-SP, TSAR and SWORDS at pinned commits; trains the word, context and sentence difficulty models.
2. `scripts/evaluate_lexical.py`: first candidate, context and ranking evaluation; builds the SWORDS feature caches.
3. `scripts/train_context_v2.py`, `scripts/train_phrases.py`: vector context checker and phrase difficulty model.
4. `scripts/train_semcor.py`, `scripts/evaluate_semcor_pipeline.py`: SemCor sense model and its evaluation.
5. `scripts/train_ensemble_senses.py`, `scripts/train_ensemble_fit.py`: sense ensemble and ensemble v1.
6. `scripts/train_ensemble_v2.py`: expanded candidates and the cross-fitted checker.
7. `scripts/build_candidate_tables.py`, `scripts/train_detector.py`, `scripts/train_ensemble_v3.py`, `scripts/train_ensemble_v4.py`: BenchLS decision model, CWI detector, specificity features.
8. `scripts/train_fit_v5.py`, then `REFINE_RUN=runs/ensemble-v5-20260927 REFINE_VERSION=v5 scripts/build_candidate_tables.py`, `scripts/train_detector_final.py`, `scripts/train_sentence_model.py`, `scripts/train_ensemble_v5.py`.
9. BERT (v7): `scripts/train_bert_senses.py` (about 2.5 hours on CPU), `scripts/train_fit_v7.py` (needs the cached SWORDS BERT scores from `scripts/experiments/bert_benchmark.py`), `REFINE_RUN=runs/ensemble-v7-20260928 REFINE_VERSION=v7 scripts/build_candidate_tables.py`, `scripts/experiments/refine_bert_decision.py`, `scripts/train_ensemble_v7.py`.

BenchLS (Zenodo, CC BY 4.0) and CWI 2018 English download automatically, pinned and checksummed, the first time a step needs them (`benchls_data()` and `cwi_download()` in `data.py`). The CWI source repository has no license file; use it for research only and do not redistribute it.

## Use

```sh
.venv/bin/python scripts/predict_ensemble.py "The technician will commence the installation tomorrow."
.venv/bin/python scripts/predict_ensemble.py --max-edits 2 "..."   # allow a second word edit
.venv/bin/python scripts/predict_ensemble.py --full "..."          # also apply phrase and structure rules
```

The output includes each edit, every scored candidate with its rejection reasons, and the reading level before and after.

For the newer locally fitted runs:

```sh
.venv/bin/python scripts/predict_ensemble.py --run runs/ensemble-context-similarity-20261002 "The technician will commence the installation."
.venv/bin/python scripts/analyze_blocks.py --text "The support vector machine (SVM) processes a convoluted input."
```

`analyze_blocks.py` accepts a text file through `--input`, limits input to 20,000 characters and refuses to overwrite an existing output file. See [BUILDING_BLOCKS.md](docs/BUILDING_BLOCKS.md) for the returned reports and local configuration recipe.

Start the demo from `models/` after rebuilding v7:

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python demo/server.py --host 127.0.0.1 --port 8770
```

Open `http://127.0.0.1:8770`. The strictness control sets the minimum acceptance score, with a default of 0.45. Higher values allow fewer edits. The server handles one request at a time and repeats scoring to populate the display, so CPU responses can take several seconds. It has no authentication and is intended for local or trusted-network use.

## Results and evaluation limits

- TSAR 2022 test, 373 sentences: 158 correct replacements from 219 edits (72.1%). v5 without BERT: 152 of 238 (63.9%). The dictionary baseline gets 85 of 196 (43.4%).
- Word meaning, SemCor test: 67.6% (61.8% without BERT). Fit, SWORDS cross-validated AUC: 0.808 (0.779 without BERT).
- Hard-word detection, CWI 2018 test: F1 0.78. The first detector got 0.38.
- Reading level, CEFR-SP test: mean absolute error 0.43 levels.
- TSAR and SWORDS test were reused while building, so those numbers are exploratory. Model choices were made on development data; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The 52-feature ranker reaches 79.9% listed-substitute precision at 69.1% edit coverage on the fixed BenchLS development comparison. All 929 BenchLS cases, including the historical holdout, have been reused for development. This result is not an independent held-out evaluation. The similarity feature adds nine net listed matches over the preceding 51-feature ranker across three grouped repeats. See [CONTEXT_SIMILARITY.md](docs/CONTEXT_SIMILARITY.md) for the protocol.

The six-block grammar ablation changed no selections in its final development check. The latest novel-candidate shadow gained 14.67 net listed matches but lowered precision by 3.34 percentage points and failed the calibration gate. Its safe configuration stays disabled. [NOVEL_CONTEXT.md](docs/NOVEL_CONTEXT.md) retains both successful checks and failed attempts.

The attributed domain audits are ungraded diagnostics. Existing Europarl excerpts come from reused CompLex component test data. No independent arXiv, SEC 10-K or GovInfo evaluation has been completed. No synthetic correctness labels are used.

Known limits include technical detail loss such as `encodes` to `converts` and `dividend` to `profit`. Mechanical checks can pass these edits. Phrase and structure rules remain unmeasured demonstrations. Predicted simplicity gain is a ranker input, rather than a hard requirement in the base pipeline.

## Check the code

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest tests -q
uvx ruff check --select F,E9 .
```

Tests exercise candidate provenance, exact target offsets, complete-span pooling, grammar fallback, protected term occurrences, output rollback and disabled novel gates. They do not substitute for independent sentence-level judgments.

Experiment caches require matching source and input fingerprints to resume. Code cleanup changes source hashes even when behavior is unchanged. Keep completed outputs with their saved source snapshots; use a new cache directory for experiments under changed source.
