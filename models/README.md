# ClearText models

Non-LLM lexical simplification for CS 4701. Given a sentence, the pipeline finds hard words, proposes simpler replacements, scores them with several small models, and applies an edit only when the ensemble judges it safe and useful. It also reports the sentence's reading level before and after.

A step-by-step account of how each block was built is in [docs/cleartext-steps.html](docs/cleartext-steps.html) (open in a browser) and [docs/report.txt](docs/report.txt). Every version and refinement round, with the numbers behind each decision, is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

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
| Meaning | SemCor, WordNet glosses | Sense model with evidence-weighted gloss backoff | `wsd.py`, `ensemble.py` |
| Ensemble decision | SWORDS, BenchLS | Stacked fit model, then an averaged logistic + boosting decision | `ensemble.py`, `refine.py`, `ensemble_pipeline.py` |

## Layout

- `src/cleartext/`: the library. `ensemble_pipeline.py` is the current pipeline; `pipeline.py`, `context_pipeline.py` and `sense_pipeline.py` are the earlier versions it grew from.
- `scripts/`: every step needed to rebuild the current model, listed in order below, plus `predict_ensemble.py`.
- `scripts/experiments/`: comparison rounds that only write result files, and the rejected v6 build.
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

## Rebuild the current model (v5)

Run with `.venv/bin/python` from this folder, in order. Later steps read earlier artifacts from `runs/` and caches from `data/cache/`.

1. `scripts/train.py`: downloads CompLex, CEFR-SP, TSAR and SWORDS at pinned commits; trains the word, context and sentence difficulty models.
2. `scripts/evaluate_lexical.py`: first candidate, context and ranking evaluation; builds the SWORDS feature caches.
3. `scripts/train_context_v2.py`, `scripts/train_phrases.py`: vector context checker and phrase difficulty model.
4. `scripts/train_semcor.py`, `scripts/evaluate_semcor_pipeline.py`: SemCor sense model and its evaluation.
5. `scripts/train_ensemble_senses.py`, `scripts/train_ensemble_fit.py`: sense ensemble and ensemble v1.
6. `scripts/train_ensemble_v2.py`: expanded candidates and the cross-fitted checker.
7. `scripts/build_candidate_tables.py`, `scripts/train_detector.py`, `scripts/train_ensemble_v3.py`, `scripts/train_ensemble_v4.py`: BenchLS decision model, CWI detector, specificity features.
8. `scripts/train_fit_v5.py`, then `REFINE_RUN=runs/ensemble-v5-20260927 REFINE_VERSION=v5 scripts/build_candidate_tables.py`, `scripts/train_detector_final.py`, `scripts/train_sentence_model.py`, `scripts/train_ensemble_v5.py`.

BenchLS (Zenodo, CC BY 4.0) and CWI 2018 English download automatically, pinned and checksummed, the first time a step needs them (`benchls_data()` and `cwi_download()` in `data.py`). The CWI source repository has no license file; use it for research only and do not redistribute it.

## Use

```sh
.venv/bin/python scripts/predict_ensemble.py "The technician will commence the installation tomorrow."
.venv/bin/python scripts/predict_ensemble.py --max-edits 2 "..."   # allow a second word edit
.venv/bin/python scripts/predict_ensemble.py --full "..."          # also apply phrase and structure rules
```

The output includes each edit, every scored candidate with its rejection reasons, and the reading level before and after.

## Results (v5)

- TSAR 2022 test, 373 sentences: 152 correct replacements from 239 edits (63.6%). The dictionary baseline gets 85 of 196 (43.4%).
- Hard-word detection, CWI 2018 test: F1 0.78. The first detector got 0.38.
- Reading level, CEFR-SP test: mean absolute error 0.43 levels.
- TSAR and SWORDS test were reused while building, so those numbers are exploratory. Model choices were made on development data; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Known limits: one word edit per sentence by default; broader replacements can still lose detail (database becomes information); the phrase and structure rules are unmeasured demonstrations.
