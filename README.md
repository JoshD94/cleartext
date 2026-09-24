# ClearText

Initial non-LLM models for CS 4701. Run and trained on Joshua's Yoga server, September 24, 2026.

## Modules

- `data.py`: pinned dataset downloads, split-preserving loaders, and source checksums.
- `features.py`: word and sentence feature extraction using wordfreq, CMUdict, and spaCy's small English pipeline.
- `models.py`: interchangeable regressors selected by development MAE.
- `lexical.py`: WordNet candidate generation, context features, candidate ranking, and explicit preservation checks.
- `pipeline.py`: frozen-model inference, one accepted edit per sentence.

## Run

The existing Yoga checkout is `/home/joshuadirga/Documents/projects/cleartext`.

```sh
bash scripts/setup.sh
bash scripts/train.sh
bash scripts/lexical.sh
.venv/bin/python scripts/sample.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/verify_report.py
.venv/bin/python scripts/predict.py 'The company purchased additional equipment.'
```

Use `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4` for direct Python runs. Training scripts already set these limits. Run long jobs in named Yoga tmux sessions. Browser QA requires Playwright Chromium, installed with `.venv/bin/python -m playwright install chromium`.

`requirements-lock.txt` records the installed versions. For exact environment recreation, install that lock into Python 3.12; `setup.sh` is a bootstrap and may otherwise resolve newer package versions. The small spaCy model is pinned to 3.8.0. Dataset commit IDs and SHA-256 hashes are in `data/raw/manifest.json`.

## Initial results

The selected word-only model is gradient boosting, test MAE 0.0696 on CompLex. Context features select random forest, test MAE 0.0690. This small difference is not clearly established by the example bootstrap interval. Sentence gradient boosting reaches MAE 0.4573 on the mean of the two CEFR educator ratings.

The context-fit classifier is weak. Its custom SWORDS binary diagnostic has test AUC 0.658 and precision 0.151 at the development-selected operating threshold. Dictionary substitution has official TSAR Precision@1 0.2600; the full guarded variant reaches 0.2064. KEEP scores zero in that metric. The added modules do not yet improve replacement quality.

Preservation checks catch six of eight deliberately incorrect rewrites, but miss an antonym and lost specificity. They accept both controls. Passing checks is not proof of semantic equivalence.

## Evaluation boundaries

Preserve all official splits. Choose configurations using development results, never test results. SWORDS development contexts are split for training/validation; official test contexts are separate. The context classification target is TRUE votes divided by all judgments, thresholded at 0.5. These are custom classification diagnostics, not official SWORDS ranking metrics.

The ranker's difficulty margin 0.02, context weight 0.03, and full pipeline's difficulty cutoff 0.30 are initial heuristic choices. Only the context-fit threshold is development-tuned. TSAR supplies target words, so its table does not evaluate difficult-word detection. The report's authored examples exercise the full detector and one-edit pipeline.

Do not retrain to improve the displayed test scores. Use new development evidence and independent human review for the next iteration. Sentence restructuring, an ordinal CEFR model, comprehensive jargon detection, phrase replacement, and general meaning verification are not implemented.

## Artifacts

- `outputs/cleartext-model-review.html`: standalone review page with formulas, comparisons, examples, and exportable ratings.
- `runs/initial-20260924/`: frozen models, selected configurations, predictions, evaluation output, and report data.
- `logs/`: setup, training, and evaluation logs.
- `work/`: browser QA screenshots and temporary verification files.

The HTML works offline. Ratings are stored locally in the reviewer's browser and exported as JSON. They do not automatically retrain models. Review examples contain attributed public dataset text; source licenses are listed in the report. Do not redistribute the datasets under a blanket project license.
