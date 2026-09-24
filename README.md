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

Do not retrain to improve the displayed test scores. Use new development evidence and independent human review for the next iteration. The initial pipeline has no sentence restructuring or phrase replacement. The experiment below adds limited rules. An ordinal CEFR model, comprehensive jargon detection, and general meaning verification are not implemented.

## Artifacts

- `outputs/cleartext-model-review.html`: standalone review page with formulas, comparisons, examples, and exportable ratings.
- `runs/initial-20260924/`: frozen models, selected configurations, predictions, evaluation output, and report data.
- `logs/`: setup, training, and evaluation logs.
- `work/`: browser QA screenshots and temporary verification files.

The HTML works offline. Ratings are stored locally in the reviewer's browser and exported as JSON. They do not automatically retrain models. Review examples contain attributed public dataset text; source licenses are listed in the report. Do not redistribute the datasets under a blanket project license.

## Context and phrase experiment

The separate context_pipeline.py adds static-vector context features, a trained candidate checker, ten authored phrase pairs ranked by a CompLex word/phrase regressor, and two narrow syntax rules. The current checker abstains from all word substitutions. It does not fix word meaning reliably. Phrase and structural examples are demonstrations, not an accuracy estimate. Longer phrases extrapolate beyond the predominantly two-word training targets.

Run scripts/train_context_v2.py, scripts/train_phrases.py, then scripts/review_context_v2.py with .venv/bin/python. Prerequisite: the initial training and lexical evaluation have run. Inference: scripts/predict_context.py TEXT. Tests: pytest tests -q, scripts/verify_context_v2.py, and scripts/verify_context_report.py. The report is outputs/context-model-review.html and experiment artifacts are in runs/context-20260924. The older pipeline and report remain unchanged.

## SemCor sense experiment

The separate SenseClearText pipeline learns word meanings from SemCor and filters WordNet candidates to the chosen sense. It restores word edits, while the v2 conservative checker remains available unchanged. Custom held-out SemCor accuracy is 69.1% versus 67.7% for a training-frequency baseline. On reused TSAR data, reference matches among edited cases rise from 43.4% to 53.6%, but total correct replacements fall from 85 to 52. This is an experimental precision/coverage tradeoff, not an overall win.

Run scripts/train_semcor.py, scripts/evaluate_semcor_pipeline.py, scripts/verify_semcor.py, and scripts/build_semcor_report.py using .venv/bin/python. Inference uses scripts/predict_sense.py TEXT, with --full to add earlier phrase and structure rules. The report is outputs/semcor-model-review.html. Artifacts are in runs/semcor-20260924. SemCor is installed by scripts/setup.sh.
