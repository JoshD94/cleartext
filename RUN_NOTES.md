# Run handoff

Objective: train the initial modular ClearText system on Yoga and deliver a concise HTML review with actual results and interactive grading.

Host: Yoga, `/home/joshuadirga/Documents/projects/cleartext`, branch `main`. No Git remote configured. Sessions used: `cleartext-setup-20260924`, `cleartext-train-20260924`, `cleartext-lexical-20260924`, `cleartext-report-20260924`, `cleartext-browser-20260924`. These one-shot jobs exit when complete.

Completed: two word feature sets, sentence scorer, WordNet generation, context-fit comparison, ranker, explicit preservation checks, unchanged/dictionary/learned/context/guarded configurations, inference CLI, offline report, rating export, and desktop/mobile browser QA.

Validation: five invariant tests passed. Independent result checks verified data hashes, development-only model selection, recomputed held-out errors, official TSAR agreement, no SWORDS development/test context overlap, and sample counts. Browser checks covered both screen sizes, no page overflow or JavaScript errors, model-table switching, sample switching, rating validation and persistence, and JSON export.

Known limitations: context filtering underperforms dictionary baseline; preservation checks miss antonyms and specificity loss; sentence regression uses equally spaced CEFR levels; repeated CompLex targets use the first matching token for context; full pipeline applies at most one edit with a heuristic target cutoff. Do not claim meaning preservation is solved.

Report source: `scripts/build_report.py`. New user grades must be explicitly exported from the browser and treated as new review evidence, not automatically added to a test set or training run.

The initial tmux completion wrappers used zsh's read-only `status` variable after the Python jobs completed, so their panes reported nonzero wrapper exits. The training/evaluation/sample artifacts and independent verification confirm completion. `scripts/run_job.sh` now uses bash and `job_exit_code` for future launches. Raw logs are retained; no model was retrained in response to the wrapper issue.

## Context and phrase prototype, 2026-09-24

Yoga sessions cleartext-vectors-install-20260924, cleartext-context-20260924, cleartext-phrases-20260924 and cleartext-context-review-20260924 completed with exit 0. Initial spacy download command failed because pip was absent; the installed uv tool successfully installed pinned en_core_web_md 3.8.0. No old model artifacts were changed.

Context checker adds static-vector and WordNet-gloss features to SWORDS candidate classification. It failed the development precision goal and currently abstains from every word edit. Test AUC 0.6646 versus old 0.6583 is exploratory on a previously used test set. Bank definition probes still show failure. The phrase regressor uses both single and multiword CompLex splits and yields multiword test MAE 0.08377 on 184 phrases. Ten authored pairs and two parser rules are only a small prototype, not a general rewriting model. Nine of twenty authored examples changed; these are demonstrations.

Validation: ten pytest tests, saved metrics and data hashes, context separation, and Playwright desktop/mobile report checks passed. Artifacts: runs/context-20260924 and outputs/context-model-review.html. New report copied to the current Mac task outputs. No persistent services launched.
