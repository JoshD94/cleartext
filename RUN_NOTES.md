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

## SemCor experiment, 2026-09-24

Yoga sessions cleartext-semcor-data-20260924, cleartext-semcormodel-20260924 and cleartext-sense-eval-20260924 completed. Trained on 141,689 labels across document-separated SemCor data; dev 28,505 and test 30,089. The model combines training meaning frequencies, naive-Bayes context counts, and static-vector sense centroids. Held-out WSD accuracy improved 1.33 percentage points, document-bootstrap interval 0.89 to 1.82. TSAR and SWORDS were reused for exploratory external evaluation only. Filtering raises precision but reduces recall and total useful edits. Bank-money, alleviate/facilitate and stipulate/qualify remain failures. No new human ratings collected.

Fourteen pytest tests passed. Verification checks document and exact sentence separation, count provenance, development choice, test accuracy, and reproduction of the original TSAR baseline. The report has desktop/mobile browser QA and all twenty original sample sentences. Original model variants remain unchanged. Artifacts: runs/semcor-20260924 and outputs/semcor-model-review.html.

## Ensemble experiment, 2026-09-27

Yoga sessions cleartext-senses-20260927 and cleartext-fit-20260927 through cleartext-fit5-20260927 completed with exit 0; each later session reran stage 2 after a diagnosed fix. Stage 1 refits SemCor on the same train documents and reproduces the earlier 67.7% baseline and 69.1% model on test. Stage 2 trains the stacker on 4,957 SWORDS dev pairs (416 generated-candidate pairs) held out from the v2 checker, using 5-fold grouped CV.

Diagnosed during the run: an evaluation tie bug (fixed before any selection was kept); SenseEnsemble evidence weighting did nothing when other member weights were zero (caught by a unit test, fixed, rerun); a candidate-frequency member mixed simplicity into fit (removed); fit*gain and fit-only ranking both let weak candidates win (replaced with the expected-utility rule).

Validation: 22 pytest tests pass, including a check that old SenseModel pickles rank identically. The old SenseClearText still outputs "facilitates" for the alleviate sentence. No new human ratings. TSAR and SWORDS test remain reused exploratory checks. Next decision: the error_cost operating point, ideally chosen on new development evidence.

## Ensemble v2, 2026-09-27

Yoga session cleartext-v2-20260927 computed SWORDS member scores (the first run failed in report code after the models were frozen; reruns used cached features and completed with exit 0). Tiering and a/an repair were added after reviewing v2 samples; the tiering decision was made on SWORDS dev only. 29 pytest tests pass. The a/an repair also applies when v1 is loaded, which can change v1 outputs where an article precedes the target. Remaining sample errors: exacerbate -> exasperate, deposited -> gave, encodes -> converts.

## Refinement rounds and v3, 2026-09-27

Yoga sessions cleartext-refine-20260927, cleartext-refine2-20260927 (candidate tables) and cleartext-targets-20260927 (end-to-end target choice) completed with exit 0; other rounds ran in the foreground in under a few minutes each. The pipeline with the dev-only decision model reproduces the table evaluation exactly on TSAR (116 of 218). 33 pytest tests pass. Downloads: BenchLS from Zenodo record 2552393 (SHA-256 pinned in data.py) and CWI 2018 English from sheffieldnlp/cwisharedtask2018-teaching at commit 74816aec. Remaining sample errors: stipulates -> contracts, deposited -> gave, encodes -> converts, database -> information when a noun wins the single edit. alleviates is now left unchanged (relieves ranks first at acceptance 0.30, below 0.40).

## v4, 2026-09-27

Rounds 6 and 7 and the v4 build ran in the foreground on Yoga (each under 15 minutes). 34 pytest tests pass. deposited and exacerbate are now left unchanged. Remaining sample errors are noun edits the detector flags and the one-edit choice prefers: database -> information, scientist -> individual, gene -> sequence, committee -> commission, and stipulates -> contracts. No labels here judge edits to non-target words, so this is not yet measurable.

## v5 and rounds 8 to 12, 2026-09-27/28

Yoga tmux sessions: cleartext-fitv5, cleartext-extras, cleartext-tables5, cleartext-cwitargets, cleartext-targets5, cleartext-meaning, cleartext-round12, cleartext-choice (all 20260927). Pipeline and table evaluations match exactly for every built run. 34 pytest tests pass. Remaining visible errors with v5 defaults: database -> information, scientist -> individual, encodes -> converts, postponed -> deferred; "an information" (mass nouns take no article; WordNet does not mark countability).
