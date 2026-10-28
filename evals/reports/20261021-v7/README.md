# v7 evaluation run, October 21, 2026

Executed the runbook from [PR #6](https://github.com/JoshD94/cleartext/pull/6) at commit `644dd6a602e312b6e5c801a8d850b575fe3f08c3` on Yoga. Existing trained v7 artifacts were served by the running demo through the PR's HTTP adapter. No model was retrained, and model binaries remain outside Git.

## Files

- `eval_results.md` is the aggregate table for original text, the dictionary baseline and v7.
- `eval_results.json` contains the per-passage and aggregate metrics.
- `detection_baseline.json` contains Kea-Roy's separate linear CWI detection baseline.
- `provenance.json` records code, data and artifact fingerprints, evaluation model snapshots, package versions and model flags.
- `reproduce.sh` contains the commands. Run it from the repository root with the documented dependencies installed and an accessible demo URL. A teammate can instead use the PR's local adapter if they have the saved artifacts.

## Scope and interpretation

The six committed sample passages are authored placeholders. These results check that the model and evaluator work together. They are not independent CS, corporate or political-domain benchmark results, human quality scores or evidence of statistical significance.

The adapter names the model `full_pipeline`. This run uses default v7 word replacement, with at most one edit per passage. Optional phrase and structure rewriting, the newer 52-feature ranker and the six-block wrapper are off. Unchanged outputs count in the aggregate preservation and readability metrics.

On these six passages, v7 lowers the mean Flesch-Kincaid grade from 20.71 to 19.70. The dictionary baseline reaches 16.57. Mean backward NLI entailment is 0.691 for v7 and 0.993 for the dictionary baseline. Two v7 cases need review for possible lost meaning, `corp-02` at 0.012 and `polsci-01` at 0.164. These automatic scores flag cases to inspect; they do not establish which output is correct.

The CWI result uses `evals/trained_weights.json` and the evaluator's linear A-E word scorer. It is not an evaluation of v7's separate trained hard-word detector. The threshold is selected on 27,299 CWI training instances and reported on 3,328 development instances. The development result is precision 0.483, recall 0.862 and F1 0.619 at threshold 0.55.

The dataset step fetched and parsed CWI, TSAR, MultiLS trial and ASSET files. It did not run TSAR/MultiLS replacement ranking or correlate scores against ASSET human ratings. Those are still follow-up tasks in PR #6.

Embedding cosine and bidirectional NLI are automatic preservation signals. Explicit checks for numbers, dates, entities, negation and tense are heuristic signals. Passing them does not establish factual or semantic correctness. Short-passage readability formulas also have limits.

## Execution notes

The demo kept its existing model and defaults. Evaluation-only dependencies were installed in a temporary overlay instead of changing the demo's virtual environment. The first evaluation attempt stopped on missing `pyphen`; installing that dependency allowed the evaluation to run. The successful dataset fetch and CWI baseline were retained.

The job ran in Yoga tmux session `cleartext-eval-20261021`, capped at 2.5 GiB with two CPU threads for evaluation. The existing demo used its existing thread settings. Logs and progress receipts are under `~/.local/state/cleartext/eval-20261021/` on Yoga.
