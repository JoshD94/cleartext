# Combined pipeline evaluation

October 27, 2026. This work connects the current sentence generator and word
models to Kea-Roy's readability and preservation evaluator. The running demo
and its default models stay unchanged. Raw passages, predictions, model files
and resumable caches remain outside Git.

The October 21 run used the evaluator on six authored sample passages. Its
adapter called the system `full_pipeline`, but phrase and structure stages were
off. That checked the old word model's connection to the evaluator. It did not
measure the newly trained sentence generator or the combined pipeline.

## What counts as 90 percent

The measurable lexical target is 90% of accepted replacements matching a
published substitute, with at least 20% edit coverage in each of three
development repeats. The coverage floor prevents a policy that meets the
precision target by making almost no edits. It is an engineering target,
chosen before this pass's calibration, rather than a project requirement.

This target does not define full-sentence accuracy. TSAR supplies the target
word. Its substitute lists do not grade additional word edits, sentence splits,
whole-sentence grammar or retained technical meaning. A listed match is useful
evidence about that replacement, rather than proof that a combined output is
correct. An absent match may also be a reasonable substitute.

No human quality labels were added. NLI probabilities, embedding similarity,
reading grades and reference overlap are reported separately. They are not
percentages of correct simplifications. Copies and changed outputs have separate
denominators, so returning the original cannot inflate the changed-output score.

## Frozen inputs and systems

The shared comparison contains 190 inputs:

| Panel | Inputs | Evidence available |
| --- | ---: | --- |
| DeSSE broad development | 96 | Published split references; reused development |
| ASSET validation | 64 | Ten human simplification references per source |
| Project domain development | 24 | Previously inspected real text; no reference or quality ratings |
| Authored smoke examples | 6 | Teammate placeholders; no quality ratings |

ASSET selection sorts source hashes with seed 4701 before prediction. Exact
sources found among the supplied earlier training/evaluation sources or full
targets are excluded. References and grammar warnings do not choose the sample.
The other three panels are retained in full. No reserved project-domain inputs
are predicted. Exact matching cannot rule out near duplicates or pretraining
overlap. The new ASSET sample is development material, rather than a new test
used to establish general accuracy.

The comparison runs identity, the teammate dictionary baseline, lexical-only,
structural-only, lexical plus structural, full without parser guards, full with
guards, the old structure-rule combination and a stricter v7 combination. The
full combinations include phrase rules, checked sentence generation and trained
word selection, with the existing limit of one word edit per input. Grammar
preprocessing remains off, as in the current default.

`full_no_guardrails` disables the generated rewrite checks, reparsed lexical
checks, lexical rejection reasons and combined-word check. It retains the
detector's eligible targets, learned scores, acceptance threshold and existing
phrase-rule restrictions. It is an ablation of these checks, rather than an
unconstrained language model. Optional inspection blocks and the separate learned
meaning classifier are not silently enabled.

The sentence generator processes each sentence in a passage and preserves the
spacing between them. Its own token limit still applies. The old single-sentence
ranker retains its original limit. Both guarded and unguarded generation use the
same underlying T5 predictions. The experimental T5 checkpoint is screened-data
step 441, SHA256 `3de186eee031e8812b019a9040775884798d01baabb072728bd03ac89f1503ac`.

Kea-Roy's readability and preservation functions score every system/input pair.
The full comparison requests both MiniLM sentence embeddings and DeBERTa NLI in
both directions. Every metric includes its available-value count. Overlong NLI
pairs abstain instead of silently dropping text. Unknown entailment labels also
abstain. The existing reference-overlap proxy is averaged across the available
references; this is explicitly not a standard multi-reference SARI result.

The evaluator now requires complete numeric items and word boundaries for
explicit preservation. Previously `20` could count as preserved by `120`, and
`not` by a word containing those letters. Four-digit plain numbers are also
recognized. These fixes improve the check, not the underlying simplification
model. Named-entity detection, tense matching, units and reference quality still
have substantial limitations.

## Stricter word policies

For v7, the decision family was refit in three 10-fold development repeats,
grouped by target word. The existing cached candidate features and published
labels were reused. The threshold grid and minimum coverage were fixed before
the new test comparison. No threshold met 90% in every repeat at 20% coverage.
The fallback selected 0.70, with development precision 81.9%, 83.0% and 82.4% at
22.0%, 23.5% and 22.6% coverage.

The saved policy was hashed before opening the reused TSAR cache:

| v7 policy | Listed matches / edits | Edit precision | Input coverage |
| --- | ---: | ---: | ---: |
| Existing 0.45 threshold | 158 / 219 | 72.1% | 58.7% |
| Development-selected 0.70 | 55 / 63 | 87.3% | 16.9% |

The stricter result has a Wilson 95% interval of 76.9% to 93.4%. It avoids 53 of
the 61 old unlisted edits, but also gives up 103 listed matches. This is a
precision/coverage tradeoff on a reused target-word benchmark, not an overall
system win. No test-driven threshold adjustment follows it.

The stronger 52-feature contextual ranker also has frozen development
probabilities. All 929 historical selections were reproduced in all three
repeats before using those arrays. The same target, coverage floor and threshold
grid select 0.80. It reaches 90.85%, 90.75% and 90.30% development precision at
32.9%, 31.4% and 32.2% coverage. These repeats share cases and are not three
independent experiments.

The separate saved-model check retains all 373 reused TSAR cases. It recomputes
the contextual proposal support and replacement similarity, then compares three
fixed cases with the native pipeline. Candidate pools and chosen words match;
the largest acceptance-probability difference is below 0.00000008.

| Contextual policy | Listed matches / edits | Edit precision | Input coverage |
| --- | ---: | ---: | ---: |
| Existing 0.45 threshold | 172 / 204 | 84.3% | 54.7% |
| Development-selected 0.80 | 80 / 84 | 95.2% | 22.5% |

At its original threshold, the contextual model finds 14 more listed matches
than v7 and makes 15 fewer edits on these cases. Its unlisted edits fall from
61 to 32. The stricter policy meets the 90% precision and 20% coverage targets
on this reused test, but accepts far fewer replacements. Its Wilson 95% interval
is 88.4% to 98.1%, so the result does not establish that true precision exceeds
90%. Across all 373 inputs it finds 80 listed matches; 95.2% is conditional on
the 84 accepted edits. Neither policy measures full-pipeline accuracy.

This check ran on the Mac while Yoga generated pipeline variants. The first
native verification attempt found a missing `en_core_web_md` dependency after
the 373 contextual predictions were saved. Copying Yoga's identical spaCy 3.8.0
model into the isolated job dependencies allowed verification to resume. No
predictions, threshold or examples changed. Transformers, tokenizer, NumPy,
spaCy and scikit-learn versions match Yoga. Torch differs by a patch version,
2.14.1 on Mac and 2.14.0 on Yoga. The native score checks pass despite that
difference. The Mac's primary source and existing environment were preserved.

## Reproduction

Merge the model and evaluator branches and use the model environment plus the
evaluation dependencies. Existing evaluation-only dependencies were available
in a local overlay for this run; no package change was made to the demo's
environment. Download the teammate benchmark cache with `evals/datasets.py` if
it is missing. The preparation CLI only uses ASSET validation files.

```sh
python evals/prepare_pipeline_benchmark.py \
  --structure-data /path/to/screened/data.json \
  --domain-dev /path/to/domain-dev-inputs.json \
  --authored evals/data/sample_passages.json \
  --asset-dir /path/to/benchmarks \
  --exclude /path/to/aligned/data.json \
  --exclude /path/to/aligned/reserved_test.json \
  --out /path/outside-git/passages.json

python evals/pipeline_benchmark.py generate \
  --passages /path/outside-git/passages.json \
  --out /path/outside-git/generation \
  --lexical-run models/runs/ensemble-v7-20260928 \
  --structure-run /path/to/screened/seq2seq

python evals/pipeline_benchmark.py score \
  --predictions /path/outside-git/generation/predictions.jsonl \
  --out /path/outside-git/scoring \
  --embedding-model /path/to/pinned/MiniLM \
  --nli-model /path/to/pinned/DeBERTa
```

For precision policies, run `models/scripts/experiments/calibrate_precision.py`
and `select_cached_precision.py`, each with a new `--out` directory. The latter
requires the existing contextual-similarity development cache. Then run
`evaluate_precision.py --policy-dir /path/to/contextual-policy --out /new/check`.
The contextual check retains all 373 TSAR cases and verifies three fixed cached
rows against native model scoring before reporting a result.

Generation freezes input, source and model fingerprints before prediction.
Scoring separately freezes prediction, evaluation-model and metric-code hashes.
Interrupted runs can resume only with matching fingerprints. Predictions are
flushed after each case. Equal system/input panels are checked before scoring.
Exact rank-result reuse requires the same input, target offset, original context,
guard setting and acceptance threshold. Mutable trace fields are copied.

Execution records are under
`~/.local/state/cleartext/pipeline-eval-20261027/` on Yoga. The contextual encoder
sets four Torch CPU threads. Generation initially had a two-CPU quota; this was
raised to four during inference to match the existing thread count. Models,
inputs, code and model thread settings did not change. Scoring uses two threads.
Jobs use named tmux sessions and 4 GiB memory-capped scopes. Resource monitoring
stops this experiment if available host RAM falls below 2 GiB. Unrelated jobs
and the existing demo are preserved.
