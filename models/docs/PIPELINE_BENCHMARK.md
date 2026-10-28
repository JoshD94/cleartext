# Combined pipeline evaluation

Started October 27, 2026; completed October 28. This work connects the current sentence generator and word
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
guards and the old structure-rule combination. Two contextual-model combinations
use the existing 0.45 threshold and the development-selected 0.80 threshold.
Every system receives the same 190 inputs. The stricter v7 policy remains a
component-only result because it failed the predeclared development target.
The full combinations include phrase rules, checked sentence generation and trained
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
pairs abstain instead of silently dropping text. Embedding inputs also abstain
when either text exceeds the encoder's token limit, counting any default prompt.
Unknown entailment labels also abstain. The existing reference-overlap proxy is averaged across the available
references; this is explicitly not a standard multi-reference SARI result.

The evaluator now requires complete numeric items and word boundaries for
explicit preservation. Previously `20` could count as preserved by `120`, and
`not` by a word containing those letters. Four-digit plain numbers are also
recognized. These fixes improve the check, not the underlying simplification
model. Named-entity detection, tense matching, units and reference quality still
have substantial limitations.

## Full comparison result

All ten systems completed the same 190 inputs. MiniLM and both NLI directions
returned scores for every pair. Reference overlap covers 160 inputs, the 96
DeSSE and 64 ASSET cases. It is a proxy, not standard SARI. The table uses
all-input reading grades and changed-only preservation means.

| System | Changed / 190 | Mean reading grade | Embedding, changed | NLI, changed | Reference proxy, 160 inputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0 | 11.708 | n/a | n/a | 25.069 |
| Dictionary | 13 | 11.523 | 0.952 | 0.995 | 26.300 |
| Words, v7 | 140 | 11.318 | 0.963 | 0.902 | 35.873 |
| Structure, T5 | 20 | 11.153 | 0.991 | 0.989 | 28.450 |
| Words and T5 | 141 | 10.758 | 0.963 | 0.905 | 37.233 |
| Full, checks disabled | 155 | 9.622 | 0.958 | 0.904 | 41.254 |
| Full, checked v7 | 141 | 10.773 | 0.963 | 0.906 | 37.233 |
| Full, structure rules | 140 | 11.314 | 0.963 | 0.903 | 35.939 |
| Full, contextual 0.45 | 137 | 10.867 | 0.966 | 0.947 | 36.972 |
| Full, contextual 0.80 | 76 | 11.000 | 0.982 | 0.965 | 33.062 |

Changed-only denominators are the second column. They differ between systems,
so a higher changed-only mean is not a paired quality improvement. Grade
estimates and preservation probabilities are also not accuracy percentages.
The explicit number and negation checks return 1.0 throughout; these include
vacuous cases and do not establish perfect preservation.

The trained structure stage changes only 20 of 190 inputs. On those inputs it
reduces the reading-grade estimate by 5.27 on average, with mean NLI 0.989.
Across the full panel its grade reduction is 0.56. Its coverage is a bottleneck.
Adding trained structure to the full v7 combination reduces the panel's mean
grade from 11.31 with structure rules to 10.77 with T5. That is a modest
0.54-grade additional reduction, rather than a measured accuracy gain.

The contextual model at 0.45 is the more balanced experimental candidate.
On the 130 inputs changed by both it and checked v7, mean NLI improves by
0.031 and the reading-grade estimate rises by 0.077. It retains most coverage,
137 changed inputs versus 141. On the reused domain panel its changed-output
NLI is 0.903 across 15 outputs, versus 0.865 across 16 for v7. These are
small development panels, not independent correctness labels.

The strict contextual model changes 76 inputs. All 76 are also changed by
the usual contextual setting. On that common subset, stricter selection raises
mean NLI by only 0.0034 and the reading-grade estimate by 0.110. Most of the
larger difference between their overall averages comes from abstention and
which inputs they change. Its separate 95.2% target-word precision result
below does not transfer into 95.2% sentence accuracy.

Disabling checks produces the lowest grades and highest reference overlap.
That does not settle whether its outputs are safe. Published decomposition
references can omit information, and NLI misses some detail changes. Two
low-NLI ASSET cases, `asset-valid-565` and `asset-valid-219`, lose geographic
directional detail even in all three guarded model combinations. The earlier
predicate-role failure at `desse-train-10769` also remains in each combination.
These assistant observations are diagnostic notes, not added gold labels.
No thresholds or weights changed after inspecting them.

Keep the contextual model and strict policy as explicit experimental choices.
The current evidence supports a better precision/coverage tradeoff for words,
but does not justify a claim of 90% correct complete simplifications or a demo
model switch. The next model work should target retained detail and predicate
roles, then increase safe sentence splitting. More epochs alone have no measured
benefit from this comparison.

The [aggregate evidence](../evidence/pipeline-20261027/README.md) includes all
system/panel denominators, per-example hashed metrics, common-changed comparisons,
model/source fingerprints and execution records. Raw corpus text remains local.

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

This is the existing contextual model configuration, which also enables target-
centered sense windows for long inputs. This pass selects an acceptance policy
and tests the saved model. It does not add T5 training epochs.

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

## Word scorer memory use

The legacy masked-word scorer produced vocabulary logits at every token, then
used only the masked positions. The scorer now applies the same BERT vocabulary
head to the masked-position vectors. This avoids the batch × sequence ×
vocabulary allocation. Candidate tokenization, the six-piece cap, context
windows, attention masks and batches remain the same. The optional complete-
candidate scorer remains a separate experiment.

Fixed real-BERT checks cover short text, a 4,006-character context and a literal
padding token. The largest log-probability difference is 0.00000477. Three native
contextual rankings still reproduce the cached scores and selected words. All
81 strict-policy outputs saved before a Mac memory stop match the fresh rerun.
The regression test prevents full-sequence vocabulary projection and checks the
legacy piece cap, attention behavior and batch boundary. Fifteen related tests
pass. Cold-load timings in the numeric verification are not a speed comparison.

All 455 overlapping rows from the bounded Yoga attempt match the fresh Mac
outputs. Against the older legacy Yoga attempt, 1,068 of 1,070 rows match.
The two differences concern one input, `desse-train-4253`, in lexical-only and
lexical-plus-structure variants. Its cause is not isolated across the host,
Torch patch and scorer changes. The checks above do not establish whole-pipeline
bitwise equivalence across those runtimes. All final systems use fresh outputs
from the same Mac runtime; no old rows were mixed into the comparison.

The experiments use a recorded runtime override of the same function.
The production function and experimental helper have identical executable ASTs.
Source, wrapper and helper hashes accompany the output evidence. This change
reduces an allocation; it does not make a substitute more accurate.

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

For the additional full combinations, generate with `--systems full_pipeline`
once using `--lexical-run models/runs/ensemble-context-similarity-20261002`,
and once using the strict selector's output `run` directory. Use the same
passages and structure checkpoint, with separate output directories. Alias these
records as `contextual_default` and `contextual_precision` before concatenating
with the eight core variants. Preserve every source generation freeze alongside
the combined prediction hash, as the published provenance does. The scorer
checks matching input IDs, text, panels and references before accepting the
combined file. The completed run contains exactly 1,900 unique system/input pairs.

Generation freezes input, source and model fingerprints before prediction.
Scoring separately freezes prediction, evaluation-model and metric-code hashes.
Interrupted runs can resume only with matching fingerprints. Predictions are
flushed after each case. Equal system/input panels are checked before scoring.
Exact rank-result reuse requires the same input, target offset, original context,
guard setting and acceptance threshold. Mutable trace fields are copied.

Execution records are under
`~/.local/state/cleartext/pipeline-eval-20261027/` on Yoga. Initial generation
ran in a 4 GiB memory-capped Yoga scope. The original attempt stopped for the
word scorer allocation change. The second attempt stopped when Yoga's 15-minute
load exceeded 12. Both partial outputs and logs remain available for comparison.
No unrelated job was stopped.

Final generation runs sequentially on the Mac under Yoga tmux coordinators
`cleartext-bounded-pipelines-20261027` and
`cleartext-baseline-mac-20261027`. It uses an isolated source snapshot, model
assets and dependency overlay. Four CPU threads serve generation. The scoring
coordinator is `cleartext-score2-mac-20261027`; scoring uses two CPU threads and a
separate frozen evaluation source snapshot. Both evaluation models selected the
Mac's MPS device. The scoring runtime file records this and the package versions.
The first scoring launch failed during import because the isolated snapshot
lacked the baseline dictionary file. Restoring the identical public file allowed
a fresh launch; no predictions or metric rows changed. The Mac worker checks memory
pressure and disk space every 20 seconds. It stops its own job if memory
pressure rises or free disk falls below 2 GiB. Completed outputs are copied
back to Yoga. The Mac's primary checkout, shared environment, user applications
and current demo remain intact.

## Subsequent word-ranking experiment

The October 28 [candidate sense usage experiment](SENSE_USAGE.md) tested nine additional WordNet usage and verb-pattern features on frozen grouped development data. It failed the adoption bar, so the models measured in this benchmark remain the retained models. Its separate candidate availability audit identifies losses during generation and eligibility checks for the next investigation.
