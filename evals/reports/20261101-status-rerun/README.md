# Status-report benchmark replay, November 1, 2026

The submitted report's six-passage experiment reproduces. Its core readability,
embedding, bidirectional entailment and explicit-check means match the saved
October 21 results within `1e-12`. Current v7 lexical output is unchanged on all six.
Adding the trained sentence stage barely changes mean reading grade, while the
contextual lexical variant trades weaker simplification for stronger backward
entailment scores. These are automatic diagnostics on authored passages.

## Match to the submitted report

The source is Section 3.3 of the
[project status report](https://docs.google.com/document/d/1xlG3QSOl0MVIsYRYL2rm37vXQerNhgzjM_-9n2WEbsc/edit?tab=t.ivdwwl8wj2yb).
It names the [October 21 GitHub evidence](../20261021-v7/README.md), six authored
passages, v7 lexical simplification, one word edit and disabled phrase/structure
stages. Its `full_pipeline` label predates the trained structural pipeline.

The replay imports model source from `0a17b49`, uses the same local v7 artifacts
and compares it with source at `085015f` plus the explicit replay harness.
All eight model hashes recorded on October 21 match. All 63 implicit encoder,
corpus and parser artifacts recorded in the October 27 supplement also match.
The October 21 record did not freeze every implicit resource, so historical
byte equality for unrecorded resources cannot be established.
All 107 resources in the more complete October 28 native manifest also match;
the dictionary baseline bytes match the October 21 Git object.

Every model arm permits one word edit. Grammar preprocessing, the optional
dictionary lemma repair and source-anchored split fallback remain off.
`T5` below means the experimental screened step 441 checkpoint, SHA-256
`3de186eee031e8812b019a9040775884798d01baabb072728bd03ac89f1503ac`, matching
the larger October 27 benchmark. It has not replaced the demo's default rules.
The separate step 906 split-selector experiment is not substituted. Its saved
[128-input comparison](../../../models/docs/IMPROVEMENT_SPRINT_20261028.md)
reports 55 edits versus 54 with the optional selector, while preserving the
32 copy references. That is a separate reused-development result; this replay
does not exercise that selector or establish a new gain for it.
The contextual weights date to October 2. This replay compares configurations
and performs no new training.

[Interactive metric comparison](comparison.html) ·
[Verification receipt](verification.json)

## Six authored passages

All values are means over the same six inputs; all metrics below are available
for all six. Original text has grade 20.7139. Lower grade is easier. Embedding and
entailment values run approximately from zero to one and are preservation signals.

| Configuration | Changed / 6 | Reading grade ↓ | Embedding ↑ | NLI forward ↑ | NLI backward ↑ |
|---|---:|---:|---:|---:|---:|
| Dictionary baseline | 6 | 16.5735 | 0.9156 | 0.9964 | 0.9934 |
| October 21 source, v7 words only | 6 | 19.6955 | 0.9905 | 0.9629 | 0.6910 |
| Current source, v7 words only | 6 | 19.6955 | 0.9905 | 0.9629 | 0.6910 |
| Current v7 + phrase rules + T5 | 6 | 19.6862 | 0.9871 | 0.9731 | 0.7266 |
| Current v7 + phrase/structure rules | 6 | 20.1694 | 0.9889 | 0.9732 | 0.7264 |
| Contextual variant + phrase rules + T5, threshold .45 | 6 | 20.1336 | 0.9773 | 0.9526 | 0.9866 |
| Same contextual variant, threshold .80 | 6 | 20.3958 | 0.9830 | 0.9530 | 0.9880 |

The v7 trained-stack reading-grade delta is **−0.0093**, a negligible change.
Backward entailment increases by **0.0356**. The contextual configuration raises
that signal by **0.2956**, with a **+0.4381** reading-grade change relative to the
old v7 run. The higher threshold further reduces lexical editing, although
phrase or structure edits still make all six final passages differ from source.
None of these figures measures a percentage of correct simplifications.

### Evaluator changes and controls

The same current evaluator scores all before/after arms. Forty-nine of 51
historically recorded means reproduce within `1e-12`; the other two are TF-IDF
means. The later tokenization fix includes one-character tokens, changing the
dictionary score from 0.6212 to 0.6287 and v7 from 0.8940 to 0.8985. These are
evaluator changes. The archived October 21 table remains intact.

The separate evaluator CWI baseline also reproduces exactly: 27,299 training
and 3,328 development instances, train-selected threshold .55, development
precision 0.482856, recall 0.862392 and **F1 0.619085**. This baseline is the
evaluator's linear A–E word scorer, separate from v7's hard-word detector.
No CWI test inference, replacement-ranking experiment or new model fit was run.

### Qualitative failure retained

For `cs-02`, T5 produces:

> The methodology uses stochastic gradient descent to mitigate overfitting.
> empirical results corroborate a substantial amelioration in convergence speed.

The missing capital at the new sentence boundary remains an output error.
Readability and entailment can improve despite that error. The full authored
outputs and per-input metrics are committed beside this report.

## Frozen 190-input comparison with October 27

This is the same panel used in the [previous ten-system benchmark](../../../models/docs/PIPELINE_BENCHMARK.md):
96 reused DeSSE development inputs, 24 reused target-domain inputs, six authored
examples and 64 ASSET validation inputs. No reserved-domain inference or new
filtering was performed. All ten configurations generated all 190 outputs.
The full rerun also includes twelve additional Week 7 inputs, discussed below.

The table compares the saved October 27 results with fresh November 1 scoring.
Every grade and backward-NLI mean has **n = 190**, including copies. The same
saved lexical/T5 weights are used before and after; this measures code/runtime
changes, rather than another training run.

| Configuration | Changed before → now / 190 | Grade before ↓ | Grade now ↓ | Backward NLI before ↑ | Backward NLI now ↑ |
|---|---:|---:|---:|---:|---:|
| Original | 0 → 0 | 11.7080 | 11.7080 | 0.9896 | 0.9896 |
| Dictionary baseline | 13 → 13 | 11.5235 | 11.5235 | 0.9899 | 0.9899 |
| v7 words only | 140 → 140 | 11.3179 | 11.3060 | 0.9027 | 0.9028 |
| T5 structure only | 20 → 20 | 11.1529 | 11.1529 | 0.9894 | 0.9894 |
| v7 words + T5 | 141 → 141 | 10.7584 | 10.7465 | 0.9038 | 0.9039 |
| v7 + phrases + T5, unchecked experiment | 155 → 155 | 9.6222 | 9.6222 | 0.8971 | 0.8971 |
| v7 + phrases + T5, checked | 141 → 141 | 10.7733 | 10.7614 | 0.9050 | 0.9051 |
| v7 + phrase/structure rules | 140 → 140 | 11.3135 | 11.3016 | 0.9041 | 0.9042 |
| Contextual + phrases + T5, .45 | 137 → 138 | 10.8669 | 10.8576 | 0.9508 | 0.9508 |
| Contextual + phrases + T5, .80 | 76 → 77 | 10.9999 | 10.9947 | 0.9780 | 0.9780 |

The checked v7 stack moves from grade **10.7733 to 10.7614** (delta **-0.0119**), retaining 141 edits. Its final outputs differ on 3 inputs. This is a small readability movement, not evidence of a broad quality breakthrough. The contextual default changes 138 inputs, versus 137 before.

Current changed-only results make the copy/coverage tradeoff visible:

| Configuration | Changed n | Grade ↓ | Embedding ↑ | Forward NLI ↑ | Backward NLI ↑ |
|---|---:|---:|---:|---:|---:|
| v7 words only | 140 | 12.0814 | 0.9634 | 0.9346 | 0.8714 |
| v7 + phrases + T5, checked | 141 | 11.3263 | 0.9628 | 0.9383 | 0.8753 |
| v7 + phrase/structure rules | 140 | 12.0755 | 0.9634 | 0.9351 | 0.8733 |
| Contextual + phrases + T5, .45 | 138 | 11.4881 | 0.9662 | 0.9592 | 0.9357 |
| Contextual + phrases + T5, .80 | 77 | 11.9650 | 0.9821 | 0.9721 | 0.9593 |

Matched comparisons limited to inputs both configurations edit are stored in
[matrix_comparison.json](matrix_comparison.json), along with every changed output ID.
The matched sample sizes against current v7 words are v7 + phrases + T5, checked: 140, Contextual + phrases + T5, .45: 129, Contextual + phrases + T5, .80: 75.

Readability and preservation metric-source hashes match October 27. The old scoring runtime used Mac/MPS with Torch 2.14.1; this run uses Yoga/CPU with Torch 2.14.0+cpu. On identical outputs, maximum absolute residuals are 2.98e-07 for embedding and 0.00249 for backward NLI. No historical metric cache was reused. The current dedicated six-passage and larger runs agree on all 42 repeated output comparisons; metric residuals are recorded beside that check.

The tiny before/after NLI movements in this table cannot establish a semantic
improvement: the observed runtime residual on identical text is larger. The
readability delta follows the three changed outputs, with identical-text
readability scores unchanged.

None of the 202 inputs contains backticks, and grammar preprocessing is off.
This panel therefore does not measure the separate literal-preservation or
grammar-correction mechanisms. Their tests and earlier native controls remain
separate evidence. There is no verified 90% full-pipeline accuracy result.

## Kea-Roy's Week 7 authored inputs

[PR #7](https://github.com/JoshD94/cleartext/pull/7), commit
`b1aa16666b298a2d44c1b5b02ae2256a873bcd29`, supplies 18 authored passages.
Six overlap the historical panel and twelve are additional, so generation uses
202 unique inputs rather than 208. This table independently validates **18 × 10**
outputs and reports all-input means with **n = 18** for each displayed metric.
These are authored smoke checks, separate from the 190-input development panel.

| Configuration | Changed / 18 | Grade ↓ | Embedding ↑ | Backward NLI ↑ |
|---|---:|---:|---:|---:|
| Original | 0 | 19.1685 | 1.0000 | 0.9889 |
| Dictionary baseline | 9 | 17.6379 | 0.9708 | 0.9915 |
| v7 words only | 18 | 18.6233 | 0.9849 | 0.7114 |
| T5 structure only | 4 | 18.0719 | 0.9987 | 0.9895 |
| v7 words + T5 | 18 | 17.5258 | 0.9839 | 0.7149 |
| v7 + phrases + T5, unchecked experiment | 18 | 15.6729 | 0.9763 | 0.7297 |
| v7 + phrases + T5, checked | 18 | 17.7154 | 0.9832 | 0.7271 |
| v7 + phrase/structure rules | 18 | 18.8129 | 0.9843 | 0.7236 |
| Contextual + phrases + T5, .45 | 18 | 17.8825 | 0.9816 | 0.9363 |
| Contextual + phrases + T5, .80 | 16 | 18.0380 | 0.9847 | 0.9367 |

This uses the same ten configurations as the larger replay. It does not complete
PR #7's different eight-arm runbook: its rule-based `structure_only` and standalone
phrase arm differ from this benchmark's trained `structural_only` arm. The teammate
PR remains unchanged and unmerged.

[Numeric per-input records](matrix_190_metrics.jsonl),
[Week 7 numeric records](week7_18_metrics.jsonl), all/changed-only/panel summaries,
source/model freezes, [runtime controls](runtime-controls.json) and
[post-run integrity checks](post-run-integrity.json) are committed beside this report.
Larger-panel raw text and predictions stay in the local state directory.

## Execution and verification

Yoga ran the inference in named tmux jobs, with 4 GiB process caps. Generation
and external-model scoring used separate processes. The larger generation jobs
were `cleartext-pipeline-rerun-20261101` and
`cleartext-context-rerun-20261101`; scoring used
`cleartext-score-warm-20261101`, `cleartext-strict-tail-score-20261101` and
`cleartext-final-score-20261101`,
with 3 GiB scoring caps. Model weights, training
data, thresholds and the existing demo were preserved. Exact input, source,
model and evaluator snapshots are recorded in the adjacent manifests. The
local state directory retains raw larger-panel inputs, predictions, timing
logs and completion receipts:
`~/.local/state/cleartext/status-benchmark-pr-20261101/`.

`scripts/verify` passed **419 tests and 13 subtests** in 23.80 seconds, with no
skips reported and 0.82 GiB peak RSS. It covers parser checks, code preservation,
rollback traces, grammar composition, candidate morphology, manifest pinning,
structural alignment/search and evaluation completeness. It does not estimate
semantic accuracy.

## Reproduction

Model binaries remain outside Git. Copy the saved lexical runs and their
configuration-referenced dependencies, the screened T5 step 441 checkpoint,
the local parser/corpus resources and the two pinned evaluation-model snapshots
before reproducing. Their hashes are in the manifests. Using these saved
artifacts requires no retraining.

After checking resources, run the wrapper in a named Yoga tmux job with a
4 GiB memory cap. `CLEARTEXT_EVAL_DEPS` optionally points to an evaluation
dependency overlay; this run used textstat 0.7.13, pyphen 0.18.1 and
sentence-transformers 6.2.0 without changing the model environment.

```sh
export CLEARTEXT_REPLAY_OUT="$HOME/.local/state/cleartext/report-replay-fresh"
export CLEARTEXT_EVAL_DEPS=/path/to/evaluation/dependencies
evals/reports/20261101-status-rerun/reproduce.sh six
```

The `matrix` mode additionally requires `CLEARTEXT_PASSAGES` pointing to the
frozen 190-input file with SHA-256
`2d39da8438317166a24e69d6a822e29039c5ae3031b8a23da95b912244ae2d4f`.
The prior [data manifest](../../../models/evidence/pipeline-20261027/data_manifest.json)
records its original preparation inputs and exclusions. Optional
`CLEARTEXT_WEEK7_PASSAGES` points to Kea-Roy's 18-passage JSON at commit
`b1aa16666b298a2d44c1b5b02ae2256a873bcd29` in
[PR #7](https://github.com/JoshD94/cleartext/pull/7). The wrapper appends only
its twelve additional passages, for 202 unique inputs. Preserve the reserved
domain set; reuse the frozen panel for this comparison.

`reproduce.sh detector` runs the separate frozen evaluator CWI control using
the existing local training/development TSVs. It checks the saved metrics and
writes a fresh result outside Git. The wrapper's shell syntax and Python lint
checks pass; native inference/scoring commands were executed individually for
this evidence, and the portable detector command was also executed directly.

For the exact 190-input table, leave `CLEARTEXT_WEEK7_PASSAGES` unset. With that
variable set, the wrapper scores the combined 202-input matrix; this report
partitions its per-input metrics into the independently validated 190- and
18-input tables by source identity.

Scoring began with two CPU threads and resumed its 65 completed fresh pairs with
four threads after resource checks. Formula/model bytes and thresholds stayed
fixed. The interrupted attempt and cache fingerprint are retained in the local
records and `runtime-controls.json`. Five disjoint strict-only pairs were scored
in a separate complete five-input batch on Yoga at two threads. Both fresh
caches were checked against every source/output key in the full matrix before
final aggregation; all 2,020 rows remain included. No historical metric cache
was used. Final scoring completed successfully.
