# Checked candidate search

This opt-in experiment reuses the continued T5 checkpoint. It generates four
deterministic beams and tries them in model order. Each beam must pass the
existing content, scope, tense and clause checks, plus two new boundary checks.
If a comma in the first beam separates independently parsed complete clauses, the selector can
replace it with a period and capitalize the next character. The repaired text
must pass every check too. A valid preferred copy stays a copy. If all candidates
fail, the selector returns the source.

Lower-ranked beams must pass as generated. The selector does not repair them.
Development inspection found that composing a lower-ranked candidate with a
punctuation repair could expose a noun-list fragment that the parser accepted.
Restricting repairs to the preferred beam avoids that observed combination;
it does not solve the underlying parser ambiguity.

The comma repair changes no words or word order. It stops at four sentences or
three repairs and abstains on sentences containing quotations, brackets or
dashes. Valid coordinating commas and semicolons stay unchanged. A source-root
check rejects some detached subordinate fragments even when the output parser
incorrectly calls them complete. These are parser diagnostics, not proof of
grammar or preserved meaning. Relative clause scope, coreference, discourse
relations and parser errors remain limitations.

The existing demo, `SentenceGenerator`, shared ranker features and saved weights
keep their behavior. Import `CandidateGenerator` from
`cleartext.structure.search` to use the experiment with a local checkpoint.
The returned dictionary includes selected beam, repair origin, attempted
candidates and failed checks. There is no reference-dependent selection.

From the repository root, run one sentence with an extracted checkpoint:

```sh
OMP_NUM_THREADS=2 models/.venv/bin/python models/scripts/predict_structure_search.py \
  --checkpoint /path/to/seq2seq --device cpu --details \
  "Alice reviewed the report and Bob checked the figures."
```

Omit `--details` to print only the selected sentence. This loads local weights;
it does not start training or download new data.

## Fixed comparison protocol

Run `python -m cleartext.structure.search_experiment prepare --help` with
`PYTHONPATH=models/src` in the structural-model environment. Preparation verifies
the pinned published DeSSE files and saves a data hash and settings manifest.
Supply all prior selected datasets with repeated `--previous` flags, in order
with the latest continued-training data last. Supply prior inspected inputs
with `--exclude`. Only domain development inputs go in `--domain-dev`.

The comparison has three variants using the same checkpoint:

- Current two-beam generation and current checks.
- The same two-beam raw text with the additional boundary checks.
- Four-beam search with those checks and bounded comma repair.

Development uses the existing 128 published pairs, 24 domain development inputs
without references and six authored diagnostics. The fresh test sample has 128
previously unscored official DeSSE test pairs. Sampling uses fixed source hashes,
length and markup filters, exact source/full-target exclusions and duplicate
removal. It does not filter on reference correctness or any model check. All
official DeSSE test sources and full targets were excluded from the preceding
training run. Exact matching does not rule out near duplicates or pretraining
overlap. Document grouping is unavailable.

Run the explicit `evaluate --phase dev` command first, then inspect development
results. Freeze settings and source before `evaluate --phase test`. Both require
a fresh output directory. Each writes source/module, model and data hashes before
inference, candidate traces, timings and metrics. No fitting or threshold tuning
is part of this pass. Reserved project domain test inputs stay unused.

Report SARI as single-reference token overlap, with a paired row bootstrap for
the difference. Also report changed outputs, annotated copies retained, sentence
length, check failures and latency. Check-pass rates cannot establish quality,
especially when the same checks select candidates. Copied malformed inputs can
still fail diagnostics. Count failures on changed outputs separately. No human
ratings or domain quality scores are supplied by this experiment.

Run persistent inference on Yoga in a named tmux session with a 3 GB memory cap
and two CPU threads. Allow roughly 0.5 GB of new disk space. Preserve all previous
artifacts. The results determine whether to retain this as an experimental
option; they do not automatically change demo defaults.

## October 27 result

The [saved comparison](../evidence/structure-search-20261027/README.md) did not
establish an overall quality gain. On 128 fresh test pairs, current decoding
scored 37.03 SARI and search scored 36.36, with an uncertainty interval spanning
a tie. Both changed 19 inputs and kept all 31 annotated copies. Inspection found
false alarms and missed meaning changes despite fewer diagnostic failures.
Keep this option experimental. The useful technical comma repair and rejection
of observed fragments do not establish general sentence simplification quality.
