# Non-LLM compound splitting

Joshua's compound-sentence task uses the existing spaCy `en_core_web_sm`
dependency parser and deterministic character edits. There is no T5, decoder,
LLM call or new training. The parser itself is a small learned model.

```sh
models/.venv/bin/python models/scripts/predict_compound.py --details \
  'The store closed early, and the workers went home.'
```

The output is `The store closed early. The workers went home.` Contrast and
consequence keep `But` and `So` at the second sentence's start. Some finite
shared-subject predicates can repeat an existing definite subject, such as
`The server checks each request and sends a response.`

Use `DependencyCompoundSplitter()` as `structure_rewriter` with
`EnsembleClearText.analyze(..., structure=True)`. It is opt-in: the demo and
the old `structural()` rules are unchanged. Existing passive conversion is
untouched; this splitter does not rewrite passives or negation.

For Kea-Roy's evaluation runner, select `compound_only` or
`compound_full_guarded`. `structure_only` still means the old rules and
`structural_only` in the separate T5 benchmark still means T5. Compare each
new variant with its matching old-rule arm on the same inputs.

This version attempts at most one boundary per original sentence, retaining
inter-sentence whitespace. Quoted text, code, embedded clauses, uncertainty,
negation and ambiguous shared auxiliaries/modifiers abstain conservatively.
Only comma + `so` is currently supported. Parser mistakes can reject valid
splits: the authored technical fixtures preserve those failures. Acceptance
checks are diagnostics, not proof of semantic equivalence. Authored regression
cases are mechanism checks, never held-out quality labels. The first benchmark results below measure coverage as well as automatic scores.


## November 2 benchmark

The [Week 7 ladder evidence](../../evals/reports/20261102-week7-ladder/aggregate-results.json)
runs Kea-Roy's exact eight systems plus `compound_only` and
`compound_full_guarded` on 18 authored and 90 sourced passages. All 1,080
expected outputs returned. [Provenance](../../evals/reports/20261102-week7-ladder/provenance.json)
records implementation commit e0b9e94, input/model/resource hashes, native Mac
package versions and successful hash checks. Evaluation used the existing v7
lexical artifacts, pinned BERT, MiniLM embeddings and DeBERTa NLI on CPU;
no training or T5 decoding ran. Full verification passed 463 tests plus 13 subtests.
The Mac setup required explicit NLTK/Hugging Face cache paths and a byte-exact
BERT revision reference; initial failed setup/test attempts are retained locally.

The new splitter changed **0/108 passages** and its guarded pipeline produced
exactly the existing guarded outputs. This pass therefore demonstrates no added
benchmark benefit from compound splitting. Standalone phrase rewriting changed
3/18 authored and 1/90 sourced passages with the phrase model present.

The parser-only coverage audit found 215 sentences, 131 containing `and`, `but`
or `so`. Only nine of those survive the initial conservative filters; these
include noun lists, a sentence-initial connective and shared-auxiliary forms.
The three known-positive `and`/`but`/`so` controls split correctly. The counts
of embedded-clause, negation/modality and literal flags overlap: they are
coverage diagnostics, not semantic error labels or exact attribution of every
rejection. The controls are authored mechanism checks, not independent accuracy
measurements.

Coverage is the next implementation gap: support independent clauses that
contain their own embedded clause or scoped modifier, with targeted preservation
regressions, before considering any broader rollout. Do not relax all guards
or tune thresholds on these inspected evaluation panels. Existing lexical gains
must not be attributed to the new splitter. Raw outputs, per-passage metrics,
review packets and failure logs remain local on Yoga and Mac; only aggregates,
public authored controls and provenance are committed.
