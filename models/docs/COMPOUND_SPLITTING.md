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
inter-sentence whitespace. Quoted text, code, uncertainty, negation and ambiguous
shared auxiliaries/modifiers abstain conservatively. Reported speech, conditions
and other embedded scope still abstain. An attached relative or participial
clause may stay wholly on one side when both main clauses have their own finite
predicate and subject. Shared-subject splits with attached relatives still abstain.
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

Coverage remains an implementation gap. The follow-up below supports a bounded
subset of independent clauses containing relatives. Do not relax all guards
or tune thresholds on these inspected evaluation panels. Existing lexical gains
must not be attributed to the new splitter. Raw outputs, per-passage metrics,
review packets and failure logs remain local on Yoga and Mac; only aggregates,
public authored controls and provenance are committed.

## Boundary checks and optional length policy

The follow-up implementation can split this authored mechanism example:

> Alice, who wrote the report, checked the figures, and Bob signed the document.

into:

> Alice, who wrote the report, checked the figures. Bob signed the document.

Coordination inside the relative stays intact. A conjunction inside
`who wrote and revised the report` is rejected as an internal boundary while
the later matrix boundary can still be accepted. Relative attachment and joint
subject checks supplement the existing argument, operator and discourse checks.
In particular, `Alice and Bob lifted the piano.` cannot pass validation as
`Alice lifted the piano. Bob lifted the piano.` The shared checks previously
missed that joint-action distinction. These additions apply to this opt-in
splitter; they do not certify semantic equivalence or change the T5 validator.

Use `DependencyCompoundSplitter(min_sentence_words=20, min_clause_words=6)`
or CLI flags `--min-sentence-words 20 --min-clause-words 6` to attempt only longer
sentences with enough words on both sides. These illustrative thresholds are
not tuned or validated quality cutoffs. Defaults are zero, so existing short
controls still run. Counts exclude punctuation/space; clause counts exclude the
boundary conjunction and any copied subject. The sentence count includes the
conjunction. Length is an eligibility filter, not a safety/confidence score.

`--details` now reports `boundaries` with source character offsets, acceptance,
rejection reasons and available length measurements. `candidate_count` counts
considered noninitial coordinator tokens, including rejected noun coordination;
`accepted_count` counts surviving sentence edits. Processing stops after the
first accepted boundary in each sentence. A passage rollback marks provisional
acceptances as rejected. No calibrated probability is exposed. Global literal,
negation and modality exclusions still apply to the whole source sentence.

Parser failures remain: the small parser mistakes the main verb in
`The server, which stores records, checks requests, and the client displays results.`
for part of a noun phrase, so the splitter abstains. Authored examples and tests
establish mechanism behavior only. The earlier 0/108 result belongs to the first
implementation; a fresh comparison is required before claiming benchmark gain.
