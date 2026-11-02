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
cases are mechanism checks, never held-out quality labels. No measured
benchmark gain is claimed before the queued comparison completes.
