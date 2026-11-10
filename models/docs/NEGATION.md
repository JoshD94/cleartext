# Negation simplification

Kea-Roy Ong, model side. Rule from the Sentence Structure Rule tab
(team spec): double negatives and "not un-" constructions.
"It is not uncommon" to "It is common"; "Not an uncommon behavior"
to "A common behavior"; "Not hard" to "easy"; prefixes Un, In, Im, Ir.

## What the rule does

`negation_rewrite(text)` in `models/src/cleartext/rewrites.py`, at most
one edit per sentence. It fires only when the negated element is the
predicate:

- Copula predicate adjective: "It is not uncommon" to "It is common"
  ("not" plus a negative-prefixed adjective as the complement of "be").
  Covers the un/in/im/ir/il prefix map (84 curated pairs) and the
  plain whitelist (hard to easy, the doc's example only).
- Fragment noun phrase: "Not an uncommon behavior" to "A common
  behavior", with the article fixed (an to a when the replacement
  starts with a consonant sound; "an unusual event" to "a usual
  event").
- Double negatives with a positive counterpart: "It is not nothing"
  to "It is something", "He is not nobody" to "He is somebody",
  "There is not no reason" to "There is some reason".

The prefix map is curated, not mined: each pair is a genuine negative
prefix plus its positive stem, both verified as real words. Plain
adjectives are a separate small whitelist; general antonym substitution
is deliberately not done, because "not good" is not "bad" and "not
big" is not "small".

## What it refuses, and why

- Negation over a verb with the adjective in an object: "She did not
  give an uncommon answer" stays, because removing "not" would change
  what she did, not what the answer was.
- Correlatives: "not only", "not just", "not merely", "not simply",
  and any "but" after the negation ("not only X but Y").
- Quantifiers between the negation and the adjective: "Not all answers
  are uncommon" stays, because the negation scopes over "all".
- Degree-modified adjectives: "not very uncommon" stays.
- "not never": no safe flattening (ambiguous between "always" and
  "sometimes").
- Quoted text, code literals, and named entities, like the phrase
  stage.
- Comparative and superlative adjectives (base form only).

## Litotes caveat

Flattening strengthens the claim: "not uncommon" is weaker than
"common". The rule applies the flattening anyway, per the team spec.
Consequence for measurement: the evals preservation suite's
`negations_preserved` check counts negation markers, so it scores
0.00 on every flattened output by design. That check compares
markers, not truth conditions, and needs a carve-out before this
stage is ever measured through the evals ladder.

## Wiring

`EnsembleClearText.analyze(..., negation=False)`: opt-in, off by
default. New stages stay off until the team adopts them. The stage
runs before structure, because `structural()` abstains on any sentence
containing "not": de-negating first lets it fire on sentences it
would otherwise skip (verified: "It is not uncommon, and the workers
went home" becomes "It is common. The workers went home."). Details
in the returned `negation` field, same shape as `structure`. The demo
server accepts a `negation` boolean and traces the stage. The
technical-term rollback and the code-literal rollback mirror the
structure stage.

## Measurement

Coverage on the Week 7 panels (run Nov 8, this machine): **0 of 108
passages changed** (0 of 18 authored, 0 of 90 sourced). The panels
contain 55 negation tokens, all verb negations ("does not
guarantee", "not follow", "not include"). No predicate-adjective
litotes occurs in either set, so the rule's targets are rare on real
prose, the same coverage story as structure_only (0 of 108) and
phrase_only (1 of 90).

Probe set (16 authored litotes sentences, one per pattern, scored
with the evals readability and preservation functions, full suite):

- Readability: FKGL 4.19 to 3.91, Flesch 75.39 to 75.62.
- Preservation means: TF-IDF 0.515, embedding 0.766, NLI forward
  0.918, NLI backward 0.929. Entities, numbers, and tenses 1.00.
- NLI accepts the flattening in both directions for 15 of 16 cases
  (backward 0.98 to 1.00). The exception is the "not no" case:
  "There is not no reason to leave" to "There is some reason to
  leave" scores backward 0.03, forward 0.63. Either the NLI model
  struggles with double negation (it gave 0.012 for a benign
  substitution in the Week 7 review) or the flattening genuinely
  weakens the litotes reading ("not no reason" often implicates a
  strong reason). That sub-rule is the one to revisit if the team
  tightens the rule; the prefix and pronoun patterns are clean.

## Adoption

Not on by default. To adopt: turn the `negation` flag on in the demo
and the evals ladder arm, and decide the "not no" sub-rule (keep with
the NLI flag noted, or drop it). The evals ladder needs a negation
arm plus a carve-out in the negations check before this stage can be
measured there; that is evals work, parked while the team is on
models.

## Tests

`models/tests/test_negation.py`, 11 tests: the four doc examples, all
prefix classes, the contraction, the double negatives, multi-clause
and multi-sentence handling, one edit per sentence, the attributive
case, and every abstention above. Run with the focused checks in
`scripts/verify` (added there) or directly with pytest.
