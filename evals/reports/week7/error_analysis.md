# Sentence-level error analysis — Week 7 (Oct 26–Nov 1)

Kea-Roy Ong, evals side. Deliverable: record and categorize simplification
errors for the combined system.

## Method (the "LLM check" question, decided)

Two stages, in this order. Deterministic signals first so the review is
reproducible; LLM/manual review second, with a fixed rubric, on the texts
actually available.

**Stage 1 — deterministic flags.** Every changed output with backward NLI
< 0.50, or any explicit check (numbers, dates, entities, negations,
tenses) < 1.0. Applied to all four model arms over the 190-input benchmark
panel (evidence file `evidence_error_flags.json`, 68 flags, 27 unique
inputs), and to the local baseline run on the 18 authored passages.

**Stage 2 — rubric review.** Each flagged case is coded against the
taxonomy below. Model output *texts* for the 190-input benchmark panel
are not available on this machine: the benchmark evidence stores
output hashes only, and predictions live on Yoga. Model cases from
that panel below are therefore **signal-coded** (flag pattern +
the benchmark report's diagnostic notes), not text-verified; the rubric
is written so Joshua or Chhoeng can re-apply it to the texts on Yoga and
confirm or overturn each code. The dictionary baseline's outputs *were*
reviewed as text here, since they run locally. Output texts for the
ladder runs on the 18 and 90 Week 7 sets were shared Nov 3 and every
flagged case in them was read as text; those verdicts are in Finding 6
and are text-verified, not signal-coded.

## Taxonomy

| Code | Category | Signal pattern |
| --- | --- | --- |
| E1 | Detail loss | Backward NLI low, forward NLI high: the output is entailed by the original but drops a qualifier, quantity, direction, or condition |
| E2 | Meaning shift | A substitute changes the sense (wrong word sense, near-synonym with a different denotation) |
| E3 | Predicate-role error | Subject/object/agent-patient relations swapped or misassigned, including split/clause-extraction errors |
| E4 | Mechanical check failure | An explicit check (number, date, entity, negation, tense) < 1.0 |
| E5 | Grammar breakage | Output ungrammatical: split artifact, agreement, article, or word-order error |
| E6 | Under-simplification (coverage) | No edit on an input that needed one; tracked separately, not a correctness error |
| E0 | Check artifact | Explicit-check failure with NLI ~0.99: the check, not the output, is at fault |

## Finding 1: the flagged meaning-loss cases are lexical, not structural

19 of the 27 flagged inputs carry *identical* scores in `lexical_only`,
`full_pipeline`, and `full_no_guardrails` — the same output in all three
arms. The structure stage and the guardrails did not touch those outputs.
The errors that survive into the combined system arrive with the word
edits. Examples:

- **corp-02** (authored smoke, Oct 21 case; original: "The company
  will commence a restructuring initiative in order to expedite
  operational efficiency and ameliorate shareholder value, effective
  January 15, 2026."): backward NLI 0.012 in all arms. Forward 0.88.
  Classic E1 signature: the output is entailed by the original but the
  reverse fails, meaning some detail of the original does not survive.
  Which detail is unknown: the output text was not stored.
  Text-verified Nov 3 (Finding 6): the only edit is ameliorate to
  improve, and no detail is lost. Recoded E0. The E1 signature was a
  metric failure, not a text failure.
- **quic-rfc9000-3** (domain, CS): forward 0.83, backward 0.015. E1.
- **quic-rfc9000-1** (domain, CS): forward 0.41, backward 0.49. Both
  directions low: the output diverges from the original more broadly
  than a single dropped detail; needs text review (E1/E2 unconfirmed).
- **asset-valid-565**: forward 0.85, backward 0.08 in all arms. The
  benchmark report's diagnostic note records loss of geographic
  directional detail (E1, signal + note agree).
- **asset-valid-663** (forward 0.13, backward 0.07), **asset-valid-278**
  (0.38/0.003), **asset-valid-1782** (0.68/0.006): both directions low;
  text review required before coding beyond "severe divergence."

One counter-case: **polsci-01** (Oct 21 case; original: "The
legislation seeks to disseminate pragmatic reforms that mitigate
ubiquitous regulatory fluctuations affecting a large number of
constituents.") scores backward 0.165 lexical-only but 0.374 in the
combined arm. The combined output differs from the lexical-only output
and preserves more. The structure stage (or the combined edit selection)
helped here; it is still flagged, still E1-candidate, but less badly.
Text-verified Nov 3 (Finding 6): the lexical error is ubiquitous to
present, E2 mild, and the combined output adds the clean phrase edit
(a large number of to many). The difference is the phrase stage, not
structure: `structure_only` changed nothing on this set.

## Finding 2: guardrails catch a distinct subset

Five inputs are flagged *only* in `full_no_guardrails`:

- **desse-train-10769**: forward 0.14, backward 0.01. The benchmark
  report identifies this as a predicate-role failure (E3) that remains
  in unguarded combinations; the guarded arms do not produce it.
- **desse-train-5830** (backward 0.036), **desse-train-5938** (0.002),
  **asset-valid-1270** (0.063): severe low-backward cases the checks
  block.
- **asset-valid-219**: guarded backward 0.57 (above the 0.50 flag
  threshold), unguarded 0.47. The benchmark note records geographic
  directional detail loss in the guarded combinations too. This case is
  the reason Stage 1 thresholds cannot stand alone: the guarded output
  is still wrong, just less wrong than the flag cutoff.

Net: on the 190 inputs, guardrails change 14 outputs (155 vs 141
changed) and remove exactly the 5 worst unguarded-only failures, at no
measurable cost to average preservation. Their value is concentrated,
not average. Ladder guardrail verdicts on the 90 sourced passages
(text-verified, Finding 6) show the same shape with a qualification:
guardrails blocked 2 significant errors and 3 mild ones, but missed
3 cases where the guarded output is worse, all technical-term
substitutions the meaning check passed.

## Finding 3: the structure-only arm is clean but narrow

`structural_only` changed 20/190 inputs and tripped **zero** flags: no
low-backward case, no failed check. Its errors, if any, are below the
detection floor of these signals — or absent. Either way, the T5
structure stage is not a source of flagged errors in this run. Its
problem is coverage (20/190), recorded under E6, not correctness. The
main-branch old-rules structure stage is narrower still: executed here
on the 18 Week 7 passages, it changed 0 (see ablation_table.md,
Source C).

## Finding 4: most entity-check failures are check artifacts (E0)

Seven flagged inputs fail only `entities_preserved` with NLI ~0.99 in
both directions (desse-train-5012/3664/7215/5487, asset-valid-781/1723,
plus asset-valid-781's twin cases). The explicit entity check is
string-based: an acceptable substitution of an entity mention fails it
while NLI judges meaning intact. These are coded E0, not model errors.
The check needs the boundary fix the benchmark report notes (complete
numeric items and word boundaries), extended to entity spans.

The one genuine mechanical failure: **biden-inaugural2021-0** fails
`tenses_preserved` in all arms at NLI ~0.97 (E4, tense shift in the
output; text review to confirm).

Two sourced ladder flags (cs-s01, cs-s07) fail only
`entities_preserved` at NLI ~0.99, the same signature, coded E0.

## Finding 5: dictionary baseline — no correctness errors, severe under-coverage

All 9 changed outputs on the 18 authored passages were reviewed as text.
Every substitution preserves meaning:

- cs-01: utilize → use, heterogeneous → varied, facilitate → help.
- cs-02: methodology → method, mitigate → reduce, empirical → observed,
  corroborate → confirm, substantial → large.
- corp-01: proliferation → spread, exacerbate → worsen, notwithstanding
  → despite.
- corp-02: commence → start, in order to → to, expedite → speed up,
  ameliorate → improve.
- polsci-01: disseminate → share, pragmatic → practical, mitigate →
  reduce, ubiquitous → common, a large number of → many.
- polsci-02: paradigm → model, nevertheless → still, prior to → before.
  The paradigm → model substitution is the closest call in the set
  (a narrowing), but bidirectional NLI stays at 0.99; coded acceptable,
  not an error.
- cs-05: mitigate → reduce.
- corp-05: prior to → before, notwithstanding → despite.
- corp-06: approximately → about, in order to → to.

No E1–E5 errors. But 9 of 18 passages are unchanged (E6), and on the
24 real domain inputs the dictionary changed **0**. The baseline's
failure mode is coverage, and it is total on real text.

The run's single explicit-check failure, `entities_preserved` on
corp-05, is an E0 artifact, verified directly: the entity check is a
capitalized-span heuristic, and it treats the sentence-initial words
"Prior" (original) and "Before" (output) as entities. Neither is a
named entity; the substitution is correct and nothing was lost.

## Finding 6: ladder text review (text-verified, Nov 3)

Every flagged output from the Nov 2 ladder runs on the 18 authored
and 90 sourced passages was read as text, plus all four phrase_only
edits and all ten passages where guarded and unguarded outputs
differ. Codes below are text-verified. The pattern from Finding 1
holds and sharpens: every genuine error is a single-word lexical
substitution. No structure or phrase edit produced a flagged error.

**Authored set (lexical arms, 5 low-backward flags).** Four are real
sense errors, all E2: polsci-01 ubiquitous to present (mild),
cs-04 retransmit to carry (significant: a sender retransmits packets,
it does not carry them), cs-06 compiler to program (the compiled
program does not eliminate its own dead code), corp-06 subsidiary to
company (entity shift). One is a false alarm: corp-02 ameliorate to
improve, coded E0 (see Finding 1 correction). By domain: CS 2, corp 2
(one E0), polsci 1. `full_guarded` and `full_no_guardrails` produce
identical outputs on all 18, so guardrails changed nothing on this
set; the polsci-01 guarded improvement over lexical-only is the
phrase edit, not a guardrail.

**Sourced set (lexical arms, 14 low-backward flags).** Twelve are
genuine errors and two are borderline. Technical-term confusion:
cs-s08 handshake to acknowledgement (significant), cs-s14
authenticating to certifying (significant), cs-s30 authenticate to
demonstrate, cs-s06 algorithms to rules. Near-synonym
over-broadening: corp-s09 reliability to responsibility (significant,
inside a 10-K assertion), corp-s28 attractiveness to quality,
corp-s30 estimated to calculated, cs-s23 interoperability to ability.
Rhetorical diction: polsci-s19 generous to big, polsci-s18 eternity to
time (also E5: the output repeats "time and time"), polsci-s22
hostages to prisoners (significant: the sentence already
distinguishes released prisoners from released hostages), polsci-s30
tutoring to teaching (mild). Borderline: cs-s04 bidirectional to
duplex (the inline definition, "which allow both endpoints to send
data", preserves the term's meaning) and corp-s12 procure to secure.
NLI magnitudes exaggerate throughout (0.02 to 0.45 for one changed
word) but the direction is right in nearly every case.

**Guardrail divergences (10 sourced passages).** Guardrails blocked
2 significant errors: cs-s09 (unguarded rewrote the quoted spec term
"Probing" to "Examining"; guarded output changes only initial to
first) and corp-s18 (unguarded rewrote "accruals" to "increases";
guarded output changes only evaluate to assess). They blocked 3 mild
ones (cs-s29, corp-s25, polsci-s17). They missed 3 where the guarded
output is worse: cs-s08 (guarded output carries the handshake error;
unguarded output changes only containing to including), cs-s05
(guarded output reads "An end SHOULD", from endpoint to end;
unguarded output is clean), corp-s04 (guarded output changes the
term of art "performance obligations" to "responsibilities";
unguarded output changes only allocated to assigned). One divergence
is neutral (corp-s01: each run changes a different occurrence of
"restricted" to "limited", both fine) and one favors unguarded
(polsci-s04: unguarded peril to danger is a clean simplification;
guarded restore to improve is a mild shift). Net: guardrails help
more than they hurt, with a technical-term blind spot in the meaning
check.

**Phrase and dictionary.** All four phrase_only edits are clean:
polsci-01 a large number of to many, polsci-02 prior to to before,
corp-05 Prior to to Before, corp-s24 with the exception of to except.
The dictionary baseline's two degrading sourced edits reproduced
exactly on Yoga (cs-s06 "an carrying out should", cs-s26 "MUST carry
out the Connection header field"), so those failures are stable
across machines, not environment quirks.

## What still needs text review (on Yoga)

1. The 8 "both directions low" cases (quic-rfc9000-1, asset-valid-663/
   278/1782/790, desse-train-8123, asset-valid-778/1222): code E1 vs E2
   vs E3 from the output texts.
2. asset-valid-219's guarded output: confirm the geographic-detail loss
   the benchmark note records.
3. biden-inaugural2021-0: confirm the tense shift.
4. The PR #9 compound-split passage: the single changed sourced
   passage is metrics-only in the committed evidence; its split text
   is unreviewed.

Done (text-verified Nov 3, Finding 6): polsci-01's combined output,
all ladder flags on the 18 and 90, all phrase edits, all guardrail
divergences.

## Error counts (signal-coded, 190 inputs)

| Arm | Changed | Flagged inputs | Low backward NLI | Check-only failures |
| --- | ---: | ---: | ---: | ---: |
| lexical_only | 140 | 22 | 15 | 7 |
| structural_only | 20 | 0 | 0 | 0 |
| full_no_guardrails | 155 | 24 | 17 | 7 |
| full_pipeline (guarded) | 141 | 22 | 15 | 7 |

Counts overlap: most flagged inputs are shared across arms (Finding 1).
"Check-only" = failed explicit check with backward NLI >= 0.50, mostly
E0 entity artifacts.

## Error counts (text-verified, ladder runs, Nov 2)

| Set / arm | Changed | Flagged inputs | Real errors (E2/E5) | Borderline | E0 artifacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Authored / lexical_only | 18 | 5 | 4 | 0 | 1 |
| Authored / full_guarded | 18 | 6 | 4 | 0 | 2 |
| Sourced / lexical_only | 85 | 16 | 12 | 2 | 2 |
| Sourced / full_guarded | 85 | 16 | 12 | 2 | 2 |
| Sourced / full_no_guardrails | 85 | 17 | 13 | 2 | 2 |

Authored full_guarded adds the corp-05 entity artifact to the lexical
flags; its low-backward flags are the same five. Sourced
full_no_guardrails adds cs-s09 and corp-s18 (both blocked by
guardrails) and drops cs-s08 (guarded-only flag). "Real errors" counts
low-backward flags read as genuine sense errors; borderline counts
cs-s04 and corp-s12.
