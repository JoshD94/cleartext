# Round 8 protocol (structure v4; written November 23, 2026, before any round-8 output)

Joshua's brief: push sentence-level rewrites further with exactly the
structure v3 protocol. In order: raise by-phrase passive → active recall and
its Good rate outside the Federal Register as a new versioned op
(`passive_v6`) without changing earlier ops' outputs; nominalization → verb
(`nominal_v3`) with pooled fresh windows to get enough items; a stronger
parser only if it needs no new download (none is installed:
`en_core_web_trf` is absent and nothing was downloaded, so none is used).
Bar for each new op: **majority Good ≥ 90% and majority Changes meaning
≤ 8%** of three blind raters on fresh text, with Wilson 95% intervals.

## Diagnosis (read before this file)

- **Where `passive_v5` loses recall outside the Federal Register.** On
  never-rated sentences with "by" in the old pools (build, A–G; 960 by-phrase
  passive clauses in Wikinews/Wikipedia text, 917 in openFDA labels), the
  v2–v4 skips stop all but 51 and 142 of them, and v5 keeps 7 and 5. The
  largest single skip is "coordinated predicate shares the subject" (244 of
  the 960 news/wiki clauses): "The episode was written by X and directed by
  Y", "The bridge was designed by X and opened in 1932", "…, but his methods
  were questioned by Y". Relaxing individual v5 reasons would gain a few
  items each with mixed labels (labelled rated items, rounds 1–7: no single
  v5 reason blocks more than 7 Good items), so v6 does not relax them.
- **Rated v5 failures of rounds 6–7** (all seven read; see the structure v3
  README): a trailing "whilst working" clause, "applause … met", a later
  "their" that can pick up a coordinated agent, "MTA Bridges and Tunnels"
  made plural, a procedure as the doer, "the discharges caused the
  evacuations to water", the light verb "be done by".
- **Nominalization:** every rated `nominal`/`nominal_v2` failure of rounds
  1–5 (29 of 179 items) read: the noun's preposition left on the verb
  ("decided regarding", "suggested on"), a comma-interrupted complement, a
  later pronoun that lost its antecedent ("unable to pay when they are
  due"), a modifier of the noun left on the verb, WordNet-mined pairs
  (5 of 5 rated mined items failed), and *provide/offer* pairs that change
  the claim ("providing measurements of" is not "measuring").

## Code

**`passive_v6`** (`ActiveVoice(version=6)`, `stage.OPS["passive_v6"]`).
v2–v5 outputs are unchanged (the new code paths need `version >= 6`; the
v3 tests and a replay check this). On top of every v5 rule and check:

- **Clause coordination in place (`agentive_conj`):** a passive that is the
  second of two coordinated clauses and has its own subject ("…, but his
  methods were questioned by other members") is no longer skipped; every v5
  rule and generic check (second parser, output parse, fluency, entailment)
  applies to it.
- **Coordination split (`agentive_split`):** "S was V-ed by A [...] and/but
  VP2." → "A V-ed S [...]. [But] S VP2." The subject must start the sentence
  (no fronted phrase scoping over both halves), be a pronoun or a definite or
  named phrase of at most 5 words with no possessive or "of", and have no
  negation before the verb; exactly one coordinated verb phrase, ending the
  sentence, joined by *and*/*but*; no brackets, quotes, colons or dashes.
  A second verb that only looks past-tense ("and erected in 1843" vs "and
  served until 1913") takes the shared helper only when both parsers tag it
  as a participle, or when the pipeline's BERT prefers one reading by at
  least 2 nats; otherwise no edit. Both halves must parse, under both
  parsers, as one clause with that subject. Each half is then flipped by the
  v6 in-place rules (every v5 rule and check plus the v6 skips); at least one
  half must flip, or there is no edit.
- **v6 skips** (from the rated round 6–7 failures and the dev check below):
  a participle clause after the agent ("whilst working", ", taking"); a later
  plural pronoun after a coordinated agent; a name containing "and" as the
  agent; light or reception verbs (*do, make, have, give, take, get, meet,
  greet*); an action or procedure noun as the agent (*-tion, -sion, -ment,
  -ance, -ence, -ing, -ure, -ysis* without a possessor); medical or technical
  means (*caesarean section, injection, route, device …*).
- Rated configuration: the 0.5 entailment floor (as v5).

**`nominal_v3`** (`Nominalization(version=3)`): v2 plus curated table pairs
only (no WordNet mining); light verbs *make, conduct, perform, carry out,
undertake, have (impact/effect), reach* only; no modifier of the noun left
behind; the dropped complement directly after the noun, not coordinated, not
a gerund; no comma-interrupted complement; no later pronoun; no complement
on the light verb; no negation or polarity determiner; verbs that need an
object ("recommend", "state", "pay" …) must get one; the second parser
(`en_core_web_md`) agrees on the light verb, its object and the complement's
attachment; the rewrite parses with the same subject and the complement's
object as the new verb's object; the BERT fluency drop is at least −16 nats;
and the 0.5 entailment floor.

## Dev check (old pools, never rated; one dev rater)

`passive_v6` and `nominal_v3` (first versions) over never-rated old-pool
sentences (1,831 news/wiki/openFDA sentences with "by"; 3,862 sentences with
a light verb and a table noun): 87 new passive outputs (v5 changed 7 of those
sentences) and 33 nominal outputs. One Opus dev rater (brief and wrapper
verbatim; cards in `dev/devrate8/` of the state directory): split 67/74
(90.5%), in-clause 13/13, nominal 32/33. The failures (a wrongly shared "was"
for "served/started/responded", "caesarean section" as the doer, repeated
possessive subjects, "affected hindering adoption") gave the shared-helper
rule, the means list, the subject-length rule and the gerund-complement rule
above. These dev numbers are in-sample for the final rules and are not
evidence for the bar.

## Fresh windows (unread)

- **Pool H** (`structure_v2_fetch.py --round 8`, `download8`): Wikinews and
  Good-article titles not in any earlier window (400 and 800 documents, more
  than before so that more items come from outside the Federal Register);
  Federal Register significant rules 2019-01-01..06-30 (20 documents);
  openFDA prescription labels effective in 2017 (65 documents left after the
  earlier-label and generic exclusions).
- **Pool I** (`--round 10`, `download10`): Federal Register significant rules
  2018-01-01..06-30 only (85 documents), for nominalization items.
- Pool builder `pool-c --name H|I`: exclusions as in round 7 plus pools G
  and G-wide (and H for I); the wide pools now keep every clean sentence of
  every fresh document. `prefilter` then drops wide-pool sentences that
  none of the rated ops can change (no "by"; no light verb before a table
  noun), so the wide apply fits the queue. Edits per 100 sentences come
  from the unfiltered standard pools H + I. Pool sizes and hashes are in
  [r8-precommit.json](r8-precommit.json). The pool files were built before
  this file; no sentence has been read or rewritten. The locked test split
  is never read. `download9` (round 9 window) is fetched, unread, for a
  possible confirmation round.

## Arms, panel and rule

- `passive_v6` and `nominal_v3`, each with the 0.5 entailment floor, and
  `passive_v5` with the floor (as rated in rounds 6–7) as a same-window
  reference, on the standard pools (`r8_apply.sh`) and the prefiltered wide
  pools (`r8_apply_wide.sh`).
- Panel (`r8_panel.sh`): items from the wide apply file, caps `passive_v6`
  100, `nominal_v3` 60, `passive_v5` 50, round robin over sources, seed
  20261122. Identical before/after pairs share one card, so raters cannot
  tell the arms apart.
- Three new blind raters (Opus subagents),
  [RATER_BRIEF.md](../20261122-structure-v2/RATER_BRIEF.md) and
  [RATER_PROMPT.md](../20261122-structure-v2/RATER_PROMPT.md) verbatim, each
  with its own shuffle and only its own cards; majority mark, the most severe
  on a three-way split.
- **Pass (each new op, round 8 alone):** n ≥ 40, majority Good ≥ 90%, and
  majority Changes meaning ≤ 8%. An op with n < 40 is inconclusive (n ≥ 20
  is still reported). Reported: Wilson 95% intervals, every mark, per-kind
  (`agentive`, `agentive_conj`, `agentive_split`) and per-source results,
  Good outside the Federal Register, per-rater rates, agreement, edits per
  100 standard-pool sentences.
- `passive_v6`'s recall claim is the per-100 rate against `passive_v5` on the
  same standard pools, and its non-Federal-Register Good rate is reported
  next to v5's rounds 6–7 figure (44/49 = 89.8%).
- A passing op joins `KEPT_V4 = KEPT_V3 + (passing ops)`; `KEPT_V3` is not
  changed. Then a descriptive dev check on top of F2 (DEV_V4.md, written
  before those outputs).
- Nothing ships; F2 stays the default.
