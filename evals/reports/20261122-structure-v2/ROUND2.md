# Round 2 protocol (written November 22, 2026, before any round-2 output)

Round 1 ([round1.json](round1.json)) kept two operations under its rule:
`fronted_split` (95% Good) and `nominal` (90%). The others missed:
`passive` 72%, `passive_recovered` 77% (22 items), `appositive` 75%,
`frames` 43% (14 items), `fronted_move` 0% (33 of 40 Not simpler).
Chhoeng's converter: agentive 42%, `placeholder="auto"` 3%.

## What had been read before this file

- Every round-1 card, mark and failure (pool A). The round-2 rules below were
  written from those failures, so a replay of round-1 items through them is
  optimistic and is reported only as development evidence:
  passive 25/26 Good of the items it still converts; nominal 35/35;
  fronted_split 38/38; appositive_v2 23 converted, 0 of the 10 bad ones.
- The synthetic round trip (below) on development sentences (documents in
  buckets 0–2). No pool-B sentence has been read or rewritten.

## Code changes since round 1 (round-1 code: commit `537491b`)

- **passive** (all from round-1 failures or the synthetic round trip):
  no relative-pronoun subjects (they become harder object relatives);
  no pronoun or possessive inside the agent when the subject is a noun
  phrase (cataphora: "her guards assassinated his grandmother"); no
  "any/no/either/neither" agent; no second by-phrase inside the agent; no
  noun, number or adverb right after the agent (misparse, floating "alike");
  run-in headings skipped; pronoun case changed only on phrase heads
  ("the research that he would publish" keeps "he"); one-word sentence-
  initial "entities" that are dictionary words are lowercased; the final
  period is kept after an abbreviation; a clause or comma before the
  subject head, or an "as/like" phrase inside the agent, is skipped.
- **passive_recovered**: author-verb list narrowed (no find, determine,
  establish, develop, use, provide, include, apply, perform, conduct,
  implement, issue, set, design, build, construct, prepare, consider, seek,
  request); subjects over 10 tokens skipped.
- **appositive_v2**: lists, anchors with their own PP or inside an adverbial
  clause, pobj anchors of a noun, finite verbs outside a relative clause,
  number mismatch between a relative's verb and its anchor, and "which"
  that points at a clause (anchor with its own clause, inside a gerund
  phrase, "which in turn") are skipped; definitions use "is/are" unless
  the appositive says "former/late/then".
- **frames_v2**: existential "there" only (round-1 "it is X that" items were
  referential "it"); no adverb on "be"; reduced relatives only with a past
  participle; no noun phrase starting with a digit.
- **nominal**: no "decide about"; no intransitive "change"; "have an impact
  on X of Y" skipped.
- **fronted_split**: no "but" in the main clause; no "it/they/this/these"
  subject in the fronted clause when the main subject is a noun phrase.
- `fronted_move` is dropped (Not simpler), and so is Chhoeng's "auto" arm.

## Learned gate (passive)

- **Labels:** synthetic active→passive pairs. Development sentences with a
  simple transitive main clause are passivized by rule
  (`structure_v2/gate.py: passivize`); the round-2 converter is run on each
  passive; label 1 when it returns the original sentence exactly.
  1,589 converted pairs, 84.4% exact.
- **Model:** gradient boosting on the converter's 38 candidate features
  (`gate.vectorize`), 5-fold out-of-fold AUC 0.77. Threshold: the lowest
  out-of-fold score whose kept pairs are ≥ 92% exact (0.839; keeps 1,227 of
  1,589 at 92.0%). File `gate-synth-r2.pkl`, sha256 in `r2-precommit.json`.
- **Test:** the gate never writes text; `passive_gated` is the subset of
  rated `passive` cards whose gate score passes, so it needs no extra cards.

## Fresh window

- **Pool B:** documents hashed into buckets 7–9, the same exclusions and
  sources as pool A, built together with it (`pool-manifest.json`) and
  never read. Rare arms (fewer than 40 changed in pool B) draw extra items
  from pool B-wide; edits per 100 sentences always use pool B.
- **Arms and caps:** `passive` 60, `chhoeng_agentive` 30 (comparator on the
  same pool), `passive_recovered` 40, `appositive_v2` 40, `fronted_split` 40,
  `nominal` 40, `frames_v2` 40. Same seed rule, card sharing and hashed card
  IDs as round 1.
- **Rating:** three new blind Opus raters, [RATER_BRIEF.md](RATER_BRIEF.md)
  and [RATER_PROMPT.md](RATER_PROMPT.md) verbatim; same majority rule.

## Decision rule (unchanged from round 1)

- **Kept:** at least 20 rated items, majority Good ≥ 85%, Changes meaning
  ≤ 10%. Round 2 decides; round 1 is reported beside it.
- **Gate:** `passive_gated` replaces `passive` in the combined stage only if
  `passive` misses the rule and `passive_gated` meets it with ≥ 20 items.
- **Combined stage:** the kept operations, in the order passive,
  passive_recovered, nominal, frames_v2, appositive_v2, fronted_split,
  after F2 on the test set v3 dev split, judged by ROUND1.md's dev success
  rule (all gates and the over-edit gate; Δ passage NLI CI lower end ≥
  −0.005; ΔFK ≤ 0).
- Nothing ships; F2 stays the default.
