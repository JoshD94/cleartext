# Round 6 protocol (structure v3; written November 23, 2026, before any round-6 output)

Joshua's follow-up: push sentence-level edits further, mainly ordinary
by-phrase passive → active, to **≥ 90% Good and ≤ 8% Changes meaning**
(majority of three blind raters) on fresh text with usable coverage. No LLM
in the pipeline. Round 5 left `passive_v4` at 78.9% Good [70–86] with 12.8%
meaning changes over rounds 4 + 5 (109 items).

## Diagnosis (rated v4 failures, rounds 4–5: 23 of 109)

| Failure class | Items | Examples |
|---|---:|---|
| Trailing participle or depictive re-attaches to the new subject | 6 | "…, becoming the most watched show", "…, taking third place", "…, compiling a 4–14 record", "…, blinded to treatment arm", "…, pinned to the floor" |
| Parse error around the by-phrase (agent span or attachment) | 6 | "Government has not killed civilians shelling", "Merryday of the hearing …", "Leisure Suit Larry … in the Land of the Lounge Lizards", "the Air Ministry for offices", "the investigator based on …" |
| Means, process or time as the agent | 6 | "by gravity", "by strangulation", "by both diffusion and active transport", "by the game's release", "by the use of metal sheets …", "by excitation and stimulation" |
| Relation verbs, not actions | 3 | "a test dose … should precede repeat doses" (twice), "manifested by" |
| Focus word over the agent | 2 | "only by certified risk assessors", "originally primarily used … by" |
| Tense | 1 | "has been appointed … since 2024 by" |

Rounds 1–3 failures fall into the same classes (plus run-in headings and
capitalized drug names, which v3–v4 already skip). The rule fixes of rounds
2–5 were blacklists: each fixed the failures it saw, and each fresh window
showed new ones. Version 5 instead whitelists the clause shape and adds two
generic checks that do not depend on any one failure.

## Code: `passive_v5` (`ActiveVoice(version=5, kinds=("agentive",))`)

On top of v4 (which still runs first), a by-phrase passive is converted only
if every condition holds (`passive.v5_reasons`):

- **Agent is a plain noun phrase:** at most 8 words; only determiners,
  adjectives, compounds, possessives, numbers, coordination and "of" phrases
  inside (no relative or participle clause, no other preposition, no "of"
  phrase on a name, no punctuation but hyphens); no focus word over it
  (*only, both, mainly, primarily, partly, even, also, alone …*); its head is
  not a means, process, rule or time noun (a list plus the first WordNet
  sense: process, time, quantity, shape).
- **Verb:** not a relation verb (*precede, follow, accompany, manifest,
  mediate, relate, characterize, distinguish, mark, represent, reflect,
  indicate, base, determine, define, measure …*); no perfect with "since";
  no adverbs on both sides of the participle.
- **Clause after the agent:** empty, or starting with a preposition,
  subordinator or adverb of the verb; no participle, adjective or non-finite
  clause after a comma; nothing after the agent attaching to it; no bracket
  or dash; no capitalized phrase after a name (titles). No non-finite
  adverbial clause, complement, object, conjunct or predicative on the verb
  at all; the clause is not inside brackets.
- **Old subject:** no bracket; no wh-word or complementizer; not right
  after a preposition, nor right after a verb when it is coordinated or
  starts with "that" (misparses); a sentence-initial capitalized word that
  would stay capitalized must be a named entity of the same type under both
  parsers and must not look like a drug name.
- **Other skips** from the dev checks below: relation verbs also include
  *derive, intend, mean*; process nouns unknown to WordNet (*-ation, -sis,
  -ment*); location nouns; a negated passive with a bare evidence noun ("is
  not supported by explanation"); an infinitive after the verb ("in order to
  be published"); a modifier after the agent's head other than an "of"
  phrase or a conjunct; an "of" phrase with *notify/inform/advise …*;
  *rather than / instead of / as well as* in the agent; quantifiers in both
  phrases; a pronoun object after an agent of more than 6 words; a
  hyphenated word split at the agent's end.
- **Second parser agrees** (generic): `en_core_web_md` 3.8.0, trained
  separately from the pipeline's `en_core_web_sm` 3.8.0, finds the same
  passive with the same subject span and agent span, and attaches every
  later word to the agent (or not) the same way.
- **Output parses as intended** (generic): under either parser, the
  rewrite has a verb of the same lemma whose subject is the new subject and
  whose object starts with the old subject.
- **Entailment floor** (generic): Chhoeng's NLI scorer (the F2 split
  floor, `cross-encoder/nli-deberta-v3-small`, minimum of both directions)
  must reach 0.5; Good dev rewrites score 0.98 or more at the 10th
  percentile, so the floor only removes garbled output.
- **Fluency floor** (generic): the pipeline's own `bert-base-uncased`
  (`cleartext.contextual`) pseudo-log-likelihood of the rewrite minus that
  of the source (`structure_v2/fluency.py`) must be at least −16 nats.
- **Agreement and realization:** codes and acronyms take singular verbs
  ("CYP2C9 metabolizes", "OSHA cites"); "a number of studies" is plural;
  "cannot" stays one word; a sentence-initial word before an identifier
  keeps its capital.
- The F2 hard checks (numbers, negation, entities, technical phrases, code
  literals) still apply to every edit.

## `appositive_v4` (`AppositiveSplit(version=4)`)

Version 3 plus a whitelist: the anchor is a name or a definite noun phrase
(no *one, another, some, only*, numbers, bare common nouns; "a/an" anchors
are repeated with "the"), it has no bracket or hyphen and cannot start
earlier (misparse); a relative clause is not progressive ("who was
accompanying him"); and `en_core_web_md` labels the same span as the same
appositive or relative clause of the same anchor. Same entailment floor.
Replay on rated round 1–3 items (in-sample): keeps 30 of the 84 Good items
and none of the 30 failures (version 3 reproduces 40 Good and 10 failures).
On old-pool sentences (dev check 2 below), one dev rater marked all 12 of its
outputs Good.

## What had been read before this file

- Every round 1–5 card, mark and failure.
- **Dev check 1** (old pools, never rated): the first v5 over the unrated
  sentences of the standard pools build/A–E and the wide pools C–E with "by"
  (9,431 sentences) changed 158; one Opus dev rater (brief verbatim) marked
  143 Good (90.5%), 5 meaning, 9 ungrammatical, 1 not simpler; news/wiki
  26/26, Federal Register 94/105, openFDA 23/27. A relaxed variant (agents
  of up to 12 words with prepositional phrases) scored 154/172 (89.5%) and
  was dropped.
- **Dev check 2** (old wide pools build/A/B, never rated), after fixing the
  dev-1 failures: 82 changes, mostly Federal Register; a second dev rater
  marked 65 Good (79.3%), 7 meaning, 8 ungrammatical, 2 not simpler. Its
  failures gave the last group of rules above (scope words in the verb
  group, criterion verbs, later pronouns, footnote markers, adjectival
  participles, lemmatizer disagreement).
- **In-sample replay of the final code** over the 446 labelled items (206
  rated v4 outputs and both dev checks): 284 kept, 281 Good. This is
  optimistic; dev check 2 shows how far a fresh window can fall.
- **Fluency floor:** `bert-base-uncased` pseudo-log-likelihood was computed
  for 395 labelled dev items (rewrites of earlier versions included). The
  drop from source to rewrite separates Good from not Good with AUC 0.68;
  of the 15 rewrites below −16, 13 were not Good (garbled parses such as
  "the national UK teachers will … close over 1200 schools strike", "Provider
  Z. owns them"). The floor is set there; it drops about 2 of the 281 Good
  items the rules keep.
- No pool-F sentence has been read or rewritten.

## Fresh window (pool F)

`structure_v2_fetch.py --round 6`: Wikinews and Good-article titles not in
any earlier window; Federal Register significant rules 2020-01-01..06-30;
openFDA prescription labels effective in 2019, excluding every earlier label
document and every test set v3 generic. Pool builder `pool-c --name F`
(exclusions: word-stage panels, test set v3 `panel/all.json`, pools A–E and
their wide pools, 8-gram overlap): 5,597 standard and 29,847 wide sentences.
The pool files were built before this file; no sentence was read. The locked
test split is never read.

## Arms, panel and rule

- `passive_v5` and `appositive_v4` with the 0.5 entailment floor, and
  `passive_v4` exactly as rated in rounds 4–5 (no floor) as a same-window
  reference. All three run on pool F (edits per 100 sentences); the two new
  arms also run on pool F-wide (`r6_apply.sh`, `r6_apply_wide.sh`).
- Panel: `passive_v5` and `appositive_v4` items from the F-wide apply file
  (cap 80 and 60), `passive_v4` items from the standard pool F (cap 60),
  round robin over sources, seed 20261122. Identical before/after pairs
  share one card, so raters cannot tell the arms apart.
- Three new blind raters (general-purpose Opus subagents),
  [RATER_BRIEF.md](RATER_BRIEF.md) and [RATER_PROMPT.md](RATER_PROMPT.md)
  verbatim, each with its own shuffle; majority mark, the most severe on a
  three-way split.
- **Pass (each new arm, round 6 alone):** n ≥ 40, majority Good ≥ 90%,
  majority Changes meaning ≤ 8%. Reported: Wilson 95% intervals, every
  mark, per-source results, per-rater rates, agreement, edits per 100 pool-F
  sentences. An arm with n < 40 is inconclusive.
- If `passive_v5` misses, a round 7 on a new window (download 7, fetched
  now, unread) may test a revised version under its own protocol, written
  before its outputs.
- Nothing ships; F2 stays the default.
