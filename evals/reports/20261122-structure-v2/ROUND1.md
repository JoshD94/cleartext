# Round 1 protocol (written November 22, 2026, before any round-1 output)

Joshua's ask: improve the sentence-structure stage beyond splitting,
starting with passive to active. This round measures each new operation
on its own, on fresh public text, with blind ratings. No LLM is used in any
operation. Code: `models/src/cleartext/structure_v2/`; harness:
`evals/structure_v2.py`. Sources and thresholds are hashed in
`r1-precommit.json`, committed with this file.

## What had been read before this file

- The November 10 ensemble study (Chhoeng's converter: `placeholder="auto"`
  fails the meaning gate; agentive-only passes with a tiny effect).
- F2 dev outputs (100 passages) and a build pool (1,020 sentences from
  documents hashed into buckets 0–2) were read to write the rules.
  Their rule fixes are in the code before this file. No round-A or round-B
  sentence has been read or rewritten.

## Operations (arms)

| Arm | What it does |
|---|---|
| `passive` | passive with a by-phrase → active (my converter: token-faithful realization, skip rules) |
| `chhoeng_agentive` | comparator: Chhoeng's converter, `placeholder=None`, unchanged |
| `chhoeng_auto` | reference: Chhoeng's default (`placeholder="auto"`), unchanged |
| `passive_recovered` | agentless action passive → "we" when the 3 preceding sentences use "we" as an active subject, for an author-action verb list, main clause only |
| `appositive` | long appositive or non-restrictive who/which clause → its own sentence, anchor repeated |
| `fronted_split` | fronted because/although/though/after clause → own sentence + "So"/"But"/"Then" |
| `fronted_move` | fronted if/when/unless/once/until/before clause → after the main clause |
| `frames` | "there is/are X who/that …" and subject clefts "it is X that …" → plain clause |
| `nominal` | light verb + nominalization → verb ("conduct an evaluation of" → "evaluate") |

Every arm applies the F2 hard checks (numbers, negation, entities,
technical phrases, code literals) to each edit and drops failing edits.
Round 1 uses **no NLI floor** (hard checks only); the dev run reports NLI.
Double negation: Kea-Roy's existing rule (`negation_rewrite`) is reported
for coverage only; it is not re-implemented.

## Fresh window

- **Pool A:** documents hashed into buckets 3–6, excluding every document
  in the word-stage panels and test set v3, and every sentence sharing a word
  8-gram with them. Sources: Wikinews, Wikipedia (Featured leads, word-stage
  corpus; Good-article leads, downloaded now), Federal Register (word-stage
  corpus; 2023-07..12 significant rules, downloaded now), openFDA, RFCs.
  Sentences of 8–45 words, at most 40 per document, at most 1,500 per source.
  `chhoeng_auto` (his default "someone"/"people" placeholder) is rated as a
  reference arm with up to 30 items.
- Each arm runs on every pool-A sentence (raw sentence, with its three
  preceding sentences as context; raters see the same context). Operations are rated on raw sentences,
  not on F2 output; most F2 sentences are unsplit, so this is the same input
  class.
- **Panel:** up to 40 changed items per arm, round robin over sources, seed
  20261122. Identical before/after pairs from two arms share one card.
- **Rare arms:** an arm with fewer than 40 changed items in pool A draws its
  remaining items from **pool A-wide** (every clean sentence of the same
  pool-A documents). Its edits per 100 sentences always use pool A.

## Rating

- Three blind raters: separate general-purpose Opus subagents, each with
  [RATER_BRIEF.md](RATER_BRIEF.md) verbatim and its own shuffle. They see
  context, before and after; never the arm.
- Marks: Good / Not simpler / Changes meaning / Ungrammatical.
- **Primary mark:** the majority of three; when all three differ, the most
  severe (meaning > ungrammatical > not simpler).

## Decision rule (per arm, majority marks)

- **Kept:** at least 20 rated items, Good ≥ 85%, and Changes meaning ≤ 10%.
- Reported for every arm: Good rate with Wilson 95% interval, every mark,
  per-rater Good rates, agreement, edits per 100 pool sentences, per-rule
  and per-source rates, top failure types.
- `passive` against `chhoeng_agentive`: Good rates and coverage on the same
  pool; the arms are rated on the same cards where their outputs coincide.

## After round 1

- Kept arms form the combined "structure v2" stage, in a fixed order
  (passive, passive_recovered, nominal, frames, appositive, fronted_split,
  fronted_move), applied after F2 on the test set v3 dev split.
- **Dev success (combined stage against F2, fixed now):**
  - all four selection gates and the over-edit gate pass;
  - Δ passage NLI 95% CI lower end ≥ −0.005 (the v3 "no worse in meaning"
    bound);
  - ΔFK ≤ 0 (point estimate).
- Round 2 (if run) gets its own protocol, written before its outputs: a
  learned gate or rule fixes for arms that missed, tested on pool B.
- Nothing ships here; F2 stays the default. Adoption is Joshua's decision.
