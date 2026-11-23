# Sentence structure beyond splitting (structure v2), November 22, 2026

Joshua asked how to improve the sentence-structure stage other than by
splitting, starting with passive to active. This study built seven
rule-based operations (no LLM), rated each one blind on fresh public text in
five pre-registered rounds, and ran the passing ones after F2 on the test set
v3 dev split. Raw text and marks stay in
`~/.local/state/cleartext/structure-20261122/`; this folder holds protocols,
precommits and aggregates.

## Answer

- **Two operations pass the bar** (majority Good ≥ 85%, Changes meaning ≤ 10%,
  n ≥ 20, on fresh text):
  - **`fronted_split`**: a fronted *because / although / though / after*
    clause becomes its own sentence, joined by "So" / "But" / "Then".
    92.9% Good [84–97] over rounds 2 + 3 (65/70; round 1: 38/40). It fires on
    about **0.4 sentences per 100**.
  - **`passive_recovered`**: an agentless passive becomes active with "we"
    when the three sentences before it speak as "we" and the verb is a
    document or method verb ("These comments are summarized in this
    section" → "We summarize these comments in this section"). 91.4% Good
    [83–96] (64/70). Almost all items are Federal Register rules: **0.05–0.07
    per 100** sentences of mixed text.
- **Passive with a by-phrase does not reach the bar.** My converter goes from
  72% Good (round 1) to 61%, 78% and 76% across four versions on fresh text;
  a fifth, confirmation round of the last version gave 82%, so v4 pools to
  **78.9% [70–86]** over 109 items with 12.8% meaning changes (81.9% outside
  drug labels). Chhoeng's converter scored 42% and 37% on the same pools,
  and 3% with his default "someone/people" placeholder. Each round fixed the failures it
  saw, and each fresh window showed new ones (parse errors around the
  by-phrase, trailing participles that re-attach to the new subject, means
  phrases like "by a slow infusion", drug-label headings).
- **The rest miss:** appositive extraction 71–75%, nominalization →
  verb 90% and 88% but 77% on the third window (pooled 82.9%: dropped),
  "there is … who" frames 77% on only 13 items, moving a fronted *if/when*
  clause after the main clause 0% (raters: not simpler).
- **A learned gate did not help.** Gradient boosting on 1,589 synthetic
  active → passive → active round trips (out-of-fold AUC 0.77) keeps the 77%
  of flips most likely to round-trip exactly, but on rated items the kept
  subset scored 63% Good against 61% for all flips. A gate trained on the
  blind marks themselves (146 rated v4 outputs, 18 failures) scored AUC 0.44
  on round 4 when fit on rounds 1–3 ([rated-gate.json](rated-gate.json)),
  so none was pre-registered.
- **On top of F2 (dev, 100 passages)** the two kept operations change 3
  passages: ΔFK −0.011 [−0.028, 0], Δ passage NLI +0.0002 [0, +0.0004];
  every gate and the over-edit gate pass, so the pre-registered dev rule is
  met. The effect on readability scores is small because the operations fire
  rarely.

## Per operation (blind majority marks, fresh text)

Good rate with Wilson 95% interval; edits per 100 sentences of the standard
pool of that round. Code versions differ by round (see the round protocols);
"r2+r3" pools the two rounds that ran identical code.

| Operation | Round 1 | Round 2 | Round 3 | Round 4 | Per 100 | Decision |
|---|---|---|---|---|---:|---|
| **Fronted clause split** | 38/40 = 95% [83–99] | 37/40 = 92% [80–97] | 28/30 = 93% [79–98] | – | 0.38–0.49 | **kept** (r2+r3 92.9% [84–97]) |
| **Passive → "we" (recovered agent)** | 17/22 = 77% [57–90] | 37/40 = 92% [80–97] | 27/30 = 90% [74–97] | – | 0.04–0.07 | **kept** (r2+r3 91.4% [83–96]) |
| Passive → active, by-phrase (mine) | 29/40 = 72% [57–84] | 35/57 = 61% [48–73] | v3 47/60 = 78% [66–87] | v4 37/49 = 76% [62–85]; round 5: 49/60 = 82% [70–89] | 0.9–1.8 | not kept (v4 rounds 4+5: 86/109 = 78.9% [70–86]) |
| … with the synthetic gate | – | 12/19 = 63% [41–81] | – | – | 0.49 | not kept |
| Chhoeng's converter, by-phrase only | 17/40 = 42% [29–58] | 11/30 = 37% [22–54] | – | – | 3.9–4.1 | comparator |
| Chhoeng's converter, default (someone/people) | 1/30 = 3% [1–17] | – | – | – | 26.9 | comparator |
| Nominalization → verb | 36/40 = 90% [77–96] | 35/40 = 88% [74–95] | 23/30 = 77% [59–88] | v2 12/15 = 80% [55–93]; round 5: 16/24 = 67% [47–82] | 0.05–0.18 | dropped (r2+r3 82.9% [72–90]; v2 rounds 4+5 71.8% [56–83]) |
| Appositive / who-which → own sentence | 30/40 = 75% [60–86] | v2 30/40 = 75% [60–86] | v3 24/34 = 71% [54–83] | – | 0.3–1.7 | not kept |
| "There is … who" / "it is … that" | 6/14 = 43% [21–67] | v2 5/7 | v2 5/6 | – | 0.0–0.1 | inconclusive (13 items) |
| Fronted *if/when* clause moved after | 0/40 = 0% [0–9] (33 not simpler) | – | – | – | 0.32 | dropped |
| Kea-Roy's double negation (existing rule) | 1 edit in 20,316 sentences | – | – | – | 0.02 | coverage only |

Raters: three separate general-purpose Opus subagents per round (fifteen in
all), [RATER_BRIEF.md](RATER_BRIEF.md) and [RATER_PROMPT.md](RATER_PROMPT.md)
verbatim, each with its own shuffle and no arm names. All three agreed on
82%, 94%, 91%, 92% and 92% of cards in rounds 1–5. These are model raters, not
people: their agreement is high and they may share blind spots.

### Top failure types (from the rated failures)

- **Fronted split** (5 of 70 failed): mostly *after* clauses: "Then" with a
  perfect or future tense ("X has been established. Then Y may be
  given"), and one "compared to" phrase cut off.
- **"We" recovery** (6 of 70): a doer the context does not support ("Two
  methods ... were tested" by contractors, not the agency), "we all
  explain", and a news article whose "we" came from quoted speech.
- **By-phrase passive** (round 4, 12 of 49): a trailing participle now
  modifies the new subject ("19.9 million viewers watched the episode,
  becoming the most watched show"), "only" dropped from "only by certified
  assessors", temporal or means "by" ("by the game's release", "by
  gravity"), misparsed agents ("government ... shelling"). Earlier rounds:
  drug names left capitalized mid-sentence, run-in label headings, object
  relatives, cataphoric pronouns ("her guards assassinated his
  grandmother"). By source, rounds 2–4 were 78–86% Good outside drug labels
  and 25–63% on openFDA labels.
- **Appositives**: sentence-final material describing another noun or the
  whole event ("…Monday, the 2nd attack in two days"), tense of the added
  "is/was" sentence, team-name agreement.
- **Nominalization**: verbs left with the noun's preposition ("suggested
  on", "decided regarding", "affect … or on the private sector").

### Examples (rounds 2–3, majority marks)

Fronted split, good:

- "Although this metabolite has similar EGFR-TK activity to gefitinib in
  the isolated enzyme assay, it had only 1/14 of the potency of gefitinib in
  one of the cell-based assays." → "This metabolite has similar EGFR-TK
  activity to gefitinib in the isolated enzyme assay. But it had only 1/14
  of the potency of gefitinib in one of the cell-based assays."
- "Because delinquent and defaulted borrowers were not enrolling in the IDR
  plans at the rate we expected, the Department considered changes to
  REPAYE …" → "Delinquent and defaulted borrowers were not enrolling in the
  IDR plans at the rate we expected. So the Department considered changes
  to REPAYE …"
- "Although this population receives services within their own homes, they
  can transfer the virus to their caretakers …" → "This population receives
  services within their own homes. But they can transfer the virus to their
  caretakers …"

Fronted split, failed:

- (meaning) "After the diagnosis of hyperaldosteronism has been established
  by more definitive testing procedures, spironolactone tablets may be
  administered …" → "The diagnosis … has been established … Then
  spironolactone tablets may be administered …"
- (meaning) "After all the items have been scanned, Watkins plans to donate
  them …" → "All the items have been scanned. Then Watkins plans to donate
  them …"
- (ungrammatical) "After Condé Nast and Skadden Arps moved out of the
  building during the 2010s, a variety of office tenants have occupied 4
  Times Square." → "… during the 2010s. Then a variety of office tenants
  have occupied 4 Times Square."

"We" recovery, good:

- "The charge inflation and CCR inflation factors are discussed in detail
  in the FY 2021 IPPS/LTCH PPS final rule (84 FR 45542)." → "We discuss the
  charge inflation and CCR inflation factors in detail in …"
- "These comments and our responses are summarized in this section of the
  rule." → "We summarize these comments and our responses in this section
  of the rule."
- "No comments were received on the proposed approach." → "We received no
  comments on the proposed approach."

"We" recovery, failed:

- (meaning) "Two methods of imputing missing data were tested: …" → "We
  tested two methods …"
- (meaning) "These items are all explained in greater detail in the Net
  Budget Impact section of this RIA." → "We all explain these items …"
- (meaning) "The mascot was introduced in 1907, brought in as a costume
  character in 1921, …" → "We introduced the mascot in 1907, …"

Example sentences are public text: Federal Register (public domain), openFDA
labels (public domain), Wikinews (CC BY 2.5), Wikipedia (CC BY-SA 4.0).

## On top of F2 (test set v3 dev split)

Each arm is applied to F2's cached dev outputs (100 passages); unchanged
passages reuse F2's cached metrics, which this harness reproduced exactly on
87 of 87 rescored passages. Paired deltas against F2 with the
domain-stratified bootstrap ([dev-judge.json](dev-judge.json)).

| System (after F2) | Passages changed | FK | ΔFK [95% CI] | Passage NLI | ΔNLI [95% CI] | Gates |
|---|---:|---:|---:|---:|---:|---|
| F2 | 0 | 12.851 | – | 0.9746 | – | pass |
| **Kept stage** (`passive_recovered` + `fronted_split`) | 3 | 12.840 | −0.011 [−0.028, 0.000] | 0.9748 | +0.0002 [0.0000, +0.0004] | **pass** |
| Kept stage + by-phrase passive v4 | 7 | 12.839 | −0.011 [−0.028, +0.000] | 0.9755 | +0.0009 [+0.0001, +0.0022] | pass |
| All round-3 versions combined | 10 | 12.791 | −0.059 [−0.126, −0.011] | 0.9756 | +0.0010 [+0.0002, +0.0023] | pass |
| Chhoeng's converter, by-phrase only | 12 | 12.837 | −0.013 [−0.023, −0.005] | 0.9729 | −0.0017 [−0.0062, +0.0017] | pass (misses the −0.005 NLI bound) |

On the 3 passages the kept stage changes, FK falls from 12.87 to 12.51 and
passage NLI rises from 0.985 to 0.991. NLI is an automatic signal, not a
percentage of correct outputs.

## How it was run

- **Code.** `models/src/cleartext/structure_v2/` (operations,
  `StructureV2` stage, `KEPT`, synthetic-pair gate), documented in
  [models/docs/STRUCTURE_V2.md](../../../models/docs/STRUCTURE_V2.md);
  harness `evals/structure_v2.py`; fetcher `evals/structure_v2_fetch.py`;
  tests `models/tests/test_structure_v2.py`, `evals/test_structure_v2.py`.
  Every edit must pass F2's hard checks (numbers, negation, entities,
  technical phrases, code literals). Chhoeng's `passive/` code is
  unchanged and ran through `teammate_stages.PassiveStage`.
- **Pre-registration.** Each round's protocol and precommit (source and
  input hashes, thresholds) were committed before that round produced any
  output: [ROUND1.md](ROUND1.md) `537491b`, [ROUND2.md](ROUND2.md)
  `53c5f12`, [ROUND3.md](ROUND3.md) `a66af31`, [ROUND4.md](ROUND4.md)
  `e9e6f24`, [ROUND5.md](ROUND5.md) `9a697b5` (confirmation, no code
  change). Results: [round1.json](round1.json) … [round5.json](round5.json).
- **Fresh text.** Five disjoint windows of 4,713, 3,870, 5,630, 5,492 and
  5,544 sentences (plus "wide" pools of the same documents for rare operations),
  from Wikinews, Wikipedia Featured and Good-article leads, Federal
  Register significant rules (2020-07..12, 2021-01..06, 2021-07..12,
  2023-07..12 and the word stage's unused 2022–23 documents), openFDA labels
  (2020, 2021, 2023 and the word stage's unused 2022 labels) and RFCs. Every document used by the word
  stage's panels or test set v3, and every sentence sharing a word 8-gram
  with them or with an earlier window, was excluded. The locked test split
  was never read.
- **Rounds.** Each round's rule fixes came from the previous round's rated
  failures (and, for passives, a synthetic round trip), so replaying old
  items through new code is optimistic: passive v4 replayed at 90–96% Good
  and scored 76% on fresh text.

## Disclosures

- Round 1's panel was regenerated once before any rater saw it, because the
  first card IDs followed arm order; cards and items were otherwise
  identical.
- Dev outputs for the round-1 single operations were made before the
  round-1 commit's last edits, so dev rows are reported only for round 2+
  code and Chhoeng's unchanged converter.
- Operations ran on raw sentences in the rating rounds, and on F2's output
  in the dev run; F2 leaves most sentences unsplit.
- "We" recovery and nominalization items come mostly from Federal Register
  rules; their rates say little about news or encyclopedic text.
- The new modules under `models/src/cleartext/` change
  `ensemble_study.generation_sources()`, so a state frozen before this branch
  sees them as new sources.

## Reproduce

`source ~/.local/state/cleartext/structure-20261122/env.sh`, then per round:
`python -m evals.structure_v2 pool|pool-c …`, `apply`, `panel`, rate,
`tally --rates-from …`; dev: `dev`, `score` (queue), `judge`. Queue jobs
(Yoga, personal, `later`): dev scoring `9739ec70322d`, `e7320da1bcb6`,
`adac92bf2a33`, `f9e6b404c759`, `6c55eed1c673`; `scripts/verify` `014d07d2bb0c` and, on the final commit, `bc290ce9e860` and after round 5 `e56e6a45b9ac` (687 passed each).
