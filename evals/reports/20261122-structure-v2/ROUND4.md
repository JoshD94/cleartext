# Round 4 protocol (written November 22, 2026, before any round-4 output)

Round 3 ([round3.json](round3.json)): `passive_v3` 78% Good (n = 60; round
2's `passive` 61%), `appositive_v3` 71% (34), `frames_v2` 5 of 6;
confirmations `fronted_split` 93%, `passive_recovered` 90%, `nominal` 77%.
Under ROUND3.md, `nominal` is dropped (round 2 + 3 pooled 58/70 = 82.9%)
and `passive_recovered` (64/70 = 91.4%) and `fronted_split` (65/70 =
92.9%) stay kept. `frames_v2` stays inconclusive (13 items pooled).

This is the last round. It tests two new versions written from the
round-3 failures; nothing else changes.

## What had been read before this file

- Every card, mark and failure of rounds 1–3. Replays through the round-4
  code (optimistic, development only): `passive_v4` converts 48 round-3
  items at 96% Good and 31 round-2 items at 90%; `nominal_v2` converts 19
  round-3 items at 95% and 25 round-2 items at 96%.
- The kept operations' outputs on every round-3 item are unchanged.
- No pool-D sentence has been read or rewritten.

## Code changes since round 3 (round-3 code: commit `a66af31`)

- **passive_v4** (`ActiveVoice(version=4)`, on top of v3): skips a
  title-case run of three or more capitals in a sentence-initial subject
  (drug-label headings, titles), a capitalized sentence-initial word with
  no WordNet sense, a by-phrase inside the subject, more than one "by" in
  the sentence, a colon or semicolon after the verb, means/procedure agents
  ("by a 420-3 vote", "by a slow infusion", "by the Cockcroft-Gault
  formula"), a modifier before the subject outside its phrase, and a
  relative clause hung on the verb; puts a pronoun object before its
  particle ("laid him off"); plural agreement for plural acronyms
  ("LSLs"); keeps the capital of a title-case subject ("Wings for My
  Flight").
- **nominal_v2** (`Nominalization(version=2)`): skips any preposition on
  the light verb or noun that the table does not list ("make a statement
  regarding", "offered suggestions in response"), a complement or adverb
  on the light verb ("make payments ... more consistent"), a "to/regarding"
  phrase left behind, and plural "impacts/effects".

## Fresh window (pool D)

- `structure_v2_fetch.py --round 4`: Wikinews and Good-article titles not
  in any earlier window; Federal Register significant rules 2021-01..06;
  openFDA prescription labels effective in 2021, excluding every earlier
  label document and every test set v3 generic. Pool rules as in round 3
  (`pool-c` builder, exclusions extended with pools A–C).

## Arms, caps and rule

- `passive_v4` 60 items, `nominal_v2` 40 (pool D-wide if rare). Same three
  fresh blind raters, brief and wrapper verbatim, majority rule.
- **Kept:** on round 4 alone, n ≥ 20, majority Good ≥ 85%, Changes
  meaning ≤ 10%.
- **Combined stage:** the final kept set in the order passive_v4,
  passive_recovered, nominal_v2, fronted_split, after F2 on the test set v3
  dev split, judged by ROUND1.md's dev success rule.
- Nothing ships; F2 stays the default.
