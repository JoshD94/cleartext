# Round 3 protocol (written November 22, 2026, before any round-3 output)

Round 2 ([round2.json](round2.json)) kept `passive_recovered` (93% Good,
n = 40), `fronted_split` (93%) and `nominal` (88%). It did not keep the
agentive `passive` (61%; Chhoeng's agentive 37%), `appositive_v2` (75%,
17.5% Changes meaning), `frames_v2` (7 items) or the synthetic gate
(`passive_gated` 63% on 19 cards: the round-trip gate does not predict
human Good).

## What had been read before this file

- Every round-1 and round-2 card, mark and failure. By source, round-2
  `passive` was 25% Good on drug labels (openFDA, 20 items) and 81% on the
  rest (37 items). The round-3 rules were written from the round-2
  failures; a replay of round-2 items is optimistic and reported only as
  development evidence: `passive_v3` converts 34 of them at 91% Good;
  `appositive_v3` converts 7 at 100%.
- The kept operations' outputs on every round-2 item are byte-identical
  under the round-3 code (checked before this file).
- No pool-C sentence has been read or rewritten.

## Code changes since round 2 (round-2 code: commit `53c5f12`)

New versions only; the kept operations are unchanged.

- **passive_v3** (`ActiveVoice(version=3)`): skips drug-label subheadings
  and run-in headings (lower frequency bar), demonstrative subjects,
  commas or wh-words in the subject, a capitalized sentence-initial name of
  unknown type ("Lidocaine HCl is metabolized ..."), measure phrases as
  agents ("by about 60%"), anything after the agent that depends on it, a
  coordination after the agent, and bare "-age" count nouns; uses plural
  agreement for "both ..." and quantifier agents ("some have criticised").
- **appositive_v3**: only material between the main subject and its verb
  ("X, a Y, did Z"); sentence-final appositives and relatives are dropped;
  a PERSON anchor in a past-tense sentence gets "was"; "The ...s" proper
  names take "are".

## Fresh window (pool C)

- Downloaded now into its own directory (`structure_v2_fetch.py --round 3`):
  Wikinews and Wikipedia Good-article titles not used by the word stage or
  rounds 1–2; Federal Register significant rules 2021-07..12 (the word
  stage used 2022-01..2023-06, test set v3 2024-25, round 1–2 2023-07..12);
  openFDA prescription labels effective in 2023, one per generic name,
  excluding every generic in test set v3 and every earlier label document.
- Same pool rules as rounds 1–2 (8–45 words, at most 40 per document, at
  most 1,500 per source, 8-gram exclusion against every earlier panel and
  test set v3). Every bucket of the new documents is fresh, so pool C uses
  all of them; rare arms draw extra items from pool C-wide.

## Arms and caps

| Arm | Cap | Role |
|---|---:|---|
| `passive_v3` | 60 | decides on its own (round-3 code) |
| `appositive_v3` | 40 | decides on its own |
| `frames_v2` | 40 | same code as round 2; judged on round 2 + 3 pooled |
| `passive_recovered`, `nominal`, `fronted_split` | 30 each | confirmation of round 2, same code |

Same three-rater blind procedure, brief and wrapper verbatim, majority
rule, hashed card IDs.

## Decision rule

- New versions (`passive_v3`, `appositive_v3`): kept if, on round 3 alone,
  n ≥ 20, majority Good ≥ 85% and Changes meaning ≤ 10%.
- `frames_v2`: the same rule on round 2 + 3 pooled.
- Confirmation arms stay kept unless their pooled round 2 + 3 Good rate
  falls below 85% or Changes meaning above 10%.
- **Combined stage:** the final kept set in the order passive_v3,
  passive_recovered, nominal, frames_v2, appositive_v3, fronted_split, after
  F2 on the test set v3 dev split, judged by ROUND1.md's dev success rule.
- Nothing ships; F2 stays the default.
