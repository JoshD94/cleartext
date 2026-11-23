# Round 10 protocol: `passive_v7` (written November 23, 2026, before any round-10 output)

One new operation, one fresh round, the round 8–9 rating protocol unchanged.

## Why

`passive_v6` is not kept: without the six round-8 items that repeat earlier
text (README correction), round 8, its pre-registered test, is 84/94 = 89.4%,
under 90%. The review also showed one live meaning loss of the split shape:
"The song was written and produced by Torres." → "The song was written.
Torres produced the song." (the by-phrase belongs to both participles).

## The operation (`passive_v7`, `ActiveVoice(version=7)`)

`passive_v6` plus one skip, `shared_agent_participles_v7` in `passive.py`:
the coordination split abstains when the first participle has no by-phrase
and nothing but adverbs before the conjunction, and the by-phrase belongs to
the participle chain that starts at the second verb ("S was V1-ed [and
V2-ed] and V3-ed by A"). The by-phrase then reads as shared, and the split
would drop who did V1. Every other v6 rule, check and output is unchanged
(`version >= 7` gates the skip; `passive_v6` outputs are identical).

The rule comes from the shape the review named and from the grammar of
coordinated participles (a trailing by-phrase scopes over the coordination),
tested on sentences written for the tests. No rated item's mark from any
round was read to write or tune it. No other narrowing is added: every other
known v6 failure was found from rated marks, and a rule written from those
cannot be validated on them (AGENTS.md).

## Fresh window (pool L)

- `structure_v2_fetch.py --round 12` (`download12`, fetched before this file):
  Good-article titles and Wikinews documents not in any earlier window,
  Federal Register significant rules 2016-01-01..06-30, openFDA labels
  effective in 2015, every earlier download excluded by title, label, generic
  name and date window.
- `pool-c --name L`: documents and sentences sharing an 8-gram with **every**
  earlier pool (A–K, standard, wide and prefiltered, both state directories),
  every rating key (rounds 1–9), the dev pool and the dev rating key are
  excluded (`earlier_rated`; the round-8 gap is closed and tested), plus the
  word-stage panels and test set v3 as before. `prefilter --ops passive_v7`.
  Pool hashes are in [r10-precommit.json](r10-precommit.json); the pools were
  built before this file was committed and no sentence of them has been read
  or rewritten. The locked test split is never read.

## Arm, panel and rule

- `r10_apply.sh`: `passive_v7` with the 0.5 entailment floor on the
  prefiltered L wide pool (rated items); `passive_v7` and `passive_v6` on the
  standard pool L for edits per 100 sentences (v6 only for comparison).
- `r10_panel.sh`: up to 100 `passive_v7` items, round robin over sources,
  seed 20261122; three new blind Opus raters (native subagents),
  [RATER_BRIEF.md](../20261122-structure-v2/RATER_BRIEF.md) and
  [RATER_PROMPT.md](../20261122-structure-v2/RATER_PROMPT.md) verbatim, each
  with its own shuffle and only its own cards; majority mark (ties to the
  most severe mark, as before).
- **Bar (round 10 alone):** n ≥ 40, Good ≥ 90% and Changes meaning ≤ 8%.
  Pass: `KEPT_V4 = KEPT_V3 + ("nominal_v3", "passive_v7")` and the dev check
  on F2 (`dev`, `score`, `judge`; every gate must pass). Fail: reported as
  is, and `KEPT_V4` stays `KEPT_V3 + ("nominal_v3",)`.
- Every rated item is checked again for 8-gram overlap with earlier material
  (expected none); the bar is judged without any overlapping item, and the
  result with every item is reported beside it.
