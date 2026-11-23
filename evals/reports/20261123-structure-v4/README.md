# Sentence structure v4: passive coordination and nominalization (rounds 8–9), November 23, 2026

Structure v3 (rounds 6–7) left two by-phrase passive results open: `passive_v5`
passed the bar (Good ≥ 90%, Changes meaning ≤ 8%, three blind raters) but
fired on only 0.39 sentences per 100, and outside the Federal Register it was
89.8% Good (44/49). Nominalization → verb had been dropped because it fired
too rarely to fill a round. Structure v4 adds two versioned operations,
rated with exactly the v3 protocol on fresh text. No LLM is in the pipeline.
Raw text, cards and marks stay in `~/.local/state/cleartext/structure-20261123/`;
this folder holds protocols, precommits and aggregates.

## Answer

- **Both new operations pass the bar on fresh text** (pre-registered; three
  new blind Opus raters per round, majority mark; rounds 8 and 9 ran
  identical operation code on disjoint fresh windows):

| Operation | Round 8 | Round 9 | Pooled 8 + 9 | Meaning, pooled | Per 100 sentences | Decision |
|---|---|---|---|---:|---:|---|
| **`passive_v6`** | 90/100 = 90.0% [83–94] | 92/100 = 92.0% [85–96] | **182/200 = 91.0% [86–94]** | 3.5% [1.7–7.0] | 0.51 (v5: 0.39) | **passes** (round 8 test, round 9 confirmation) |
| **`nominal_v3`** | 20/20 (n < 40) | 60/60 = 100% [94–100] | **80/80 = 100% [95–100]** | 0% [0–4.6] | 0.03 mixed; 0.07 in Federal Register text | **passes** (pooled n = 80, as ROUND8.md pre-registered) |
| `passive_v5` (reference) | 46/50 = 92.0% [81–97] | 47/50 = 94.0% [84–98] | 93/100 = 93.0% [86–97] | 2.0% [0.6–7.0] | 0.39 | reference (unchanged) |

- **`passive_v6` raises recall, not the Good rate.** On the unfiltered
  standard pools (11,847 sentences) it fires on 0.51 sentences per 100
  against `passive_v5`'s 0.39 (+33%); outside the Federal Register 0.67
  against 0.46 (+47%; 53 vs 36 edits in 7,910 sentences), and in the
  rated wide pools 259 vs 178 edits. Its Good rate outside the Federal
  Register is **133/147 = 90.5% [85–94]**, about what v5 scored there in
  rounds 6–7 (44/49 = 89.8%) and in rounds 8–9 (61/67 = 91.0%). So the first
  goal of the brief is half met: more conversions in news and encyclopedic
  text at the same quality, not a higher quality.
- **By shape** (pooled): in-place flips (v5's shape with the v6 skips)
  121/132 = 91.7% [86–95]; clause coordination flipped in place 12/13;
  **coordination split 49/55 = 89.1% [78–95]**, just under 90% on its own.
  Four of the six split failures are one shape round 9 exposed and the dev
  check never showed: two participles sharing one by-phrase ("The song was
  written and produced by Torres" → "The song was written. Torres produced
  the song."; "She is shot and killed by …"). A skip for coordinated
  participles before the agent is the obvious `passive_v7` fix; it is not in
  `passive_v6` because the rated code cannot change, and it is unrated.
- **`nominal_v3`**: every one of 80 rated items was Good (all three raters
  agreed on all of them). The price is coverage: 76 of the 80 items are
  Federal Register sentences, and the operation fires on about 0.07 of 100
  Federal Register sentences (180 in 244,384) and almost never elsewhere
  (4 in 7,910 news, encyclopedic and label sentences of the standard pools).
- **On top of F2** (test set v3 dev split, [DEV_V4.md](DEV_V4.md),
  [dev-judge-v4.json](dev-judge-v4.json)): `KEPT_V4` (= `combo-v4-all`)
  gives exactly `KEPT_V3`'s outputs: neither new operation fires on any of
  F2's 100 dev passages. ΔFK −0.031 [−0.067, −0.003], Δ passage NLI +0.0003
  [+0.0001, +0.0005], every gate and the over-edit gate pass, as for
  `KEPT_V3`. The dev rule is met, but it says nothing about the new
  operations.
- **No stronger parser** was used: `en_core_web_trf` is not installed and
  nothing was downloaded into the shared venv.

Raters: six new general-purpose Opus subagents (three per round),
[RATER_BRIEF.md](../20261122-structure-v2/RATER_BRIEF.md) and
[RATER_PROMPT.md](../20261122-structure-v2/RATER_PROMPT.md) verbatim, each with
its own shuffle and only its own cards; all three agreed on 89% (round 8,
132 cards) and 99% (round 9, 166 cards). Results: [round8.json](round8.json),
[round9.json](round9.json), [pooled-r8-r9.json](pooled-r8-r9.json).

### Rated failures (all of them)

`passive_v6`, round 8 (10 of 100): in-place v5 realization errors — "Historians
have praised fish" (a name lowered), "Architect Charles Klauder designed The
Beaux Arts style building" and "… suspended The Parliament of Canada" (a
capital kept mid-sentence), "DEA does no longer regulate", "States is already
performing", "a second earthquake hit Chile, that reached …" (relative clause
left after the new object), "Daniel Romanchuk won the men's and women's
wheelchair races in 1:21:36 and Manuela Schär …" (*respectively*), "Multiple
organizations or representatives signed many submittals" (quantifier scope);
one split starting with a number ("8.07 million viewers watched the
episode"); one clause coordination ("and that the repeal of this measure does
not impact availability", where "that availability" was a noun phrase).
Round 9 (8 of 100): four coordinated-participle splits (above), "The ship
renamed HMSAS Jan van Riebeeck" (a missing "was"), "Emperors of the Komnenos
dynasty ruled The Byzantine Empire", "The Japanese fleet intercepted the
squadron … after a lengthy voyage from the Baltic" (the voyage now reads as
the fleet's; 2 of 3 raters), and one not simpler ("claims that that film
influenced Inseminoid"). Twelve of the eighteen are the in-place v5 shape;
six of them were also drawn for the `passive_v5` reference arm, where v5 made
the identical rewrite and it failed the same way.

`nominal_v3`: none.

### Examples (unanimous Good, public text)

- `passive_v6` (split, Wikipedia, CC BY-SA 4.0): "The music video was
  directed by Nabil Elderkin and featured an orange car with an atmospheric
  background." → "Nabil Elderkin directed the music video. The music video
  featured an orange car with an atmospheric background."
- `passive_v6` (clause coordination, Wikipedia): "Another significant ethnic
  group were the Jews, but almost the entire Jewish population was murdered
  during World War II by Nazi Germany." → "…, but Nazi Germany murdered almost
  the entire Jewish population during World War II."
- `nominal_v3` (Federal Register, public domain): "Wyoming would conduct an
  evaluation of the 3-way splits in two zones in accordance with established
  criteria." → "Wyoming would evaluate the 3-way splits in two zones in
  accordance with established criteria."
- `nominal_v3` (openFDA label, public domain): "Based on animal data,
  azacitidine could have an effect on male or female fertility." → "…
  azacitidine could affect male or female fertility."

## What was built

- **`passive_v6`** (`ActiveVoice(version=6)`): every v5 rule and generic
  check, plus the coordination shapes that v2–v5 always skipped. About a
  quarter of the by-phrase passives in news and encyclopedic text share their
  subject with a second verb phrase ("The episode was written by X and
  directed by Y"); no single v5 rule blocked more than a handful of Good
  items, so v6 does not relax v5 rules but adds two shapes:
  - `agentive_conj`: the passive is the second of two clauses with its own
    subject ("…, but his methods were questioned by other members" → "…, but
    other members … questioned his methods"), flipped in place under every v5
    check.
  - `agentive_split`: "S was V-ed by A [...] and VP2." → "A V-ed S [...]. S
    VP2." (each half flipped when v5 allows it, at least one must). The
    subject must start the sentence and be a pronoun or a short definite or
    named phrase; a second verb that only looks past-tense takes the shared
    "was" only when both parsers call it a participle or BERT prefers one
    reading by 2 nats; both halves must parse as complete clauses under both
    parsers.
  - v6 skips for the rated v5 failures of rounds 6–7 (trailing "whilst
    working", plural pronouns after coordinated agents, names with "and",
    light verbs, procedure nouns, medical means).
  - `passive_v5`, `appositive_v4` and `nominal_v2` outputs are unchanged
    (the new paths need `version >= 6` / `version >= 3`; replay below).
- **`nominal_v3`** (`Nominalization(version=3)`): curated pairs only (no
  WordNet mining: 5 of 5 rated mined items failed), whitelisted light verbs,
  no leftover modifier, preposition, comma-interrupted complement or later
  pronoun, verbs that need an object get one, second-parser agreement,
  output parse check, BERT fluency floor and the 0.5 entailment floor.
- **No stronger parser:** `en_core_web_trf` is not installed and nothing was
  downloaded into the shared venv; v4 keeps v3's `en_core_web_sm` +
  `en_core_web_md` agreement.

## How it was run

- **Code:** `models/src/cleartext/structure_v2/passive.py`
  (`ActiveVoice(version=6)`, `v6_reasons`, `split_coordinated_v6`),
  `nominal.py` (`Nominalization(version=3)`, `v3_reason`), `stage.py` (ops
  `passive_v6`, `nominal_v3`, preset `KEPT_V4`); harness `evals/structure_v2.py`
  (`pool-c --name H…K`, `prefilter`); fetcher `evals/structure_v2_fetch.py`
  (`--round 8, 9, 10, 11`); tests in `models/tests/test_structure_v2.py` and
  `evals/test_structure_v2.py` (in `scripts/verify`; BERT patched out).
- **Unchanged earlier ops:** the current code replays every changed round-7
  `passive_v5` and `appositive_v4` record and every changed round-5
  `nominal_v2` record, plus 300 unchanged records each, byte-identically
  (1,005 of 1,005; [v5-unchanged.json](v5-unchanged.json)).
- **Pre-registration:** [ROUND8.md](ROUND8.md) and
  [r8-precommit.json](r8-precommit.json) at `85e71c7` (operation sources,
  pools, scripts, thresholds) before any round-8 output;
  [ROUND9.md](ROUND9.md) and [r9-precommit.json](r9-precommit.json) at
  `3785101` (operation sources byte-identical to round 8) before any round-9
  output; [DEV_V4.md](DEV_V4.md) at `5e4b7e5` before the dev outputs.
- **Dev material** (before round 8): rated by-phrase items of rounds 1–7 (to
  find which rules block Good items) and never-rated old-pool sentences, one
  Opus dev rater on 120 outputs of the first v6/v3 versions (split 67/74,
  clause coordination 13/13, nominal 32/33); its failures gave the last rules
  (ROUND8.md). Rated items of rounds 8–9 were not used to change any rule.
- **Fresh windows:** pool H (Wikinews, Good articles, Federal Register
  2019-01..06, openFDA 2017), pool I (Federal Register 2018-01..06), pool J
  (Wikinews, Good articles, Federal Register 2018-07..12, openFDA 2016), pool
  K (Federal Register 2016-07..2017-12, 400 rules); titles, labels and date
  windows disjoint from every earlier download, plus the word-stage panels,
  test set v3 `panel/all.json` and earlier pools by document and 8-gram. The
  locked test split was never read.
- **Queue jobs** (Yoga, personal, `later`, `--lock cleartext`): pools
  `d8c0e10ca417` (H, I), `500e5e75e8aa` (J, K); round 8 applies `c9c5b6bdb3b4`
  (wide), `b020b3abc0c1` (standard + replay); round 9 applies `b2cb1b40455d`,
  `ec71b75bca9f`, `cf5bd57dbfcc` (wide, checkpointed, resumed twice),
  `5095caeaced3` (standard); dev on F2 `92e8de96a8a6` (judge run inline: F2
  metrics reproduced); `scripts/verify` VERIFY_PLACEHOLDER.

## Disclosures

- Raters are Opus models, not people; agreement is high and they may share
  blind spots. The dev check used one rater.
- `passive_v6`'s round-8 pass was at the boundary (90.0%, lower bound 83%);
  the pooled result (91.0%, lower bound 86%) is the stronger evidence, and
  its coordination-split shape alone is 89.1%.
- The rated items are mostly Good-article sentences (118 of 200 `passive_v6`
  items; many are film, TV and music credits, "written by X and directed by
  Y"), because round robin over sources draws on whichever source has
  items. Other genres are less tested.
- `nominal_v3`'s 80 items come mostly from Federal Register rules (76), many
  of them "make changes to", "have an impact on", "conduct an evaluation of".
- The wide pools were prefiltered to sentences an operation can change
  ("by"; a light verb before a table noun); edits per 100 sentences come from
  the unfiltered standard pools. Pools H and I were excluded by 8-gram
  against pools build/A/B/F/G but not C–E (documents were already disjoint
  through the fetcher; the same gap existed for pools F and G); pools J and K
  add C–E. The builder's source labels use its own suffixes (pool I is
  `fedreg9`, pool J `*10`, pool K `fedreg11`), not the download numbers.
- The round-9 wide apply was resumed twice from its checkpoint (queue time
  limit; the nominal step's time budget was raised from 800 to 1,500 s after
  the precommit, which changes no output).
- New modules change `ensemble_study.generation_sources()` for any newly
  frozen plan.

## Reproduce

`source ~/.local/state/cleartext/structure-20261123/env.sh`; fetch with
`fetch.sh`, `fetch10.sh`, `fetch11.sh`; then `r8_pool.sh`, `r8_apply.sh`,
`r8_apply_wide.sh`, `r8_panel.sh`, three raters (`rater_prompt.py`), `tally
--name rating-r8 --rates-from r8-apply.jsonl`; the same with `r9_*` and
`rating-r9`; dev on F2: `dev_v4.sh`, then `judge --arms combo-v3 combo-v4-p6
combo-v4-n3 combo-v4-all`.
