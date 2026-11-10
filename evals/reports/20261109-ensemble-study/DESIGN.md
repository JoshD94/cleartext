# Study design (pre-registered before phase A scores were read)

Written 2026-11-09 while phase A was running and before any phase A
aggregate was computed or inspected. The selection rule itself is frozen
separately in [SELECTION_RULE.md](SELECTION_RULE.md).

## Factors

| Factor | Levels | Notes |
|---|---|---|
| words | v7, contextual, off | contextual = `ensemble-context-similarity-20261002` (PR #8) |
| phrases | authored, off | SimplePPDB is not on main, so that level is skipped |
| structure | rules, t5, compound_tagger, off | t5 = screened step-441 checkpoint; compound_tagger = PR #12 splitter, scoped guards, WikiSplit tagger |
| negation | off, on | PR #11 rule |
| grammar | off, on | local LanguageTool policy v3 |
| terms | off, on | technical-term protection |

Passive-to-active is excluded for now (teammate branch pending). It can be
added to `FACTORS` with legacy level `off` in `LEGACY`; existing config IDs and
cached cells are unchanged, so a rerun only adds the passive cells.

The full factorial is 3·2·4·2·2·2 = 192 configs. It is not run.

## Phase A: one-factor-at-a-time screen (10 configs)

The v7 baseline (`words=v7, phrases=authored, structure=rules`, the rest off,
identical to Week 7 `full_guarded`) plus each single-level change from it,
plus the `original` and `rule_based` references, on all 90 sourced passages.

## Phase B rule

1. A level is **active** if, against the v7 baseline over the 90 passages,
   it changes mean FK grade by at least 0.02, or mean passage-level NLI or
   mean sentence-level NLI by at least 0.005, or changes the output of at
   least 3 passages. Inactive levels are dropped from phase B: their outputs
   are (near-)identical to the baseline's and cannot change the ranking.
2. Phase B is the full factorial over the active levels (with each factor's
   baseline level) when that is at most 40 configs. Otherwise it is greedy
   forward selection from the v7 baseline: at each step apply the single
   factor change that most improves the frozen objective (all gates pass,
   then lowest FK; while nothing passes, highest passage NLI subject to FK
   not above the baseline), stopping when no change improves it, followed by
   a full factorial over the top three factors to check two-way interactions.
3. **Finalists** for ASSET test SARI: the top five configs under the frozen
   rule. If fewer than five pass, the remaining slots go to the Pareto
   frontier configs (FK versus passage NLI) closest to the NLI gate. The v7
   baseline, `original` and `rule_based` are always scored on ASSET.

## Caching

Outputs are cached per (passage, config) under a generation signature; metrics
per exact (source, output) pair under a scoring signature, so configs with the
same output share scoring. Lexical decision rows (keyed by word-stage text,
target offset, original text and threshold), T5 sentence outputs and grammar
corrections are cached and reused across configs and batches. Term
protection, guardrails and selection still run live for every config.
