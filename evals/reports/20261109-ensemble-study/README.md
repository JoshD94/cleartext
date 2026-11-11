# Non-LLM ensemble study, November 9, 2026

Which combination of the existing non-LLM stages simplifies the 90 sourced
development passages most, without failing the meaning gates? Fourteen configs
plus `original` and `rule_based` ran on the same 90 passages as Week 7
(30 each from computer science, corporate communication and political
science). The finalists also ran on all 359 ASSET test sentences. No LLM is
used in any stage. Raw outputs stay local; this folder holds aggregates only.

## Result

Follow-up: [ensemble v2, November 10](../20261110-ensemble-v2/README.md) adds Chhoeng's passive and phrase stages, T5 + compound stacking and ablations.


**Winner under the [frozen rule](SELECTION_RULE.md):
`words=off, phrases=authored, structure=t5`** (negation, grammar and terms
off). That is the screened step-441 T5 sentence splitter plus the authored
phrase rules, with no word substitutions. Opt-in file:
[recommended-config.json](recommended-config.json). No default changed.

| 90 passages | Original | v7 (Week 7 `full_guarded`) | **Winner** |
|---|---:|---:|---:|
| Flesch–Kincaid grade ↓ | 14.115 | 13.948 | **13.618** |
| Flesch reading ease ↑ | 38.32 | 39.49 | 39.49 |
| Words per sentence | 24.96 | 24.94 | 23.62 |
| Passage NLI mean | 0.9859 | 0.8373 | 0.9836 |
| Passages with passage NLI < 0.80 | 0 | 20 | 0 |
| Sentence-level NLI mean | 0.9881 | 0.9129 | 0.9873 |
| Changed passages | 0 | 85 | 16 |
| ASSET test SARI (Xu 0/0=0) | 20.45 | 39.95 | 24.23 |

The winner lowers FK by 0.33 grade more than v7 does and keeps NLI at the
unchanged-text level. On the 16 passages it changes, FK falls from 13.81 to
11.02 and passage NLI is 0.971 (lowest 0.851). It passes all four gates;
the next passing config (`words=off, structure=compound_tagger`, FK 13.930)
is outside the 0.1 tie band, so SARI did not decide.

**The trade-off.** The winner changes only 16/90 passages and few ASSET
sentences, so its SARI (24.2) is far below the configs with word
substitution (~40). Every config that edits words fails the NLI gates: the
substitution errors are real (see the diagnosis below).

## Step 1: the word-stage NLI drop is real meaning damage

[nli-diagnosis.json](nli-diagnosis.json) rescored the saved Week 7
`lexical_only` outputs with a new **sentence-level aligned NLI**. Sentences
are aligned by shared words, so a 1→2 split forms one group. Each group gets
bidirectional DeBERTa NLI; a dropped sentence scores 0. Passages are 40–77
words, so passage-level NLI never abstained and truncation is not the cause.

- On the edited sentence, sentence-level NLI tracks passage-level NLI with
  **Pearson r = 0.97**. All **20** passages below 0.80 at passage level are
  also below 0.80 at sentence level (3 more are low only at sentence level;
  0 are low only at passage level). Unchanged sentences score ≥ 0.967.
- The worst swaps are wrong-sense or unrelated substitutions:
  estimated→calculated (edited-sentence NLI 0.005), handshake→acknowledgement
  (0.029), authenticating→certifying (0.035), authenticate→demonstrate
  (0.055), attractiveness→quality, reliability→responsibility,
  interoperability→ability, nonce→present, elected→taken. A second group
  generalizes and loses detail: forward entailment stays high, backward is
  low (tutoring→teaching 0.94/0.08, smugglers→crooks, pandemic→epidemic,
  eternity→time, hostages→prisoners).

So passage-level NLI is not an artifact. Sentence-level NLI is **not more
faithful**; its passage mean is higher (0.913 vs 0.837) only because
unchanged sentences dilute one bad swap. It is reported as an additional
column, and passage-level NLI stays the frozen gate. The contextual weights
from PR #8 avoid most of these swaps (passage NLI 0.931, 7 passages < 0.80)
but still fail the gates.

## Design

See [DESIGN.md](DESIGN.md). It was committed before any phase A score was
read, and the selection rule was committed before any scoring run.

- **Phase A (one factor at a time):** v7 plus each single change, 10 configs
  ([design-phase-a.json](design-phase-a.json)). Pre-registered activity test:
  a level is active if it changes FK by ≥ 0.02, NLI by ≥ 0.005, or ≥ 3
  outputs.
  - Active: words=contextual/off, structure=t5 (−0.49 grade, 15 passages),
    structure=compound_tagger (−0.18, 6 passages).
  - Inactive, identical to v7 or within one passage: phrases=off (1),
    structure=off (0; the old rules never fire on this panel),
    negation=on (0: 30 passages contain negations but none match the PR #11
    patterns), grammar=on (0; LanguageTool changes 0/449 inputs) and
    terms=on (0).
- **Phase B (full factorial over the active levels):** words {v7, contextual,
  off} × structure {rules, t5, compound_tagger}, 9 configs, 4 of them new
  ([design-phase-b.json](design-phase-b.json)). Effects are additive: T5 cuts
  FK by 0.49 / 0.49 / 0.50 under v7 / contextual / off words, and the NLI
  change from words is the same with every structure backend.
- **ASSET finalists** ([design-asset.json](design-asset.json)): the three
  configs that pass, the two Pareto configs nearest the NLI gate, and the
  v7 and contextual baselines.
- **Not run:** 178 of the 192 configs. SimplePPDB phrases are not on main, so that
  level was skipped. Passive-to-active is excluded until the teammate branch
  lands (see "Adding passive" below).

## All configs (90 passages)

Full table with gates and SARI: [results-table.md](results-table.md); every
metric, changed-only means, stage counts, paired deltas from v7, gates and
selections: [aggregate-results.json](aggregate-results.json). ◆ marks the
FK/passage-NLI Pareto frontier. Gate failures: 1 = mean NLI < 0.95,
2 = more than 4 passages < 0.80, 4 = entities below `rule_based`. No config
failed gate 3 (numbers/negations).

| Config (non-default factors) | Changed | FK ↓ | Flesch ↑ | Words/sent | NLI passage | NLI<0.8 | NLI sentence | Sent<0.8 | Entities | Gates | SARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| v7 + t5 ◆ | 89 | 13.460 | 40.62 | 23.62 | 0.8399 | 19 | 0.9140 | 16 | 0.9861 | fail 1,2,4 | 40.54 |
| contextual + t5 ◆ | 84 | 13.519 | 40.20 | 23.62 | 0.9320 | 7 | 0.9586 | 4 | 0.9889 | fail 1,2,4 | 39.81 |
| **words off + t5** ◆ | 16 | **13.618** | 39.49 | 23.62 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | **pass** | 24.23 |
| v7 + compound_tagger | 85 | 13.764 | 39.93 | 24.45 | 0.8374 | 20 | 0.9129 | 17 | 0.9861 | fail 1,2,4 | |
| contextual + compound_tagger | 83 | 13.822 | 39.51 | 24.45 | 0.9312 | 7 | 0.9591 | 4 | 0.9889 | fail 1,2,4 | |
| words off + compound_tagger ◆ | 7 | 13.930 | 38.74 | 24.45 | 0.9860 | 0 | 0.9881 | 0 | 1.0000 | pass | 21.34 |
| v7 baseline (= structure off, negation on, grammar on, terms on) | 85 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | fail 1,2,4 | 39.95 |
| v7, phrases off | 85 | 13.949 | 39.51 | 24.96 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | fail 1,2,4 | |
| contextual | 83 | 14.007 | 39.07 | 24.94 | 0.9309 | 7 | 0.9590 | 4 | 0.9889 | fail 1,2,4 | 39.03 |
| rule_based (reference) ◆ | 14 | 14.049 | 38.79 | 24.96 | 0.9867 | 0 | 0.9886 | 0 | 0.9907 | ref | 22.34 |
| words off | 1 | 14.114 | 38.29 | 24.94 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | pass | 20.79 |
| original (reference) | 0 | 14.115 | 38.32 | 24.96 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | ref | 20.45 |

Notes:
- "v7 baseline" covers five configs whose outputs are identical on all 90
  passages.
- Numbers and negations are preserved on every passage for every config, and
  no NLI score abstained.
- SARI is HF `evaluate`'s port of Xu et al. 2016 with 0/0 = 0, as in the
  original script. Under `evaluate`'s 0/0 = 1 convention the untouched
  source scores 53.8 and outranks every system; both are in the JSON.
- **Disclosure:** the harness committed before scoring (`31f2334`) used
  0/0 = 1 for the tie-break. It was switched to 0/0 = 0 in `c940ad9`, after
  the ASSET scores were in. The frozen winner does not depend on SARI (no tie),
  but alt 2 below does: under the pre-registered 0/0 = 1 convention, words
  off + t5 (52.50) beats contextual + t5 (47.98).

## Gate calibration and labeled alternative rankings

The frozen file is unchanged. Two calibration problems showed up:

1. **Gate 4 (entities) measures a heuristic artifact.** The capitalized-span
   check counts sentence-initial words as entities. Every entity any config
   loses is "Similarly", "Coalescing" or "However"; with sentence-initial
   spans excluded, all configs score 1.000. Fixing this changes nothing here
   (alternative 1 below): gates 1–2 bind first.
2. **Gates 1–2 bind on real errors.** Per the diagnosis, they correctly
   reject v7 word substitution. At 7 passages below 0.80, contextual words
   are 3 passages over gate 2.

| Ranking (post hoc, labeled) | Winner | Note |
|---|---|---|
| Frozen rule | words off + t5 | Passing: words off + {t5, compound_tagger, rules} |
| Alt 1: gate 4 on non-initial entities | words off + t5 | Same passing set |
| Alt 2: alt 1 + gates 1–2 on sentence-level NLI | contextual + t5 (0/0 = 0) or words off + t5 (pre-registered 0/0 = 1) | Tied with words off + t5 within 0.1 grade; the SARI tie-break flips with the convention: 39.8 vs 24.2 (0/0 = 0), 47.98 vs 52.50 (0/0 = 1) |

Alt 2 is looser than it looks: sentence-level means are diluted by
unchanged sentences. It is the ranking to use if Joshua decides the
contextual model's remaining errors (7 passages under 0.80 at passage level)
are an acceptable cost for 84/90 changed passages and much higher SARI.

## What to ablate in step 3

- **Words, by error type.** Add a meaning floor or a backward-entailment-style
  veto inside the contextual decision so the generalization swaps
  (high forward, low backward) are rejected. That is the gap between alt 2
  and the frozen gates.
- **T5 split acceptance.** It delivers all of the winner's gain. Ablate its
  generation checks and per-sentence length cut-offs. Its one weak case is
  "which" resolved to a specific referent (NLI 0.85).
- **T5 + compound_tagger stacking.** Each is one structure backend today;
  the compound splitter also adds −0.18 grade alone, on different passages.
- **Passive-to-active** once the teammate branch lands.
- **Drop from the search:** negation, grammar, terms, phrases and the old
  structure rules change ≤ 1 of 90 passages on this panel. Ablate them on a
  panel where they fire, or not at all.

## Adding passive later

Add `"passive": ("off", "on")` to `FACTORS` and `{"passive": "off"}` to
`LEGACY` in `evals/ensemble_study.py`. Existing config IDs and cells keep
their identity. A new source tree needs a new frozen plan and state
directory, then
`report --state OLD --state NEW` merges them and refuses any cell whose
outputs differ between the two runs.

## Reproduction and provenance

[provenance.json](provenance.json) records the commits, frozen plan hash,
generation/scoring signatures, every artifact/source/panel hash, the ASSET
download hashes, runtime and per-job queue receipts. Yoga CPU, one thread per
process, `PYTHONHASHSEED=0`, offline models: 36 bounded batches (≤ 25 min
each) on the personal queue, checkpointed per cell; every batch re-hashed all
frozen inputs and saw constant signatures.

- **Reproduction:** the v7 baseline, `original` and `rule_based` match Week 7's
  `full_guarded`/`original`/`rule_based` outputs on **90/90** passages each.
  `compound_tagger` differs from Week 7 `compound_full_guarded` on 5, as
  expected from PR #12.
- **Diagnosis job:** its optional second pass (Week 7 `full_guarded`) was
  stopped by the queue at 45 min. The v7 baseline's sentence-level NLI comes
  from the study scorer instead.
- **Rerunning:** `python -m evals.ensemble_study freeze|run|report` (see the
  module docstring). Local state, raw outputs and stage memos:
  `$HOME/.local/state/cleartext/ensemble-study-20261109`.

These are inspected development passages. Automatic readability, NLI, entity
and SARI scores are signals, not percentages of correct simplifications.
