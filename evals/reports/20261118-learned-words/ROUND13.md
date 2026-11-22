# Round 13 protocol (written November 22, 2026, while round 12's window was still being generated, before any round-12 mark or round-13 swap)

Round 13 fits no new model. It tests the frozen round-12 arm p on a second
fresh window, and adds one generator change aimed at yield.

## Why

- **One window is thin evidence.** Round 12 rates arm p on about 60 swaps,
  so a 95% interval is roughly ±8 points. The same model, thresholds and
  priors on a second window test whether the result replicates.
- **Yield is capped by the generator.** On round 11's window the stage
  featurized up to 10 targets per passage, yet only 176 of them produced a
  swap; 46 of 135 passages had none. Arm p only filters, so each rejected
  top candidate costs a swap.
  - **Fallback:** when a target's top candidate yields no accepted swap, let
    the stage decide again on its runner-up candidate. The runner-up then
    faces the same gate, checker, margin and arm-p threshold.

## Evidence (development rows, read while designing this round)

- No rated development swap came from this runner-up generator, so the
  fallback cannot be cross-validated.
- The nearest rated data are round 5's arm f, where a judge picked among the
  stage's candidates. Its 161 rated rows were 32% Good.
  - Passage-grouped out-of-fold arm-p scores at the round-12 threshold kept
    12 of them, and 11 were Good.
  - At the 65% threshold it kept 41, at 71% Good.
- Expectation: the fallback at arm p's threshold adds few swaps, at high
  precision. At the 65% threshold it is risky.

## Generators and arms

One featurization per target gives three generators:
- **all:** the current stage;
- **sense:** round 11's sense-limited generator (cover ≥ 0.25);
- **next:** every candidate except the all-generator's top one.

The stage, the margin (≥ 0.1), the per-passage limits (3 swaps, one per
sentence) and every model and threshold are round 12's, unchanged (sha256 in
`r13-precommit.json`).

| Arm | Swaps considered | Filter |
|---|---|---|
| b | all | none |
| s | all | arm s ≥ 0.5127 |
| gs | sense | arm s ≥ 0.5127 |
| t | all | arm s ≥ 0.6976 (round 12) |
| p | all | frozen arm p ≥ 0.6979 (round 12, replication) |
| p65 | all | frozen arm p ≥ 0.4787 |
| **pn** | all, else next for the same target | frozen arm p ≥ 0.6979 |
| p65n | all, else next | frozen arm p ≥ 0.4787 |

Filters and the fallback apply before the per-passage limits.

## Gate (fixed now)

No new model is fitted, so the gate checks the frozen model and the code
change. All three must hold before any round-13 swap is generated:

1. Round 12's CV gate passed ([r12-cv.json](r12-cv.json), `go: true`).
2. The round-12 model file is unchanged (sha256 match).
3. **Smoke:** on 3 spent round-11 passages, the new code's all and sense
   generators reproduce round 11's stored swaps exactly. This shows that
   arms b, s, gs and p are unaffected by the change.

## Fresh window

- **Panel:** round 12's rule, with seed 20261213 and 45 passages per source.
  - No document used in round 11 or 12 is reused.
  - A passage is dropped if it shares a word 8-gram with the round-4, 5, 8,
    11 or 12 panels, any rated development sentence, or the round-7
    rewrite corpora.
- **Rating:** as round 12. Three fresh blind Opus raters each get the brief
  verbatim, all arms mixed in their own shuffle, and 20 old round-5 items.
  The majority mark counts; a three-way split counts as the most severe
  mark.
- **Primary success (arm pn, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b;
  - more swaps than arm p.
- **Replication (arm p):** the round-12 bars (Good ≥ 90%, meaning ≤ 8%,
  ≥ 40% of arm b).
- **Also reported:**
  - every arm, with Wilson 95% intervals;
  - per-rater results and agreement;
  - swaps per 100 sentences;
  - distinct pairs;
  - Good rate on unseen pairs;
  - Good rate on runner-up swaps.
- **Results:** `round13.json`. Nothing ships.
