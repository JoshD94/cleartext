# Round 17 protocol (written November 22, 2026, after round 16's no-go, before the round-17 CV check or any round-17 swap)

## Why

Two fresh windows have rated round 12's frozen arm p, which keeps 50% of
round-11-like arm-b swaps:

| Window | Swaps | Good | Changes meaning | Yield vs b |
|---|---:|---:|---:|---:|
| Round 12 | 51 | 84.3% | 15.7% | 48% |
| Round 13 | 64 | 89.1% | 9.4% | 56% |

Both miss the 90% bar. A stricter threshold is the remaining lever with clean
evidence behind it:
- Round 15 pre-registered **p40**, arm p refitted on 620 rows at a 40%
  keep, as a comparator, before round 13 was rated.
- Round 16 scored p40 on round 13's 115 rated arm-b swaps, which it had
  never seen: 92.3% Good (48/52) at 45% kept.
- That number is a held-out filter test. It was read before this protocol
  was written, so it is evidence rather than a blind gate.
- Over the same rows, round 16's veto arm reached only 84.1%, so the veto
  is dropped.

## Arm p40

- **Model:** round 12's 4-feature arm p, unchanged (logistic C = 0.1 on the
  arm-s logit, sense cover, and smoothed pair and target-lemma priors).
- **Refit:** on all 750 rated rows from rounds 3–8 and 11–13 (round 16's
  frozen pool; sha256 checked). Rounds 12 and 13 add 244 rated technical
  swaps to the priors.
- **Thresholds:**
  - **p40:** keeps 40% of the rounds 11–13 arm-b rows by passage-grouped
    out-of-fold score.
  - **p50:** the same model at 50% (secondary).
  - The frozen round-12 arm p is also rated, a third replication.
- **Excluded:** Joshua's 60, round 2, `stacker-joshua60.json` and the locked
  test split.

## CV check ([`r17-cv.json`](r17-cv.json))

All must hold:

1. **Held out (read before writing; reported for completeness):** round
   15's p40 scored on round 13 reaches ≥ 90% Good with ≥ 40% kept.
2. **Passage-grouped nested CV** (10 repeats × 5 folds, 750 rows): p40
   reaches ≥ 90% Good on the 348 rounds 11–13 arm-b rows.
3. In the same CV, p40 keeps ≥ 38% of those rows.

## Fresh window (only if the check passes)

- **Panel:** round 12's rule at full size. 45 passages per source (135),
  seed 20261217.
  - No document used in rounds 11–13 is reused.
  - A passage is dropped if it shares a word 8-gram with the round-4, 5, 8,
    11, 12 or 13 panels, any rated development sentence, or the round-7
    rewrite corpora.
- **Generators:** round 11's (all and sense).
- **Arms:** b, s, gs, p (frozen round 12), **p40**, p50. Filters apply
  before the per-passage limits.
- **Pre-commitment:** `r17-precommit.json`, committed and pushed before any
  swap.
- **Rating:** three fresh blind Opus raters, as in rounds 12–13. Each gets
  the brief verbatim and all arms mixed in its own shuffle, plus 20 old
  round-5 items. The majority mark counts; a three-way split counts as the
  most severe mark.
- **Success (arm p40, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals;
  - per-rater results;
  - swaps per 100 sentences;
  - distinct and unseen pairs;
  - results by source.
- **Results:** `round17.json`. Nothing ships.
