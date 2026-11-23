# Round 19 protocol (written November 22, 2026, after round 18's marks, before the round-19 CV check or any round-19 swap)

## Why

Rounds 17 and 18 rated the same refitted arm p on two fresh windows:

| Arm | Round 17 | Round 18 | Pooled | Yield vs b |
|---|---:|---:|---:|---:|
| p40 | 96.3% (52/54) | 88.5% (46/52) | 92.5% (98/106; CI 86–96) | 44%, 39% |
| p50 | 96.9% (63/65) | 87.3% (62/71) | 91.9% (125/136; CI 86–95) | 53%, 53% |

Precision is about the same at either keep, and p50 makes about 30% more
swaps. Round 18's p40 also fell below the 40% yield bar. Round 19 therefore
makes **p50 the primary arm** and refits with rounds 17 and 18's 274 rated
swaps, which add pair history.

## Arm p50

- **Model:** round 12's 4-feature arm p, unchanged.
- **Refit:** on all 1,024 rated rows (rounds 3–8, 11–13, 17, 18).
- **Thresholds:**
  - **p50** keeps 50% of the arm-b rows of the technical windows (rounds
    11–13, 17, 18) by passage-grouped out-of-fold score.
  - p40 keeps 40% (secondary).
  - The frozen round-12 arm p is also rated.
- **Excluded:** Joshua's 60, round 2, `stacker-joshua60.json` and the locked
  test split.

## CV check ([`r19-cv.json`](r19-cv.json))

All must hold:

1. **Passage-grouped nested CV** (10 repeats × 5 folds): p50 reaches ≥ 90%
   Good on the technical-window arm-b rows.
2. In the same CV, p50 keeps ≥ 45% of those rows.
3. **Time split:** fit on all rounds before 18, with the threshold from
   their arm-b rows; score round 18. p50 must reach ≥ 85% Good with ≥ 40%
   kept. This is a weak sanity bar, because round 18's fresh result (87.3%)
   is already known.

## Fresh window (only if the check passes)

- **Panel:** 45 passages per source (135), seed 20261219.
  - No document used in rounds 11–13, 17 or 18 is reused.
  - A passage is dropped if it shares a word 8-gram with those panels or
    the rounds 4, 5 and 8 panels, any rated development sentence, or the
    round-7 rewrite corpora.
- **Generators:** round 11's (all and sense).
- **Arms:** b, s, gs, p, p40, **p50**. Filters apply before the per-passage
  limits.
- **Pre-commitment:** `r19-precommit.json`, pushed before any swap.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all
  arms mixed in each rater's own shuffle, and 20 old items. The majority
  mark counts; a three-way split counts as the most severe mark.
- **Success (arm p50, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals;
  - per-rater results;
  - swaps per 100 sentences;
  - unseen pairs;
  - results by source.
- **Results:** `round19.json`. Nothing ships.
