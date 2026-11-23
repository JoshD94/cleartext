# Round 26 protocol (written November 22, 2026, after round 25's marks, before the round-26 CV gate or any round-26 swap)

## Why

Round 25 ([round25.json](round25.json)) replicated round 24's frozen arms on a
hard window: arm b reached 58.9% Good with 24.2% Changes meaning, the worst
since round 12. Arm f reached 85.5% (59/69) with 13.0% meaning and p50 86.6%
(58/67); every arm missed its bars. Over rounds 24–25, f is 127/141 (90.1%)
and p50 122/135 (90.4%): the same. f's errors were frames on well-rated pairs
("discontinued from the clinical trials" → "stopped from", twice; "medals were
awarded in" → given), a repeated word ("had experienced" → "had had"; "later
… subsequently" → "later … later"), and senses (escalated, admitted, acquired).
The frame prior helps only where the same pair recurs in the same frame, which
fresh text rarely repeats. Rounds 24–25 add 260 rated swaps, many of them
news and encyclopedic pairs, so a refit is due.

Unseen pairs remain the open gap. On round 24's fresh unseen-pair swaps, arm
f's score ranked better than p50's (AUC 0.82 against 0.78), but its threshold,
set on technical rows where most pairs are seen, kept only 7 of 41 (28 Good).
In passage-grouped CV on recent windows, f keeps unseen-pair rows above its
threshold at about 93% Good, and the next band down is about 85% Good
([r25-unseen-check.json](r25-unseen-check.json)). So a separate unseen
threshold could add recall only by accepting a lower marginal rate; whether
the total unseen rate stays near 90% needs a fresh window.

## Arms

- **f2 (refit):** arm f's class, unchanged (`F_ARGS`), refitted on every rated
  row through round 25 (1,454 + rounds 24–25). Its threshold keeps 50% of the
  technical arm-b rows by passage-grouped out-of-fold score (mean of 10
  repeats).
- **f2u (primary):** f2, plus an unseen-pair threshold for pairs absent from
  f2's pool. It is the lowest score at which the recent windows' (rounds
  17–22, 24, 25) unseen-pair arm-b rows at or above it are ≥ 90% Good by
  out-of-fold score, with at least 20 kept; it is never stricter than f2's
  threshold. The recent windows share the current generator, three raters
  and the current pool's coverage; rounds 3–13 are left out of this
  calibration only.
- **Frozen comparators:** round 24's f (a third window), p50, arm b.
- **Excluded from everything:** Joshua's 60, round 2, `stacker-joshua60.json`,
  his round-21 pages and keys, and the locked test split.

## CV gate ([`r26-cv.json`](r26-cv.json))

10 repeats × 5 passage-grouped folds on all pooled rows. In each outer split,
both thresholds come from the training folds (inner out-of-fold scores).

- f2 is rated only if it reaches ≥ 92% Good on held-out technical arm-b rows
  with ≥ 45% kept.
- f2u is rated only if, in addition, on the recent windows' held-out arm-b
  rows:
  1. it keeps ≥ 1.3 × f2's unseen-pair rows;
  2. those are ≥ 88% Good;
  3. all its kept rows are ≥ 91% Good.

A preview on the 1,454 rows plus round 24's 133 (before round 25's marks
existed; 3 repeats) gave f2 93.6% Good on technical rows, and on recent unseen
rows f2 34 kept at 94% against f2u 47 at 89%. With round 25 added, the gate
decides.

## Fresh window (only if f2 passes)

- **Panel:** seed 20261226: 45 Wikinews and 55 featured-article leads from the
  extended title range, 45 unused openFDA labels. Every document of every
  earlier window (rounds 11–25, both Joshua windows) is skipped, and any
  passage sharing a word 8-gram with an earlier panel or window (round 25
  included), a rated development sentence, test-set v3's panel file
  (exclusion only) or the round-7 rewrite corpora is dropped.
- **Generator, rating, majority rule:** as in rounds 24–25.
- **Success (f2u, majority marks):** Good ≥ 90%; Changes meaning ≤ 8%; ≥ 40%
  of arm b's swaps. **Unseen goal:** f2u keeps ≥ 1.3 × f2's unseen-pair swaps
  (pairs absent from f2's pool), ≥ 90% of them Good. f2 has the same three
  success bars.
- **Also reported:** every arm with Wilson intervals; unseen pairs against
  f2's pool and against the 1,454 rows; frame errors; per source, per rater.
- **Results:** `round26.json`. Nothing ships; Joshua's page is unchanged.
