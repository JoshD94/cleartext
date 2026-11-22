# Round 16 protocol (written November 22, 2026, after round 15's no-go and before any round-13 mark or round-16 swap)

## Why

Round 15's arm pv is round 12's arm p, refitted, at a 60% keep, followed by
a learned corpus-collocation veto. It failed its CV gate by one criterion:
- **Passage-grouped CV:** arm pv reached 94.7% Good at 45.6% kept. Arm p40
  reached 92.7% at 40.6%. The paired gain was +2.05 points; the bar was 3.
- **Time split onto round 12:** 90.0% against 85.0% at the same 37% keep.
- **The problem:** the veto was found on the very rows used in that gate, so
  neither number is clean.

Round 13's window has been generated but not rated yet. Its marks will be
the first data that no design step has seen. Round 16 uses them as the gate
for arm pv **exactly as round 15 specified it**, before spending any new
window.

## Gate (fixed now, on round 13's majority marks; [`r16-cv.json`](r16-cv.json))

1. Fit round 15's arm pv on round 15's 620-row pool (rounds 3–8, 11, 12),
   with round 15's rules:
   - arm p refitted;
   - thresholds at 60% (pv) and 40% (p40) of the rounds 11–12 arm-b rows'
     passage-grouped out-of-fold scores;
   - veto cut-off at the 80th percentile of `max_drop` on Good
     technical-window rows.
2. Score round 13's rated arm-b swaps. The veto feature removes each swap's
   source document from the corpus counts.
3. **Go only if all three hold:**
   - arm pv reaches ≥ 90% Good;
   - it is ≥ 3 points above arm p40 on the same rows;
   - it keeps ≥ 35% of round 13's arm-b swaps.

This is a filter test on rated swaps. It does not include the per-passage
limits or swaps that arm b did not choose.

## If it passes

- **Refit:** the same procedure on all rated rows (620 + round 13).
  Thresholds come from the rounds 11–13 arm-b rows, and the cut-off is
  refitted.
- **Fresh window:** 90 passages (30 per source), seed 20261216. No
  document used in rounds 11–13 is reused, and passages with 8-gram
  overlaps are excluded as in round 15.
  - Round 11's two generators.
  - Arms b, s, gs, p (frozen round 12), p40 and **pv**. Filters apply
    before the per-passage limits.
- **Pre-commitment:** `r16-precommit.json`, committed and pushed before any
  swap.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all
  arms mixed, and 20 old items. The majority mark counts; a three-way split
  counts as the most severe mark.
- **Success (arm pv, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals;
  - per-rater results;
  - swaps per 100 sentences;
  - unseen pairs;
  - results by source.

If the gate fails, no fresh text is spent and the report says so.
