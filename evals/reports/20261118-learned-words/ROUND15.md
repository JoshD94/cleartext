# Round 15 protocol (written November 22, 2026, after round 14's no-go, before the round-15 CV gate or any round-15 swap; round 13's window was still being generated)

## Why

Round 12's arm p made 8 errors, all Changes meaning, and mostly broken
collocations ("enough numbers of", "orally given", "long administration").
Round 14 added the stage's meaning signals (`ctx_cos`, checker) and failed
its gate (+1 point).

A different signal does separate these errors on the technical windows.
Round 6's corpus collocation drop, `max_drop`, is the largest fall in
corpus association between target and replacement across the frames around
the swap.
- **Counts:** the 12M-token RFC, Federal Register and openFDA corpus, with
  the swap's whole source document removed.
- **Separation on rounds 11–12 arm-b swaps:**
  - Good vs Changes meaning: AUC 0.71;
  - among swaps that arm p keeps: AUC 0.84 (round 12 alone: 0.79).
- **On non-technical text (rounds 3–8) it does not work** (0.56), which is
  why round 6's arm k failed on that replay set. The reference corpus has to
  match the text.

Added linearly to arm p, the signal gained only 0.5–1.5 points, because it
matters only among swaps arm p already likes. As a **veto after arm p** it
moved the precision/yield frontier on the pooled development rows. These
numbers are in-sample and exploratory:
- arm p at a 60% keep, then vetoing `max_drop` above about 3, kept 43% at
  97% Good;
- arm p alone at a 40% keep reached about 92.5%.

## Arm pv

1. **Arm p refitted:** the same 4-feature logistic as round 12, on round
   14's frozen 620-row pool (rounds 3–8, 11, 12). Its threshold keeps 60% of
   the rounds 11–12 arm-b rows by passage-grouped out-of-fold score.
2. **Veto:** reject the swap if `max_drop` exceeds the 80th percentile of
   `max_drop` among Good-rated technical-window rows in the training pool.
   The cut-off is learned from ratings. It is set to keep 80% of Good swaps
   and is never tuned on errors.
3. **Comparator p40:** the same refitted arm p at a 40% keep (no veto),
   about the yield arm pv is expected to reach. The frozen round-12 arm p is
   also rated.

The veto feature uses only corpus counts with the passage's whole source
document removed, as it would be for a user's document. Excluded: Joshua's
60, round 2, `stacker-joshua60.json` and the locked test split.

## CV go/no-go ([`r15-cv.json`](r15-cv.json))

All must hold:

1. **Passage-grouped nested CV** (10 repeats × 5 folds, all 620 rows). Each
   outer fold sets the keep threshold and the veto cut-off from its
   training folds only. On the 233 arm-b rows of rounds 11–12:
   - arm pv reaches ≥ 92% Good;
   - it is ≥ 3 points above arm p40 (paired);
   - it keeps ≥ 38%.
2. **Time split:** fit on rounds 3–11, with the threshold and cut-off from
   them; score round 12. Arm pv reaches ≥ 90% Good with ≥ 35% of round 12's
   arm-b rows kept.

The veto was found on these rows, so the gate is optimistic. The fresh
window decides.

## Fresh window (only if pv passes)

- **Panel:** round 14's rule. 30 passages per source (90 in all), seed
  20261215.
  - No document used in rounds 11–13 is reused.
  - A passage is dropped if it shares a word 8-gram with the round-4, 5, 8,
    11, 12 or 13 panels, any rated development sentence, or the round-7
    rewrite corpora.
- **Generators:** round 11's (all and sense).
- **Arms:** b, s, gs, p (frozen round 12), p40, **pv**. Filters apply before
  the per-passage limits.
- **Pre-commitment:** `r15-precommit.json` holds the thresholds, the
  cut-off and the sha256 of the model, pool, corpus counts, round-12 model,
  sense table, arm s and sources. It is committed and pushed before any
  swap.
- **Rating:** three fresh blind Opus raters, as in round 12. Each gets the
  brief verbatim and all arms mixed in its own shuffle, plus 20 old round-5
  items. The majority counts; a three-way split counts as the most severe
  mark.
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
- **Caveat:** 90 passages give arm pv roughly 30–40 swaps, so its 95%
  interval spans about ±10 points.
- **Results:** `round15.json`. Nothing ships.
