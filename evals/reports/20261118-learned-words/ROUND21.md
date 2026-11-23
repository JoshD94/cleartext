# Round 21 protocol (written November 22, 2026, after round 20's marks, before the round-21 CV gate or any round-21 swap)

## Why

Arm p50's precision rests on pairs it learned from past marks. Before this
round, an honest unseen-pair measurement was run on all 1,259 rated rows
(rounds 3–20) with arm p's model class ([`r21-unseen.json`](r21-unseen.json)):

- **Passage-grouped CV:** about 150 of the 827 technical-window arm-b rows
  have a pair never seen in the training folds. Arm p keeps only about 13% of
  them, at about 82% Good. On seen pairs it keeps about 59%, at about 93%.
- **Pair-grouped CV** (no held-out pair was ever seen): about 87% Good at a
  51% keep. **Target-lemma-grouped CV:** about 87%.
- **No available signal separates unseen-pair swaps well.** Their base Good
  rate is about 49%. Exploratory variants all reach AUC of about 0.72–0.75 on
  them, and none reaches 90% Good even on its top 10–20%. The variants were:
  - the arm-s score alone;
  - arm p;
  - back-off priors on the replacement lemma, the stem family and the target
    stem family;
  - BERT static-embedding neighbours of the pair;
  - a 31-feature refit (logistic or boosted);
  - a separate leave-pair-out scorer with its own threshold.

  The exploratory aggregates are in [`r21-explore.json`](r21-explore.json);
  `r21-unseen.json` re-runs arm p, arm q and the arm-s score.

So the technical windows hold out little hope of keeping more unseen pairs at
≥ 90% Good. The open question is what happens on text where most pairs are
unseen. Every fresh window so far was technical. Round 21 is the first
window on other genres.

Meaning residue in p50's rounds 17–20 errors (19 of 255 swaps):
- context frames on well-rated pairs, 13 swaps:
  - "has occurred" → "has happened" ×3;
  - "it is necessary to";
  - "orally administered";
  - "incorporate (by reference)";
  - "induces (metabolism)" ×2;
  - indicate→show ×2;
  - others;
- the replacement repeats a word already in the sentence ("Also … also",
  "contain … have"), 2 swaps;
- a swap inside quoted regulatory text, 1 swap;
- a broken article ("an wrong"), 1 swap;
- Not simpler, 3 swaps.

Over all 1,259 rows:
- swaps inside quotation marks: 6, all Changes meaning;
- the replacement's stem already in the sentence: 30, 47% Good against 60%
  overall;
- a mismatched a/an after the swap: 5, all Changes meaning.

## Changes

- **Surface realization (all arms):** after a swap, the indefinite article in
  front of the replacement is made to agree (`cleartext.learned_words.realize`).
  The swap decision is unchanged; raters see the realized sentence.
- **Arm r50:** round 12's arm p, unchanged, refitted on all 1,259 rated rows,
  keeping 50% of the technical-window arm-b rows (rounds 11–13, 17–20) by
  passage-grouped out-of-fold score. It is a comparator for q50, and is
  rated only if its passage-grouped CV reaches ≥ 90% Good at ≥ 45% kept.
- **Arm q50** (the unseen-robust refit): arm p's four features plus
  - back-off priors (replacement lemma, stem-family pair, target stem family);
  - the pair's count, and an unseen-pair × arm-s-score interaction, so unseen
    pairs can lean on the arm-s score;
  - two learned context features (inside quotation marks; replacement stem
    already in the sentence). They are features with fitted weights, not
    vetoes;
  - two stage meaning signals (`ctx_cos`, checker score).

  It is a standardized logistic (C = 0.1), refitted on 1,259 rows at a 50%
  keep chosen like r50's.
- **Frozen comparators:** p50 and p40's model (round 19, unchanged), the
  frozen round-12 arm p, arms b, s and gs. p50 is the frozen arm under test
  on the new genres.

## CV gate for q50 ([`r21-cv.json`](r21-cv.json))

The variant was chosen after the exploratory look above on the same rows, so
the gate is weak evidence. 10 repeats × 5 folds, nested thresholds, on the
technical-window arm-b rows. All must hold:

1. passage-grouped CV: q50 ≥ 90% Good;
2. passage-grouped CV: q50 keeps ≥ 45%;
3. passage-grouped CV: q50 loses at most 1 point against r50 (paired);
4. pair-grouped CV: q50 gains ≥ 1 point over r50 (paired);
5. target-lemma-grouped CV: q50 gains ≥ 1 point over r50 (paired).

If the gate fails, q50 is not rated, and the frozen-arm genre test still runs.

## Fresh window (round 21)

- **Panel:** 135 passages of 4 consecutive clean sentences, one per document,
  seed 20261221:
  - 45 news (English Wikinews, Category:Published);
  - 45 encyclopedic (English Wikipedia featured articles, lead sections);
  - 23 Federal Register and 22 openFDA documents never used before.

  The genre texts were downloaded as untrusted data
  (`evals/learned_words_genre.py`, generic client name). A passage is dropped
  if it shares a word 8-gram with:
  - any earlier panel or window (rounds 4, 5, 8, 11–13, 17–20);
  - any rated development sentence;
  - test set v3's panel file, read for exclusion only;
  - the round-7 rewrite corpora.
- **Generators:** round 11's (all, sense).
- **Arms:** b, s, gs, p, p50, r50 and q50 (if go). Filters apply before the
  per-passage limits.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all arms
  mixed in each rater's own shuffle, and 20 old items. The majority mark
  counts; a three-way split counts as the most severe mark.
- **Unseen pair:** a pair that is absent from all 1,259 rated rows.

### Success bars (majority marks)

- **Genre test, frozen p50** (news + encyclopedic passages):
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b on those passages.

  Expectation, written down now: precision may hold, but yield will likely
  fall below 40%, because p50 keeps few unseen pairs.
- **Arm q50** (whole window, only if rated):
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b;
  - at least 1.25 times as many unseen-pair swaps as p50, with ≥ 85% Good
    among q50's unseen-pair swaps.
- **Also reported:**
  - every arm, with Wilson intervals, per part (genre, technical) and per
    source;
  - unseen-pair counts and Good;
  - per-rater results;
  - swaps per 100 sentences;
  - how many items the a/an realization changed.
- **Results:** `round21.json`. Nothing ships.

## Joshua's window (prepared, not rated by any AI rater)

- **Panel:** a second window, seed 20261231, excluding round 21's passages
  and documents. It has 45 news, 45 encyclopedic, 30 Federal Register and 30
  openFDA passages, with the same exclusions.
- **Generation:** the same apply job, after round 21's window.
- **Best-arm rule:** q50 if it was rated and meets all its round-21 bars;
  otherwise p50.
- **Page** (`~/Documents/cleartext-rating/rate-learned-swaps-r21.html`):
  - about 60 best-arm swaps and about 40 arm-b swaps that the best arm
    rejects;
  - seeded mix, no scores or arm labels;
  - the key stays in `joshua-r21/`, and marks download as CSV.

  Joshua's marks are the adoption decision. They are never used to train,
  tune or choose anything.
