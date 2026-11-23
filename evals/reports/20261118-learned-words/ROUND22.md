# Round 22 protocol (written November 22, 2026, after round 21's marks, before the round-22 CV gate or any round-22 swap)

## Why

Round 21 ([round21.json](round21.json)) put frozen p50 on news and
encyclopedic text for the first time. It held its precision there, 94.4%
Good (17/18) with no meaning errors, but made only 26% as many swaps as arm b
(bar 40%), about 6 per 100 sentences, against 12 on technical text.
- Arm b's 68 genre swaps were 65% Good, and 40 of them used a pair never
  rated before. p50 kept 3 of those 40.
- q50, the unseen-robust refit, chose the same swaps as p50.

The thresholds were set on technical rows, where most pairs are seen, so
genre scores fall below them.

Ranking round 21's 68 genre arm-b swaps by the q50 score puts 95% Good in the
top 30% and 92.6% in the top 40% (p50's score: 95% and 88.9%). That look
used round 21's marks, which are now development data. A fresh genre window
must confirm it.

## Arms

- **qg (primary):** round 21's arm q, unchanged as a model class, refitted
  on all 1,377 rated rows (rounds 3–21). The pair priors now include round
  21's genre pairs.
  - The genre threshold keeps 40% of round 21's news and encyclopedic arm-b
    rows, by passage-grouped out-of-fold score.
  - A technical threshold (50% of the technical arm-b rows) is recorded but
    not used on this all-genre window.
- **rg (ablation):** arm p refitted the same way, with the same genre
  threshold rule.
- **Frozen comparators:** p50 (round 19) and q50 (round 21) at their own
  thresholds, arm b and arm gs.
- **Surface realization:** the a/an realization applies, as in round 21.
- **Excluded:** Joshua's 60, round 2, `stacker-joshua60.json`, the locked
  test split and Joshua's round-21 window.

## CV gate ([`r22-cv.json`](r22-cv.json))

10 repeats × 5 passage-grouped folds. In each outer split, the genre
threshold keeps 40% of the training folds' genre arm-b rows, by inner
out-of-fold score. All must hold:

1. qg ≥ 90% Good on held-out genre arm-b rows (68 rows, so the result is
   noisy);
2. qg keeps ≥ 35% of them;
3. on the technical arm-b rows with the technical threshold, qg reaches
   ≥ 90% Good (sanity).

The rule was chosen after the look above, so a pass is weak evidence. If the
gate fails, no fresh text is spent.

## Fresh window (only if the gate passes)

- **Panel:** news and encyclopedic only, seed 20261222: 45 Wikinews and 60
  featured-article leads. About 30% of encyclopedic passages are lost to the
  rewrite-corpus overlap check. One passage per document is taken, and no
  document from round 21's or Joshua's window is used. A passage is dropped
  if it shares a word 8-gram with:
  - any earlier panel or window, including round 21 and Joshua's;
  - any rated development sentence;
  - test set v3's panel file (exclusion only);
  - the round-7 rewrite corpora.
- **Generators:** round 11's (all, sense).
- **Arms:** b, gs, p50, q50, rg and **qg**. Filters apply before the
  per-passage limits.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all arms
  mixed in each rater's own shuffle, and 20 old items. The majority mark
  counts; a three-way split counts as the most severe mark.
- **Success (arm qg, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals and per source;
  - unseen pairs, against the 1,377 rows and against the 1,259 rows before
    round 21;
  - per-rater results;
  - swaps per 100 sentences.
- **Results:** `round22.json`. Nothing ships.
- **Caveat:** the genre threshold requires the text type to be known at run
  time. Here it comes from the source corpus.
