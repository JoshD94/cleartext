# Round 8 protocol (written November 21, 2026, before any round-8 model or swap)

Rounds 6 and 7 trained on proxies (corpus counts, editor behaviour) and the
proxies did not predict blind marks. Round 8 trains on the only data that has
the right label and the right distribution: **the stage's own kind of swaps,
with blind Good / Not simpler / Changes meaning marks.**

## This changes the role of the replay set

- Rounds 6 and 7 kept the 334 rated swaps from rounds 3–5 for validation only.
  **Round 8 trains on them**, so they can no longer validate anything by
  themselves.
- Validation is now nested cross-validation (below), then a fresh window with
  a fresh blind rater.
- **Risk:** the marks come from blind AI raters, so a model fitted to them can
  learn rater habits. A fresh AI rater may share those habits. The check that
  does not share them is Joshua's 60 human marks, read once, after the fresh
  window is rated. They are never used to fit, tune or decide go/no-go.
- Round-2 swaps and Joshua's 60 stay out of training.

## Model (arm s)

- **Rows:** all 334 rated swaps. Label 1 = Good, 0 = Not simpler or Changes
  meaning.
- **Features, fixed now (31):** stage checker, window NLI both ways, BERT
  contextual and sentence similarity, masked probabilities, frequency, length,
  syllables, senses, the learned simplicity margin, complex-word and term
  probabilities, part of speech, and seven rewrite-table statistics from
  round 7.
- **Classifier:** standardized logistic regression, C = 0.1. No model search.
- **Threshold, fixed now:** keep the top 65% of arm-b swaps by out-of-fold
  score.

## Cross-validation go/no-go (fixed now)

Twenty repeats of 5-fold cross-validation, **grouped by target lemma**, so a
word is never in both train and test. Each outer fold sets its threshold from
inner out-of-fold scores on its training rows only. On the 153 arm-b swaps,
averaged over repeats, arm s earns a fresh window only if all of these hold:

1. Good rate among kept swaps is at least the baseline (65%) plus 5 points.
2. It keeps at least 60% of the swaps.
3. It removes at least a third of the 54 errors.

If it fails, no fresh text is spent and the report says so.

## Fresh window (only if s passes)

- **Passages:** new unused windows of test-set-v3 dev documents, excluding
  rounds 4 and 5. The locked test split is not read.
- **Pre-commitment:** model and threshold hashes in `stacker-precommit.json`,
  committed before any swap is generated.
- **Rating:** a fresh blind rater sees arm b and arm s swaps together with no
  scores.
- **Success:** at least 78% Good, at least 60% of arm b's swaps kept, and
  Changes meaning at or below 15%, with Wilson 95% intervals.
