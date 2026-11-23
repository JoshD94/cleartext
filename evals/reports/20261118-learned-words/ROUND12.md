# Round 12 protocol (written November 22, 2026, before the round-12 CV gate or any round-12 swap)

Round 11's arm gs reached 94.7% Good but kept only 30% of arm b's swaps
(bar: 40%). Round 12 tries to keep most of that precision at a higher yield.

## What the development rows show (exploratory, read while designing this round)

The round-11 swaps are now development data: 506 rated rows in all (371 from
rounds 3–8, single-rater marks; 135 from round 11, three-rater majority).

- **The sense cut is blunt.** Of arm s's 95 round-11 swaps, 60 had sense
  cover < 0.25. Of those, 45 were Good (relocated→moved, illustrate→show,
  identified→found), and arm gs threw all of them away.
  - WordNet relations do not rescue them: hypernym and verb-group links
    connect the bad ones (composed→made, expanded→increased) as often as
    the good ones.
  - Within arm s, a high arm-s score separates them instead. In rounds
    3–8, the 15 arm-b swaps with cover < 0.25 and an out-of-fold arm-s
    score ≥ 0.8 were all Good.
- **Pairs recur.** 71 of the 135 round-11 swaps had a lemma pair already
  rated in rounds 3–8, and that history was predictive. For example,
  multiple→several had one earlier Not simpler and failed 5 of 5 times in
  round 11; additional→extra had 6 earlier Good and passed 6 of 6.
- **A time-split look** (fit on rounds 3–8, score round 11's 126 arm-b
  swaps) was run while designing this round, so the gate below is not
  blind. A 4-feature model at a keep set on rounds 3–8 kept 53% of round
  11's arm-b swaps at 97% Good. The arm-s score alone at the same keep reached
  87%.

## Arms (one featurization per target, rated together)

| Arm | Candidates | Filter (after the margin ≥ 0.1) |
|---|---|---|
| b | all (current stage) | none |
| s | all | arm-s score ≥ 0.5127 (round 8, unchanged) |
| gs | sense cover ≥ 0.25 | arm-s score ≥ 0.5127 (round 11, unchanged) |
| t | all | arm-s score ≥ threshold keeping 50% of round-11 arm-b swaps |
| **p** | all | arm-p score ≥ threshold keeping 50% of round-11 arm-b swaps (out of fold) |
| p65 | all | the same arm-p model at a 65% keep (secondary) |

All filters apply before the per-passage limits (3 swaps per passage, one per
sentence), as in rounds 8–11. Arm t is the ablation: it shows whether arm p's
gain is more than a stricter arm-s threshold.

**Arm p** ([`evals/learned_words_r12.py`](../../learned_words_r12.py)) is a
standardized logistic regression (C = 0.1, no model search) on 4 features:

1. the logit of the committed arm-s score;
2. the target's in-context sense cover for the replacement (round 9 table);
3. the smoothed Good rate of the (target lemma, replacement lemma) pair in
   earlier ratings: (Good + 2m) / (n + 2), with m the pool's Good rate;
4. the same for the target lemma alone.

- **Pool:** all 506 rated rows.
  - Lemmas come from the stage's own lemmatizer.
  - In training, each row's priors leave the row itself out.
  - The 334 round 3–5 rows trained arm s, so their arm-s score is out of
    fold: the mean of 20 lemma-grouped 5-fold runs. Round 8 and 11 rows use
    the committed model's score.
- **Thresholds** (pre-committed): p and p65 at the 50% and 35% quantiles of
  passage-grouped out-of-fold arm-p scores on round 11's arm-b rows. Arm t is
  at the 50% quantile of the arm-s score on the same rows.
- **Excluded:** Joshua's 60, the round-2 swaps, `stacker-joshua60.json` and
  the locked test split.

## CV go/no-go (fixed now, [`r12-cv.json`](r12-cv.json))

**A. Passage-grouped nested CV.** 20 repeats of 5 folds over all 506 rows,
grouped by passage. Held-out rows get priors only from the training folds,
as fresh passages would. Each outer threshold comes from inner out-of-fold
scores on the training folds' round-11 arm-b rows. The test is on round 11's
126 arm-b rows.

**B. Time split.** Fit on rounds 3–8, with thresholds from their arm-b
rows; score round 11.

**C. Diagnostic only.** Lemma-grouped CV, where no held-out pair or target
has a prior.

Arm p earns a fresh window only if all of these hold:

1. In A, its mean kept Good rate is ≥ 92%.
2. In A, it is ≥ 3 points above arm t at the same keep (paired over repeats).
3. In A, it keeps ≥ 45% of round 11's arm-b swaps.
4. In B, it reaches ≥ 90% Good with ≥ 40% kept.

The gate is optimistic. Its design came after the time-split look above, and
round 11's window is in both A and B. The fresh window decides.

## Fresh window (only if p passes)

- **Panel:** 135 new passages, 45 per source (RFC, Federal Register,
  openFDA). Each is 4 clean consecutive sentences from one document.
  - The selection uses a new seed (20261212).
  - No document used in round 11 is reused.
  - A passage is dropped if it shares a word 8-gram with the round-4, 5, 8
    or 11 panels, any rated development sentence, or the round-7 rewrite
    corpora.
- **Pre-commitment:** `r12-precommit.json` holds the thresholds, the
  sha256 of model, pool, sense table, arm-s model and sources, and the
  panel seed. It is committed and pushed before any swap is generated.
- **Rating:** three fresh blind raters, each a separate general-purpose
  Opus subagent with the round-11 brief verbatim. Each sees every arm's swaps
  in its own shuffle, with no scores, plus 20 old round-5 items.
  - **Primary mark:** the majority of the three. A three-way split counts
    as the most severe mark.
- **Success (arm p, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson 95% intervals;
  - per-rater results and agreement;
  - swaps per 100 sentences, so a change to the base stage cannot inflate
    yield;
  - distinct pairs;
  - the Good rate on arm-p swaps whose pair was never rated before (the
    priors cannot help there).
- **Results:** `round12.json`. Nothing ships; Joshua's marks remain the
  adoption decision.
