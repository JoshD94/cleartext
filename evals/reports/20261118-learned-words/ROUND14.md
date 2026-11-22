# Round 14 protocol (written November 22, 2026, after round 12's marks, before the round-14 CV gate or any round-14 swap; round 13's window was still being generated)

## Why

On round 12's fresh window, arm p reached 84.3% Good (43/51) at 48% yield.
All 8 of its errors were Changes meaning:
- broken collocations and frames, such as "enough numbers of", "orally
  given" and "long administration";
- term senses, such as "permitted vessels" → "allowed".

Most of them came from openFDA drug labels. None was Not simpler.

Among round 12's rated swaps with an arm-p score ≥ 0.48 (65% keep), the
two stage signals that best separate Good from Changes meaning are:
- the BERT contextual cosine of target and replacement, `ctx_cos` (AUC
  0.72);
- the stage checker's score, `check` (AUC 0.73).

Arm p receives them only through the arm-s logit, which also rewards a large
frequency gain. The frequency gain runs the other way here: large jumps such
as sufficient→enough and prolonged→long changed meaning more often (AUC 0.30).
Round 10 found that adding many features hurts, so round 14 adds just these
two. They were found by reading round 12's marks, so the gate below is
optimistic.

## Arm m

- **Features:** arm p's four (logit of the arm-s score, sense cover,
  smoothed pair and target-lemma priors) plus `ctx_cos` and `check`.
- **Model:** a standardized logistic regression, C = 0.1, no model search.
- **Pool:** 620 rated rows: 371 from rounds 3–8, 135 from round 11 and 114
  from round 12. Each training row's priors leave the row itself out.
  - The 31 features of round 11's swaps come from re-featurizing them
    (`r13 features11`). The committed arm-s model reproduces their stored
    scores.
- **Keep rule (fixed now):** choose the widest keep in {40, 45, 50, 55,
  60}% at which passage-grouped CV gives at least 92% Good on round 12's
  arm-b rows, the harder window. If no keep qualifies, the gate fails.
  - Arm m's threshold is that quantile of out-of-fold scores on the rounds
    11–12 arm-b rows.
  - Arm mw uses the same model at a keep 10 points wider (secondary).
- **Excluded:** Joshua's 60, round 2, `stacker-joshua60.json` and the locked
  test split.

## CV go/no-go ([`r14-cv.json`](r14-cv.json))

At the chosen keep, all must hold:

1. The keep rule is met.
2. In passage-grouped nested CV (10 repeats × 5 folds), arm m reaches ≥ 90%
   Good on the 233 arm-b rows of rounds 11–12.
3. In the same CV, arm m is ≥ 2 points above arm p refitted on the same
   pool (paired).
4. **Time split:** fit on rounds 3–11, with thresholds from round 11's
   arm-b rows, then score round 12. Arm m must be ≥ 3 points above arm p,
   with ≥ 35% of round 12's arm-b rows kept.

A lemma-grouped diagnostic is reported too, but it decides nothing.

## Fresh window (only if m passes)

- **Panel:** round 12's rule with seed 20261214, but **30 passages per
  source** (90 in all), so the window fits before 11:00.
  - No document used in rounds 11–13 is reused.
  - A passage is dropped if it shares a word 8-gram with the round-4, 5, 8,
    11, 12 or 13 panels, any rated development sentence, or the round-7
    rewrite corpora.
- **Generators:** round 11's (all and sense). There is no runner-up
  fallback, because round 13's result is not known yet.
- **Arms:**
  - b, s, gs;
  - p (the frozen round-12 model, a second replication);
  - **m**;
  - mw.

  Filters apply before the per-passage limits.
- **Pre-commitment:** `r14-precommit.json` holds the keep, thresholds and
  sha256 of the model, pool, round-12 model, sense table, arm s and
  sources. It is committed and pushed before any swap.
- **Rating:** three fresh blind Opus raters, as in round 12. Each gets the
  brief verbatim and all arms mixed in its own shuffle, plus 20 old round-5
  items. The majority counts; a three-way split counts as the most severe
  mark.
- **Success (arm m, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals;
  - per-rater results;
  - swaps per 100 sentences;
  - distinct and unseen pairs;
  - results by source.
- **Caveat:** with 90 passages, arm m will make roughly 30–40 swaps, so its
  95% interval spans about ±10 points.
- **Results:** `round14.json`. Nothing ships.
