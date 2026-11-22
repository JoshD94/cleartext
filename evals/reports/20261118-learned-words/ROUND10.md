# Round 10 protocol (written November 21, 2026, after the round-9 no-go, before any round-10 model or swap)

## Why

Round 9's diagnostic: on arm-b rows, `ctx_cos` alone separates Changes
meaning with AUC 0.79, yet the combined filters reach only 0.57–0.58. The
reason is that the two errors move in opposite directions along the same
signals:
- Changes meaning falls as contextual similarity rises;
- Not simpler rises with it.

One logistic model of "Good" is linear, so it cannot express "reject both
ends", and the signal cancels out.

## Arm h: two heads

- **Rows and features:** the same 371 rows and the same 44 features as
  round 9 (features sha256 `55e85ba1…`, checked).
- **Heads:** head M predicts Changes meaning vs the rest, and head N
  predicts Not simpler vs the rest. Each is the same standardized
  logistic regression, C = 0.1.
- **Score:** (1 − P(meaning)) × (1 − P(not simpler)).
- **Threshold:** keep the top 55% of arm-b swaps by out-of-fold score, as in
  round 9. Nothing else changes.

## Gate (round 9's gate, unchanged)

Twenty repeats of nested 5-fold CV, grouped by lemma, against arm s refitted
at 55% keep on the same folds. On the 189 arm-b rows:

1. Paired over repeats, arm h's Good rate is at least 3 points above arm s.
2. It removes at least half of the Changes meaning swaps.
3. Its Good rate is at least the base (61%) plus 10 points.

**Caveat:** this is the second gate on the same 371 rows, and arm h was
chosen after looking at round-9 diagnostics on them, so the CV result is
optimistic. The fresh window decides.

## Fresh window (only if h passes)

Exactly as specified in [ROUND9.md](ROUND9.md):
- **Panel:** 135 technical-corpus passages.
- **Arms:** b, s and h rated together. In the code and the raw files, arm
  h's score and arm are stored under the key `w`.
- **Rating:** three fresh blind raters; the majority mark counts, and a
  three-way split counts as the most severe mark.
- **Success:** at least 78% Good, Changes meaning at or below 15%, and at
  least 45% of arm b's swaps kept.
- **Pre-commitment:** `r10-precommit.json`, committed before any swap.
- **Results:** `round10.json`.
