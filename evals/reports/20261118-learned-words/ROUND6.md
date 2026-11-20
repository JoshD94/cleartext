# Round 6 protocol (written November 20, 2026, before any round-6 model or swap)

Planning advice came from a separate model (Fable 5.2). It was given only
results, data and constraints. Its plan: an offline replay gate, then
context-sensitive domain-sense protection, then a pairwise simplicity model.

## Replay set (validation only)

- `learned-words replay build` collects the blind-rated swaps from rounds
  3–5. It leaves out round-2 swaps, Joshua's 60 marked swaps and the
  human-gold controls.
- 334 swaps in total. The 153 from the current stage (arm b) have marks
  99 Good, 23 Not simpler and 31 Changes meaning, for a Good rate of 65%.
- The replay set is never used to fit a model or choose a threshold.
- An arm is evaluated on replay as a filter on arm b's rated swaps.

## Arms (each one is b plus a filter)

- **p: pairwise simplicity.** A gradient-boosting classifier for
  P(replacement simpler than target), trained on:
  - pairs between BenchLS-train substitutes, labelled by rank;
  - plainlanguage.gov hard→plain pairs, with the reversed pairs as
    negatives;
  - each BenchLS-train target → its rank-1 substitute.

  Folds are grouped by lemma. The threshold maximizes out-of-fold balanced
  accuracy, as in rule R2. Swaps with P below the threshold are dropped.
- **k: context and domain fit.** A learned veto using counts from the
  round-6 technical reference corpus (`learned-words corpus`). Its
  features, labels and threshold rule go into an addendum below before it is
  trained and before its replay numbers are computed.
- **pk:** both filters.

## Replay go/no-go (per arm, fixed now)

An arm earns a fresh window only if, on the 153 arm-b replay swaps, all of
these hold:

1. Its Good rate among the swaps it keeps is at least the baseline (65%)
   plus 5 points.
2. It keeps at least 60% of the baseline swaps.
3. Its targeted error falls enough: p removes at least 40% of the
   Not-simpler swaps; k removes at least a third of the Changes-meaning
   swaps.

If no arm passes, round 6 spends no fresh text, and the report says so.

## Fresh window (only for arms that pass)

- **Passages:** `panel build-r6` takes unused windows of test-set-v3 dev
  documents. It excludes every round-4 and round-5 window and never reads a
  test document.
- **Pre-commitment:** thresholds and model hashes go into
  `round6-precommit.json` before any swap is generated.
- **Rating:** a fresh blind rater sees every arm's swaps shuffled
  together, with no scores.
- **Success:** the arm reaches at least 78% Good, keeps at least 60% of arm
  b's swaps on the same window, and has Changes meaning at or below 15% of
  its swaps. Wilson 95% intervals are reported.
