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

## Addendum k (written before arm k was trained or replayed)

### Corpus

- **Sources:** RFC standards from 2015–2019 (300 documents), significant
  Federal Register rules and proposed rules from 2022 to mid-2023 (395), and
  openFDA prescription labels effective in 2022 (585, one per generic name).
- **Excluded:** every v3 generic name, v3 document and round-4/5 panel
  document.
- **Sentences:** 391k kept. 1,393 were dropped because they share a word
  8-gram with a v3 or round-4/5 passage.
- **SEC 10-K filings were skipped:** SEC refuses clients that send no contact
  address, and none is sent.

### Model

- **Features:** corpus statistics only (`context_fit.py`). They cover
  keyness, plus counts and association of target and replacement in five
  frames around the swapped word.
- **Labels:**
  - A, human labels from SWORDS dev: accepted by at least 50% of
    annotators means fit; accepted by at most 10% means misfit.
  - B, distant labels from the corpus: the same word inside a multiword
    NIST, US Courts or MeSH term is a misfit, and outside any term is a fit.
    Rows are matched per word, at most 20 each. The swapped-in candidate is a
    more frequent WordNet or plainlanguage.gov candidate.
- **Threshold rule:** keep at least 90% of out-of-fold fits, with folds
  grouped by word. Swaps scoring below the threshold are vetoed.
- **Reported, not gating:** leave-one-glossary-out AUC on B.
- **Replay go/no-go:** as above. Arm k must remove at least a third of the
  Changes-meaning swaps.
