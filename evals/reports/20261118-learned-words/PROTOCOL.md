# Learned word stage: protocol (pre-registered November 18, 2026)

Committed before any model is fitted or any test number exists. Git history
is the evidence. No LLM is used anywhere. Raw text, candidate rows and
predictions stay in `~/.local/state/cleartext/learned-words-20261118/`; this
folder holds aggregates, word-level decisions keyed by hash, and this file.

## What had been read before this file

- [Joshua's 60 marked swaps](../20261118-human-swap-marks/README.md): 13 Good,
  36 Changes meaning, 11 Not simpler; nouns 1/24 good. Used **only** as a
  held-out technical-domain check. Their sentences are featurized with no
  gold and no candidate supplied, and nothing is tuned on them.
- [Locked-test finalists](../20261116-locked-test/FINALISTS.md): F2 (splits
  only, words off) is the shipped default and stays so unless the new stage
  is confirmed by human marks.
- The swap-veto study (window NLI cannot reach 0.90 precision on gold pairs).

## Stages (code: `models/src/cleartext/learned_words/`)

1. **Eligibility** (`eligibility.py`): complex-word classifier (CWI 2018
   Train; threshold = best F1 on CWI Dev; reported on CWI Test) and a term
   classifier (NIST CSRC, US Courts and MeSH single-word terms vs. words with
   attested simpler substitutes; threshold 0.5). Eligible = complex and not a
   term.
2. **Candidates** (`features.py`): WordNet relations (v7 generator), masked
   BERT (LSBert paired input), plainlanguage.gov pairs.
3. **Ranker + no-swap gate** (`decide.py`): gradient boosting on BenchLS (minus
   NNSeval's instances, LexMTurk counts attached) and TSAR-2022 trial.
4. **Per-edit checkers**: word swaps (`decide.py`; rated SWORDS dev + LS
   train) and sentence splits (`splits.py`; WikiSplit human splits vs.
   labelled synthetic corruptions).

## Splits

- Fit and thresholds: train only (BenchLS-train, TSAR trial, SWORDS dev,
  CWI Train/Dev, WikiSplit tune/validation), out-of-fold where needed
  (5 folds grouped by target word).
- Test, read once after thresholds are written: TSAR-2022 test (373),
  NNSeval (239), SWORDS test (762 targets), CWI 2018 Test, WikiSplit test.
  TSAR test and SWORDS test were used before in this repo (v7 exploratory
  numbers; the swap-veto window calibration), never by these models.
- Held out: Joshua's 60. New human marks: the round-2 rating page.

## Thresholds (fixed here)

- Word swap = eligible and gate ≥ tg and checker ≥ tc. (tg, tc) on a 0.05
  grid: the most out-of-fold train swaps with precision against LS gold
  ≥ 0.90 and ≥ 30 swaps; ties to higher thresholds. If unreachable, the
  highest precision at ≥ 5% coverage, and the report says so.
- Split checker: the highest validation recall of human splits with
  specificity ≥ 0.90 on corrupted splits.

## Reported

- Per-edit precision (swaps whose word is in the gold list; TSAR also with
  ≥ 2 annotators), coverage, Wilson 95% intervals, on TSAR test and NNSeval;
  checker AUC and precision on SWORDS test; split-checker test results by
  corruption type.
- Ablations, each stage on/off: eligibility off / complex-only; candidates
  WordNet only, BERT only, WordNet+BERT; gate off; LSBert-style ranker;
  checker off; checker replaced by window NLI ≥ 0.95.
- Outside baseline: an LSBert-style reimplementation (rank average of masked
  probability, contextual cosine, loss change and frequency) next to the
  official TSAR-2022 English numbers (LSBert baseline ACC@1 0.5978).
- Joshua's 60: per item, abstain / same replacement (known mark) / different
  replacement (no mark). Known-label precision and bad swaps avoided.
- Split checker: decisions on the 60 F2 splits of the first rating page
  (part 2) are committed in `split-checker-precommit.json` before those
  marks exist.

## Adoption

The new word stage is not adopted by this study. Gold-list precision
underestimates acceptability (lists are incomplete) and public data are
general-domain. Adoption on top of F2 needs Joshua's round-2 marks: a Good
rate ≥ 90% (and the Wilson lower bound reported), with F2 otherwise
unchanged. Until then F2 stays the shipped default.
