# Round 7 protocol (written November 21, 2026, before any round-7 model or swap)

Round 6 showed that corpus statistics do not detect the remaining errors.
Round 7 asks a different question: **when human editors rewrote hard text
into plain language, did they replace this word, in this context, with this
replacement, or did they keep it?**

## Training data (public, none from any rated round)

| Source | Domain tag | Pairs | License |
|---|---|---:|---|
| WikiAuto train + validation (`GEM/wiki_auto_asset_turk`) | general | 503,801 sentence pairs | CC BY-NC 3.0 |
| Contract simplification (same dataset, `test_contract`) | technical | 659 sentence pairs | as published with the dataset |
| Cochrane abstracts → plain-language summaries (`GEM/cochrane-simplification`, train + validation) | technical | 3,979 documents | CC BY 4.0 |

- SHA-256 of each file is pinned in `evals/learned_words_rewrite.py`.
- Cochrane documents are sentence-aligned by content-lemma overlap (Jaccard at
  least 0.35, best target for each source sentence).
- **Excluded before counting:** every pair where either side shares a word
  8-gram with, or exactly equals, any of: test-set-v3 passages (dev and test),
  round-4/5 panels, OneStop dev, Joshua's 60, round-2 swaps, the replay set,
  TSAR test, NNSeval, SWORDS test. The locked OneStop test file is not opened.
  The number dropped is recorded.
- CC BY-NC data means a model trained on it is for this class project, not a
  commercial release.

## Model (`rewrite.py`, arm w)

- **Events:** each content word of a source sentence is *kept*, *replaced* (a
  1:1 aligned word with a different lemma) or *dropped*.
- **Features:** cross-fitted counts of those events: how often editors keep
  or replace the word, overall and in technical text, with this previous or
  next word; how often this exact replacement was made, and how often the
  reverse was; plus frequency, length and WordNet relations. A row's own
  fold is always subtracted from its counts.
- **Labels:**
  - 1: an editor replaced w with a more frequent r.
  - 0 (keep): an editor kept w; r is a random more-frequent WordNet or
    plainlanguage.gov candidate.
  - 0 (wrong word): an editor replaced w, but with a different word than r.
- **Weights:** general and technical text count equally; classes are
  balanced within each.
- **Threshold (fixed now):** keep 70% of out-of-fold technical-domain human
  replacements. Swaps scoring below it are dropped.

## Replay go/no-go (fixed now)

On the 153 arm-b replay swaps (99 Good, 23 Not simpler, 31 Changes meaning),
arm w earns a fresh window only if all of these hold:

1. Good rate among kept swaps is at least 70% (baseline 65% + 5 points).
2. It keeps at least 60% of the swaps.
3. It removes at least a third of the 54 errors (Not simpler + Changes
   meaning).

The replay set is never used to fit or tune. If w fails, no fresh text is
spent, and the report says so.

## Fresh window (only if w passes)

Same rules as round 6: unused test-set-v3 dev windows, thresholds and hashes
pre-committed before any swap is generated, a fresh blind rater with no
scores, and success at **at least 78% Good, at least 60% of arm b's swaps
kept, and Changes meaning at or below 15%**, with Wilson 95% intervals.
