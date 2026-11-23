# Round 24 protocol (written November 22, 2026, after round 22's marks, before the round-24 CV gate or any round-24 swap)

## Why

p50 (round 19, frozen) is the best arm. Over rounds 20–22 it kept 132 swaps
(120 Good, 10 Changes meaning, 2 Not simpler) and rejected 177 arm-b swaps,
79 of which raters marked Good. Two gaps remain.

1. **Frame errors on well-rated pairs.** The replacement is fine alone but wrong
   in a fixed frame or term: "orally administered" → given (four times over
   the rounds), "terminated for default" → ended, "has operated Hawk 127" → run,
   "induces its own metabolism" → causes, "occurred in 5.3% of patients" →
   happened, migrate → move. In the 1,454 rated rows, 40 Changes-meaning marks
   sit on pairs whose other rows are ≥ 70% Good over ≥ 3 rows. Pair priors cannot
   see the frame.
2. **Unseen pairs.** p50 keeps about 13% of unseen-pair technical swaps, at
   about 83% Good. On news and encyclopedic text about 57% of arm-b swaps
   use a pair never rated, and p50 rejected clearly good ones (aircraft→plane,
   deceased→dead, cope→deal, commanded→led).

## Why the earlier attempts failed, and what is different here

- Round 5's frame probe masked every word within 3 of the slot (AUC 0.64);
  round 6's and round 15's corpus statistics used fixed windows in a technical
  corpus. The corpus signal tracked the corpus genre (0.56 off-genre) and the
  round-15 veto was overfit to its rows (round 16). Round 14's stage meaning
  signals added under a point because the pair prior already decides those rows.
- Round 24's main signal is **anchored on the dependency frame and scored
  relative to the pair**: a learned prior over (target lemma, replacement lemma,
  dependency relation, neighbour lemma), e.g. (administer, give, child:advmod,
  orally), shrunk to the pair's own rate. Its feature is the most negative
  frame deviation from the pair prior. It is a learned rate with a fitted
  weight, not a veto, and it needs no corpus, so it carries across genres.
- Two further learned inputs are new:
  - **masked-LM frame features** on the same dependency neighbours
    (`cleartext.learned_words.depframe`): how much less BERT predicts each
    neighbour once the replacement is in place, and the slot preference in
    the bare frame;
  - **public pair priors** for unseen pairs (`cleartext.learned_words.pairlex`):
    SimplePPDB++'s relative complexity and PPDB 2.0 score for the exact pair
    (Maddela and Xu 2018, MIT license; 1.63M single-word rules), the human
    word-complexity lexicon from the same release (about 15,000 words, 11
    raters each), and a spelling-similarity ratio (centred → centered). An
    unseen-pair indicator interacts with the arm-s score, the complexity score,
    the lexicon difference and `ctx_cos`.
- Human-gold attestation (does the exact pair appear in BenchLS, LexMTurk, TSAR
  or SWORDS gold) was checked first and dropped: on the rated rows, attested
  pairs were as often Good as unattested ones (BenchLS 60.6% vs 59.6%).
  WordNet relation type (synonym, hypernym, hyponym, sister) did not separate
  meaning errors either (AUC 0.48–0.62).

## Development look (all 1,454 rows were seen by earlier design steps)

[`r24-explore.json`](r24-explore.json) holds the diagnostics. Passage-grouped
nested CV (5 repeats × 5 folds) on all rows, each arm keeping 50% of the 870
technical arm-b rows; "genre" is the 143 news and encyclopedic arm-b rows of
rounds 21–22; "frame errors" are held-out Changes-meaning rows whose pair is
≥ 70% Good over ≥ 3 training-fold rows (26.8 such rows per run).

| Variant | Tech Good | Frame errors kept | Unseen kept (Good), tech | Genre Good | Unseen kept (Good), genre |
|---|---:|---:|---:|---:|---:|
| arm p refitted (r50) | 91.8% | 25.4 | 22.2 (18.8) | 94.4% | 12.4 (10.8) |
| arm q refitted | 91.6% | 25.6 | 21.4 (17.0) | 94.7% | 13.6 (11.6) |
| + frame prior | 92.7% | 19.2 | 29.6 (23.4) | 95.1% | 14.2 (12.2) |
| + masked-LM frame features | 92.7% | 21.4 | 24.4 (20.6) | 96.9% | 14.0 (13.4) |
| + frame prior + pair lexicon | 93.2% | 19.6 | 27.4 (23.2) | 94.8% | 13.6 (11.8) |
| **+ frame prior + masked LM + pair lexicon (arm f)** | **93.0%** | **18.4** | **29.6 (24.4)** | **95.9%** | **14.0 (13.2)** |
| arm f + unseen interactions | 93.2% | 18.6 | 26.8 (22.2) | 95.9% | 12.6 (11.8) |

- **The frame prior is the main gain:** about +1 point and a quarter fewer
  frame errors kept. Its smoothing (1, 2 or 4 pseudo-counts) and key set
  (neighbour lemma, relation only, both) changed results by under 0.5 points;
  the first-written version (neighbour lemma, 2 pseudo-counts) is used.
- **Single masked-LM features are weak** (best: bare-frame slot preference, AUC
  0.57 for Good vs Changes meaning, 0.60 inside well-rated pairs; neighbour
  drops 0.42–0.49), but together they add on genre rows.
- **Pair lexicon:** SimplePPDB++ complexity separates Good from Not simpler at
  AUC 0.77 and the human lexicon difference at 0.81, but neither separates
  meaning errors inside well-rated pairs.
- **Unseen pairs stay hard.** No variant keeps unseen pairs at ≥ 90% in CV:
  arm f keeps 33% more of them than r50 at about the same rate (85% vs 85%).
  Unseen interactions, and an unseen threshold set for 85–90% inner precision,
  did not help: realized unseen precision stayed at 79–86%.
- **A genre threshold** (arm f keeping 40% of the genre rows): 93.0% Good with
  6.3% meaning on 60 of 143 genre rows, unseen 18 kept at 86%; arm p with the
  same rule reached 93.7% on 57 rows.

All of this used rows that every earlier round also used, so the gate below is
optimistic; the fresh window decides.

## Arms

- **f (primary):** `ArmF` with the frame prior, the masked-LM frame features
  (`dep_max_drop`, `dep_top_drop`, `dep_head_drop`, `frame_slot_d`) and the six
  pair-lexicon features, plus arm p's four features and round 21's two context
  features; no unseen interactions. Standardized logistic, C = 0.1, refitted
  on all 1,454 rated rows (rounds 3–22). Priors and frame rates leave the row
  out. Its threshold keeps 50% of the technical arm-b rows by passage-grouped
  out-of-fold score (mean over 10 repeats), and applies to every passage.
- **fg (secondary):** the same model with a genre threshold keeping 40% of the
  genre arm-b rows (qg's rule) on news and encyclopedic passages, and f's
  threshold on openFDA. Rated only if its genre check passes.
- **Frozen comparators:** p50 (round 19) and qg (round 22, its genre threshold
  on genre passages and its technical threshold on openFDA), and arm b.
- The a/an realization applies to all arms, as since round 21.
- **Excluded from everything:** Joshua's 60, round 2, `stacker-joshua60.json`,
  his round-21 pages and keys, and the locked test split (test-set v3's panel
  file is read for exclusion only).

## CV gate ([`r24-cv.json`](r24-cv.json))

10 repeats × 5 passage-grouped folds on all 1,454 rows. In each outer split
the threshold keeps 50% of the training folds' technical arm-b rows by inner
out-of-fold score. r50 is arm p refitted in the same folds. Go only if all hold:

1. arm f reaches ≥ 92% Good on held-out technical arm-b rows;
2. it keeps ≥ 45% of them;
3. its paired gain over r50 is ≥ +0.5 points;
4. it keeps at most 0.85 × r50's frame errors (as defined above);
5. on technical and genre arm-b rows whose pair is unseen in training, it
   keeps at least as many as r50, at a Good rate no more than 2 points below
   r50's.

fg is rated only if, in addition, its genre-threshold CV reaches ≥ 90% Good at
≥ 35% of the genre arm-b rows kept. If the gate fails, no fresh text is spent.

## Fresh window (only if the gate passes)

- **Panel:** seed 20261224, one passage per document, four consecutive clean
  sentences:
  - 45 Wikinews and 60 Wikipedia featured-article leads from **new titles**
    (positions 220–520 of round 21's seeded title shuffle, fetched November 22
    with the same code into `genre/news2` and `wiki2`; no earlier window can
    contain them);
  - 45 unused openFDA labels.
  - A document used by any earlier window (rounds 11–22, both Joshua windows)
    is skipped. A passage is dropped if it shares a word 8-gram with any
    earlier panel or window (rounds 4, 5, 8, 11–22, Joshua's two windows), any
    rated development sentence, test-set v3's panel file (exclusion only) or
    the round-7 rewrite corpora.
- **Generator:** round 11's (all candidates) with arm b's margin; filters apply
  before the per-passage limits.
- **Rating:** three fresh blind Opus raters, each with the round-9 brief
  verbatim (paths and counts filled in), all arms mixed in its own shuffle
  with no scores, plus 20 old round-5 items. The majority mark counts; a
  three-way split counts as the most severe mark.
- **Success (arm f, majority marks):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Secondary goals (arm f, reported against p50 on the same window):**
  - frame errors (Changes meaning on a pair ≥ 70% Good over ≥ 3 pool rows)
    no more than p50's;
  - at least 1.25 × p50's unseen-pair swaps (pairs absent from the 1,454 rows),
    with ≥ 90% of them Good.
- **Also reported:** every arm with Wilson intervals, per source, per rater,
  swaps per 100 sentences, distinct pairs, rater agreement and consistency.
- **Results:** `round24.json`. Nothing ships; Joshua's page is unchanged.

## Round 23

Round 23's pre-registered replication (frozen qg, rg, q50, p50 on a genre
window) is folded in rather than run separately: round 24 rates frozen qg and
p50 on a new genre window, which answers its question (does qg's genre
threshold hold up?) without spending round 23's remaining genre documents.
Round 23's own pre-commitment stays unused; its arms rg and q50 are not
rated, because q50 never differed from p50 on fresh text and rg is qg's
ablation.
