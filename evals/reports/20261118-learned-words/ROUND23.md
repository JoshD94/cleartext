# Round 23 protocol (written November 22, 2026, after round 22's marks, before any round-23 swap)

Round 22's arm qg reached 91.7% Good at 48% of arm b's yield on news and
encyclopedic text, but missed the meaning bar by one swap: 8.3% against 8%.
Frozen p50 there reached 90.9% at 44%, against 26% yield on round 21's
genre passages. Windows move these numbers as much as the arms do. Round 23
is a pure replication of round 22's arms on a new genre window. **Nothing
is refitted and no threshold changes.**

- **Models and thresholds:** round 22's qg and rg with their genre
  thresholds, round 21's q50 and round 19's p50, all exactly as in
  [r22-precommit.json](r22-precommit.json). `r23-precommit.json` records
  their sha256.
- **Gate:** round 22's check passed, and all four model files are
  unchanged. `r23 precommit` refuses otherwise.
- **Panel:** news and encyclopedic only, seed 20261223: 45 Wikinews and 60
  featured-article leads, one passage per document.
  - No document from round 21's, round 22's or either Joshua window is
    used.
  - A passage is dropped if it shares a word 8-gram with any earlier panel
    or window (both Joshua windows included), any rated development
    sentence, test set v3's panel file (exclusion only) or the round-7
    rewrite corpora.
  - The genre corpora are nearly used up, so fewer passages may be found.
- **Generators and arms:** as in round 22: b, gs, p50, q50, rg and **qg**.
  Filters apply before the per-passage limits, and the a/an realization
  applies.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all arms
  mixed in each rater's own shuffle, and 20 old items. The majority mark
  counts; a three-way split counts as the most severe mark.
- **Success (arm qg, majority marks; round 22's bars):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals and per source;
  - unseen pairs;
  - per-rater results;
  - swaps per 100 sentences;
  - qg and p50 pooled over rounds 22–23.
- **Results:** `round23.json`. Nothing ships, and Joshua's page is unchanged.
