# Round 20 protocol (written November 22, 2026, after round 19's marks, before any round-20 swap)

Round 19's arm p50 met its bar: 94.9% Good (56/59) at 53% of arm b's yield.
Round 18 showed that a single success can fail to replicate: round 17's
arms dropped from 96% to 88%. Round 20 is a pure replication of round 19's
arms. **Nothing is refitted and no threshold changes.**

- **Models and thresholds:** round 19's refitted arm p with its p50
  (primary) and p40 thresholds, plus the frozen round-12 arm p, all exactly
  as in [r19-precommit.json](r19-precommit.json). `r20-precommit.json`
  records their sha256.
- **Gate:** round 19's check passed, and both model files are unchanged.
  `r20 precommit` refuses otherwise.
- **Panel:** 45 passages per source (135), seed 20261220.
  - No document used in rounds 11–13 or 17–19 is reused.
  - A passage is dropped if it shares a word 8-gram with those panels or
    the rounds 4, 5 and 8 panels, any rated development sentence, or the
    round-7 rewrite corpora.
- **Generators and arms:** as in round 19. The arms are b, s, gs, p, p40
  and **p50**, and filters apply before the per-passage limits.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all
  arms mixed in each rater's own shuffle, and 20 old items. The majority
  mark counts; a three-way split counts as the most severe mark.
- **Success (arm p50, majority marks; round 19's bars):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals;
  - per-rater results;
  - swaps per 100 sentences;
  - unseen pairs;
  - results by source.
- **Results:** `round20.json`. Nothing ships.
