# Round 18 protocol (written November 22, 2026, after round 17's marks, before any round-18 swap)

Round 17's arm p40 met its bar on one fresh window: 96.3% Good (52/54) at
44% of arm b's yield. But fresh windows vary a lot:

| Window | Arm b Good |
|---|---:|
| Round 11 | 66% |
| Round 12 | 56% |
| Round 13 | 66% |
| Round 17 | 63% |

Round 12's harder window held round 12's arm p to 84%. Round 18 is a pure
replication. **Nothing is refitted and no threshold changes.**

- **Models and thresholds:** round 17's refitted arm p with its p40 and p50
  thresholds, plus the frozen round-12 arm p, all exactly as in
  [r17-precommit.json](r17-precommit.json). `r18-precommit.json` records
  their sha256.
- **Gate:** round 17's check passed, and both model files are unchanged.
  `r18 precommit` refuses otherwise.
- **Panel:** round 12's rule at full size. 45 passages per source (135),
  seed 20261218.
  - No document used in rounds 11–13 or 17 is reused.
  - A passage is dropped if it shares a word 8-gram with the round-4, 5,
    8, 11, 12, 13 or 17 panels, any rated development sentence, or the
    round-7 rewrite corpora.
- **Generators and arms:** as in round 17. The arms are b, s, gs, p, **p40**
  and p50, and filters apply before the per-passage limits.
- **Rating:** three fresh blind Opus raters with the brief verbatim, all
  arms mixed in each rater's own shuffle, and 20 old round-5 items. The
  majority mark counts; a three-way split counts as the most severe mark.
- **Success (arm p40, majority marks; the round-17 bars):**
  - Good ≥ 90%;
  - Changes meaning ≤ 8%;
  - at least 40% as many swaps as arm b.
- **Also reported:**
  - every arm, with Wilson intervals;
  - per-rater results;
  - swaps per 100 sentences;
  - distinct and unseen pairs;
  - results by source;
  - pooled results over rounds 17 and 18.
- **Results:** `round18.json`. Nothing ships.
