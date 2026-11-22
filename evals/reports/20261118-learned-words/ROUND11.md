# Round 11 protocol (written November 21, 2026, before any round-11 swap)

Joshua chose this direction after rounds 9 and 10 failed their gates.
Instead of filtering the stage's swaps, **limit what the generator may
propose** to words that share the target's sense in context.

## Evidence from the development rows (371 rated swaps, not fresh)

- The round-9 sense table disambiguates the target in its sentence.
  `sense_t_cov` is the target's probability mass on senses the replacement
  also has.
- Arm-b swaps with `sense_t_cov` ≥ 0.25:
  - 92 of 189;
  - 68% Good, 13% Changes meaning, 18% Not simpler;
  - all arm-b rows by comparison: 61%, 21% and 18%.
- Cut-offs of 0.1 and 0.5 gave nearly the same split, so 0.25 is a
  middle value, not a tuned optimum.
- The sense limit reduces meaning errors but not Not simpler. The arm-s
  filter does the opposite: on round 8's fresh text it removed 10 of 11 Not
  simpler and 1 of 8 Changes meaning. **The prediction is that they
  combine.**
- This round differs from a filter: when the top candidate is outside the
  sense, the ranker can pick the best candidate inside it.

## Arms (one featurization per target, rated together)

| Arm | Candidates | Then |
|---|---|---|
| b | all (current stage) | simplicity margin ≥ 0.1 |
| s | all | margin, then the round-8 arm-s filter at its committed threshold |
| g | only those with in-context sense cover ≥ 0.25 | margin |
| **gs** | sense-limited | margin, then the arm-s filter |

- The ranker, no-swap gate, checker and per-passage limits are unchanged.
  The sense table is round 9's (sha256 `443ef2bf…`). The arm-s model and
  threshold are round 8's, unchanged.
- Candidates with no WordNet sense, such as many BERT-only multi-word
  outputs, are outside the sense and are dropped in arms g and gs.

## Fresh window

- **Panel:** round 9's specification (`r9 panel`), built now for the first
  time:
  - 45 passages per source from the RFC, Federal Register and openFDA
    technical corpus;
  - each passage is 4 clean consecutive sentences;
  - 8-gram exclusions as in ROUND9.md.
- **Rating:** three fresh blind raters (general-purpose Opus subagents)
  with the round-5/8 brief verbatim. Each sees every arm's swaps, in its
  own shuffle, plus 20 old round-5 items as a consistency check.
  - **Primary mark:** the majority of the three; when all three differ,
    the most severe mark counts.
- **Success (arm gs, majority marks):**
  - at least 78% Good;
  - Changes meaning at or below 15%;
  - at least 40% as many swaps as arm b.
- **Also reported:** every arm, per-rater results, agreement, and Wilson
  95% intervals.
- **Results:** `round11.json`. Nothing ships; Joshua's marks remain the
  adoption decision.
