# Round 25 protocol (written November 22, 2026, after round 24's marks, before any round-25 swap)

## Why

Round 24's arm f met its pre-registered bars on a fresh window
([round24.json](round24.json)): 94.4% Good (68/72, CI 87–98), 2.8% Changes
meaning, 55% of arm b's swaps. Frozen p50 reached 94.1% (64/68) at 52%. But
the window was easy and says little about the two gaps:

- **Arm b reached 75.6% Good,** against 63–66% on every earlier window, and it
  made only one frame error on a well-rated pair. f and p50 both kept it
  ("occurred in 4 (1.3%) of 297 patients" → happened).
- **Unseen pairs:** arm b made 41 swaps with a pair absent from the 1,454 rows
  (28 Good). f kept 7 (all Good), p50 8 (7 Good), qg 14 (13 Good). f's
  secondary unseen goal (1.25 × p50's count) was missed.
- Since only round 24's arms are under test, a second window with more
  technical text, where frame errors occur, is the informative next step.
  Nothing is refitted.

A development check after round 24 (passage-grouped CV on the 1,454 rows, arm
f) shows why a lower unseen threshold is not tried: on rounds 17–22's unseen
arm-b rows, f's score ≥ 0.78 keeps 29 at 93% Good, ≥ 0.74 keeps 36 at 89%, and
≥ 0.66 keeps 50 at 86%.

## Arms (all frozen, as in [r24-precommit.json](r24-precommit.json))

- **f (primary, replication):** round 24's model and technical threshold on
  every passage.
- **fg:** f with its genre threshold on news and encyclopedic passages.
- **p50** (round 19) and **qg** (round 22), frozen; **arm b**.
- `r25 precommit` refuses unless round 24's gate passed, round 24 is scored
  with f meeting its bars, and every model file matches round 24's
  pre-commitment ([r25-precommit.json](r25-precommit.json)).

## Window

- **Panel:** seed 20261225, one passage per document, four consecutive clean
  sentences: 40 Wikinews and 50 featured-article leads from the extended title
  range (`genre/news2`, `wiki2`), 50 unused openFDA labels and up to 20 unused
  Federal Register rules (the corpus is nearly used up).
  - Every document of every earlier window (rounds 11–24, both Joshua windows)
    is skipped. A passage is dropped if it shares a word 8-gram with any
    earlier panel or window (round 24 included), any rated development
    sentence, test-set v3's panel file (exclusion only) or the round-7 rewrite
    corpora.
- **Generator, rating and majority rule:** as in round 24 (three fresh blind
  Opus raters, the round-9 brief verbatim, 20 old items, most severe mark on a
  three-way split).

## Success (arm f, majority marks; round 24's bars)

- Good ≥ 90%;
- Changes meaning ≤ 8%;
- at least 40% as many swaps as arm b.

**Also reported:** the secondary goals as in round 24 (frame errors against
p50; unseen-pair swaps against p50 and their Good rate), every arm with Wilson
intervals, per source, per rater, and f and p50 pooled over rounds 24–25.
Results: `round25.json`. Nothing ships; Joshua's page is unchanged.
