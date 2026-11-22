# Joshua's blind marking page (written November 22, 2026, before the supplementary window is generated)

[ROUND21.md](ROUND21.md) fixed the page rule:
- the best arm is q50 if it met all its round-21 bars, otherwise p50;
- about 60 best-arm swaps, plus about 40 arm-b swaps that the best arm
  rejects;
- all from a window that no AI rater sees.

q50 missed its bars ([round21.json](round21.json)), so **the page uses
p50**, round 19's frozen model and threshold.

## Supplement

Round 21's Joshua window will fall short of 60 p50 swaps, for two reasons:
- the Federal Register corpus had only 16 unused documents (30 planned);
- p50 makes about 6 swaps per 100 sentences on news and encyclopedic text.

A partial count after 50 of its 124 passages found 10 p50 swaps.

One supplementary window is therefore added (`evals/learned_words_joshua.py`,
[joshua-precommit.json](joshua-precommit.json)). It has 40 openFDA passages,
seed 20261232. No document from any earlier panel or window is reused,
including round 21's, Joshua's and round 22's. A passage is dropped if it
shares a word 8-gram with any of them, any rated development sentence, test
set v3's panel file (exclusion only) or the rewrite corpora.

## Page

- **Selection:**
  - up to 60 p50 swaps and 40 arm-b swaps that p50 rejects, drawn from both
    Joshua windows;
  - one seeded shuffle (seed 1231);
  - the a/an realization applies;
  - no scores, arm labels or sources are shown.
- **Location:** `~/Documents/cleartext-rating/rate-learned-swaps-r21.html`.
  Marks download as `cleartext-marks-learned-swaps-r21.csv`, and the key is
  in `joshua-r21/key.json`. [joshua-page.json](joshua-page.json) records the
  counts only.
- **Arm b's Good rate:** it can be rebuilt from the two groups, weighted by
  the window counts in the key.
- **Use of the marks:** Joshua's marks decide adoption. They are never used
  to train, tune or choose anything.
