# Round 9 protocol (written November 21, 2026, before any round-9 model or swap)

Round 8's filter learned "is it simpler" but not "does it keep the meaning":
it removed 10 of 11 Not simpler and 1 of 8 Changes meaning on fresh text.
Round 9 adds signals aimed at meaning, from an analysis of the 371 rated
development swaps (rounds 3–5 and the round-8 window).

## What the analysis found

By one reader's tally of the 137 Changes meaning marks:

- about 55%: the replacement is a synonym from a different sense, or a looser
  or more general word (recovered→gone back, regime→government,
  granted→gave, arch→structure);
- about 12%: connotation or intensity shifts (adored→loved, urge→want);
- about 10%: the verb frame or grammar breaks (referred to→called to,
  struggle with→trouble with);
- about 12%: a fixed term breaks (clinically significant, finance lease);
- the rest look like rater noise: on word pairs rated more than once, marks
  agree 88 of 114 times (occurred→happened: 5 Good, 2 Changes meaning).

Crude WordNet and parse indicators, added to the 24 stage features in grouped
cross-validation, raised AUC for Changes meaning from 0.77 to 0.80 and for
Good from 0.76 to 0.81. These indicators were designed after reading the
errors, so the gain is optimistic. The fresh window is the test.

## New features (13, fixed now)

**Sense match**
[`cleartext/learned_words/senses.py`](../../../models/src/cleartext/learned_words/senses.py):

- **Sense vectors.** Each WordNet sense gets a vector from BERT-base (the
  stage's own model): the sum of its last 4 layers, averaged over SemCor
  tokens tagged with that sense. Each vector is blended with the vector of
  the lemma in the sense's WordNet example or gloss, weighted
  n / (n + 2), so every sense has one.
- **Disambiguation.** The target is disambiguated in the original sentence,
  and the replacement in the swapped sentence: a softmax of cosine scores
  over the lemma's senses, temperature 0.05.
- **Features:**
  - the probability that both words express the same sense;
  - each side's probability mass on shared senses;
  - whether the target's top sense is one of the replacement's senses;
  - cross-fit: how well the replacement's vector in context fits the
    target's top sense, minus how well the target's own vector fits it;
  - whether the replacement has any senses;
  - whether the replacement is a hypernym (up to 2 levels) of the target's
    top sense.

**Parse stability:** re-parse the swapped sentence and compare the replaced
token with the target. The features are changes in part of speech,
dependency label, head lemma and children's labels; the number of tokens
whose dependency or head changed (capped at 5); and a multi-word
replacement.

Idea 3 from the analysis (back-substitution with a masked LM) is dropped. A
single masked slot gives the same context before and after the swap, so it
would repeat the existing `mask_logp`. Cross-fit replaces it.

## Model (arm w)

- **Rows:** all 371 development swaps. Label 1 = Good. Joshua's 60, the
  round-2 swaps and the locked test split are not used.
- **Features:** the 31 arm-s features plus the 13 above, 44 in total.
- **Classifier:** the same standardized logistic regression, C = 0.1. No
  model search.
- **Threshold:** keep the top 55% of arm-b swaps by out-of-fold score.
  Fable's planning estimate was 80–85% Good at about half the swaps, so
  arm w aims at a smaller, cleaner set than arm s (65%).

## Cross-validation go/no-go (fixed now)

Twenty repeats of nested 5-fold cross-validation, grouped by target lemma.
Each outer fold sets its threshold from inner out-of-fold scores on its own
training rows. Arm s (31 features) is refitted on the same rows and folds at
the same 55% keep, as the comparator. On the 189 arm-b rows (base Good rate
61%), arm w earns a fresh window only if all of these hold:

1. Paired over repeats, its Good rate is at least 3 points above arm s.
2. It removes at least half of arm b's Changes meaning swaps.
3. Its Good rate is at least the base plus 10 points.

If any condition fails, no fresh text is spent and the report says so.

## Fresh window (only if w passes)

- **Passages:** 45 per source from the round-6 technical corpus (RFC
  2015–19, Federal Register 2022–23, openFDA 2022). Each is four consecutive
  clean sentences from one document, picked by a seeded shuffle (seed
  20261126). Passages are dropped if they share a word 8-gram with:
  - the round-4, round-5 or round-8 panels;
  - any development swap sentence;
  - the round-7 rewrite corpora.

  The corpus already excludes 8-gram overlaps with test-set v3. No arm-s or
  arm-w feature uses corpus counts.
- **Pre-commitment:** the model, sense table and threshold hashes go in
  `r9-precommit.json`, committed before any swap is generated. The arm-s
  model and threshold are unchanged from round 8.
- **Rating:** three fresh blind raters, each a separate general-purpose
  Opus subagent with the round-5/8 brief verbatim. Each sees arms b, s and w
  together, in its own shuffle, with no scores, plus 20 old round-5 items as
  a consistency check.
  - **Primary mark:** the majority of the three. When all three differ, the
    most severe mark counts (Changes meaning, then Not simpler).
  - Per-rater results and pairwise agreement are reported too.
- **Success (majority marks):** arm w is at least 78% Good, Changes meaning
  is at or below 15%, and arm w keeps at least 45% of arm b's swaps. Wilson
  95% intervals are reported.
- **Secondary, not a decision:** arm s on the same window; and arm w against
  arm s at matched keep, by ranking arm s's scores.

Joshua's marks remain the adoption decision. Nothing ships from this round.
