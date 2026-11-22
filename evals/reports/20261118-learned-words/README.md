# Learned word stage (November 18–19, 2026)

A trained four-stage word simplifier, with no LLM and no rules-only decision.
The protocol and thresholds were fixed before fitting ([PROTOCOL.md](PROTOCOL.md)).
Raw text, candidate rows and predictions stay in `~/.local/state/cleartext/learned-words-20261118/`.
Code is in `models/src/cleartext/learned_words/`, and it runs through `scripts/learned-words`.

**Status: not adopted.** F2 stays the shipped default. Joshua's own blind marks decide adoption (≥ 90% Good).

**Best arm (November 22): arm p50.** It is round 12's 4-feature filter (arm-s score, sense cover, learned pair and target priors), refitted on all rated rows, keeping 50% of arm-b swaps. Round 19's model is frozen since then.
- Round 19: 94.9% Good (56/59), 3.4% Changes meaning, 53% of arm b's swaps. It met its pre-registered bar.
- Frozen replication in round 20: 90.0% (54/60), 8.3% meaning (bar ≤ 8%), 55% yield.
- **News and encyclopedic text** (rounds 21–22, frozen): 92.2% Good (47/51, CI 81–97) at 36% of arm b's yield. Precision holds; yield falls, because most genre pairs were never rated.
- Pooled over rounds 17–22: 92.0% Good (301/327, CI 89–95), 6.7% meaning, 49% yield.
- Round 22's qg (refitted on 1,377 rows, genre threshold): 91.7% (33/36) at 48% yield on genre text, 8.3% meaning (bar 8%).

Marks are a majority of three blind AI raters on fresh text: technical (RFC, Federal Register, openFDA) through round 20, then news and encyclopedic text in rounds 21–22. Joshua's blind page from a window no rater has seen is ready ([JOSHUA-PAGE.md](JOSHUA-PAGE.md)).

Best pre-registered arm per fresh window:

| Round | Arm | Swaps | Good (95% CI) | Changes meaning | Yield vs b | Arm b Good | Bar met |
|---|---|---:|---:|---:|---:|---:|---|
| 2–5 | stage | 33–98 | 54–71% | — | — | — | (single rater) |
| 8 | s | 24 | 62.5% (43–79) | 29% | 67% | 47% | no |
| 11 | gs | 38 | 94.7% (83–99) | 2.6% | 30% | 66% | no (yield) |
| 12 | p | 51 | 84.3% (72–92) | 15.7% | 48% | 56% | no |
| 13 | pn (frozen p + fallback) | 71 | 84.5% (74–91) | 14.1% | 62% | 66% | no |
| 17 | p40 (refit) | 54 | 96.3% (87–99) | 3.7% | 44% | 63% | **yes** |
| 18 | p40 (round 17's, frozen) | 52 | 88.5% (77–95) | 11.5% | 39% | 68% | no |
| 19 | p50 (refit) | 59 | 94.9% (86–98) | 3.4% | 53% | 66% | **yes** |
| 20 | p50 (round 19's, frozen) | 60 | 90.0% (80–95) | 8.3% | 55% | 63% | no (meaning) |
| 21 | p50 (frozen), news + encyclopedic | 18 | 94.4% (74–99) | 0% | 26% | 65% | no (yield) |
| 21 | q50 (refit, unseen-robust), whole window | 39 | 92.3% (80–97) | 5.1% | 35% | 63% | no (yield, unseen) |
| 22 | qg (refit, genre threshold), news + encyclopedic | 36 | 91.7% (78–97) | 8.3% | 48% | 65% | no (meaning) |

Rounds 14–16 failed their CV gates and spent no fresh text. Details are in
the round sections below. Round 21 also measured p50 on unseen pairs: about
13% of them kept, at 83% Good, against 93% on seen pairs
([r21-unseen.json](r21-unseen.json)).

## Stages

| Stage | Model | Training data | Check |
|---|---|---|---|
| 1a Complex word | gradient boosting on frequency, shape, WordNet, POS | CWI 2018 Train (threshold from Dev) | CWI Test F1 **0.800** (v7 detector 0.78) |
| 1b Term protection | gradient boosting on frequency, WordNet, BERT static-embedding PCA | 2,198 glossary terms (NIST, US Courts, MeSH) vs 799 words with attested simpler substitutes | Leave-one-glossary-out AUC: NIST 0.83, US Courts 0.82, MeSH 0.97 |
| 2 Candidates | WordNet relations + masked BERT (LSBert input) + plainlanguage.gov pairs | — | Gold candidate among candidates: TSAR test 98.1%, NNSeval 95.4% |
| 3 Ranker + no-swap gate | gradient boosting (62 features), then a gate on the top candidate | BenchLS − NNSeval (LexMTurk counts), TSAR trial | Ranker OOF AUC 0.85; gate OOF AUC 0.75 |
| 4a Swap checker | gradient boosting on window NLI, BERT cosines, LM scores, WordNet, frequency | rated SWORDS dev + LS train | SWORDS test AUC **0.88** (window NLI alone 0.73) |
| 4b Split checker | gradient boosting on NLI and parse features | WikiSplit human splits vs labelled synthetic corruptions | WikiSplit test AUC 0.90 (NLI alone 0.58) |

Thresholds came from out-of-fold train predictions by the pre-registered rule:
gate ≥ 0.40 and checker ≥ 0.90. Out-of-fold precision on train was 0.90 at
25% coverage.

## Per-edit precision on public test splits

The test splits are TSAR-2022 test (373 targets) and NNSeval (239 targets).
None of the models trained or tuned on them.
"Correct" means the swapped word is in the gold list (TSAR: at least 1 of 25
annotators). Gold lists are incomplete, so these numbers are lower bounds.
NNSeval's short lists are the clearest case: most of its "errors" read as fine,
e.g. gradually→slowly, various→different, contains→includes. Wilson 95% intervals in brackets.

| Configuration | TSAR swaps | TSAR precision | TSAR ≥ 2 annot. | NNSeval swaps | NNSeval precision |
|---|---:|---:|---:|---:|---:|
| **Full learned stage** | 36 (9.7%) | **94.4%** [82, 98] | 80.6% | 41 (17.2%) | 46.3% [32, 61] |
| Stage 1 off | 45 | 93.3% | 82.2% | 78 | 34.6% |
| Stage 1 complex-only (no term protection) | 45 | 93.3% | 82.2% | 45 | 51.1% |
| Stage 2 WordNet only | 25 | 100% [87, 100] | 84.0% | 28 | 53.6% |
| Stage 2 BERT only | 36 | 91.7% | 80.6% | 40 | 45.0% |
| Stage 3 gate off | 37 | 91.9% | 78.4% | 41 | 46.3% |
| Stage 3 LSBert-style ranker (unsupervised) | 30 | 80.0% | 70.0% | 30 | 46.7% |
| Stage 4 off | 188 (50%) | 79.8% | 70.7% | 134 | 53.0% |
| Stage 4 → window NLI ≥ 0.95 | 162 | 84.6% | 74.1% | 124 | 53.2% |
| All gates off | 370 | 74.1% | 64.1% | 238 | 44.5% |

- The checker (stage 4) does most of the work. The learned checker beats an
  NLI threshold by 10 points of TSAR precision.
- On rated SWORDS test substitutes, which have true human negatives, the
  checker at its threshold accepts 479 at **96.2%** precision and 19.6%
  recall. Window NLI ≥ 0.95 accepts 3,041 at 60.5%.
- Term protection changes almost nothing on TSAR, which is general news.
  It matters for technical text (below).

### Outside baseline (full coverage, TSAR-2022 English ACC@1)

| System | ACC@1 |
|---|---:|
| GPT-3 UniHD (best, LLM) | 0.810 |
| **This ranker (top candidate, no gates)** | **0.735** |
| MANTIS (best non-LLM entry) | 0.657 |
| LSBert official baseline | 0.598 |
| LSBert-style reimplementation (same candidates) | 0.539 |
| TUNER baseline | 0.340 |

Official numbers come from the TSAR-2022 overview (Saggion et al. 2022,
Table 5). TSAR test was read before in this repo, for v7's exploratory numbers
and the swap-veto calibration, but never by these models.

## Joshua's 60 marked swaps (held out, technical domain)

The pipeline ran with no gold and no candidates supplied. It saw the 60 targets
from the Nov 17 sample, which the old word stage had swapped: 13 Good, 36
Changes meaning, 11 Not simpler.

| Configuration | Abstain | Same word as before | New word (unmarked) | Known-label precision |
|---|---:|---:|---:|---:|
| **Full** | 54 | 2 (both Good) | 4 | **2/2** |
| Stage 4 off | 34 | 11 (4 Good, 7 meaning) | 15 | 4/11 |
| All gates off | 0 | 26 (6 Good) | 34 | 6/26 (23%) |

- The full stage swaps none of the 24 nouns.
- It keeps erroneously→mistakenly and terminate→end, both marked Good before.
- Its 4 new words are grateful→glad, concluded→found and decreased→reduced
  (twice). They have no marks yet.
- It repeats none of the 47 swaps marked bad. It abstains on 44 of them and
  picks a different word on 3 (decreased ×2, concluded), whose new words are
  unmarked. Of the 13 marked Good, it keeps 2, changes 1 (grateful) and
  abstains on 10.
- The old stage's estimated Good-rate was about 42%. A real Good-rate for the
  new stage needs the round-2 marks.

## Round 2: new swaps on F2's dev outputs (for Joshua's blind marks)

- **Setup:** the learned stage ran on F2's unchanged output for all 100
  test-set-v3 dev passages. It featurized the 10 best eligible words per
  passage and allowed at most 3 swaps per passage, one per sentence.
- **Output:** 98 swaps in 70 passages, 9 of them in the 16 simple controls
  ([dev-application.json](dev-application.json)).
  - By part of speech: 54 verbs, 25 adjectives, 11 adverbs, 8 nouns.
  - None repeats one of the 60 marked swaps.
  - Mean Zipf gain is +0.74. 88% of replacements are more frequent than the
    word they replace, and they are 2.6 letters shorter on average.
- **Page:** `~/Documents/cleartext-rating/rate-learned-swaps.html`, with the
  key in the state directory's `rating/key.json`. Marks download as
  `cleartext-marks-learned-swaps.csv`.
- **Known risk:** the gold data reward "simpler" only implicitly. A few swaps
  (evaluates→assesses, intentionally→deliberately) may be marked Not simpler.
  The model was not changed after these swaps were seen.

## Round 3: is the replacement simpler? (November 19)

**Hypothesis:** many round-2 failures are same-difficulty synonyms, because the
ranker learned substitutability, not simplicity. A learned word-difficulty
margin should remove them.

- **Model** (`learned_words/simpler.py`): two gradient-boosting regressors
  on word-type features (frequency, length, WordNet, BERT static embedding).
  One is trained on CompLex 2.0 ratings, the other on CWI 2018 annotator
  shares. Margin = difficulty(target) − difficulty(replacement), in z units.
- **Held-out words** ([simpler-check.json](simpler-check.json)), pair AUC:
  CompLex 0.90 and CWI 0.79 on test words. Median margin: TSAR human
  simplifications +0.81, SWORDS plain synonyms −0.31.
- **The pre-registered margin rule failed.** Its precision target on random
  word pairs gave a margin of 1.6, which passes only 14% of TSAR-test human
  simplifications and removes 28 of the stage's 34 correct TSAR swaps.
- **Revised rule R2** ([simpler-precommit-r2.json](simpler-precommit-r2.json)),
  committed before any round-2 comparison or round-3 swap. Margin = **0.1**,
  chosen for best balanced accuracy between train-side human simplifications
  and SWORDS-dev synonyms. The revision was prompted by TSAR-test numbers, so
  TSAR test under R2 is not clean: it passes 73% of gold simplifications and
  17% of SWORDS-test synonyms, and the stage's TSAR precision stays at 94%
  (33 swaps, down from 36).
- **Fresh swaps:** F2's output on the 181 OneStopEnglish dev sentences, which
  the word stage had never seen. Arm A is the round-2 stage; arm B adds the
  margin.
- **Rater:** a fresh blind subagent (Claude, not human). It saw only the
  sentences and the two words, with round-2 and round-3 swaps shuffled
  together. It is stricter than the earlier Claude marks (79% agreement).

| Swaps (blind rater) | n | Good | Not simpler | Changes meaning | Good rate |
|---|---:|---:|---:|---:|---:|
| Round 2, no margin | 98 | 53 | 34 | 11 | 54% |
| Round 2, margin (held-out comparison) | 79 | 49 | 14 | 16 | 62% |
| Round 3 arm A, no margin | 45 | 23 | 16 | 6 | 51% |
| **Round 3 arm B, margin** | **33** | **21** | **6** | **6** | **64%** |

- **Effect:** on fresh text the margin blocks 10 Not-simpler swaps, 2 Good
  and 2 Changes meaning. The hypothesis holds, but the Good rate is far
  from 90%.
- **What still fails in arm B:**
  - spelling variants: neighbouring→neighboring, sceptical→skeptical
  - near-equal synonyms: particularly→especially, international→global
  - broken grammar: artwork→art in "an art", adored→loved in "his loved music"
  - shifted claims: suggests→says, proposed→planned
- Details are in [round3.json](round3.json). Per-swap sentences and marks
  stay in the state directory (`rating-r3/`). Joshua's page is
  `~/Documents/cleartext-rating/rate-learned-swaps-r3.html`.

## Round 4: target floor and strict checker (November 19)

- **Fresh text:** 84 unused windows of test-set-v3 dev documents, never test
  documents (`learned-words panel build-r4`). The stage ran on source text,
  because no F2 output exists for these windows.
- **Hypotheses**, with thresholds fixed before any round-4 swap existed:
  - (c) A target floor: the target must be predicted complex by at least 2 of
    20 CWI annotators. Stage 1 required only 1.
  - (d) A strict checker trained on SWORDS-dev substitutes that at least 80%
    of annotators accept. Its pre-registered 0.90-precision rule found no
    threshold; fallback S2 (best F0.5, fixed before the grid was seen) gave
    0.50. On SWORDS test its AUC is 0.80, against 0.81 for the existing
    checker on the same label.
- **Rater:** a fresh blind subagent. It agreed with the round-3 rater on 94%
  of the 47 re-shown swaps.

| Arm (blind rater) | Swaps | Good | Not simpler | Changes meaning | Good rate |
|---|---:|---:|---:|---:|---:|
| b: round-3 stage | 71 | 44 | 12 | 15 | 62% |
| c: + target floor | 55 | 35 | 9 | 11 | 64% |
| d: + strict checker | 27 | 15 | 6 | 6 | 56% |

**Both hypotheses are rejected.**
- The floor also removes Good swaps (rapidly→quickly, believes→thinks).
- The strict checker does not track meaning errors.
- Across the 123 round-3 and round-4 swaps, the Good rate is 7% below margin
  0.1, 41% at 0.1–0.3, and a flat 61–67% above 0.3.
- About a quarter of the swaps above 0.3 change meaning. Most are broken
  verb frames or collocations, such as "we will be needed to", "asking an
  extension" and "it can be found that", or lost term senses, such as
  "statistically significant → large".
- Details are in [round4.json](round4.json).

## Round 5: frame probe and a fine-tuned substitution judge (November 20)

- **Frame probe (diagnostic only, on the 123 rated round-3/4 swaps):** how
  much a swap lowers BERT's masked probability of the words within 3 of it
  ([frame-probe.json](frame-probe.json)). Best AUC for meaning error vs Good
  is 0.64 (largest neighbour drop). That is too weak to build on, so it was
  not trained.
- **Judge (`cleartext.learned_words.judge`):**
  - Model: `cross-encoder/nli-deberta-v3-small` with a new one-logit head,
    fine-tuned on CPU.
  - Input: the 12-word window around the target, before and after the swap,
    with the swapped word marked.
  - Training data, public only: 11,184 pairs from SWORDS dev (soft label =
    share of annotators who accept the substitute) and from BenchLS-train and
    TSAR trial (human simplification vs generated candidate).
  - Validation: 1,876 pairs, held out by target lemma.
  - Settings and threshold rules were fixed before training (see the
    docstring of `evals/learned_words_judge.py`).
  - Training took 11 queue chunks (`172a5ab75661` … `4f2450b5633b`) because
    Yoga was at load 25–46.
  - The threshold chosen on validation is 0.78. No threshold reached 90%
    validation precision, so the strict arm g did not run.
  - Pre-committed in [judge-precommit.json](judge-precommit.json) before any
    round-5 swap existed.
- **Public tests ([judge-check.json](judge-check.json)):** the judge is weaker
  than the existing feature models.

  | Test | Judge | Existing stage |
  |---|---|---|
  | SWORDS test AUC | 0.85 | 0.88 (stage-4 checker) |
  | TSAR-2022 test ACC@1 among generated candidates | 0.53 | 0.73 (GBM ranker) |
  | TSAR test precision, stage + judge filter | 25/26 | 34/36 (stage alone) |

- **Fresh text:** 70 unused windows of test-set-v3 dev documents (31
  documents), none overlapping round 4 (`panel build-r5`).
- **Rater:** a fresh blind subagent rated all 254 mixed items. It agreed with
  round 4's marks on 18 of 20 re-shown swaps.

| Arm (blind rater) | Swaps | Good | Not simpler | Changes meaning | Good rate (95% CI) |
|---|---:|---:|---:|---:|---:|
| b: round-3 stage | 49 | 34 | 5 | 10 | 69% (55–80) |
| e: b + judge ≥ 0.78 | 42 | 30 | 4 | 8 | 71% (56–83) |
| f: judge picks the replacement | 172 | 56 | 19 | 97 | 33% (26–40) |
| control: TSAR-test top human substitute | 20 | 10 | 2 | 8 | 50% |

**The judge is rejected.**
- As a filter it removes 7 of 49 swaps with no clear gain.
- As a picker it is far worse than the GBM ranker: mostly wrong senses and
  broken frames.
- The human-gold controls are a ceiling check. Even TSAR's most frequent
  human substitute is rated Good only 50% of the time (10 of 18 once 2 a/an
  errors from building the controls are excluded). Under this rater, the
  shipped stage's 62–69% on technical text is above that level.
- Remaining stage failures are mostly technical senses ("clinically
  significant", "proposed rule", diplomatic "recognize") and same-difficulty
  synonyms.
- Details are in [round5.json](round5.json).

## Split checker (stage 4b)

[split-checker.json](split-checker.json) has the full results; all figures
below are on the WikiSplit test set.

- **Threshold:** 0.54, from the pre-registered rule (specificity ≥ 0.90 on
  validation).
- **Rejection by corruption type:**
  - broken boundary: 93% (NLI floor 0.5: 0%)
  - dropped span: 92% (86%)
  - lost subject: 96% (50%)
  - wrong referent: 52% (52%)
- **Human splits:** it accepts 70.5%.
- **On F2's 73 dev splits:** it rejects 38. That is far more than the
  roughly 15% a review found broken.
  - The lowest-scored ones are real failures, such as "Products from all
    three supply modes coming from the same plant." and duplicated clauses.
  - The middle band looks like acceptable splits.
  - It stays **opt-in** (`apply --split-filter`), so the round-2 page uses
    F2's splits as shipped.
- **Pre-commitment:** decisions on the 60 F2 splits in part 2 of the first
  rating page (30 rejected) are in
  [split-checker-precommit.json](split-checker-precommit.json) (sha256
  `beb05590…`), written before those marks exist.
  `learned-words splits score-marks` scores them once Joshua marks part 2.

## Evaluation caveats addressed

- **Passage-averaged gates:** results here are per edit. Readability is not
  measured with FK, which rewards only splitting. A word swap leaves FK
  nearly unchanged even when it helps.
- **No outside baseline:** added above.
- **Over-splitting is invisible to the control gate:** the split checker
  scores each edit instead.
- **Open:** precision against public gold is not human acceptability. The
  round-2 marks are the real test.

## Data

[data-manifest.json](data-manifest.json) records URLs, sha256 and licenses.

| Data | License | Use |
|---|---|---|
| BenchLS, CompLex | CC BY 4.0 | train |
| SWORDS | CC BY 3.0 US | dev train, test |
| TSAR-2022 | CC BY-NC-SA 4.0 | trial train, test (research, non-commercial) |
| NNSeval, LexMTurk, CWI 2018 | no license stated | research use, local only |
| plainlanguage.gov, NIST and US Courts glossaries | US government, public domain | candidates, term training |
| MeSH 2026 | NLM terms | term training |
| WikiSplit | CC BY-SA 4.0 | split checker |

No data set was skipped. None forbids this research use.

## Compute

Every heavy step ran through the job queue on Yoga at `later` priority, one
job at a time, each under 30 minutes. In order:

- featurize: `a0b5ebdeb4f7`, `d47461169d83`, `ef3b92a16ebc`
- NLI and training: `ae95d75cc0e8`, `d41a1af09617`, `9acf96d5dc95`
- split checker: `6dbcb84771fe`
- apply: `493d4bd01756`

Featurizing took about 3.4 s per target on a loaded host.

Problems along the way:

- A first featurize attempt, `8ccc46da0472`, was too slow at 29 s per target.
  After NLI was moved to only the rows the checker reads, the work was
  restarted in a new state directory.
- A first threshold pass leaked through NLI availability. Checker scores were
  missing for non-gold top candidates and defaulted to 0, which made tiny
  thresholds look precise. It was fixed before any test number was read: every
  gate row now gets an out-of-fold checker score, and training refuses to run
  if any NLI is missing.

## Round 6: offline replay gate, pairwise simplicity and context fit (November 20)

[ROUND6.md](ROUND6.md) holds the pre-registered rules, from a planning pass
by a separate model.

### Replay set

- It holds the 153 blind-rated swaps from rounds 3–5 that the current stage
  made: 99 Good, 23 Not simpler and 31 Changes meaning, a Good rate of 65%.
- It is validation only.
- An arm earns a fresh window only if it meets all three conditions:
  - its Good rate improves by at least 5 points;
  - it keeps at least 60% of the swaps;
  - it removes a large enough share of the error type it targets (at
    least 40% of Not simpler for arm p, at least a third of Changes meaning
    for arm k).

### Technical reference corpus

- It has 12.0M tokens. The sources are:
  - RFCs from 2015–2019;
  - Federal Register rules and proposed rules from 2022 to mid-2023;
  - openFDA labels from 2022.
- No test-set-v3 document or generic name is in it, and no round-4/5 panel
  document. 1,393 sentences were dropped for a word 8-gram overlap with
  those passages.
- SEC was skipped because it refuses clients without a contact address.
- Its manifest and hashes are in the state directory under `corpus/`.

| Arm | Model check | Threshold (pre-committed) | Replay: kept | Good rate kept | Targeted error removed | Replay AUC | Decision |
|---|---|---|---:|---:|---:|---:|---|
| p: pairwise simplicity | OOF AUC 0.61 (43k BenchLS/plainlanguage pairs) | 0.31 (max balanced acc.) | 150/153 | 66% | 2 of 23 Not simpler | 0.76 (Good vs Not simpler) | no-go |
| k: corpus context fit | OOF AUC 0.77 (SWORDS 0.59, distant 0.71); leave-one-glossary-out 0.54–0.64 | 0.236 (keep 90% of fits) | 148/153 | 65% | 0 of 31 Changes meaning | 0.53 (Good vs Changes meaning) | no-go |

**Neither arm earns a fresh window,** so round 6 spent no new text
([pairwise-replay.json](pairwise-replay.json),
[contextfit-replay.json](contextfit-replay.json)).

### Findings

- **Corpus collocation statistics do not detect the meaning errors.** Their
  replay AUC is 0.53. "Clinically significant", "required to" and
  "broadcast" all have the replacement attested in the same frames.
- **The errors look semantic, not distributional.** The replacement fits
  the frame, but the term of art or the claim changes.
- **BenchLS rank pairs are weak labels for simplicity.** The pairwise model
  reaches only 0.61 out of fold. On replay, raw frequency gain separates
  Not simpler from Good better (AUC 0.79) than either the learned margin
  (0.67) or the pairwise model (0.76).
- **Replay prevented two wasted windows.** Both arms would have cost
  fresh dev text and a rating pass for no expected gain.

## Round 7: learning the swap decision from human rewrites (November 21)

[ROUND7.md](ROUND7.md) holds the pre-registered rules.

- **Data:** 329,281 aligned sentence pairs after filtering, from WikiAuto,
  Cochrane plain-language summaries and contract simplifications. That is
  4.07M word events (kept, replaced or dropped); 480 pairs were excluded for
  overlapping evaluation text ([rewrite-data.json](rewrite-data.json)).
- **Model check:** out-of-fold AUC 0.98. This mostly shows that real editor
  replacements are easy to tell from random WordNet candidates. It says
  little about the stage's own proposals.
- **Replay ([rewrite-replay.json](rewrite-replay.json)):** kept 45 of 153
  swaps, Good rate 56% (baseline 65%), AUC 0.47 for Good vs error.
  **No-go.** No fresh text was spent.
- **Finding:** whether editors made a word swap does not predict whether the
  stage's swap is good. The stage already proposes plausible synonyms, and
  the rewrite statistics do not separate the good ones from the bad ones.

## Round 8: a filter trained on blind marks (November 21)

[ROUND8.md](ROUND8.md) holds the pre-registered rules. Arm s is a logistic
filter (C = 0.1, 31 fixed features) trained on the 334 blind-rated swaps
from rounds 3–5, keeping the top 65% of arm-b swaps.

- **Void first run:** a featurization bug dropped 134 of 334 rows. Its pass
  (76% Good on 89 rows, baseline 58%) is void; model and pre-commitment were
  deleted uncommitted. The fixed rerun resolves all 153 arm-b rows.
- **Cross-validation ([stacker-cv.json](stacker-cv.json)):** 20 × 5-fold,
  grouped by target lemma. Kept swaps 78% Good (range 74–81%) vs baseline
  65%, 65% kept, 60% of errors removed, AUC 0.78 on arm b. **Go.**
- **Fresh window ([round8.json](round8.json)):** arm b 17/36 Good (47%,
  95% CI 32–63%); arm s 15/24 Good (62.5%, CI 43–79%), 67% kept. Arm s
  removed 10 of 11 Not simpler but only 1 of 8 Changes meaning (meaning rate
  29%); one arm-s swap used a different candidate than arm b. **Misses the
  bar** (≥ 78% Good, meaning ≤ 15%).
- **One-time human check ([stacker-joshua60.json](stacker-joshua60.json)):**
  on Joshua's 60 older swaps the filter kept 7 (4 Good, 3 meaning), removed
  44 of 47 errors but also 9 of 13 Good; AUC 0.62. Spent: not used to decide
  anything.
- **Caveats:** the rater repeated 14 of 20 items identically, so AI marks
  are noisy. The fresh window used leftover passages, so its arm-b baseline
  (47%) is not comparable to earlier rounds.
- **Finding:** the 31 signals learn "is it simpler" but not "does it keep the
  meaning". The remaining errors are sense, verb frame, collocation and
  register (occurred→happened, concluded→found, overseen→managed,
  unrestricted→free, requires→needs).

## Round 9: sense match and parse stability (November 21)

[ROUND9.md](ROUND9.md) holds the pre-registered rules. Arm w adds 13
features to arm s:
- sense match: BERT sense vectors built from SemCor (25,805 synsets, 37,176
  sentences), with the target disambiguated in the original sentence and
  the replacement in the new one;
- parse stability: the swapped sentence is re-parsed and compared.

It keeps 55% of the swaps.

- **Cross-validation ([r9-cv.json](r9-cv.json)):** 371 rows, 189 of them
  arm-b (base 61% Good, 39 Changes meaning), 20 × nested 5-fold grouped by
  lemma, with arm s refitted at the same 55% keep. Arm w reaches 78.8% Good
  against 77.9% for arm s, a paired gain of **+0.9 points** where the bar
  was 3. It removes 51% of Changes meaning (arm s 49%). **No-go.** No fresh
  text was spent.
- **Diagnostic, not a decision:** on arm-b rows, the existing `ctx_cos`
  separates Changes meaning well on its own (AUC 0.79), and the new
  cross-fit feature does too (0.74; the two correlate at r = 0.71). Yet both
  combined models reach only 0.57–0.58 meaning AUC. The single
  Good-vs-rest label mixes two errors that pull in opposite directions:
  - Changes meaning falls as contextual similarity rises;
  - Not simpler rises with it, since near-identical words such as
    sceptical→skeptical score highest.

  A linear model of "Good" cannot express that, so the sense signal is
  mostly already present and gets cancelled.
- **Parse stability** fires rarely and does not separate on arm b (AUC
  0.50–0.53).

## Round 10: two heads (November 21)

[ROUND10.md](ROUND10.md) holds the pre-registered rules. Arm h scores
(1 − P(meaning)) × (1 − P(not simpler)) with two logistic heads on the same
44 features.

- **Cross-validation ([r10-cv.json](r10-cv.json)):** 78.1% Good against
  77.9% for arm s, a paired gain of **+0.2 points** (bar 3); it removes 49%
  of Changes meaning. **No-go.** No fresh text was spent.
- **Diagnostic, not a decision:** out of fold, the 44-feature meaning head
  reaches AUC 0.70 on arm b, *below* `ctx_cos` alone (0.79). With 137
  meaning labels, more features add noise rather than signal. Split by
  `ctx_cos` tertile, arm b's swaps (Good / Changes meaning / Not simpler)
  are:
  - low: 30 / 26 / 7;
  - middle: 43 / 10 / 10;
  - high: 43 / 3 / 17.

  Even the best two thirds reach only about 68% Good. Filters over these
  signals plateau near 78% in cross-validation and lower on fresh text.
- **What this suggests:** the limit is no longer the filter. It is the
  candidates the generator offers and the noise in single-rater labels
  (88/114 agreement on repeated pairs).

## Round 11: sense-limited candidates (November 21)

[ROUND11.md](ROUND11.md) holds the pre-registered rules
([r11-precommit.json](r11-precommit.json)). The generator may propose only
words that cover at least 25% of the target's in-context sense mass, using
the round-9 SemCor table. The fresh window is 135 technical-corpus
passages (45 RFC, 45 Federal Register, 45 openFDA). Three fresh blind
raters marked 155 items; the majority mark counts
([round11.json](round11.json)).

| Arm | Swaps | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|
| b: current stage | 126 | 65.9% (57–74) | 20.6% | 17 |
| s: b + round-8 filter | 95 | 83.2% (74–89) | 9.5% | 7 |
| g: sense-limited candidates | 51 | 72.5% (59–83) | 15.7% | 6 |
| **gs: g + round-8 filter** | **38** | **94.7% (83–99)** | **2.6%** | 1 |

- **Pre-registered success: no.** Arm gs meets the quality bars
  (≥ 78% Good, ≤ 15% Changes meaning) by a wide margin, but makes 30% as
  many swaps as arm b against a required 40%.
- **Arm s meets every round-8 bar on this window:** 83% Good, 9.5% meaning,
  75% of arm b's swaps kept. Round 8's window gave it 62.5% on leftover dev
  passages. The difference may be the text (cleaner technical prose) or
  the raters (three, with a majority), so it is not a replication.
- **The two mechanisms combine as predicted.** The sense limit cuts meaning
  errors (21% → 16%), the filter cuts Not simpler, and together they leave
  2 errors in 38: creates→makes (Not simpler) and lifetimes→lives (meaning).
- **Raters:** all three agree on 90% of items, and pairwise agreement is
  92–95%. Each agrees with the old round-5 marks on 19 of 20.
  Per-rater Good for arm gs is 95% from each rater.
- **Caveats:**
  - Arm gs is conservative and repetitive: 38 swaps over 30 distinct
    pairs, including additional→extra ×6, frequently→often ×3 and
    occurred→happened ×3.
  - All raters are AI.
  - The text is one technical genre.
  - Nothing ships; Joshua's marks decide adoption.

## Round 12: pair priors and sense cover on the arm-s score (November 22)

[ROUND12.md](ROUND12.md) holds the pre-registered rules
([r12-precommit.json](r12-precommit.json)).

**Arm p** is a 4-feature logistic (C = 0.1) trained on all 506 rated rows,
with round 11 now development data. Its features:
- the logit of the arm-s score;
- the in-context sense cover;
- smoothed past Good rates of the (target, replacement) lemma pair and of
  the target lemma.

It keeps 50% of round-11-like arm-b swaps. Arm t is the ablation: the arm-s
score alone at the same keep.

- **Why:** two findings from the development rows.
  - Round 11's sense cut threw away 45 Good arm-s swaps.
  - Pairs recur: 71 of round 11's 135 swaps had a pair already rated in
    rounds 3–8, and that history predicted the mark (multiple→several
    failed 5 of 5).
- **CV ([r12-cv.json](r12-cv.json)):** **go**, with the caveat that the
  design came after a time-split look.
  - Passage-grouped nested CV on round 11's arm-b rows: 96.2% Good at 51%
    kept, against 91.5% for arm t (+4.7 points).
  - Time split (rounds 3–8 → 11): 97.1% at 54% kept.
  - Lemma-grouped, where no prior is available: 93.1%.
- **Fresh window ([round12.json](round12.json)):** 135 new technical-corpus
  passages (546 sentences, none from round-11 documents). Three fresh blind
  raters marked 134 items. All three agreed on 92%; pairwise agreement
  was 94–96%.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 107 | 100% | 19.6 | 56.1% (47–65) | 24.3% | 21 |
| s: round-8 filter | 74 | 69% | 13.6 | 73.0% (62–82) | 24.3% | 2 |
| gs: round 11's best | 36 | 34% | 6.6 | 77.8% (62–88) | 19.4% | 1 |
| t: stricter arm s | 53 | 50% | 9.7 | 75.5% (62–85) | 24.5% | 0 |
| **p: arm p** | **51** | **48%** | **9.3** | **84.3% (72–92)** | **15.7%** | 0 |
| p65: arm p, 65% keep | 70 | 65% | 12.8 | 78.6% (68–87) | 21.4% | 0 |

- **Pre-registered success: no.** Arm p missed both quality bars (Good
  ≥ 90%, Changes meaning ≤ 8%) but met the yield bar (48% ≥ 40%).
- **Relative result:** arm p is the best arm on this window.
  - It beats arm t by 9 points at the same yield, so the priors and cover
    add precision beyond a stricter threshold.
  - It beats gs by 6.5 points at 1.4 times gs's yield.
- **The window was harder.** Arm b fell from 65.9% in round 11 to 56.1%,
  and gs from 94.7% to 77.8%.
  - The drop is in openFDA drug labels. Arm b had 19 Good of 37 there
    (round 11: 32 of 43), and arm p had 15 of 20. On Federal Register and
    RFC text, arm p reached 18 of 20 and 10 of 11.
- **Arm p's 8 errors are all Changes meaning,** mostly broken
  collocations and frames:
  - "sufficient numbers of" → "enough numbers of" (×2);
  - "orally administered" → "orally given";
  - "prolonged administration" → "long administration";
  - "potential … as well as" → "possible";
  - "permitted vessels" (licensed) → "allowed";
  - satisfactory→good;
  - originated→made.

  Arm p removed every Not simpler swap and 18 of arm b's 26 Changes
  meaning.
- **Seen pairs:** 41 of arm p's 51 swaps had a rated pair, and 36 of those
  were Good (88%). Of the 10 unseen pairs, 7 were Good.
- **Caveat:** each rater agreed with the 20 old round-5 marks only 14–16
  times (round 11: 19 of 20), so these raters may be stricter.

## Round 13: frozen arm p on a second window, plus a runner-up fallback (November 22)

[ROUND13.md](ROUND13.md) holds the pre-registered rules. Nothing was refitted.
Round 12's arm p and thresholds were frozen before round 12 was rated. The
only change is a fallback: when a target's top candidate yields no accepted
swap, the stage decides again on its runner-up. The gate was:
- round 12's CV gate had passed;
- the model was unchanged;
- a smoke test showed the new code reproduces round 11's swaps on 3 spent
  passages ([r13-smoke.log](r13-smoke.log)).

Fresh window ([round13.json](round13.json)): 134 new passages (530
sentences; one skipped for rewrite-corpus overlap), 150 items, three fresh
blind raters. All three agreed on 92%. Consistency with the old marks was
18–19 of 20.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 115 | 100% | 21.7 | 66.1% (57–74) | 13.9% | 23 |
| s: round-8 filter | 92 | 80% | 17.4 | 76.1% (66–84) | 14.1% | 9 |
| gs: round 11's best | 53 | 46% | 10.0 | 83.0% (71–91) | 15.1% | 1 |
| t: stricter arm s | 62 | 54% | 11.7 | 85.5% (75–92) | 14.5% | 0 |
| **p: frozen arm p** | **64** | **56%** | **12.1** | **89.1% (79–95)** | **9.4%** | 1 |
| p65 | 83 | 72% | 15.7 | 80.7% (71–88) | 12.0% | 6 |
| pn: p + fallback | 71 | 62% | 13.4 | 84.5% (74–91) | 14.1% | 1 |
| p65n | 94 | 82% | 17.7 | 78.7% (69–86) | 14.9% | 6 |

- **Pre-registered success (arm pn): no.** The fallback added 7 swaps and
  only 3 were Good; the others included significant→major and
  determine→find.
- **Replication (arm p): no.** It reached 89.1% and 9.4% against bars of
  90% and 8%, at 56% yield.
- **Over its two fresh windows, arm p is 100 of 115 Good: 87.0%, at about
  half of arm b's yield.** On both windows it is the best arm, at a higher
  yield than gs (round 12: +6.5 points; round 13: +6.1).
- **Arm p's 7 errors:**
  - Changes meaning: selected→chosen, terminated→ended, indicate→show,
    determine→find, maintained→had and emphasize→state;
  - Not simpler: selects→chooses.

  By source: openFDA 19 of 22 Good, Federal Register 21 of 23, RFC 17 of 19.

## Round 14: two meaning signals added to arm p (November 22)

[ROUND14.md](ROUND14.md) holds the pre-registered rules. Arm m adds two
stage signals to arm p's four features: `ctx_cos` and the checker score.
On round 12's high-scoring swaps they separated Good from Changes meaning
(AUC 0.72–0.73). Arm m was refitted on 620 rows, with round 12 added.

- **CV ([r14-cv.json](r14-cv.json)): no-go.** No fresh text was spent.
  - The keep rule needed some keep from 40% to 60% to reach ≥ 92% on round
    12's arm-b rows. None did; the best was 86.7% at 40%.
  - Arm m beat a refitted arm p by only 0.4–1.3 points.
  - On the time split onto round 12, it scored 87.2% against 85.0% (bar: +3
    points).
- **Diagnostic:** with lemma-grouped folds, where no priors exist, arm m
  scored 90.3% against 86.3%. The two signals help on unseen words, but
  priors already cover most of that gain.

## Round 15: a corpus-collocation veto after arm p (November 22)

[ROUND15.md](ROUND15.md) holds the pre-registered rules.
- **Signal:** round 6's `max_drop`, the largest fall in corpus association
  from target to replacement around the swap. Counts come from the
  technical corpus with the swap's source document removed.
- **Where it works:** on rounds 11–12 arm-b swaps it separates Good from
  Changes meaning with AUC 0.71, and 0.84 among swaps arm p keeps. On
  rounds 3–8 text, which the corpus does not match, it scores 0.56. That
  explains round 6's failure.
- **Linear version:** adding it to arm p gained only 0.5–1.5 points.
- **Arm pv:** a refitted arm p at a 60% keep, then a veto on `max_drop`
  above the 80th percentile seen on Good technical swaps.

- **CV ([r15-cv.json](r15-cv.json)): no-go**, on one of four criteria. No
  fresh text was spent.
  - Passage-grouped CV: arm pv reached 94.7% Good at 45.6% kept. Arm p40,
    the refitted arm p at a 40% keep, reached 92.7% at 40.6%. The paired
    gain was +2.05 points; the bar was 3.
  - On round 12's rows: arm pv 90.5% at 40% kept, arm p40 85.2% at 37%.
  - Time split onto round 12: arm pv 90.0% against arm p40's 85.0%, both at
    37% kept.
- **Design flaw in the gate:** it compared the arms at different yields.
  It also reused the rows where the veto was found, so the result counted
  as no-go. Round 16 tests the same arm on unseen marks instead.

## Round 16: the veto, gated on round 13's unseen marks (November 22)

[ROUND16.md](ROUND16.md), written before round 13 was rated, tests round 15's
arm pv exactly as specified on round 13's 115 rated arm-b swaps. No design
step had seen those marks.

- **Gate ([r16-cv.json](r16-cv.json)): no-go.**
  - Arm pv reached 84.1% Good at 55% kept.
  - Arm p40, the refitted arm p at a 40% keep (round 15's comparator),
    reached **92.3% (48/52) at 45% kept**.
- **The veto was overfit to rounds 11–12.** It removed Good swaps on round
  13 and let meaning errors through. No fresh text was spent.
- **A stricter threshold was the result that held on unseen data.** Round
  17 tests it on a fresh window.

## Round 17: arm p refitted on 750 rows at a 40% keep (November 22)

[ROUND17.md](ROUND17.md) holds the pre-registered rules
([r17-precommit.json](r17-precommit.json)).
- **Model:** round 12's arm p, unchanged, refitted on all 750 rated rows
  (rounds 3–8, 11–13).
- **Arms:** p40 (primary) keeps 40% of the rounds 11–13 arm-b rows by
  passage-grouped out-of-fold score; p50 keeps 50% (secondary).
- **Check ([r17-cv.json](r17-cv.json)): go.**
  - Passage-grouped CV: 93.2% Good at 42% kept.
  - Held-out evidence: round 15's p40, scored on round 13's unseen marks,
    reached 92.3% at 45%. That number was read before this protocol was
    written.

Fresh window ([round17.json](round17.json)): 135 new passages (542
sentences), 148 items, three fresh blind raters. All three agreed on 88%.
Consistency with the old marks was 15–17 of 20.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 123 | 100% | 22.7 | 63.4% (55–71) | 18.7% | 22 |
| s: round-8 filter | 87 | 71% | 16.1 | 80.5% (71–87) | 11.5% | 7 |
| gs: round 11's best | 42 | 34% | 7.7 | 95.2% (84–99) | 4.8% | 0 |
| p: frozen round-12 arm p | 66 | 54% | 12.2 | 95.5% (87–98) | 4.5% | 0 |
| **p40: refitted, 40% keep** | **54** | **44%** | **10.0** | **96.3% (87–99)** | **3.7%** | 0 |
| p50: refitted, 50% keep | 65 | 53% | 12.0 | 96.9% (89–99) | 3.1% | 0 |

- **Pre-registered success: yes.** Arm p40 reached 96.3% Good with 3.7%
  Changes meaning, at 44% of arm b's swaps; the bar was 40%.
- **Against gs:** about the same precision, with 1.3 times gs's yield
  (p50: 1.5 times).
- **Errors:** indicate→show ("overstates") and Additionally→Also ("Also
  … also"). The frozen arm p also made significant→major in a statistical
  sense.
- **Caveats:**
  - This window was easier than round 12's; arm b reached 63% here against
    56% there.
  - Arm p40 leans on memorized pairs. Only 1 of its 54 swaps had a pair
    never rated before, it used only 28 distinct pairs, and 10 of the 54
    were additional→extra. Without additional→extra it reached 42 of 44
    (95.5%).
  - Rater 3 was stricter (p40: 90.7%).
- **Round 18 replicates these exact arms, frozen, on a new window.**

## Round 18: round 17's arms frozen on a new window (November 22)

[ROUND18.md](ROUND18.md) holds the pre-registered rules. This is a pure
replication: no refit and no threshold change
([r18-precommit.json](r18-precommit.json)).

Fresh window ([round18.json](round18.json)): 134 new passages (532
sentences; one skipped for overlap), 166 items, three fresh blind raters.
All three agreed on 87%. Consistency with the old marks was 18–19 of 20.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 135 | 100% | 25.4 | 68.1% (60–75) | 12.6% | 26 |
| s: round-8 filter | 103 | 76% | 19.4 | 75.7% (67–83) | 16.5% | 8 |
| gs: round 11's best | 57 | 42% | 10.7 | 84.2% (73–91) | 14.0% | 1 |
| p: frozen round-12 arm p | 72 | 53% | 13.5 | 86.1% (76–92) | 11.1% | 2 |
| **p40: round 17's, frozen** | **52** | **39%** | **9.8** | **88.5% (77–95)** | **11.5%** | 0 |
| p50: round 17's, frozen | 71 | 53% | 13.3 | 87.3% (78–93) | 11.3% | 1 |

- **Pre-registered success: no.** Arm p40 missed all three bars: Good
  < 90%, Changes meaning > 8%, and 39% of arm b's swaps (bar: 40%).
- **The window was not harder for arm b** (68%, the highest of the night),
  but its errors sat in well-rated pairs used in the wrong frame:
  - occurred→happened ×3 in drug labels ("anemia has happened");
  - additional→extra ("extra input");
  - informs→tells;
  - established→created ("fee amounts created");
  - necessary→needed ("it is needed to");
  - incorporate→include (incorporation by reference).

  Pair priors cannot see the frame.
- **Rounds 17–18 pooled:**
  - p40: 98 of 106, 92.5% Good (CI 86–96), 7.5% meaning, 41% of arm b;
  - p50: 125 of 136, 91.9% Good (CI 86–95), 7.4% meaning, 53% of arm b;
  - gs on the same two windows: 88 of 99, 88.9%, at 38%.

## Round 19: arm p refitted on 1,024 rows, p50 primary (November 22)

[ROUND19.md](ROUND19.md) holds the pre-registered rules
([r19-precommit.json](r19-precommit.json)).
- **Model:** round 12's arm p, unchanged, refitted on all 1,024 rated rows
  (rounds 3–8, 11–13, 17, 18).
- **Arms:** p50 (primary) keeps 50% of the technical-window arm-b rows; p40
  keeps 40% (secondary).
- **Check ([r19-cv.json](r19-cv.json)): go.**
  - Passage-grouped CV: p50 91.9% Good at 51% kept.
  - Time split onto round 18: 88.4% at 51% kept.

Fresh window ([round19.json](round19.json)): 135 new passages (540
sentences), 141 items, three fresh blind raters. All three agreed on 88%.
Consistency with the old marks was 15–16 of 20.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 112 | 100% | 20.7 | 66.1% (57–74) | 14.3% | 22 |
| s: round-8 filter | 86 | 77% | 15.9 | 80.2% (71–87) | 11.6% | 7 |
| gs: round 11's best | 44 | 39% | 8.1 | 93.2% (82–98) | 4.5% | 1 |
| p: frozen round-12 arm p | 61 | 54% | 11.3 | 95.1% (87–98) | 3.3% | 1 |
| p40: refitted, 40% keep | 55 | 49% | 10.2 | 94.5% (85–98) | 3.6% | 1 |
| **p50: refitted, 50% keep** | **59** | **53%** | **10.9** | **94.9% (86–98)** | **3.4%** | 1 |

- **Pre-registered success: yes.** Arm p50 reached 94.9% Good with 3.4%
  Changes meaning, at 53% of arm b's swaps.
- **Against gs:** about the same precision, with 1.34 times gs's yield.
- **Errors:**
  - Changes meaning: indicate→show ("slight strengthening") and
    induces→causes (pharmacology sense);
  - Not simpler: receives→gets.
- **Still repetitive:** 34 distinct pairs across 59 swaps, including
  additional→extra ×8 and administered→given ×5. Only 1 swap had an unseen
  pair.
- **Round 20 replicates these exact arms, frozen, on a new window.**

## Round 20: round 19's arms frozen on a new window (November 22)

[ROUND20.md](ROUND20.md) holds the rules. This is a pure replication
([r20-precommit.json](r20-precommit.json)). The panel rule found only 118
passages: 45 Federal Register, 45 openFDA and 28 RFC. RFC documents were
nearly used up by the five earlier panels, and one passage was skipped for
overlap. That left 117 passages (478 sentences), 134 items and three fresh
blind raters. All three agreed on 87%.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Not simpler |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 109 | 100% | 22.8 | 63.3% (54–72) | 16.5% | 22 |
| s: round-8 filter | 86 | 79% | 18.0 | 75.6% (66–83) | 16.3% | 7 |
| gs: round 11's best | 40 | 37% | 8.4 | 85.0% (71–93) | 12.5% | 1 |
| p: frozen round-12 arm p | 61 | 56% | 12.8 | 86.9% (76–93) | 11.5% | 1 |
| p40: round 19's, frozen | 50 | 46% | 10.5 | 92.0% (81–97) | 8.0% | 0 |
| **p50: round 19's, frozen** | **60** | **55%** | **12.6** | **90.0% (80–95)** | **8.3%** | 1 |

- **Pre-registered success: no, by one swap.** Arm p50 reached 90.0% Good
  and 55% yield, but Changes meaning was 8.3% (5 of 60) against a bar of 8%.
  The secondary p40 would have met all three bars.
- **Rounds 19–20 pooled (same model):**
  - p50: 110 of 119, 92.4% Good (CI 86–96), 5.9% meaning, 54% yield;
  - p40: 98 of 105, 93.3% (87–97), 48% yield.

## Overnight summary (rounds 12–20)

- **Precision and yield:** refitted arm p at a 50% keep is pooled over
  rounds 17–20 at **92.2% Good (CI 88–95), 6.7% Changes meaning, 53% of
  arm b's swaps** (about 12 per 100 sentences).
  - Round 11's arm gs reached 89.1% at 38% on the same windows.
  - The frozen round-12 arm p reached 89.6% (CI 86–92) over six windows.
- **Pre-registered bars:**
  - met in rounds 17 (p40) and 19 (p50);
  - missed in rounds 12, 13, 18 and 20 (round 20 by one meaning error);
  - three rounds (14, 15, 16) failed their CV gates and spent no fresh text.

### Gaps found overnight

- **Strictness, not sense limits, drove gs.** At equal yield, the arm-s
  score alone (arm t) matched or beat gs on both later windows: 75.5% vs
  77.8% at 1.5 times gs's yield in round 12, and 85.5% vs 83.0% in round 13.
- **Learned pair priors carry most of the precision.** The (target,
  replacement) history from earlier blind marks separates swaps the stage's
  signals cannot. Most of arm p's swaps reuse rated pairs, so it acts as a
  context-checked lexicon learned from AI raters.
  - Unseen pairs are few, and arm p rejects most of them: round 12 had 10 at
    7 Good; round 13 had 10 at 7 Good; round 17 had p40 at 1 of 1.
  - Pairs that recur in a window dominate it: additional→extra made 10 of
    p40's 54 swaps.
- **Windows differ more than arms do.** Arm b ranged from 56% to 66% Good,
  and every filter moved with it. Round 12's drug-label collocations held
  every arm below 85%.
- **The fallback and the veto both failed.**
  - On round 13's fresh window, runner-up candidates were 3 of 7 Good.
  - The corpus-collocation veto looked strong on rounds 11–12 but failed on
    round 13's unseen marks. Round 16's gate stopped it before a window was
    spent on it.
- **Errors left in the best arm are context frames.** A well-rated pair
  fails in the wrong frame: "anemia has happened", "extra input", "it is
  needed to", "indicate → show" (a stronger claim). Pair priors cannot see
  the frame, and no stage signal tested overnight separated these cases on
  unseen data.
- **The base stage caps yield.** Only about 13% of featurized targets become
  a swap, and about a third of passages get none. Any filter can keep only
  a share of about 20 swaps per 100 sentences.
- **Caveats:**
  - All marks come from AI raters.
  - One genre: RFC, Federal Register and openFDA text.
  - RFC documents are nearly used up for new panels.
  - Nothing ships until Joshua's own marks decide.

## Round 21: unseen pairs, meaning residue and a genre window (November 22)

[ROUND21.md](ROUND21.md) holds the pre-registered rules
([r21-precommit.json](r21-precommit.json)).

### Unseen pairs, measured honestly ([r21-unseen.json](r21-unseen.json))

The measurement uses arm p's model class on all 1,259 rated rows (rounds
3–20), with 5 repeats of nested CV at a 50% keep. Precision and yield are on
the 827 technical-window arm-b rows.

| CV grouping | Rows scored | Good | Kept |
|---|---:|---:|---:|
| Passage, all rows | 827 | 92.3% | 51% |
| Passage, pair seen in training folds | ~673 | 92.7% | 60% |
| Passage, **pair unseen** | ~154 | **82.8%** | **13%** |
| Pair-grouped (no pair seen) | 827 | 87.3% | 51% |
| Target-lemma-grouped (no target seen) | 827 | 87.4% | 52% |

- **p50 is a context-checked lexicon.** On new pairs it keeps about 1 swap
  in 8, and those are 83% Good.
- **The unseen-pair swaps are hard in themselves.** Their base Good rate is
  50%. On them:
  - every scorer tried reaches AUC 0.73–0.75;
  - none reaches 90% Good even on its top 10% (best: 84–89%);
  - see [r21-explore.json](r21-explore.json).
- **Generalizing the prior did not fix this:**
  - back-off to the replacement lemma, the stem family (additional /
    additionally) and the target stem family: +0 to +0.5 points;
  - BERT static-embedding neighbours of the pair: +0.2 points in
    pair-grouped CV;
  - a 31-feature refit (logistic or boosted): no gain (boosted: −4 points);
  - shrinking to the arm-s score, with a separate scorer and threshold for
    unseen pairs: more kept (18–26%), but at 73–76% Good.
- **Arm q50 (pre-registered):** back-off priors, the pair count, an
  unseen × arm-s interaction, two context features, `ctx_cos` and the
  checker score. **Gate passed** ([r21-cv.json](r21-cv.json)):
  - passage-grouped CV: 92.1% Good at 51% kept (r50, arm p refitted on the
    same rows: 92.1%);
  - pair-grouped CV: 88.8% against 87.3% (+1.6 points; meaning errors 41
    against 50);
  - lemma-grouped CV: 89.1% against 87.5% (+1.7 points).

  On passage-CV unseen rows it is no better (80.4% against 83.9% at 14%).
  The variant was chosen after the exploratory look, on the same rows.

### Meaning residue (p50, rounds 17–20)

p50 made 20 errors in 255 rated swaps: 17 Changes meaning and 3 Not simpler.

| Error type | Swaps | Examples |
|---|---:|---|
| Context frame on a well-rated pair | 14 | "has occurred"→"has happened" ×3, "it is necessary to", "orally administered", "additional input", incorporate (by reference), induces (metabolism) ×2, indicate→show ×2, informs→tells, established→created, typically→often |
| Replacement repeats a word in the sentence | 1 (+1 Not simpler) | "Additionally … also"→"Also … also"; contain→have before "have" is also Not simpler |
| Inside quoted regulatory text | 1 | “determines to be sufficient” → enough |
| Broken article | 1 | "an incorrect" → "an wrong" |
| Not simpler | 3 | discussed→mentioned, receives→gets, contain→have |

Counts over all 1,259 rated rows:
- swaps inside quotation marks: 6, all Changes meaning;
- the replacement's stem already in the sentence: 30, 47% Good;
- a mismatched a/an: 5, all Changes meaning.

Two changes follow:
- **a/an surface realization** for every arm. It does not change the swap
  decision.
- **Two learned context features** in q50. In passage-grouped out-of-fold
  CV over p50's rounds 17–20 swaps, q50 keeps 15.4 of the 17 meaning errors
  against r50's 15.8. It drops the quoted swap and contain→have, but keeps
  "Also … also" and all 14 frame errors. It also drops about 5 Good swaps.
  The frame errors remain the open problem.
  ROUND21.md says 19 errors and 13 frame errors; the correct counts are 20
  and 14.

### Fresh window ([round21.json](round21.json))

- **Panel:** 135 new passages:
  - 45 news (Wikinews);
  - 45 encyclopedic (Wikipedia featured-article leads);
  - 23 Federal Register and 22 openFDA.

  The genre texts were downloaded as untrusted data
  ([r21-genre-manifest.json](r21-genre-manifest.json)). 14 passages were
  skipped for rewrite-corpus overlap, 13 of them encyclopedic, which is
  expected because WikiLarge comes from Wikipedia. That left 121 passages
  (493 sentences), 138 items and three fresh blind raters.
- **Agreement:** all three raters agreed on 89%, and pairwise agreement was
  92%. Consistency with the old marks was 14–17 of 20.
- **Realization:** the a/an fix changed no item in this window.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Unseen pairs (Good) |
|---|---:|---:|---:|---:|---:|---:|
| **News + encyclopedic (312 sentences)** | | | | | | |
| b: current stage | 68 | 100% | 21.8 | 64.7% (53–75) | 11.8% | 40 (22) |
| gs | 21 | 31% | 6.7 | 90.5% (71–97) | 4.8% | 9 (7) |
| **p50 (frozen, round 19)** | **18** | **26%** | **5.8** | **94.4% (74–99)** | **0%** | 3 (2) |
| r50 (refit, 1,259 rows) | 17 | 25% | 5.4 | 100% (82–100) | 0% | 2 (2) |
| q50 | 18 | 26% | 5.8 | 94.4% (74–99) | 0% | 3 (2) |
| **Technical (181 sentences)** | | | | | | |
| b | 43 | 100% | 23.8 | 60.5% (46–74) | 20.9% | 7 (5) |
| p50 | 21 | 49% | 11.6 | 90.5% (71–97) | 9.5% | 3 (3) |
| q50 | 21 | 49% | 11.6 | 90.5% (71–97) | 9.5% | 3 (3) |
| **Whole window** | | | | | | |
| b | 111 | 100% | 22.5 | 63.1% (54–71) | 15.3% | 47 (27) |
| p50 | 39 | 35% | 7.9 | 92.3% (80–97) | 5.1% | 6 (5) |
| r50 | 37 | 33% | 7.5 | 94.6% (82–99) | 5.4% | 4 (4) |
| q50 | 39 | 35% | 7.9 | 92.3% (80–97) | 5.1% | 6 (5) |

- **Genre test (frozen p50): precision holds, yield does not.**
  - On news and encyclopedic text, p50 reached 94.4% Good (17/18) with no
    Changes meaning. It made 26% as many swaps as arm b; the bar was 40%.
    That missed the pre-registered bar on yield alone, as predicted.
  - It made 5.8 swaps per 100 sentences, against about 12 on technical
    text.
  - Arm b on these genres reached 64.7%, about its usual technical rate.
- **Unseen pairs decide the genre gap.**
  - 40 of arm b's 68 genre swaps used a pair absent from all 1,259 rated
    rows (55% Good), against 7 of 43 on technical text.
  - p50 kept 3 of the 40, and q50 also 3.
- **q50 is not a different arm in practice.** It chose exactly p50's 39
  swaps, so it missed its yield and unseen-pair bars. Its CV gain in the
  unseen regimes did not translate into different decisions at this
  threshold.
- **Errors (p50, q50):**
  - "orally administered" → "orally given" (again);
  - "terminated for default" → "ended" (a legal term);
  - centred→centered (Not simpler).
- **Per rater:** p50 Good was 87.2%, 92.3% and 94.9%. On genre text it was
  88.9%, 94.4% and 94.4%.
- **Joshua's page:** by the pre-registered rule, it uses p50.

## Round 22: a genre threshold, on a fresh genre window (November 22)

[ROUND22.md](ROUND22.md) holds the pre-registered rules
([r22-precommit.json](r22-precommit.json)).
- **Model:** **arm qg** is round 21's arm q refitted on all 1,377 rated rows
  (rounds 3–21).
- **Threshold:** it keeps 40% of round 21's news and encyclopedic arm-b rows,
  by passage-grouped out-of-fold score.
- **Ablation:** arm rg is arm p with the same rule.
- **Motivation:** ranked by q50 score, round 21's 68 genre rows were 92.6%
  Good in their top 40%. This look used round 21's marks, now development
  data.
- **Gate ([r22-cv.json](r22-cv.json)): go.**
  - Held-out genre rows: qg 92.6% Good at 40% kept (68 rows, so noisy); rg
    92.4%.
  - Technical rows at the technical threshold: 91.9%.

Fresh window ([round22.json](round22.json)): 105 new passages (45 Wikinews,
60 featured-article leads). 19 were skipped for rewrite-corpus overlap,
leaving 86 passages (356 sentences), 97 items and three fresh blind raters.
All three agreed on 87%. Consistency with the old marks was 13–18 of 20.

| Arm | Swaps | Yield vs b | Per 100 sentences | Good (95% CI) | Changes meaning | Unseen pairs (Good) |
|---|---:|---:|---:|---:|---:|---:|
| b: current stage | 75 | 100% | 21.1 | 65.3% (54–75) | 18.7% | 43 (26) |
| gs | 22 | 29% | 6.2 | 90.9% (72–97) | 4.5% | 11 (9) |
| p50 (frozen, round 19) | 33 | 44% | 9.3 | 90.9% (76–97) | 9.1% | 10 (9) |
| q50 (frozen, round 21) | 28 | 37% | 7.9 | 92.9% (77–98) | 7.1% | 9 (8) |
| rg (refit, genre threshold) | 37 | 49% | 10.4 | 89.2% (75–96) | 10.8% | 12 (10) |
| **qg (refit, genre threshold)** | **36** | **48%** | **10.1** | **91.7% (78–97)** | **8.3%** | 11 (10) |

- **Pre-registered success: no, by one swap.** qg reached 91.7% Good and
  48% of arm b's yield, but Changes meaning was 8.3% (3 of 36) against a
  bar of 8%.
- **The window, not the model, moved yield most.** Frozen p50 made 44% of
  arm b's swaps here, against 26% on round 21's genre passages. The genre
  threshold added 3 swaps over p50 (+4 points of yield) at about the same
  precision.
- **Unseen pairs did better on this window:** qg kept 11, with 10 Good, and
  p50 kept 10, with 9 Good. On round 21's genre passages, p50 kept 3.
- **Errors (qg):**
  - migrate→move (loses the migration sense);
  - "has operated Hawk 127" → "has run";
  - "indicated a strong showing" → "showed a strong showing". The
    repeated-stem feature was on here, but its learned weight did not
    outweigh the pair prior.
- **Per rater:** qg Good was 86.1%, 91.7% and 94.4%.

## Rounds 21–22 summary

- **Gap 1, unseen pairs:**
  - **Measured:** p50 keeps about 13% of unseen-pair technical swaps, at
    83% Good, against 93% on seen pairs.
  - **No generalization tested lifts unseen-pair precision to 90% in CV.**
    The tested ideas were back-off to the replacement, the stem family and
    the target stem family; embedding neighbours; shrinkage to the arm-s
    score; and refits on 31 features. The best separate unseen threshold
    kept 18–26% at 73–76% Good.
  - **q50 gained 1.6–1.7 points in pair- and lemma-grouped CV.** It did not
    change any fresh decision in round 21.
  - **On fresh text, unseen pairs were few and mostly Good when kept:**
    round 21's p50 kept 6 (5 Good) and round 22's qg 11 (10 Good).
- **Gap 2, meaning residue:**
  - The a/an realization fixes a class seen 5 times in rounds 3–20 (all
    Changes meaning). It changed no item in rounds 21–22.
  - The quote and repeated-stem features have learned weights. They removed
    the quoted-text error in CV but not "showed a strong showing" on fresh
    text.
  - Frame errors remain the main residue: "orally administered" (again in
    round 21), "terminated for default", migrate, operated.
- **Gap 3, genre:** on news and encyclopedic text, frozen p50 reached 92.2%
  Good (47/51, CI 81–97) at 36% of arm b's yield over two windows. Arm b
  there was 65%, about its technical rate.
  - Precision transfers. Yield depends on the window: 26% in round 21 and
    44% in round 22.
- **Caveats:**
  - All marks come from AI raters.
  - The genre threshold needs the text type at run time.
  - Encyclopedic passages lose about 30% to the rewrite-corpus overlap
    check.
  - The Federal Register corpus has no unused documents left (16 found for
    Joshua's window).

## Joshua's blind marking page (prepared November 22, not yet used)

[JOSHUA-PAGE.md](JOSHUA-PAGE.md) holds the rules
([joshua-precommit.json](joshua-precommit.json),
[joshua-page.json](joshua-page.json)).
- **Arm:** p50, by round 21's pre-registered rule, because q50 missed its
  bars.
- **Windows:** no AI rater has seen either one:
  - round 21's Joshua window (124 passages after overlap checks: 43 news, 35
    encyclopedic, 16 Federal Register, 30 openFDA);
  - a pre-registered 40-passage openFDA supplement. It was added because
    the Federal Register corpus ran out and p50 swaps rarely on genre text.
- **Page:** 100 items in one seeded shuffle, with no scores or labels.
  - 60 of the 61 p50 swaps;
  - 40 of the 113 arm-b swaps that p50 rejects.

  By source, the page holds 49 openFDA, 17 Federal Register, 19 news and 15
  encyclopedic items.
- **Location:** `~/Documents/cleartext-rating/rate-learned-swaps-r21.html`.
  Marks download as `cleartext-marks-learned-swaps-r21.csv`, and the key is
  in the state directory's `joshua-r21/key.json`.
- **Arm b's Good rate on these windows** can be estimated as (60 × p50 Good
  + 113 × rejected Good) / 173, from the two groups' rates. 60 of the 61 p50
  swaps are also arm-b swaps.
- **Use of the marks:** Joshua's marks decide adoption (≥ 90% Good). They
  are never used to train, tune or choose anything.
