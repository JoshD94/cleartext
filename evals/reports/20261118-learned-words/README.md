# Learned word stage (November 18–19, 2026)

A trained four-stage word simplifier, with no LLM and no rules-only decision.
The protocol and thresholds were fixed before fitting ([PROTOCOL.md](PROTOCOL.md)).
Raw text, candidate rows and predictions stay in `~/.local/state/cleartext/learned-words-20261118/`.
Code is in `models/src/cleartext/learned_words/`, and it runs through `scripts/learned-words`.

**Status: not adopted.** F2 stays the shipped default. Joshua's round-2 blind
marks on 98 new swaps will decide adoption (≥ 90% Good).

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
