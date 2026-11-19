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
