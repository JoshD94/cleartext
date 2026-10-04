# Eval weight training — Week 3 (Kea-Roy)

Task from Working Doc: "Get trained weights for eval based on human-rated
SimpEval data (make sure as close as possible to human evals). Combine datasets?"

Base: `evaluation/evaluate.py` from PR #1 (vendored here as `evaluate_base.py`),
which implements the Week 1 formulas (C(w,s) word complexity, S(x) sentence
simplicity, information preservation).

## What was trained

**Stage A — word weights A–E** on CompLex single-word human complexity ratings
(SemEval-2021 Task 1 LCP; 7,662 train / 421 trial rows).
Features per the formula: −Zipf frequency (wordfreq), word length, syllable
count, context complexity (surrounding words), POS. F (POS) kept at PR #1's
default 0.1 — CompLex has no POS labels, so it is untrainable here.

| weights | trial Pearson r vs humans |
|---|---|
| trained A–E | **0.68** |
| PR #1 defaults | 0.57 |

→ **Use the trained A–E.** (`evals/trained_weights.json`)

**Stage B — sentence weights G/H/I** on Lens SimpEval human simplicity ratings
(simpDA_2022: 360 pairs, absolute 0–100; simplikert_2022: same 360 pairs,
relative −2…+2 Likert). Split by original-sentence id: 48 train / 12 held-out.
Updated 2026-10-04: S_lex redefined as **threshold-based** (normalized reduction
in the count of words with C(w,s) > 0.5) instead of mean-complexity difference.

| weights | train r | held-out r |
|---|---|---|
| fitted (max correlation) | 0.56 | **0.63** |
| fitted (max corr, combined datasets) | 0.56 | 0.62 |
| fitted (least squares) | 0.38 | 0.57 |
| S_read alone | 0.38 | 0.57 |
| S_lex alone (threshold-based) | 0.46 | 0.44 |
| PR #1 default (⅓, ⅓, ⅓) | 0.40 | 0.38 |
| S_jargon alone (zipf<3.0 proxy) | 0.19 | 0.16 |

→ **Use the fitted G/H/I (max correlation): G=0.453, H=0.505, I=0.042**
(`evals/trained_weights.json`). The redefinition fixed S_lex (alone: 0.05 →
0.44 held-out); the fit now clearly beats the ⅓ defaults (0.63 vs 0.38). The
defaults underperform because S_jargon — here a zipf<3.0 rarity proxy on news
text — is near-noise (r=0.16), and the fit correctly downweights it. Least
squares disagrees with max-corr (collapses toward S_read alone); correlation is
the stated goal ("as close as possible to human evals"), so max-corr stands.

**"Combine datasets?"** — pooling simpDA + simplikert gives near-identical
fitted weights (0.619 vs 0.626 held-out). The Likert view is redundant (same
360 pairs).

Sanity check: trained S ranks systems the same as humans for the top 3
(Human-1 > Human-2 > MUSS); swaps GPT-3-few-shot and T5-3B at #4/#6.

## Caveats / required code change

- ~~PR #1's `evaluate_record` called `word_complexity(w)` with **empty context**,
  zeroing the E term.~~ **Fixed 2026-10-04**: `evaluate_base.py` now passes each
  word's sentence as context (`word_complexity_in_context`), accepts optional
  `record["word_weights"]` (e.g. the trained A–E), and uses the threshold-based
  S_lex. The diff against the PR branch is saved as `evals/pr1_fixes.patch`
  (applies cleanly) — propose applying it on the PR branch before merge.
- ~~S_lex measured mean-complexity *improvement* while humans rate the simplified
  sentence *absolutely* (held-out r=0.05).~~ **Redefined 2026-10-04** as
  threshold-based reduction in the count of complex words (held-out r=0.44
  alone). Shared implementation: `evaluate_base.lexical_simplicity`, used by both
  the formula and `train_weights.py`.
- S_jargon still needs a real per-domain technical-term set; training used a
  zipf<3.0 rarity proxy on SimpEval news (weak: r=0.16), which is why the fitted
  I weight is small (0.042).
- CompLex TSVs contain unbalanced double quotes — parse with naive tab-split,
  not `csv` (see `load_complex`). Same for the benchmark TSVs (`datasets.py`).

## Diagnostics (2026-10-04): ceiling, skew fix, non-linear check

**Human-agreement ceiling.** CompLex ships only averaged ratings, so the ceiling
was estimated on CWI-2018 English, which records native (10) and non-native (10)
annotator counts separately: split-half Pearson **r=0.77** between the two
independent rater groups, **r=0.87** Spearman-Brown stepped-up to 20 raters.
So 0.68 is genuinely below the ceiling — real headroom exists.
(`evals/diagnose_stage_a.py`, `results/stage_a_diagnostics.json`)

**Train/serve skew fix.** Weight training fit B against *real* Zipf frequencies
(`wordfreq`), but the deployed `word_complexity` fell back to a length-based
proxy (`6.0 - len/2`) when no Zipf value was passed. Now it uses `wordfreq`
when installed (proxy only as fallback). Included in `pr1_fixes.patch`.

**Non-linear diagnostic** (same CompLex split, same 4 features):

| model | train r | trial r |
|---|---|---|
| linear least squares (current) | 0.666 | 0.681 |
| ridge CV | 0.666 | 0.681 |
| gradient boosting | 0.805 | **0.737** |

GBM feature importance: −Zipf 0.85, word length 0.16, context 0.08, syllables
0.06. Two conclusions: (a) the linear-sigmoid form leaves ~0.06 correlation on
the table — the features carry more signal than the formula extracts; (b) the
fitted **C (length) weight is negative (−0.0172)**, a multicollinearity artifact
(length ↔ frequency are correlated, so least squares pushes C negative to
compensate). The weights predict well but are not interpretable feature-by-feature.

**Options from here:**
- A. Keep linear + trained weights (0.68). Cost: none. Caveat: C-negative wart.
- B. Fix collinearity (drop/orthogonalize length vs frequency), refit. Cleaner
  model, maybe a small gain.
- C. Adopt GBM for C(w,s) (0.74, nearer the 0.77–0.87 ceiling). Cost: C(w,s) is
  no longer the Week 1 linear formula — but it stays non-LLM, and the
  sentence-level S = G·S_lex + H·S_read + I·S_jargon structure is untouched.
  The Working Doc task says "as close as possible to human evals", which favors
  accuracy over formula fidelity.
- D. Add features (concreteness / age-of-acquisition norms; pool CWI-2018
  probabilistic scores as auxiliary training data) — likely +0.02–0.05 on top
  of either B or C.

## Options C+D (2026-10-04): GBM word model + new features

**C: gradient boosting for C(w,s)** — 6 transparent features, still non-LLM
(no embeddings). **D: + concreteness** (Brysbaert et al. 2014, 37k words) **+
age-of-acquisition** (Kuperman et al. 2012, 31k words) via `evals/lexicons.py`;
**+ CWI-2018 train pooled** as auxiliary data (single-word targets,
probabilistic target). Ablation, held-out (never trained on):

| model / features / train | CompLex trial r | CWI dev r |
|---|---|---|
| linear / base4 / CompLex | 0.681 | 0.301 |
| gbm / base4 / CompLex | 0.715 | 0.307 |
| linear / base6 / CompLex | 0.751 | 0.324 |
| **gbm / base6 / CompLex** | **0.808** | 0.266 |
| gbm / base6 / CWI only | 0.368 | 0.757 |
| linear / base6 / +CWI | 0.576 | 0.491 |
| gbm / base6 / +CWI | 0.586 | 0.732 |

- **Winner: GBM + 6 features, CompLex only: trial r=0.808** (vs 0.68 linear;
  ceiling 0.77–0.87 — now inside the ceiling band). The jump 0.78→0.81 came
  from de-inflected lexicon lookup (see below); features and non-linearity
  combine super-additively.
- **Lexicon coverage 59%→83%**: `lexicons.py` now falls back through
  de-inflected forms (`troops`→`troop`, `stopped`→`stop`), accepting a
  candidate only if it is in the lexicon (safe: `bias`→`bia` is rejected).
  Coverage on CompLex train tokens: concreteness 0.828, AoA 0.854.
- **CWI pooling hurts the headline metric** (0.81→0.59): Likert-mean vs
  fraction-marking-complex are different constructs. Symmetric check confirms
  it: a CWI-trained GBM hits 0.76 on CWI-dev but only 0.37 on CompLex-trial.
  Not pooled; CWI-dev stays as an out-of-scheme robustness check.
- Artifact: `evals/weights/word_gbm.joblib` (+ meta); API: `evals/word_models.py`
  (`word_complexity_gbm`, `lexical_simplicity_gbm`) — the single featurization
  source for train and serve.
- **Detection** (`detection_baseline.py --word-model gbm`): CWI dev **F1=0.63**
  (P=0.47, R=0.98, threshold 0.20 tuned on train) vs 0.62 linear. The GBM needs
  the low threshold because its scores are calibrated to CompLex's stricter
  construct while CWI's label is lenient.
- **Stage B refit with GBM S_lex** (threshold 0.3, tuned on train; 0.5
  collapses since calibrated GBM scores put most news words below 0.5):
  max-corr test r=0.54 (G=0.25, H=0.68, I=0.07) vs **0.60 linear-based**.
  The GBM's win is at the word level; **sentence weights stay linear-based**
  (G=0.457, H=0.500, I=0.043).

`./venv/bin/python evals/train_word_model.py` reproduces (ablation +
Stage B refit); full table in `results/word_model.json`.

## Detection baseline (2026-10-04)

`evals/detection_baseline.py` scores CWI-2018 English with the trained A–E
weights (real context, real Zipf after the skew fix). Threshold tuned on train
(27,299), reported on held-out dev (3,328): **P=0.48, R=0.86, F1=0.62** at
threshold 0.55. (The sigmoid midpoint 0.5 over-predicts "complex" — CWI's binary
label is lenient, ≥1/20 annotators — so the detector's operating point is tuned,
not 0.5.)
TSAR-2022 (373) and MultiLS (30) gold rankings are loaded; accuracy@k / MRR
await a substitute candidate generator from the models side.

## Reproduce

```
./venv/bin/python evals/train_weights.py       # writes evals/trained_weights.json
./venv/bin/python evals/detection_baseline.py  # writes results/detection_baseline.json
./venv/bin/python evals/build_status.py        # rebuilds evals/status.html
```

## Recommended weights

```json
word:     {A: 0.6626, B: 0.0976, C: -0.0172, D: 0.0270, E: 0.0142, F: 0.1}
sentence: {G: 0.453, H: 0.505, I: 0.042}   # max-corr fit on simpDA (updated 2026-10-04)
```
