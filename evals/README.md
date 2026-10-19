# ClearText evals

Evaluation pipeline implementing **Section 3** of the ClearText project proposal.
It evaluates each system variant on two dimensions and supports the
module-by-module ablation study from Section 3.2.

## Status page

`status.html` is the live visual summary of this folder — module status, weight
training results, baseline smoke-test numbers, open items. It is **generated**,
not hand-written: `build_status.py` reads the numbers from
`results/eval_results.json` and `trained_weights.json` and the module cards
from the `.py` files themselves (docstrings via `ast`), so it can't drift from
the code. Open it in any browser (double-click the file).

```bash
python evals/build_status.py   # regenerate manually
```

It rebuilds automatically at the end of `train_weights.py` and `run_evals.py`.

## Dimensions & metrics

**Readability and simplicity** (`readability.py`)
- Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog, SMOG
- Average sentence length, average word length, technical term count

**Meaning preservation** (`preservation.py`) — full Section 3.1 suite:
- TF-IDF cosine similarity
- Embedding cosine similarity (`--embedding-model`; needs
  `sentence-transformers` + torch)
- NLI entailment, both directions (`--nli-model`; needs `transformers` +
  torch): forward = simplified entailed by original (no hallucinations),
  backward = original entailed by simplified (nothing dropped)
- Explicit checks: fraction of numbers, dates, named entities, and
  negations from the original that survive simplification

**Module-level** (`detection.py`)
- Jargon/complexity detection: precision, recall, F1 (use with CWI 2018 labels)
- Lexical simplification ranking: accuracy@k, MRR (use with MultiLS / TSAR-2022)

## Baselines

- `original` — identity baseline (from the proposal: "The original text")
- `rule_based` — dictionary substitution using `data/simplification_dict.json`
  (extend the dictionary as the team curates more mappings)

## Run it

```bash
pip install -r requirements.txt
python run_evals.py                       # original vs rule-based, sample passages
python run_evals.py --systems original,rule_based --out results
python run_evals.py --embedding-model all-MiniLM-L6-v2 \
    --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli   # full preservation suite
```

Outputs: `results/eval_results.json` (per-passage + aggregates) and
`results/eval_results.md` (results table).

## Ablation study (proposal Section 3.2)

The 7-way ladder — original, rule-based, lexical-only, structural-only,
lexical+structural, full without guardrails, full pipeline — is defined in
`runner.default_ablation_order()`. As each team module lands, register it in
`run_evals.py`:

```python
from cleartext.lexical import LexicalPipeline
register_system("lexical_only", LexicalPipeline().simplify)
```

then run:

```bash
python run_evals.py --systems original,rule_based,lexical_only,structural_only,lexical_structural,full_no_guardrails,full_pipeline
```

`runner.compare_against_baseline()` adds paired t-tests of each variant
against the original-text baseline (p-values need `scipy`).

## `train_weights.py` — human-calibrated weights (Week 3)

Trains the eval formula weights against human judgments, per the Working Doc
task ("get trained weights for eval ... as close as possible to human evals").
Built on PR #1's `evaluation/evaluate.py` (vendored here as `evaluate_base.py`).

- **Stage A:** word weights A–E on CompLex human complexity ratings → held-out
  Pearson r **0.68** vs 0.57 for PR #1 defaults. **Use the trained values.**
- **Stage B:** sentence weights G/H/I on Lens SimpEval human simplicity
  (simpDA_2022 + simplikert_2022). With the 2026-10-04 threshold-based S_lex
  redefinition, fitted max-corr weights (**G=0.453, H=0.505, I=0.042**) reach
  held-out r **0.63** vs 0.38 for the ⅓-defaults → **use the fitted values**.
- "Combine datasets" tested: pooling both SimpEval views is redundant.
- Writes `trained_weights.json`; full write-up in `WEIGHTS_REPORT.md`.
- 2026-10-04 fixes (also saved as `pr1_fixes.patch` for the PR #1 branch):
  `evaluate_record` now passes real sentence context (E term was zeroed),
  accepts `record["word_weights"]`, and S_lex is threshold-based
  (`lexical_simplicity`), shared with `train_weights.py`.

```bash
./venv/bin/python evals/train_weights.py
```

## Benchmark datasets & detection baseline

`datasets.py` loads the proposal's benchmark sets from `data/benchmarks/`
(download once with `python evals/datasets.py --fetch`):

- **CWI 2018** English train/dev (27k/3.3k): binary complex-word labels →
  `detection.py` P/R/F1.
- **TSAR-2022** English test_gold (373) and **MultiLS 2024** English trial
  (30): annotator-ranked gold substitutes → `detection.py` accuracy@k / MRR
  (needs a candidate generator from the models side).

`detection_baseline.py` runs the CWI end-to-end check
(`--word-model linear|gbm`, default linear): threshold tuned on train (27,299),
held-out dev (3,328). Linear: **P=0.48, R=0.86, F1=0.62** @0.55;
GBM: **P=0.47, R=0.98, F1=0.63** @0.20. (CWI's binary label is lenient, ≥1/20
annotators, so the detector's operating point is tuned, not 0.5.)
→ `results/detection_baseline.json`, shown on the status page.

## Word model: GBM + psycholinguistic features (options C+D)

`evals/train_word_model.py` trains the gradient-boosted C(w,s) (option C, still
non-LLM) on 6 features — the base 4 plus Brysbaert concreteness and Kuperman
age-of-acquisition from `evals/lexicons.py` (option D). CWI-2018 pooling was
tested and rejected (different annotation construct; hurts the CompLex metric).
CompLex trial r: **0.81** vs 0.68 linear (human-agreement ceiling 0.77–0.87 —
the model is now inside the ceiling band). Artifact:
`evals/weights/word_gbm.joblib`; scoring API: `evals/word_models.py`.
Full ablation in `WEIGHTS_REPORT.md`, live table on the status page.

```bash
./venv/bin/python evals/train_word_model.py   # ablation + Stage B refit, ~9 min
```

## Robustness checks

- `evals/stage_b_robustness.py` — repeats the Stage B 48/12 split 25 times
  (seeded) and reports held-out test r as mean±std: fitted 0.55±0.06 vs
  PR#1 defaults 0.39±0.06. The G/H/I recommendation is stable across splits
  (G=0.43±0.02, H=0.54±0.02, I=0.03±0.01). Results in
  `results/stage_b_robustness.json`, live section on the status page.
- `evals/lexicons.py` de-inflected lookup raised CompLex token coverage
  59%→83% (safe: candidates accepted only if in the lexicon).
- Cross-scheme transfer is characterized, not assumed: CompLex-trained models
  score 0.81 on CompLex-trial but 0.27 on CWI-dev; CWI-trained models score
  0.76 on CWI-dev but 0.37 on CompLex-trial. CWI-dev stays as an
  out-of-scheme robustness check.

## Notes

- Per the proposal, ClearText itself stays entirely non-LLM. External models
  (NLI, sentence embeddings) are used here for *evaluation only*.
- `technical_term_count` is a heuristic (rare-word proxy via `wordfreq`);
  treat it as a trend signal, not ground truth.
- Sample passages in `data/sample_passages.json` are placeholders — swap in
  the team's compiled corpus (arXiv abstracts, 10-K excerpts, Congressional
  Record speeches) when ready.
