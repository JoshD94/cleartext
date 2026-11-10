# Formula evaluation

`evaluate.py` implements the Week 1 simplicity, complexity, and information
preservation formulas as a dependency-free **temporary baseline**. It is
intended to make the metric interfaces executable before the project has its
datasets and model-backed components. The input is JSON Lines (JSONL), with one
original/simplified pair per line:

```json
{"original": "The algorithm has high latency.", "simplified": "The method is slow."}
```

Run it from the repository root:

```bash
python3 evaluation/evaluate.py evaluation/examples/sample.jsonl
```

To see the input options and an explanation of every reported metric:

```bash
python3 evaluation/evaluate.py --help
```

## What is implemented

- Word-level complexity with a sigmoid and configurable A-F weights.
- Sentence-level `S_lex`, `S_read`, and `S_jargon` components.
- Weighted sentence simplicity score `S`.
- Baseline information-preservation metrics and `loss_info`.
- Candidate utility `U` and the proposal's acceptance guardrail.
- JSONL input validation and unit tests.

The script reports word complexity, `S_lex`, `S_read`, `S_jargon`, the weighted
sentence score `S`, `M_tfidf`, `M_embedding`, `M_nli`, `P_critical`,
`P_meaning`, `loss_info`, replacement utility `U`, and the guardrail decision.
The default sentence weights are one third each; provide `sentence_weights`
with `G`, `H`, and `I` to override them. `technical_terms`, `lambda`, and
`threshold` can also be supplied per record.

### Temporary versus production components

The baseline uses transparent proxies where the proposal calls for trained or
model-backed components:

- length-based Zipf fallback instead of corpus frequency;
- numeric POS input rather than POS tagging;
- token-overlap embedding/NLI proxies;
- weighted token overlap instead of fitted TF-IDF;
- a small technical-term list instead of a domain jargon detector;
- regular-expression entity/date/quantity/negation checks.

Future agents should replace these pieces before treating results as final
metrics. The script currently evaluates supplied original/simplified pairs; it
does not train weights, generate replacements, or search over candidate
synonyms.

## Remaining work

1. Add the real formula-definition and simplification datasets, including
   train/dev/test splits.
2. Train or integrate Zipf/POS features and learn the A-F word weights.
3. Replace the embedding, NLI, and TF-IDF proxies with validated components.
4. Build a domain-specific jargon detector and protected-jargon synonym
   checks.
5. Validate the normalization rules for entities, numbers, dates, quantities,
   negations, and technical terms against labeled examples.
6. Tune `G`, `H`, `I`, `lambda`, and the acceptance threshold on development
   data, then report held-out baseline metrics.

Blank lines are ignored. Invalid JSON, missing fields, non-string values, and
empty datasets are reported as errors rather than silently producing metrics.
