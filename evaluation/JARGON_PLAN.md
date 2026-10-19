# Jargon Simplicity Plan

This plan defines how to implement the proposal's jargon simplicity component.
The goal is to reward reductions in difficult, replaceable technical language
without rewarding meaning loss or removal of necessary terminology.

## Scope

For an original text `x` and a simplified text `y`, the evaluator should:

1. Detect technical terms and multi-word technical phrases.
2. Estimate their difficulty and domain specificity.
3. Classify terms as replaceable or protected.
4. Measure whether replaceable jargon was reduced.
5. Verify that protected jargon and meaning-critical information were preserved.

The first implementation should be transparent and deterministic. Model-backed
components can be added after the baseline has data and evaluation labels.

## Stage 1: transparent baseline

### Term resources

Create domain-specific resources for:

- technical terms and phrases;
- difficulty scores;
- simpler synonym candidates;
- protected terms;
- domain and general-language frequency estimates.

Use a versioned JSON format similar to:

```json
{
  "latency": {
    "difficulty": 0.8,
    "synonyms": ["delay", "response time"],
    "protected": false
  },
  "TCP": {
    "difficulty": 0.9,
    "synonyms": ["network protocol"],
    "protected": true
  }
}
```

Keep resource terms lowercase for matching, while preserving the original
surface form for reporting. Support phrase entries of two or more words.

### Detection

1. Tokenize text while retaining character spans.
2. Match glossary phrases from longest to shortest.
3. Do not count words inside an already matched phrase separately.
4. Match case-insensitively, but preserve the original text in output.
5. Add configured domain terms even when they are not rare.
6. Avoid classifying named entities, identifiers, units, and acronyms as
   replaceable by default.

Each detected term should produce an inspectable record:

```json
{
  "term": "latency",
  "start": 21,
  "end": 28,
  "is_jargon": true,
  "difficulty": 0.8,
  "protected": false,
  "replaceable": true,
  "replacement": "delay",
  "confidence": 0.9
}
```

### Replaceability

A term is replaceable only when:

- it has at least one configured synonym;
- the synonym is simpler than the source term;
- the synonym is not protected;
- the synonym preserves the term's grammatical role where known;
- the synonym passes the information-preservation guardrail.

Terms without a safe synonym remain jargon but are protected from automatic
replacement. A protected term must not be counted as a successful reduction
unless it remains preserved in the simplified text.

### Baseline scoring

For each text, calculate weighted replaceable jargon:

```text
J(text) = sum(difficulty(term) for term in replaceable_jargon(text))
```

Then calculate the proposal's jargon simplicity score:

```text
S_jargon = clip(
    (J(original) - J(simplified)) / max(J(original), epsilon),
    0,
    1
)
```

Also report these diagnostic values:

- number of detected jargon terms;
- number of replaceable terms;
- weighted jargon before and after;
- number of protected terms;
- protected-term preservation rate;
- list of detected terms and decisions.

Counting only replaceable jargon prevents the metric from rewarding deletion
of necessary technical vocabulary.

## Stage 2: statistical improvements

After the baseline resources exist, add:

- general-language and domain-corpus frequencies;
- domain specificity:

  ```text
  log((freq_domain(term) + epsilon) /
      (freq_general(term) + epsilon))
  ```

- a combined difficulty score using rarity, word complexity, and domain
  specificity;
- phrase mining for technical n-grams not yet present in the glossary;
- POS-aware synonym filtering;
- context-aware candidate ranking.

Use labeled development examples to tune jargon and replaceability thresholds.
Do not tune against the held-out test set.

## Stage 3: model-backed validation

Replace or supplement baseline checks with:

- contextual embeddings for synonym and context fit;
- bidirectional NLI for meaning preservation;
- a domain-specific jargon classifier;
- human judgments for jargon status, simplicity, and replaceability;
- calibrated confidence scores.

The model-backed detector should retain the same inspectable term-level output
schema so baseline and production results remain comparable.

## Protected jargon and guardrails

Protect the following by default:

- named entities;
- acronyms and initialisms;
- standards, protocols, and formal identifiers;
- product names;
- legal, medical, or safety-critical terms where precision matters;
- units, symbols, and code identifiers;
- terms without a validated simpler replacement.

For every candidate simplification, require:

```text
loss_info <= threshold
and P_critical == 1
```

Any protected-term deletion, changed number/date/quantity, or changed negation
should fail the guardrail unless an explicit domain rule says otherwise.

## Data and evaluation requirements

Create examples with labels for:

- jargon versus ordinary language;
- phrase boundaries;
- difficulty;
- protected versus replaceable;
- valid synonym replacements;
- meaning preservation after replacement.

Report at least:

- jargon detection precision, recall, and F1;
- replaceability classification precision, recall, and F1;
- protected-term preservation;
- `S_jargon`;
- information-loss guardrail acceptance rate;
- human-rated simplicity and meaning preservation when available.

Include adversarial cases:

- ambiguous words such as `model`, `network`, and `cell`;
- overlapping phrases;
- acronyms and named entities;
- technical terms with no safe synonym;
- synonym replacements that change negation, quantities, or scope.

## Implementation order

1. Add glossary, protected-term, and synonym resource files.
2. Implement longest-match phrase detection with term-level diagnostics.
3. Add replaceability checks and weighted `S_jargon`.
4. Integrate protected-term preservation with `P_critical`.
5. Add labeled fixtures and targeted unit tests.
6. Add corpus frequencies and threshold tuning.
7. Integrate embedding/NLI and compare against the transparent baseline.

Until stages 1–4 are complete, jargon metrics should be labeled provisional in
reports.
