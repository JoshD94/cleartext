# Frozen selection rule (written before any scoring run)

Frozen 2026-11-09, before the ensemble sweep produced any score. Git history
for this file is the evidence that it predates the results. Do not edit it
after scoring starts; any revision appears in the report as a separately
labeled alternative ranking.

## Panel and metrics

- Panel: the 90 sourced development passages used by the Week 7 compound
  replay (`evals/reports/20261102-compound-boundaries/native`), 30 each from
  computer science, corporate communication and political science.
- Metrics come from `evals/readability.py` and `evals/preservation.py` with the
  Week 7 evaluation models (`all-MiniLM-L6-v2` embeddings,
  `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` NLI), CPU, no truncation.
- "NLI mean" is the passage-level bidirectional mean `nli_mean`
  (average of forward and backward entailment probability). Any
  sentence-level NLI added by the study is reported alongside and does not
  replace this gate.

## Winner

Among configurations that pass **all** gates, the winner has the **lowest mean
Flesch–Kincaid grade** over the 90 passages.

Gates:

1. Mean passage-level bidirectional NLI over the 90 passages is **≥ 0.95**.
2. At most **5%** of passages (≤ 4 of 90) have passage-level NLI mean **< 0.80**.
3. `numbers_preserved` and `negations_preserved` equal **1.0 on every passage**.
4. Mean `entities_preserved` is **no lower than the `rule_based` baseline's**
   mean on the same 90 passages.

An NLI abstention (pair too long for the model) counts as failing gate 2 for
that passage and is excluded from the gate 1 mean; abstention counts are
reported.

Ties: configurations whose mean FK grade is within **0.1** of the best passing
configuration are tied; the tie is broken by the higher **ASSET test SARI**.

## Reporting

- The full Pareto table (FK grade versus passage-level NLI mean, all configs
  run) is reported regardless of which configurations pass.
- If no configuration passes every gate, the report says so; it does not pick a
  winner under this rule.
- If a gate looks badly calibrated after the NLI diagnosis, the report says so
  and adds a clearly labeled alternative ranking. This file stays unchanged.
