# Combined-system test results — Week 7 (Oct 26–Nov 1)

Kea-Roy Ong, evals side. Deliverable: test the combined system on
representative passages; test results across the three target domains
(CS, corporate communication, political science).

## Test set

`evals/data/week7_passages.json`: 18 passages, 6 per domain.

- Passages 1–6 are the project's existing sample passages, copied
  verbatim from `evals/data/sample_passages.json` (the Oct 21 set).
- Passages 7–18 (cs-03–06, corp-03–06, polsci-03–06) are newly authored
  for this week: 2–4 sentence technical passages with multi-clause
  sentences, jargon, and numbers, dates, negations, and named entities,
  so the structure stage has something to split and the explicit checks
  have something to verify. Authored passages are test inputs, not
  benchmark claims; no human quality labels exist for any of them.

## Results, source A: benchmark domain panel (executed Oct 27–28)

Joshua Dirga's ten-system comparison, branch
`josh/structural-models-20261026` commit `085015f`, evidence
`models/evidence/pipeline-20261027/combined_per_example_metrics.json`.
Domain panel: 24 real-text inputs, 8 per source (QUIC RFC 9000 = CS,
Apple 10-K 2025 = corporate, Biden inaugural 2021 = political science),
scored with this eval framework's functions. FKGL is the all-input panel
mean; embedding / NLI backward are changed-only means. Full extraction:
`evals/reports/week7/evidence_domain_breakdown.json`.

| System | CS: changed / FKGL / NLI bwd | Corporate: changed / FKGL / NLI bwd | Polsci: changed / FKGL / NLI bwd |
| --- | --- | --- | --- |
| Original | 0 / 13.76 / n/a | 0 / 17.22 / n/a | 0 / 13.01 / n/a |
| Dictionary | 0 / 13.76 / n/a | 0 / 17.22 / n/a | 0 / 13.01 / n/a |
| lexical_only | 6 / 13.49 / 0.734 | 4 / 16.92 / 0.989 | 6 / 12.90 / 0.823 |
| structural_only | 1 / 13.76 / 0.992 | 1 / 16.52 / 0.988 | 1 / 12.03 / 0.980 |
| lexical_structural | 6 / 13.49 / 0.734 | 4 / 16.22 / 0.991 | 6 / 11.93 / 0.824 |
| full_no_guardrails | 7 / 10.55 / 0.837 | 5 / 14.44 / 0.989 | 7 / 10.93 / 0.876 |
| full_guarded (= benchmark full_pipeline) | 6 / 13.49 / 0.734 | 4 / 16.22 / 0.991 | 6 / 11.93 / 0.824 |

What the combined system does, per domain:

- **Computer science is the weakest panel.** Backward NLI 0.73 for every
  lexical-bearing arm, driven by two QUIC cases (quic-rfc9000-1 and -3,
  backward 0.49 and 0.02). The structure stage changes 1 of 8 CS inputs
  and does not repair either case. Detail loss on specification text is
  the open problem, not readability: FKGL barely moves (13.76 → 13.49)
  while preservation drops.
- **Corporate is the cleanest panel.** Backward NLI 0.99, 4 of 8 inputs
  changed, FKGL 17.22 → 16.22 guarded. Filing prose simplifies without
  losing the checked facts (no failed explicit checks in any arm).
- **Political science sits between.** Backward NLI 0.82, 6 of 8 changed,
  FKGL 13.01 → 11.93. One tense-check failure (biden-inaugural2021-0)
  in all arms; see error_analysis.md.
- **Guardrails matter most in CS.** Unguarded, the CS panel changes 7/8
  (vs 6) and FKGL falls to 10.55, but that extra coverage is bought with
  the unguarded-only failures documented in the error analysis. On the
  corporate and polsci panels the guarded/unguarded difference is one
  input each.
- **The dictionary baseline is inert on real text**: 0 of 24 changed.

## Results, source B: local baseline arms on the 18 authored passages

Run Nov 1 on this machine, full suite (MiniLM embeddings, bidirectional
DeBERTa NLI, explicit checks). `evals/results/week7/eval_results.json`
(+ `.md`). Model arms cannot run here; see ablation_table.md.

| System | FKGL overall | CS FKGL / changed | Corporate FKGL / changed | Polsci FKGL / changed | Embedding | NLI bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 19.17 | 18.21 / 0 | 20.27 / 0 | 19.02 / 0 | 1.00 | 0.99 |
| Dictionary | 17.64 | 16.23 / 3 | 18.40 / 4 | 18.28 / 2 | 0.97 | 0.99 |

The dictionary changes 9 of 18 authored passages (all substitutions
reviewed as text, meaning-preserving; see error_analysis.md Finding 5)
and none of the 24 real domain inputs. Its authored-passage numbers
flatter it: the authored set contains dictionary-friendly words by
construction. The single explicit-check failure (entities on corp-05)
is a verified check artifact (sentence-initial "Prior" counted as an
entity), not a model error.

Regression check: on the 6 original passages, readability and TF-IDF
values are identical to the previously stored local run (0/6 diffs for
both systems), so the Week 7 runner change (output capture) altered no
metric.

## Limitations, stated plainly

- Small panels: 8 inputs per source in Source A, 6 per domain in
  Source B. No significance claims; differences of one input move the
  means.
- Reused development inputs, not held-out benchmarks: DeSSE, ASSET,
  and the domain panel were all inspected during development. No human
  quality labels were collected.
- The combined system's *outputs* are not stored anywhere in git
  (evidence keeps hashes). Text-level verification of the flagged
  cases still has to happen on Yoga; see error_analysis.md.
- No phrase-only arm exists yet; phrases are measured only inside the
  full combinations.

## Reproduce

```bash
# Baselines on the Week 7 set (runs anywhere with evals deps):
python evals/run_evals.py --passages data/week7_passages.json \
  --systems original,rule_based --out results/week7 \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

# Model arms (needs artifacts or the demo server; see ablation_table.md):
python evals/run_evals.py --passages data/week7_passages.json \
  --systems lexical_only,phrase_only,structure_only,lexical_structural,full_guarded,full_no_guardrails \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --model-url http://127.0.0.1:8770
```
