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

## Results, source C: local baselines on the 90 sourced passages

`evals/data/week7_sourced_passages.json`: 90 passages, 30 per domain,
sampled Nov 1 with a fixed seeded protocol (seed 4701). Unit: 2
consecutive sentences, 40-100 words, no sentence overlap, no overlap
with the 18-set. Every passage is asserted verbatim against its source
file, and no window spans removed content: figures, tables, headings,
and page furniture are hard breaks in the cleaned text. Each entry
carries its source title, URL, and sentence span.

Sources:

- CS (30, 10 each): RFC 9000 (QUIC, May 2021), RFC 8446 (TLS 1.3,
  Aug 2018), RFC 9110 (HTTP Semantics, Jun 2022), from the rfc-editor
  .txt files. Body only (sections 1 through References).
- Corporate (30, 10 each): Apple FY2025 10-K (accession
  0000320193-25-000079, period ended Sep 27, 2025), Amazon FY2024 10-K
  (0001018724-25-000004, Dec 31, 2024), Microsoft FY2025 10-K
  (0000950170-25-100235, Jun 30, 2025), from SEC EDGAR. Apple and
  Amazon from the complete-submission .txt; Microsoft from the primary
  .htm document (supplied as a browser-saved .mhtml, decoded from
  Windows-1252; zero replacement characters after decoding). Body
  only (Item 1 Business through Signatures), financial tables dropped.
- Political science (30, 10 each): Biden inaugural 2021, Bush
  inaugural 2001, Biden State of the Union 2024, from the American
  Presidency Project. Stage directions, applause lines, and speaker
  labels stripped; salutation, banter, and quoted-lyric windows
  excluded.

Eligible windows per document (after cleaning): RFC 9000 1019 of 2468
sentences, RFC 8446 448 of 1159, RFC 9110 1152 of 2113, Apple 491 of
982, Amazon 688 of 1341, Microsoft 722 of 1904, Biden inaugural 40 of
160, Bush inaugural 23 of 94, SOTU 84 of 587.

Run Nov 1 on this machine, full suite.
`evals/results/week7-sourced/eval_results.json` (+ `.md`, local only,
gitignored). Model arms on the 90 are still pending (Yoga; see
RUNBOOK_JOSHUA.md). Baselines only here.

| System | Flesch | FKGL | Fog | Tech terms | TF-IDF | Embedding | NLI bwd | Entities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 38.32 | 14.11 | 17.92 | 2.63 | 1.00 | 1.00 | 0.99 | 1.00 |
| Dictionary | 38.79 | 14.05 | 17.83 | 2.63 | 1.00 | 1.00 | 0.99 | 0.99 |

Per-domain originals, with dictionary coverage:

| Domain | FKGL | Flesch | Tech terms | Dictionary changed |
| --- | ---: | ---: | ---: | ---: |
| Computer science | 13.14 | 42.60 | 4.07 | 8 / 30 |
| Corporate | 17.33 | 16.83 | 2.60 | 5 / 30 |
| Political science | 11.87 | 55.52 | 1.23 | 1 / 30 |

Findings:

- Real prose is far less dense than the authored set: FKGL 14.11
  here vs 19.17 on the 18. The authored set was built jargon-dense by
  construction and flatters the dictionary; this set is the honest
  baseline.
- The dictionary's coverage problem is the finding. It changes 14 of
  90, nearly all single-word swaps, and 2 of the 14 degrade the text:
  cs-s06 "an implementation should" becomes "an carrying out
  should", and cs-s26 "MUST implement the Connection header field"
  becomes "MUST carry out the Connection header field". NLI stays
  0.99 because the swaps are small; the grammar and meaning damage is
  visible only by reading the outputs. On real text its inventory
  rarely fires, and when it does it can misfire.
- Corporate is the hardest panel by grade level (FKGL 17.33); CS is
  densest in technical terms (4.07 per passage); polsci speeches are
  the easiest (FKGL 11.87, 1.23 terms).
- The two entity-check dips (cs-s06 0.50, cs-s25 0.67) are the same
  string-check artifact as before: bracketed references and the
  "But," swap trip the entity check while NLI stays 0.99. Not model
  errors.

## Results, source D: Joshua's Nov 1 replay on the 18 passages (PR #8)

Joshua's replay, branch `josh/structural-models-20261026` commit
`85d1c18`, PR #8, evidence
`evals/reports/20261101-status-rerun/README.md`. Ten configurations on
202 unique inputs (the 190-input frozen panel plus the 6 submitted
passages plus the 12 additional Week 7 passages), scored with this
eval framework's functions. All-input means, n = 18. This answers the
18-passage model question through Joshua's own ten-configuration
harness, NOT through the Week 7 ladder: there is no standalone
`phrase_only` arm and no old-rules `structure_only` arm in this replay.
Those ladder arms remain unexecuted; see ablation_table.md and
RUNBOOK_JOSHUA.md.

| Configuration | Changed / 18 | Reading grade | Embedding | NLI backward |
| --- | ---: | ---: | ---: | ---: |
| Original | 0 | 19.17 | 1.00 | 0.99 |
| Dictionary baseline | 9 | 17.64 | 0.97 | 0.99 |
| v7 words only | 18 | 18.62 | 0.98 | 0.71 |
| T5 structure only | 4 | 18.07 | 1.00 | 0.99 |
| v7 words + T5 | 18 | 17.53 | 0.98 | 0.71 |
| v7 + phrases + T5, unchecked | 18 | 15.67 | 0.98 | 0.73 |
| v7 + phrases + T5, checked | 18 | 17.72 | 0.98 | 0.73 |
| v7 + phrase/structure rules | 18 | 18.81 | 0.98 | 0.72 |
| Contextual + phrases + T5, .45 | 18 | 17.88 | 0.98 | 0.94 |
| Contextual + phrases + T5, .80 | 16 | 18.04 | 0.98 | 0.94 |

What this adds:

- The trained stages move the 18-set grade only slightly (v7 words
  18.62, checked full 17.72) while backward NLI stays near 0.71 for
  every lexical-bearing arm: the CS-heavy authored set loses meaning
  the same way the benchmark domain panel did (Source A). The
  contextual variant is the exception (0.94), at the cost of less
  simplification. The T5-only arm again shows the coverage problem:
  4 of 18 changed, preservation near perfect.
- Cross-checks: the dictionary (9 changed, 17.64) and original (19.17)
  reproduce the local Source B baselines exactly. The six-passage
  replay reproduces the Oct 21 numbers (49 of 51 means within 1e-12;
  the two TF-IDF differences come from a documented tokenization fix),
  and the evaluator CWI baseline reproduces exactly (F1 0.619085).
- The frozen 190-input replay moves almost nothing: checked-stack
  grade 10.7733 to 10.7614, backward NLI 0.9050 to 0.9051, 3 outputs
  differ. Runtime residuals on identical text (backward NLI up to
  0.00249 across Mac/CPU runs) exceed that movement, so the replay
  claims no semantic gain. One retained failure is noted: the T5
  output for cs-02 misses a capital at a new sentence boundary.
- Numeric per-input records for the 18 and the 190 are committed
  beside the replay report (`week7_18_metrics.jsonl`,
  `matrix_190_metrics.jsonl`). Output texts are still local-only, so
  the text review in the Yoga packet is still open.

## Limitations, stated plainly

- Small panels: 8 inputs per source in Source A, 6 per domain in
  Source B. No significance claims; differences of one input move the
  means. Source C (30 per domain) supports domain comparisons, not
  fine-grained claims.
- Reused development inputs, not held-out benchmarks: DeSSE, ASSET,
  and the domain panel were all inspected during development. No human
  quality labels were collected.
- The polsci sources were transcribed from fetched pages: passages are
  asserted verbatim against the saved files, but the saved files were
  not independently re-verified against the live pages.
- The combined system's *output texts* are not stored anywhere in
  git (benchmark evidence keeps hashes; the PR #8 replay commits
  numeric per-input records but keeps raw texts local). Text-level
  verification of the flagged cases still has to happen on Yoga; see
  error_analysis.md.
- No phrase-only arm exists yet; phrases are measured only inside the
  full combinations.

## Reproduce

```bash
# Baselines on the Week 7 set (runs anywhere with evals deps):
python evals/run_evals.py --passages data/week7_passages.json \
  --systems original,rule_based --out results/week7 \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

# Baselines on the sourced set (runs anywhere with evals deps):
python evals/run_evals.py --passages data/week7_sourced_passages.json \
  --systems original,rule_based --out results/week7-sourced \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

# Model arms (needs artifacts or the demo server; see ablation_table.md):
python evals/run_evals.py --passages data/week7_passages.json \
  --systems lexical_only,phrase_only,structure_only,lexical_structural,full_guarded,full_no_guardrails \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --model-url http://127.0.0.1:8770
```
