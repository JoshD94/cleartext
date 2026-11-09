# Module ablation — Week 7 (Oct 26–Nov 1)

Kea-Roy Ong, evals side. This table answers the Week 7 ablation deliverable:
lexical only, phrase only, structure only, combined, each with and without
guardrails.

## What ran, and where the numbers come from

Two executed sources, kept separate because their input panels differ.
Nothing in this file is estimated or simulated.

**Source A — combined-pipeline benchmark (model arms).** Executed Oct 27–28
by Joshua Dirga on Yoga, branch `josh/structural-models-20261026`, commit
`085015f`, evidence `models/evidence/pipeline-20261027/combined_summary.json`
and `combined_per_example_metrics.json`, report
`models/docs/PIPELINE_BENCHMARK.md`. 190 identical inputs per system, scored
with this eval framework's readability + preservation functions (MiniLM
embeddings, DeBERTa NLI both directions, explicit checks). Panels: 96 DeSSE
reused dev, 64 ASSET validation, 24 project-domain reused dev (8 each: QUIC
RFC 9000, Apple 10-K 2025, Biden inaugural 2021), 6 authored smoke passages.
Name mapping: the benchmark's `full_pipeline` is this harness's
`full_guarded` (all stages, checks on); `full_pipeline` keeps its PR #6
meaning (v7 words only) in `evals/model_adapter.py`.

**Source B — local baseline arms (this machine).** `original` and
`rule_based` on the 18 authored Week 7 passages
(`evals/data/week7_passages.json`, 6 per domain), run here Nov 1 with the
full suite (embedding + bidirectional NLI). Model arms cannot run on this
machine: trained artifacts are git-excluded and no demo server is reachable
from here. See "Reproduce" below.

**Source E: ladder arms on both Week 7 sets (Yoga).** Executed Nov 2
by Joshua Dirga via `RUNBOOK_JOSHUA.md`: all eight ladder arms on the
18 authored passages and the 90 sourced passages, full suite, outputs
captured per passage. Result files shared Nov 3 (not committed). This
source closes the phrase_only and structure_only gaps and puts model
arms on the 90. Full tables below; text-review verdicts in
error_analysis.md (Finding 6).

## Source A: ten-system comparison, all 190 inputs

FKGL is the all-input mean. Embedding / NLI are changed-only means
(denominators = Changed column). Reference proxy covers the 160 DeSSE +
ASSET inputs; it is an overlap proxy, not standard SARI.

| System | Changed / 190 | FKGL | Embedding | NLI mean | NLI backward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0 | 11.71 | n/a | n/a | n/a |
| Dictionary (rule_based) | 13 | 11.52 | 0.952 | 0.995 | 0.994 |
| Words, v7 (lexical_only) | 140 | 11.32 | 0.963 | 0.902 | 0.871 |
| Structure, T5 (structure_only) | 20 | 11.15 | 0.991 | 0.987 | 0.987 |
| Words + T5 (lexical_structural) | 141 | 10.76 | 0.963 | 0.905 | 0.874 |
| Full, checks disabled (full_no_guardrails) | 155 | 9.62 | 0.958 | 0.904 | 0.876 |
| Full, checked v7 (full_guarded) | 141 | 10.77 | 0.963 | 0.906 | 0.875 |
| Full, structure rules (full_rules) | 140 | 11.31 | 0.963 | 0.903 | 0.873 |

Reading the ablation:

- **Lexical is the workhorse.** Words-only changes 140/190 inputs; adding
  the structure stage changes one more input (141) and buys 0.56 FKGL.
  The structure stage alone changes only 20/190: coverage, not quality,
  is its bottleneck. On the 20 it does change, it cuts FKGL by 5.27 on
  average at NLI mean 0.989.
- **Guardrails cost coverage, not quality, on average.** Disabling checks
  changes 14 more inputs and drops FKGL by a further 1.15, at essentially
  the same changed-only preservation (NLI mean 0.904 vs 0.906). The
  averages hide the cases that matter: the error analysis names the
  failures that appear only with checks off (see error_analysis.md).
- **Phrase rules add nothing measurable here.** Full-checked (phrases on)
  and lexical+structural (phrases off) are identical on changed counts
  and means. The phrase stage's contribution is not visible at this
  granularity; the executed phrase-only arm (Source E) changes 3 of 18
  authored and 1 of 90 sourced passages, all four edits clean.
- **NLI backward is the sensitive direction.** Every model arm sits near
  0.87 backward vs 0.90+ mean: outputs entail the original less often than
  the reverse, the signature of dropped detail.

## Source A, project-domain panel only (24 inputs, 8 per source)

Changed-only preservation means; FKGL is the all-input panel mean.

| System | CS (RFC 9000) chg / FKGL / NLI bwd | Corp (Apple 10-K) chg / FKGL / NLI bwd | Polsci (inaugural) chg / FKGL / NLI bwd |
| --- | --- | --- | --- |
| Original | 0 / 13.76 / n/a | 0 / 17.22 / n/a | 0 / 13.01 / n/a |
| Dictionary | 0 / 13.76 / n/a | 0 / 17.22 / n/a | 0 / 13.01 / n/a |
| lexical_only | 6 / 13.49 / 0.734 | 4 / 16.92 / 0.989 | 6 / 12.90 / 0.823 |
| structural_only | 1 / 13.76 / 0.992 | 1 / 16.52 / 0.988 | 1 / 12.03 / 0.980 |
| lexical_structural | 6 / 13.49 / 0.734 | 4 / 16.22 / 0.991 | 6 / 11.93 / 0.824 |
| full_no_guardrails | 7 / 10.55 / 0.837 | 5 / 14.44 / 0.989 | 7 / 10.93 / 0.876 |
| full_guarded | 6 / 13.49 / 0.734 | 4 / 16.22 / 0.991 | 6 / 11.93 / 0.824 |

Notes:

- The dictionary baseline changes **0 of 24** real domain inputs: its
  substitutions do not fire on real technical, filing, or speech text.
- The CS panel carries the weakest preservation (NLI backward 0.73),
  driven by two flagged QUIC cases (see error_analysis.md). The corporate
  panel is the cleanest (0.99).
- Guarded full == lexical+structural on this panel, arm for arm: phrase
  rules and checks changed nothing beyond the lexical+structural output
  on these 24 inputs.

## Source B — local baseline arms (this machine)

`original` and `rule_based` on the 18 authored Week 7 passages
(`evals/data/week7_passages.json`), run Nov 1 with the full suite
(embedding + bidirectional NLI). `evals/results/week7/eval_results.json`.

| System | FKGL | Flesch ease | TF-IDF | Embedding | NLI fwd / bwd | Changed / 18 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 19.17 | 0.15 | 1.00 | 1.00 | 0.99 / 0.99 | 0 |
| Dictionary | 17.64 | 10.33 | 0.86 | 0.97 | 0.99 / 0.99 | 9 |

Per domain (FKGL / changed): CS 18.21 → 16.23 (3/6), corporate 20.27 →
18.40 (4/6), polsci 19.02 → 18.28 (2/6). Regression check: on the 6
original passages, readability and TF-IDF values are identical to the
previously stored local run (0/6 diffs, both systems), so the Week 7
runner change (output capture in `evaluate_system`) altered no metric.
Model arms cannot run on this machine: trained artifacts are
git-excluded and no demo server is reachable from here. See "Reproduce"
below.

## Source C: structure-only, old rules, executed here (18 passages)

The main-branch `structural()` rules (`models/src/cleartext/rewrites.py`:
comma+and independent-clause splits, explicit-agent passive to active)
were executed directly in a scratch environment (spaCy + `en_core_web_sm`
+ lemminflect; no trained artifacts needed, no repo code changed).
Positive controls verified both rules fire and the relative-clause guard
abstains. Result on the 18 Week 7 passages: **0 changed**. Every passage
trips a conservative guard (subordinate clause, negation, modal, or
complement clause; verified per-sentence for two passages). Outputs in
`evals/results/week7/structure_rules_outputs.json` (git-ignored, local
only); metrics are the originals' by construction.

This is NOT the benchmark's `structural_only` (T5 generator, 20/190
changed). It is the harness's `structure_only` arm as it behaves on the
main-branch pipeline: old-rules coverage is 0/18 here, vs the T5
generator's ~10% on the benchmark panel. The structure stage's
contribution in the combined system comes from the trained generator,
not these rules.

## Source E: ladder arms on both Week 7 sets (Yoga, Nov 2)

All-input means. The 18-set and 90-set runs used the same eight arms,
full suite, outputs captured per passage.

| System | Changed / 18 | FKGL (18) | NLI bwd (18) | Changed / 90 | FKGL (90) | NLI bwd (90) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0 | 19.17 | 0.99 | 0 | 14.11 | 0.99 |
| Dictionary | 9 | 17.64 | 0.99 | 14 | 14.05 | 0.99 |
| lexical_only | 18 | 18.62 | 0.71 | 85 | 13.95 | 0.84 |
| phrase_only | 3 | 19.38 | 0.99 | 1 | 14.11 | 0.99 |
| structure_only | 0 | 19.17 | 0.99 | 0 | 14.11 | 0.99 |
| lexical_structural | 18 | 18.62 | 0.71 | 85 | 13.95 | 0.84 |
| full_guarded | 18 | 18.81 | 0.72 | 85 | 13.95 | 0.84 |
| full_no_guardrails | 18 | 18.81 | 0.72 | 85 | 13.95 | 0.83 |

Reading the ladder:

- **Structure contributes nothing measurable on either set.**
  `structure_only` changed 0 of 108 passages, and
  `lexical_structural` outputs are identical to `lexical_only` on
  every passage in both sets. The old rules are inert here; the only
  structure contribution anywhere in the evidence is the T5
  generator's 20/190 in Source A.
- **Phrase contributes four clean edits.** `phrase_only` changed 3
  of 18 (polsci-01, polsci-02, corp-05) and 1 of 90 (corp-s24), all
  read as text and meaning-preserving. Its one sourced edit
  ("with the exception of" to "except") is the only phrase change
  on real prose. The inventory rarely fires, and when it fires it
  is safe.
- **Guardrails are inert on the authored set and selective on the
  sourced set.** `full_guarded` and `full_no_guardrails` produce
  identical outputs on all 18 authored passages. On the 90 they
  differ on 10 passages: guardrails blocked 2 significant errors
  (cs-s09, corp-s18) and 3 mild ones, but missed 3 cases where the
  guarded output is worse (cs-s08, cs-s05, corp-s04). Per-case
  verdicts are in error_analysis.md (Finding 6). The guardrail
  meaning check has a technical-term blind spot: it passed
  "handshake" to "acknowledgement" and "obligations" to
  "responsibilities".
- **The lexical stage is the whole story on these sets.** Every
  flagged meaning error in Source E is a single-word lexical
  substitution (Finding 6). The sourced panel preserves better
  than the authored panel (0.84 vs 0.71 backward NLI), because
  authored density punishes the lexical stage more than real
  prose does.

## Follow-up: non-LLM compound splitting (PR #9, Week 8)

Joshua's assigned rule (long compound sentences without an LLM) is a
spaCy-parser plus deterministic-edits splitter (`DependencyCompoundSplitter`,
opt-in as `structure_rewriter`; no T5, no new training). Evidence:
`models/docs/COMPOUND_SPLITTING.md` and
`evals/reports/20261102-compound-boundaries/` in PR #9. On the same
two panels it changed 0 of 18 authored and 1 of 90 sourced passages,
with no measured average readability gain, and its own README reports
that plainly. The single changed passage's text is not committed
(metrics only), so the split itself is unreviewed. `structure_only`
keeps its old-rules meaning; the new arms are `compound_only` and
`compound_full_guarded`. This follow-up does not change any Week 7
number above.

## Gaps (not run anywhere yet)

- **phrase + guardrails-off separately**: `full_no_guardrails` disables
  structural proposal checks, reparsed lexical checks, rejection reasons,
  and combined-word checks together. There is no arm that disables only
  the phrase stage's restrictions.
- **Benchmark-panel text review**: the 8 both-directions-low cases, the
  asset-valid-219 guarded output, and the biden-inaugural2021-0 tense
  shift (see error_analysis.md) still need output texts, which live
  only on Yoga. The ladder text review (Finding 6) does not cover
  them.
- **Compound-split passage text**: PR #9's single changed sourced
  passage is metrics-only in the committed evidence; its split text
  is unreviewed.
- **Guardrail check over HTTP**: `full_no_guardrails` over HTTP raises
  a clear error, since the demo server exposes no guardrail switch
  (see "Reproduce" below). Unaffected by the Nov 2 runs, which used
  the local transport.

Closed gaps (executed Nov 2, Source E): standalone `phrase_only`,
old-rules `structure_only`, and model arms on both Week 7 sets.

## Reproduce

On a machine with the artifacts (Joshua's runbook path), from the repo root:

```bash
git checkout josh/structural-models-20261026   # benchmark script + evidence
python evals/pipeline_benchmark.py --help      # generate + score, 10 systems
```

Through this eval harness (PR #6 branch + the Week 7 ladder in
`evals/model_adapter.py`), local import or demo server:

```bash
python evals/run_evals.py --systems original,rule_based,lexical_only,phrase_only,structure_only,lexical_structural,full_guarded,full_no_guardrails \
  --passages data/week7_passages.json \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --model-url http://127.0.0.1:8770   # or omit for local import with models/runs/
```

`full_no_guardrails` over HTTP raises a clear error: the demo server
exposes no guardrail switch. It needs the local transport with a pipeline
whose `analyze()` accepts `guardrails` (structural branch), or the
benchmark script above.
