# Is the meaning test, rather than the word swaps, the problem? November 17, 2026

Joshua's question: he has seen word swaps he judges good that the meaning
test rejects. Should its bound be lowered, for example to 0.9?

This study checks the NLI meaning test against human judgments of single-word
substitutes. It then re-gates word-swap outputs that had already been
generated and scored, and runs one pre-registered dev comparison. No LLM is
used anywhere. The locked test split is not read. Raw text, outputs and
per-pair scores stay local in
`~/.local/state/cleartext/nli-calibration-20261117/`; this folder holds
aggregates only.

## Answer

- **Partly, yes: NLI is a noisy judge of single-word swaps.**
  - Human-gold substitutes rejected by the evaluation NLI (sentence level, mean
    of both directions): **25% at 0.95** and 18% at 0.90 (3,687 gold
    substitutes).
  - It also accepts substitutes that every SWORDS annotator rejected: 37% at
    0.95 and 46% at 0.90.
  - Separation is moderate: AUC 0.78.
- **Lowering the bound to 0.90 does not fix that.** A lower threshold trades
  one error for the other.
  - From 0.95 to 0.90, false rejections fall 7 points, but false acceptances
    rise 9 points.
  - Balanced accuracy (Youden J) peaks near 0.97 (0.976 for the evaluation
    model, 0.971 for the veto model). The curve is flat near the top: J is
    0.416 at 0.95 and 0.378 at 0.90.
- **The passage gate does not overreact to accepted swaps.**
  - A swap the evaluation NLI accepts at sentence level (≥ 0.95) costs its
    passage −0.002 NLI on average.
  - A swap it rejects (< 0.80) costs −0.46, which takes that passage below
    0.80.
  - NLI rejects the pipeline's swaps at about the rate it rejects human-gold
    substitutes: 26% below 0.95, 16% below 0.80.
  - With one swap per passage, those rejections alone fail gate 2 (at most 5%
    of passages below 0.80).
- **Relaxing the passage gate (mean ≥ 0.90, with the floor variants below)
  mostly does not help.**
  - v7 at its defaults fails every variant on the old 90-passage panel: 19
    passages fall below 0.80.
  - On the 34-passage dev screen, v7 passes only the loosest variant (≤ 10%
    below 0.80), and only once a separate entity fault is guarded (below).
  - `contextual` passes some variants with no guard on the old panel. Its gain
    is about 0.10 FK grade, at −0.05 passage NLI.
- **What works on cached outputs: a per-swap gate, with the passage gates
  unchanged.**
  - The gate drops a swap whose own sentence scores below 0.97 on the veto
    model, or that breaks a number, negation or entity check.
  - v7 under this gate passes every unchanged gate on the dev screen and the
    old panel, keeping 76% and 66% of its FK gain (about 0.1 grade).
  - It fails on the 17 paper paragraphs (one paragraph below 0.80).
- **A separate fault: the entity heuristic.** On dev, two swaps replace
  sentence-initial words, which the heuristic counts as entities. These two
  account for every gate-4 failure of the word settings.
- **F2 + words, pre-registered, dev only: does not beat F2.**
  - Under the primary gate, ΔFK is −0.101, but the run fails gate 2 (6 of 100
    passages below 0.80) and the NLI rule (−0.0115 against −0.010).
  - Only the evaluator-model gate passes, and it was declared ineligible in
    advance.
  - Either way, words add about 0.1 grade on top of F2's 1.5.
- **Recommendation: no change.**
  - Keep the 0.95 passage bound.
  - If word swaps are wanted, they need a per-swap gate. Lowering the bound is
    not the fix.
  - Joshua's marks on the review page (part 3) are the in-domain human check
    this study could not supply.

## 1. NLI against human substitute judgments

### Data

- **SWORDS test** (local copy, CC BY 3.0, built on CoInCo).
  - Only single-word substitutes are used, inflected to the target's tag with
    the pipeline's `generation.inflect`.
  - Inflections that produce a non-word are dropped (zipf 0, e.g. an already
    inflected substitute inflected again).
  - Per target, the sample takes up to 4 substitutes of each kind below and 2
    borderline ones (seed 20261117).

  | SWORDS kind | Annotators accepting | Count |
  |---|---|---:|
  | Positive | ≥ 50% of 10 | 2,151 |
  | Hard negative | none of 3 (SWORDS does not re-rate these) | 2,880 |
  | Borderline | some, under 50% | 1,454 |

  "UNSURE" labels count as not accepting.
- **TSAR-2022 test gold** (373 instances, 25 annotators).
  - Positives: substitutes named by ≥ 2 annotators (1,493).
  - Singletons: one per instance (337).
  - Easy negatives: 704 random same-POS WordNet words within 0.5 zipf of the
    top gold substitute.
- **MultiLS 2024 English trial**: 43 positives, 30 singletons, 56 easy
  negatives.

### Scoring

Scoring follows the pipeline on the sentence that holds the target:

- **`base_mean`**: the evaluation NLI (DeBERTa-v3-base-mnli-fever-anli), mean of
  both directions.
- **`small_min`**: the word veto's NLI (nli-deberta-v3-small), min of both
  directions, as `EntailmentVeto`.

**Fidelity.** On 60 random pairs, the evaluation model was run in float32 for
speed. It is within **0.002** of the evaluator's float16 pipeline (forward
direction). The veto scores are within 3e-6 of `EntailmentVeto.score`. Passage
metrics in parts 2 and 4 are the evaluator's own cached scores.

**Confidence intervals** below come from a bootstrap over targets, since a
target contributes up to 4 substitutes.

### Share of human-gold substitutes the test rejects (score < threshold)

| Threshold | Eval NLI, all gold (3,687) [95% CI] | SWORDS (2,151) | TSAR ≥ 2 (1,493) | MultiLS (43) | Veto NLI, all gold [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 0.97 | **32.8%** [30.5, 35.1] | 27.7% | 40.3% | 27.9% | 28.2% [25.9, 30.3] |
| 0.95 | **25.0%** [22.9, 27.0] | 21.6% | 29.8% | 25.6% | 21.3% [19.4, 23.2] |
| 0.90 | **17.9%** [16.1, 19.6] | 16.1% | 20.6% | 16.3% | 16.5% [14.8, 18.1] |
| 0.80 | **12.3%** [10.9, 13.8] | 11.1% | 14.0% | 14.0% | 13.3% [11.7, 14.7] |
| 0.70 | **9.7%** [8.4, 11.1] | 9.4% | 10.4% | 4.7% | 12.2% [10.7, 13.6] |

By part of speech (SWORDS, eval NLI at 0.95), nouns and adjectives are
rejected most:

| Part of speech | Rejected at 0.95 |
|---|---:|
| Nouns | 33% |
| Adjectives | 28% |
| Verbs | 10% |
| Adverbs | 10% |

### Share of bad substitutes it accepts (score ≥ threshold)

| Threshold | SWORDS, no annotator accepts (2,880) | SWORDS borderline (1,454) | TSAR random (704) | Veto NLI, SWORDS no-accept |
|---:|---:|---:|---:|---:|
| 0.97 | 29.5% | 44.4% | 5.3% | 35.1% |
| 0.95 | 36.7% | 51.9% | 8.9% | 42.1% |
| 0.90 | 46.1% | 60.0% | 15.3% | 49.5% |
| 0.80 | 53.9% | 67.3% | 23.7% | 54.8% |
| 0.70 | 59.3% | 70.6% | 28.1% | 56.9% |

### How well it separates good from bad

| | Eval NLI | Veto NLI |
|---|---:|---:|
| AUC, SWORDS positives vs no-accept [95% CI, clustered by target] | **0.777** [0.763, 0.791] | 0.734 [0.720, 0.748] |
| AUC, TSAR positives vs random words | 0.896 [0.881, 0.911] | 0.869 [0.851, 0.886] |
| Best threshold (Youden J, SWORDS) | 0.976 (J 0.43) | 0.971 (J 0.36) |
| Spearman vs share of annotators accepting (10-label items only, so no hard negatives) | 0.36 | 0.28 |

NLI tells a random word from a good one well, but a plausible wrong-sense
synonym only moderately well. Its scores rise with the share of annotators
who accept a substitute. Even substitutes half the annotators accept average
only 0.90.

## 2. Re-gating the word swaps already generated

### Sources

All outputs were already scored. There are 527 unique swap sentences:

| Source | Panel | Unique swap sentences |
|---|---|---:|
| Dev split | the 34-passage sweep screen (every word setting), the survivor cells (68–87 passages, partial) and the 100 F2+v7 outputs of part 4 | 227 |
| November 9/10 studies and the sweep smoke run | old 90 passages | 283 |
| Paper run | 17 v7 paragraphs | 17 |

Each swap is compared with its `words=off` twin. The word stage runs last and
makes one edit, so a no-refill veto gives exactly the twin's output.

### Passage-NLI cost of one swap, by its own sentence score

From 706 (passage, swap) pairs:

| Sentence NLI | Swaps | Mean Δ passage NLI | Passage drops > 0.05 |
|---|---:|---:|---:|
| ≥ 0.95 | 507 | −0.002 | 4% |
| 0.90–0.95 | 38 | −0.044 | 24% |
| 0.80–0.90 | 28 | −0.088 | 71% |
| < 0.80 | 133 | **−0.458** | 92% |

### Results

ΔFK and ΔS are against the same config without words. The CI is the
domain-stratified bootstrap. The gate columns read: **current** = mean ≥ 0.95
and ≤ 5% of passages < 0.80, plus gates 3–4; **R1** = mean ≥ 0.90 and ≤ 5%
< 0.80; **R2** = mean ≥ 0.90 and ≤ 5% < 0.70; **R3** = mean ≥ 0.90 and ≤ 10%
< 0.80. ✓ = passes.

| Panel | Word setting | Swap gate | Swaps | ΔFK [95% CI] | ΔS | Passage NLI | < 0.80 | Current / R1 / R2 / R3 |
|---|---|---|---:|---:|---:|---:|---:|---|
| dev screen (34) | v7 | none | 34 | −0.166 [−0.21, −0.12] | +0.0039 | 0.920 | 3 | ✗ ✗ ✗ ✗ |
| | v7 | checks only | 33 | −0.157 | +0.0038 | 0.920 | 3 | ✗ ✗ ✗ ✓ |
| | v7 | eval ≥ 0.90 + checks | 27 | −0.132 [−0.18, −0.09] | +0.0021 | 0.988 | 0 | ✓ ✓ ✓ ✓ |
| | v7 | **veto ≥ 0.97 + checks** | 27 | **−0.127** [−0.17, −0.08] | +0.0021 | 0.986 | 0 | **✓ ✓ ✓ ✓** |
| | contextual | none | 34 | −0.153 | +0.0038 | 0.933 | 4 | ✗ ✗ ✗ ✗ |
| | contextual | checks only | 33 | −0.143 | +0.0036 | 0.933 | 4 | ✗ ✗ ✓ ✗ |
| | contextual | veto ≥ 0.97 + checks | 24 | −0.104 [−0.15, −0.06] | +0.0019 | 0.979 | 1 | ✓ ✓ ✓ ✓ |
| dev (68 of 100, partial) | v7, threshold 0.7 | checks only | 47 | −0.111 [−0.15, −0.07] | +0.0029 | 0.970 | 1 | ✓ ✓ ✓ ✓ |
| | v7, threshold 0.7 | veto ≥ 0.97 + checks | 40 | −0.095 [−0.14, −0.06] | +0.0022 | 0.984 | 1 | ✓ ✓ ✓ ✓ |
| old panel (90) | v7 | none | 85 | −0.158 [−0.22, −0.10] | +0.0066 | 0.840 | 19 | ✗ ✗ ✗ ✗ |
| | v7 | eval ≥ 0.90 + checks | 58 | −0.101 [−0.15, −0.06] | +0.0034 | 0.981 | 0 | ✓ ✓ ✓ ✓ |
| | v7 | **veto ≥ 0.97 + checks** | 57 | **−0.105** [−0.15, −0.06] | +0.0038 | 0.965 | 3 | **✓ ✓ ✓ ✓** |
| | contextual | none | 82 | −0.099 [−0.15, −0.05] | +0.0031 | 0.932 | 7 | ✗ ✗ ✓ ✓ |
| | contextual | veto ≥ 0.97 + checks | 66 | −0.090 | +0.0029 | 0.974 | 2 | ✓ ✓ ✓ ✓ |
| paper (17) | v7 | none | 16 | −0.642 [−1.51, −0.12] | +0.0167 | 0.904 | 2 | ✗ ✗ ✗ ✗ |
| | v7 | eval ≥ 0.90 + checks | 11 | −0.597 | +0.0159 | 0.983 | 0 | ✓ ✓ ✓ ✓ |
| | v7 | veto ≥ 0.97 + checks | 11 | −0.189 | +0.0030 | 0.935 | 1 | ✗ ✗ ✗ ✓ |

Notes on the table:

- **Old panel.** It is the November 10 `v2` state with v7 on the incumbent;
  the sweep smoke copy gives identical numbers.
- **"checks".** The evaluator's own number, negation and entity checks, run on
  the swap sentence.
- **v7 threshold 0.8** (82 of 100 passages, partial) passes every gate with or
  without a gate, but gains only 0.04–0.05 grade.
- **The evaluator-model gate (eval ≥ 0.90)** filters on the metric that the
  gates then score, so its rows are optimistic by construction.
- **Paper paragraphs** are short, so one swap moves FK a lot.

The full table, every word setting × 16 swap gates × 4 passage gates, is in
[aggregate.json](aggregate.json) (`regate`).

## 3. Joshua's review page (local)

The page is at
`~/.local/state/cleartext/nli-calibration-20261117/review.html`. It works
offline at phone width.

- **What it shows.** 60 swaps, 15 from each eval-NLI band (≥ 0.95,
  0.90–0.95, 0.80–0.90, < 0.80), drawn from the 527 above.
- **How to mark.** Each swap shows the sentence before and after, with the
  swap highlighted. Mark it Good or Bad, then export `word-swap-marks.csv`.
  The CSV carries each swap's ID.
- **Scoring the marks.**
  `python -m evals.nli_calibration marks --state <state> --marks word-swap-marks.csv`
  - It refuses marks from a different sample.
  - It reports precision and recall of each NLI score per threshold, with
    Joshua's "good" as the positive class. Each mark is weighted by its band's
    share of all swaps (74% of swaps are ≥ 0.95).
  - It also reports the share marked good in each band.

The human-matched thresholds above come from SWORDS, a different domain.
Joshua's marks would show whether they hold on our technical passages,
especially in the 0.80–0.95 band.

## 4. F2 + words under the calibrated gate (pre-registered, dev only)

[PROTOCOL.md](PROTOCOL.md) and [design-F2-words.json](design-F2-words.json)
were pushed (`64ac20e`) before any F2 + words output existed. At that point
the dev re-gating in part 2 had been read; the protocol says so.

- **Generation.** v7 swaps were generated on F2 for all 100 dev passages, with
  the sweep harness on its frozen dev plan (same signatures). That took 6 of
  the 8 budgeted chunks.
- **Gates.** The word gates were applied per swap afterwards. `judge` checks
  that every passage without a word edit equals F2.
- **Judgement.** It is in [judge-dev.json](judge-dev.json); deltas are against
  F2.

| Arm | Word gate | Swaps kept / proposed | ΔFK vs F2 [95% CI] | ΔS | Δ passage NLI [95% CI] | NLI < 0.80 | Gates 1–4 | Over-edit | Beats F2 |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **F2+v7 (primary)** | **A1: veto ≥ 0.97 + checks** | 78 / 100 | −0.101 [−0.133, −0.070] | +0.0026 | **−0.0115** [−0.0257, −0.0001] | **6** | **✗ (gate 2)** | ✓ | **no** (a, d) |
| F2+v7 | A2: eval ≥ 0.90 + checks (ineligible) | 81 / 100 | −0.115 [−0.149, −0.084] | +0.0024 | −0.0067 [−0.0183, +0.0028] | 4 | ✓ | ✓ | ineligible; would pass |
| F2+v7 | none | 100 / 100 | −0.134 [−0.168, −0.102] | +0.0036 | −0.0574 | 12 | ✗ | ✗ | no |
| F2+v7, threshold 0.7 (secondary) | A1 | 64 / 77 | −0.095 [−0.126, −0.066] | +0.0019 | −0.0008 | 3 | ✓ | ✓ | no (b: ΔFK above −0.10) |
| F2+v7, threshold 0.7 | A2 (ineligible) | 70 / 77 | −0.108 | +0.0023 | −0.0032 | 3 | ✓ | ✓ | ineligible; would pass |

For reference, F2 alone on dev: FK 12.851, S 0.2700, passage NLI 0.9746, two
passages below 0.80.

**Result: F2 + words does not beat F2 under the pre-registered rule.**

- **The primary arm fails twice.**
  - It fails gate 2: 6 of 100 passages fall below 0.80.
  - Its NLI cost, −0.0115, exceeds the −0.010 limit.
  - The veto model accepts some swaps that the evaluation NLI rejects.
- **The secondary arm (threshold 0.7)** passes every gate and costs no NLI.
  Its gain of −0.095 misses the −0.10 bar.
- **The A2 arms would pass**, but they filter with the evaluator's own model,
  so they were declared ineligible in advance.
- **On every arm, words add about 0.1 FK grade** on top of F2's 1.5.

No new locked-test run is proposed.

**Equivalence caveat.** The post hoc gate scores the sentence from
`preservation.split_sentences`. The in-pipeline veto scores spaCy's sentence,
so where the two split differently, an in-pipeline A1 could decide
differently.

## Limits

- **What is measured.** These are automatic NLI scores against crowd
  judgments of substitutability. Annotators judged whether a substitute fits
  in context, not strict meaning equivalence.
- **Domains.** SWORDS and TSAR are news and fiction. Our passages are
  technical.
- **Hard negatives.** SWORDS hard negatives carry 3 labels, not 10.
- **Selection and test status.**
  - The dev passages already chose F2.
  - The dev re-gating was read before the protocol.
  - Nothing here touches the locked test split, and no default changed.

## Provenance

- **Code:**
  - [`evals/nli_calibration.py`](../../nli_calibration.py): gold pairs, swap
    extraction, scoring, analysis, review page, marks scorer, judgement.
  - Offline checks: `evals/test_nli_calibration.py`.
  - `scripts/verify` passes (590 tests, job `8dc23b41ef29`).
- **Queue:** Yoga, personal queue at `later` priority, one job at a time.
  - NLI scoring: `45ecc9ea9962`, `6defee14fbbf`, `35968e022d01` and
    `2595cc2a99b1`.
  - F2 + words generation and scoring: `7b6149b8f30c`, `0d2266541711`,
    `817b8777b96c`, `f8b419f807e3`, `8a24fd023299` and `9332a41f035d`.
  - Swaps, judgement and aggregates: `dbd02e1a9ab7`, then `edd328b964ec`
    after the independent review's fixes. Those fixes dropped non-word
    inflections, added clustered CIs, a sample-checked marks scorer and an
    eligibility flag; no verdict changed.
- **F2 + words run.** It ran with the sweep's frozen dev plan and state
  (`param-sweeps-20261116/dev`):
  - generation signature `223acb6f0e3f…`, scoring signature `b818b07ac341…`;
  - from a temporary detached worktree at the plan's pinned path
    (`cleartext-wt-sweeps`), whose 90 sources match `main` byte for byte.
    The worktree was removed afterwards.
- **Rerun:**
  - `python -m evals.nli_calibration gold|swaps --out …`
  - then `score --state … --items gold.jsonl swaps.jsonl`
  - then `analyze --state … --out aggregate.json`
  - then `judge --state … --design design-F2-words.json --out judge-dev.json`
