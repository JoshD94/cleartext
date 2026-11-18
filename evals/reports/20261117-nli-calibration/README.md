# Is the meaning test, rather than the word swaps, the problem? November 17, 2026

Joshua's question: he has seen word swaps he judges good that the meaning
test rejects. Should its bound be lowered, for example to 0.9?

This study checks the NLI meaning test against human judgments of single-word
substitutes. It then re-gates word-swap outputs that earlier studies had
already generated and scored. No LLM is used anywhere. The locked test split
is not read. Raw text, outputs and per-pair scores stay local in
`~/.local/state/cleartext/nli-calibration-20261117/`; this folder holds
aggregates only.

## Answer

- **Partly, yes: NLI is a noisy judge of single-word swaps.**
  - Human-gold substitutes rejected by the evaluation NLI (sentence level, mean
    of both directions): **25% at 0.95** and 18% at 0.90 (3,721 gold
    substitutes).
  - It also lets through substitutes that annotators all rejected: 37% at 0.95
    and 46% at 0.90.
  - Agreement with humans is moderate: AUC 0.78, Spearman 0.36 against the
    share of annotators who accept a substitute.
- **Lowering the bound does not fix that.** A lower threshold trades one error
  for the other.
  - The threshold that best matches human judgments is about **0.97** for both
    models (Youden), not 0.90.
  - From 0.95 to 0.90, false rejections fall 7 points, but false acceptances
    rise 9 points.
- **The passage gate does not overreact to good swaps.**
  - A swap that NLI accepts at sentence level (≥ 0.95) costs its passage only
    −0.002 NLI on average.
  - A swap it rejects (< 0.80) costs −0.47 and drags that passage below the
    0.80 floor.
  - The pipeline's swaps are rejected at about the rate of human-gold
    substitutes: 25% below 0.95, 16% below 0.80.
  - With one swap per passage, those rejections alone fail gate 2 (at most 5%
    of passages below 0.80).
- **Relaxing the passage gate to 0.90 passes no v7 setting.** v7 fails the
  floor at any of the tested bounds (old panel: 19 of 90 passages below 0.80).
  Only `contextual` settings pass a relaxed gate, and only once a separate
  fault (below) is guarded. They gain about 0.10–0.14 FK grade, at a cost of
  −0.02 to −0.06 passage NLI.
- **What does work: a calibrated per-swap gate.**
  - Drop a swap whose own sentence scores below 0.97 on the veto model (the
    human-calibrated value).
  - Also drop it if it breaks a number, negation or entity check.
  - Under this gate, v7 passes the unchanged passage gates on the dev screen
    and the old 90-passage panel, and keeps 65–80% of its FK gain.
- **A separate fault: the entity heuristic.** On dev, the only remaining gate
  failure is gate 4. Two swaps replace sentence-initial words ("Discovering",
  "Decreases") that the heuristic counts as entities.
- **F2 + words (pre-registered, dev only): does not beat F2.**
  - The primary gate (veto ≥ 0.97 + checks) gives ΔFK −0.101, but fails gate 2
    (6 of 100 passages below 0.80) and the NLI rule (−0.0115).
  - Only the evaluator-model gate (ineligible by design) passes, at ΔFK
    −0.115.
  - Either way, words add about 0.1 grade to F2's 1.5.
- **Recommendation: no change.**
  - Keep the 0.95 passage bound.
  - If word swaps are wanted, they need a per-swap gate. Lowering the bound is
    not the fix.
  - Joshua's marks on the review page are the independent check this study
    could not supply.

## 1. NLI against human substitute judgments

### Data

- **SWORDS test** (local copy, CC BY 3.0, built on CoInCo).
  - Each substitute was rated in context by up to 10 annotators.
  - Only single-word substitutes are used, inflected to the target's tag with
    the pipeline's `generation.inflect`.
  - Per target, the sample takes up to 4 substitutes of each kind below and 2
    borderline ones (seed 20261117).

  | SWORDS kind | Annotators accepting | Count |
  |---|---|---:|
  | Positive | ≥ 50% of 10 | 2,185 |
  | Hard negative | none | 2,967 |
  | Borderline | some, under 50% | 1,484 |

- **TSAR-2022 test gold** (373 instances, 25 annotators).
  - Positives: substitutes named by ≥ 2 annotators (1,493).
  - Singletons: one per instance (337).
  - Easy negatives: 2 per instance (726), random same-POS WordNet words within
    0.5 zipf of the top gold substitute.
- **MultiLS 2024 English trial**: 43 positives, 30 singletons, 56 easy
  negatives.

### Scoring

Scoring follows the pipeline on the sentence that holds the target:

- **`base_mean`**: the evaluation NLI (DeBERTa-v3-base-mnli-fever-anli), mean of
  both directions.
- **`small_min`**: the word veto's NLI (nli-deberta-v3-small), min of both
  directions, as `EntailmentVeto`.

**Fidelity.** On 60 random pairs, the evaluation model was run in float32 for
speed. It is within **0.002** of the evaluator's float16 pipeline. The veto
scores are within 3e-6 of `EntailmentVeto.score`.

### Share of human-gold substitutes the test rejects (score < threshold)

| Threshold | Eval NLI, all gold (3,721) [95% CI] | SWORDS (2,185) | TSAR ≥ 2 (1,493) | MultiLS (43) | Veto NLI, all gold |
|---:|---:|---:|---:|---:|---:|
| 0.97 | **32.7%** [31.2, 34.2] | 27.6% | 40.3% | 27.9% | 28.1% |
| 0.95 | **24.9%** [23.5, 26.3] | 21.6% | 29.8% | 25.6% | 21.3% |
| 0.90 | **17.8%** [16.6, 19.1] | 16.0% | 20.6% | 16.3% | 16.4% |
| 0.80 | **12.3%** [11.2, 13.3] | 11.0% | 14.0% | 14.0% | 13.2% |
| 0.70 | **9.7%** [8.8, 10.7] | 9.4% | 10.4% | 4.7% | 12.1% |

By part of speech (SWORDS, eval NLI at 0.95), nouns and adjectives are
rejected most:

| Part of speech | Rejected at 0.95 |
|---|---:|
| Nouns | 33% |
| Adjectives | 28% |
| Verbs | 10% |
| Adverbs | 10% |

### Share of bad substitutes it accepts (score ≥ threshold)

| Threshold | SWORDS, no annotator accepts (2,967) | SWORDS borderline (1,484) | TSAR random (726) | Veto NLI, SWORDS no-accept |
|---:|---:|---:|---:|---:|
| 0.97 | 29.8% | 44.5% | 5.5% | 34.9% |
| 0.95 | 36.8% | 52.0% | 9.2% | 41.9% |
| 0.90 | 46.2% | 60.2% | 16.0% | 49.4% |
| 0.80 | 53.7% | 67.5% | 24.2% | 54.7% |
| 0.70 | 59.1% | 70.8% | 28.8% | 56.9% |

### How well it separates good from bad

| | Eval NLI | Veto NLI |
|---|---:|---:|
| AUC, SWORDS positives vs human-rejected [95% CI, clustered by target] | **0.775** [0.762, 0.789] | 0.734 [0.720, 0.747] |
| AUC, TSAR positives vs random words | 0.893 [0.878, 0.908] | 0.866 [0.848, 0.882] |
| Best threshold (Youden J, SWORDS) | **0.976** (J 0.43) | **0.971** (J 0.36) |
| Spearman vs share of annotators accepting | 0.36 | 0.28 |

NLI tells a random word from a good one well, but a plausible wrong-sense
synonym only moderately well. Even 50%-acceptable substitutes average only
0.90. The strict threshold is not miscalibrated: 0.97 is where NLI agrees best
with people.

## 2. Re-gating the word swaps already generated

### Sources

All outputs were already generated and scored (472 unique swap sentences):

| Source | Panel | Unique swap sentences |
|---|---|---:|
| Sweep dev screen | 34 dev passages, every word setting | 172 |
| Survivor cells | 49–87 dev passages, partial | — |
| November 9/10 studies and the sweep smoke run | old 90 passages | 283 |
| Paper run | 17 v7 paragraphs | 17 |

Each swap is compared with its `words=off` twin. The word stage is last and
makes one edit, so a veto (no refill) gives exactly the twin's output.

### Passage-NLI cost of one swap, by its own sentence score

From 622 (passage, swap) pairs:

| Sentence NLI | Swaps | Mean Δ passage NLI | Passage drops > 0.05 |
|---|---:|---:|---:|
| ≥ 0.95 | 443 | −0.002 | 4% |
| 0.90–0.95 | 32 | −0.038 | 25% |
| 0.80–0.90 | 24 | −0.094 | 71% |
| < 0.80 | 123 | **−0.472** | 92% |

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
| | contextual_veto (no refill) | checks only | 29 | −0.129 | +0.0025 | 0.971 | 2 | ✗ ✗ ✓ ✓ |
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
- **Partial survivor cells** (v7 threshold 0.7 on 49 dev passages, 0.8 on 77)
  pass every gate once the checks are added. They gain only 0.03–0.09 FK grade.

What re-gating shows:

- **Relaxing the passage gate passes no v7 setting.** v7 fails the floor at
  all of R1–R3 everywhere; the old panel has 19 passages below 0.80.
- **`contextual` does pass a relaxed gate** (R2 or R3, here and on the old
  panel), once the entity checks are added on dev, for ΔFK about −0.10 to
  −0.14 at −0.02 to −0.06 passage NLI.
- **A per-swap gate passes the unchanged gates** and keeps about 75% (dev) and
  65% (old panel) of v7's FK gain.

The full table, every word setting × 22 swap gates × 4 passage gates, is in
[aggregate.json](aggregate.json) (`regate`).

## 3. Joshua's review page (local)

The page is at
`~/.local/state/cleartext/nli-calibration-20261117/review.html`. It works
offline at phone width.

- **What it shows.** 60 swaps, 15 from each eval-NLI band (≥ 0.95,
  0.90–0.95, 0.80–0.90, < 0.80), drawn from the 472 above.
- **How to mark.** Each swap shows the sentence before and after, with the
  swap highlighted. Mark it Good or Bad, then export `word-swap-marks.csv`.
- **Scoring the marks.**
  `python -m evals.nli_calibration marks --state <state> --marks word-swap-marks.csv`
  prints precision and recall of each NLI score at each threshold, with
  Joshua's "good" as the positive class.

Thresholds that best match human judgment come from SWORDS, a different
domain. Joshua's marks would show whether that holds on our technical
passages, especially in the 0.80–0.95 band.

## 4. F2 + words under the calibrated gate (pre-registered, dev only)

[PROTOCOL.md](PROTOCOL.md) and [design-F2-words.json](design-F2-words.json)
were pushed (`64ac20e`) before any F2 + words output existed.

- **Generation.** v7 swaps were generated on F2 for all 100 dev passages, with
  the sweep harness on its frozen dev plan (same signatures). That took 6 of
  the 8 budgeted chunks.
- **Gates.** The word gates were applied per swap afterwards.
- **Judgement.** It is in [judge-dev.json](judge-dev.json); deltas are against
  F2.

| Arm | Word gate | Swaps kept / proposed | ΔFK vs F2 [95% CI] | ΔS | Δ passage NLI [95% CI] | NLI < 0.80 | Gates 1–4 | Over-edit | Beats F2 |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **F2+v7 (primary)** | **A1: veto ≥ 0.97 + checks** | 78 / 100 | −0.101 [−0.133, −0.070] | +0.0026 | **−0.0115** [−0.0257, −0.0001] | **6** | **✗ (gate 2)** | ✓ | **no** (a, d) |
| F2+v7 | A2: eval ≥ 0.90 + checks (ineligible) | 81 / 100 | −0.115 [−0.149, −0.084] | +0.0024 | −0.0067 [−0.0183, +0.0028] | 4 | ✓ | ✓ | passes, but ineligible |
| F2+v7 | none | 100 / 100 | −0.134 [−0.168, −0.102] | +0.0036 | −0.0574 | 12 | ✗ | ✗ | no |
| F2+v7, threshold 0.7 (secondary) | A1 | 64 / 77 | −0.095 [−0.126, −0.066] | +0.0019 | −0.0008 | 3 | ✓ | ✓ | no (b: ΔFK above −0.10) |
| F2+v7, threshold 0.7 | A2 (ineligible) | 70 / 77 | −0.108 | +0.0023 | −0.0032 | 3 | ✓ | ✓ | passes, but ineligible |

For reference, F2 alone on dev: FK 12.851, S 0.2700, passage NLI 0.9746, two
passages below 0.80.

**Result: F2 + words does not beat F2 under the pre-registered rule.**

- **The primary arm fails twice.**
  - It fails gate 2: 6 of 100 passages fall below 0.80.
  - Its NLI cost, −0.0115, exceeds the −0.010 limit.
  - The veto model accepts some swaps that the evaluation NLI then rejects.
- **The secondary arm (threshold 0.7)** passes every gate and costs no NLI.
  Its gain of −0.095 misses the −0.10 bar.
- **The A2 arm passes**, but it filters with the evaluator's own model, so it
  was declared ineligible in advance.
- **On every arm, words add only about 0.1 FK grade** on top of F2's 1.5.

No new locked-test run is proposed.

## Limits

- **What is measured.** These are automatic NLI scores against crowd
  judgments of substitutability. Annotators judged whether a substitute fits
  in context, not strict meaning equivalence.
- **Domains.** SWORDS and TSAR are news and fiction. Our passages are
  technical.
- **Selection and test status.**
  - The dev re-gating results were read before the protocol was written; the
    protocol says so.
  - These dev passages also chose F2.
  - Nothing here touches the locked test split, and no default changed.

## Provenance

- **Code:**
  - [`evals/nli_calibration.py`](../../nli_calibration.py): gold pairs, swap
    extraction, scoring, analysis, review page, marks scorer, judgement.
  - Offline checks: `evals/test_nli_calibration.py`.
- **Queue:** Yoga, personal queue at `later` priority, one job at a time.
  - NLI scoring: `45ecc9ea9962`, `6defee14fbbf`, `35968e022d01` and
    `2595cc2a99b1` (12,940 rows, 4 chunks).
  - F2 + words generation and scoring: `7b6149b8f30c`, `0d2266541711`,
    `817b8777b96c`, `f8b419f807e3`, `8a24fd023299` and `9332a41f035d`.
  - Swaps, scoring, judgement and aggregates: `dbd02e1a9ab7`.
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
