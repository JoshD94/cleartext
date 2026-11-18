# Does a per-swap window veto make word swaps worth adding to F2? November 18, 2026

This study follows the [November 17 NLI calibration](../20261117-nli-calibration/README.md),
which found three flaws:

- **Scoring unit.** Whole-sentence NLI rejects some good swaps.
- **Gating unit.** One bad swap fails a whole passage.
- **Metric.** FK barely credits word swaps.

It adds an opt-in veto that scores each candidate swap on a local window and
skips failing swaps one at a time. The window and threshold are chosen on
human gold sets only. The study adds a lexical simplicity metric, then runs
F2 against F2 + words on dev under a pre-registered rule
([PROTOCOL.md](PROTOCOL.md)).

No LLM is used anywhere. The locked test split is not read. Raw text,
outputs and per-swap records stay local in
`~/.local/state/cleartext/swap-veto-20261118/`; this folder holds aggregates
only.

## Answer

- **The window veto is a better judge of single swaps than whole-sentence
  NLI, but F2 + words still does not beat F2 on dev.**
- **Calibration on gold sets.**
  - The chosen setting is a **±2-word window at T = 0.945**. It has
    balanced accuracy 0.755 (precision 0.748, recall 0.775).
  - The best whole-sentence setting reaches 0.721.
  - No setting reaches the hoped-for precision of 0.90; the highest in the
    grid is 0.889.
- **Dev results.** Both word arms (v7, contextual) under the veto:
  - make text simpler: ΔFK −0.13 and −0.11, and the new word-complexity
    metric falls with a CI below 0;
  - fail the meaning rule: 7 of 100 passages fall below 0.80 NLI (F2 has
    2), and the Δ passage-NLI CI reaches −0.039 and −0.057 (limit −0.005).
- **Why (post hoc diagnostics).**
  - The swaps the window rescues (20 of 193 applied, whole-sentence NLI
    < 0.95) account for 8 of the 12 new below-0.80 cases (10 distinct
    passages; 5 net new per arm against F2's 2).
  - Passage NLI also swings on single swaps that both the sentence and
    the window accept: one swap scoring 0.994 on its sentence costs its
    passage −0.63.
  - A swap the veto accepts can still fail the passage test.
- **Recommendation.**
  - Keep F2 without words as the strongest candidate.
  - Keep the veto as an opt-in tool; no default changes.
  - If word swaps are wanted, the open question is the metric, not the
    gate. Joshua's eyeball of the applied swaps (page below) is the
    in-domain check: are the rescued swaps actually good?

## 1. The veto (`cleartext.swap_veto.WindowSwapVeto`, opt-in)

For each candidate swap, the word stage passes the sentence before and after
that swap (`analyze(word_veto=...)`). The veto works in four steps:

1. It finds the changed words.
2. It cuts the same W whitespace words of context on each side from both
   sentences.
3. It scores the evaluation NLI (DeBERTa-v3-base-mnli-fever-anli, float32)
   on the two windows, in both directions.
4. It passes the swap only if the mean is at least T **and** the evaluator's
   number, negation and entity checks hold on the whole sentence.

A vetoed swap is skipped on its own, and the next-ranked word may take the
slot (the existing refill). An illustrative window at W = 2: *"are often
**integrated**, with products"* against *"are often **combined**, with
products"*.

The binding to the evaluation model and checks lives in `evals/swap_veto.py`;
the core module loads no model. Tests:

- `models/tests/test_swap_veto.py`: windowing, decisions, and the pipeline
  hook skipping one swap and taking the next;
- `evals/test_swap_veto.py`: the selection rule, design IDs, lexical metric
  and swap matching.

## 2. Choosing W and T on gold sets only

**Pairs.** The pairs are the November 17 gold set:

| Class | Source | Pairs |
|---|---|---:|
| Human-good | SWORDS ≥ 50% acceptors | 2,151 |
| | TSAR ≥ 2 annotators | 1,493 |
| | MultiLS ≥ 2 annotators | 43 |
| Human-bad | SWORDS, no annotator accepts | 2,880 |
| | Random same-POS, frequency-matched words for TSAR | 704 |
| | Random same-POS, frequency-matched words for MultiLS | 56 |

That gives 3,687 human-good and 3,640 human-bad pairs.

**Grid.** W ∈ {2, 3, 4, 6, 8, 10, sentence} × T ∈ [0.500, 0.995] in
steps of 0.005. The checks are part of the decision; they fail on 46 gold
pairs.

**Rule (pre-registered).** Take the highest pooled balanced accuracy with
precision ≥ 0.90. If no setting reaches 0.90, take the highest balanced
accuracy overall. No setting reached 0.90, so the fallback applied. The
choice was committed (`de7d61f`) before any dev output existed.

| Window | Best T | Pooled precision | Recall | Balanced acc. | SWORDS balanced acc. |
|---|---:|---:|---:|---:|---:|
| **±2 (chosen)** | **0.945** | **0.748** [0.733, 0.763] | **0.775** [0.755, 0.794] | **0.755** [0.745, 0.766] | 0.730 [0.717, 0.742] |
| ±3 | 0.960 | 0.751 | 0.760 | 0.752 | 0.731 |
| ±4 | 0.960 | 0.735 | 0.758 | 0.741 | |
| ±6 | 0.970 | 0.747 | 0.726 | 0.738 | 0.720 |
| ±8 | 0.975 | 0.754 | 0.696 | 0.733 | |
| ±10 | 0.970 | 0.741 | 0.712 | 0.730 | |
| whole sentence | 0.940 | 0.703 | 0.771 | 0.721 | 0.706 (at 0.945) |

CIs come from a bootstrap over targets (2,000 resamples).

**By data set, at the chosen setting:**

| Data set | Precision | Recall |
|---|---:|---:|
| SWORDS (hard negatives) | 0.643 | 0.784 |
| TSAR | 0.977 | 0.762 |
| MultiLS | 1.000 | 0.767 |

**Same threshold, different unit.** Against the whole sentence at the same
T = 0.945:

| Measure | Whole sentence | ±2 window |
|---|---:|---:|
| Precision | 0.706 | 0.748 |
| Recall | 0.762 | 0.775 |

At this threshold the ±2 window beats the whole sentence on both counts.
Wider windows (3–8 words) recall more (0.79–0.80) but are less precise. The
window still cannot tell a plausible wrong-sense synonym well: SWORDS
precision is 0.64.

The full grid is in [calibration.json](calibration.json).

## 3. Lexical simplicity metric (new)

These measures are computed over content words (alphabetic, not a spaCy
stopword) of each output passage:

- **Word complexity**: mean of Kea-Roy's GBM word model
  (`evals/word_models.py`), with the sentence as context. Lower is simpler.
- **Rare share**: share of content words with Zipf < 3.5.
- **Changed-word Zipf**: the Zipf of inserted words minus that of deleted
  words, against the source. It is also given per applied swap: the
  replacement's Zipf minus the replaced word's.

## 4. Dev run: F2 against F2 + words (pre-registered)

**Setup.**

- Dev split: 100 passages (84 standard plus 16 simple controls).
- Word stage at its defaults: one edit per passage, refill on.
- Generation reused the parameter-sweep dev state's caches only after a
  byte-identity check of all 90 sources and every artifact.
- Regenerated F2, F2+v7 (no veto), `original` and `rule_based` equal the
  sweep state's outputs on 100 of 100 passages.

| System | FK | Kea S | Word complexity | Rare share | Passage NLI | NLI < 0.80 | Entities |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 14.627 | 0.2502 | 0.2845 | 0.1588 | 0.9861 | 0 | 1.00 |
| **F2** | 12.851 | 0.2700 | 0.2840 | 0.1594 | 0.9746 | 2 | 1.00 |
| F2+v7, window veto | 12.721 | 0.2732 | 0.2825 | 0.1564 | 0.9546 | **7** | 1.00 |
| F2+contextual, window veto | 12.738 | 0.2729 | 0.2828 | 0.1557 | 0.9461 | **7** | 1.00 |
| F2+v7, no veto (reference) | 12.717 | 0.2737 | 0.2821 | 0.1544 | 0.9172 | 12 | 0.99 |
| F2+contextual, no veto (reference) | 12.722 | 0.2738 | 0.2823 | 0.1537 | 0.9241 | 11 | 0.99 |

**Paired against F2.** All 100 passages; the CI is the v3 domain-stratified
bootstrap (10,000 resamples).

| Arm | Swaps applied (vetoed) | ΔFK [95% CI] | ΔS | Δ word complexity [95% CI] | Δ rare share | Δ passage NLI [95% CI] | Gates 1–4 | Over-edit | Beats F2 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| **F2+v7, veto** | 94 (50) | −0.130 [−0.164, −0.098] | +0.0032 | −0.0014 [−0.0018, −0.0011] | −0.0031 | **−0.020** [−0.039, −0.004] | ✗ gate 2 | ✓ | **no** (a, b) |
| **F2+contextual, veto** | 99 (23) | −0.113 [−0.143, −0.083] | +0.0028 | −0.0012 [−0.0015, −0.0008] | −0.0037 | **−0.028** [−0.057, −0.005] | ✗ gates 1, 2 | ✓ | **no** (a, b) |
| F2+v7, no veto | 100 | −0.134 | +0.0036 | −0.0019 | −0.0050 | −0.057 | ✗ | ✗ | no |
| F2+contextual, no veto | 99 | −0.129 | +0.0037 | −0.0017 | −0.0058 | −0.050 | ✗ | ✗ | no |

**Rule parts** (PROTOCOL.md). Both veto arms fail (a) and (b) and pass the
rest:

- (a) NLI CI lower end ≥ −0.005: failed;
- (b) gates 1–4: failed;
- (c) over-edit gate: passed;
- (d) Δ word complexity CI below 0: passed;
- (e) ΔFK ≤ 0: passed.

**What the veto did:**

| | v7 | contextual |
|---|---:|---:|
| Swaps considered | 144 | 122 |
| Vetoed on window NLI | 46 | 20 |
| Vetoed on checks | 4 | 3 |
| Passages with a veto | 27 | 19 |
| Of those, still swapped through refill | 21 | 19 |

The veto removes the entity failures (entities 0.99 → 1.00). It also cuts
the passage-NLI cost: −0.057 → −0.020 for v7 (65% less) and −0.050 →
−0.028 for contextual (44% less). The over-edit gate then passes.

**Zipf per applied swap:**

| Arm | Replaced word | Replacement | Gain [95% CI] |
|---|---:|---:|---:|
| v7 | 3.92 | 4.71 | +0.79 [0.64, 0.94] |
| contextual | 3.88 | 4.49 | +0.61 [0.47, 0.74] |

The swaps reach common words, but one swap per passage moves passage-level
word complexity by only about 0.5% (−0.0014 on 0.284).

### Where the remaining NLI loss comes from

The table bins applied swaps by their **whole-sentence** NLI (post hoc,
aggregate). "Newly < 0.80" counts passages that F2 kept at or above 0.80.

| Arm | Sentence NLI band | Swaps | Mean Δ passage NLI | Newly < 0.80 |
|---|---|---:|---:|---:|
| v7, veto | ≥ 0.95 | 85 | −0.010 | 3 |
| | 0.90–0.95 | 3 | −0.115 | 1 |
| | 0.80–0.90 | 3 | −0.171 | 1 |
| | < 0.80 | 3 | −0.087 | 1 |
| contextual, veto | ≥ 0.95 | 88 | −0.004 | 1 |
| | 0.90–0.95 | 4 | −0.095 | 1 |
| | 0.80–0.90 | 1 | −0.192 | 1 |
| | < 0.80 | 6 | −0.325 | 3 |

- **Rescued swaps.** 20 of the 193 applied swaps score below 0.95 on the
  whole sentence but pass on the window.
  - They are about 10% of the swaps, but account for 8 of the 12 new
    below-0.80 cases. Counted per arm: 10 distinct passages, of which
    `sci-v3-013` falls in both arms.
  - Passage NLI agrees more with whole-sentence NLI than with the window
    on these swaps. Only a human judgment can say whether they are good.
- **Accepted swaps that still fail.** Swaps above 0.95 on the sentence
  push 4 cases (3 distinct passages) below 0.80.
  - These are not borderline passages. F2 had them at 0.82–0.99, and they
    fall to 0.33–0.71.
  - Example: one swap scores 0.994 on its sentence and 0.985 on its window,
    yet costs the passage −0.63.
  - Passage NLI is unstable on some single-word changes. Neither the
    sentence nor the window veto can predict that.

## 5. Applied-swaps page (local)

The page is at `~/.local/state/cleartext/swap-veto-20261118/applied.html`.
It is phone-readable and works offline.

- **Applied swaps.** All 193 swaps applied on dev, for both arms. Each card
  shows:
  - the sentence before and after, with the swap highlighted;
  - the window and its NLI in both directions;
  - the whole-sentence NLI, for comparison.
- **Order.** Lowest window NLI first.
- **Vetoed swaps.** The 73 vetoed swaps are in a collapsed section.

## Limits

- **What is measured.** These are automatic signals: NLI, readability
  formulas, heuristic checks, and a word-complexity model trained on
  CompLex. They are not human judgments of the outputs.
- **Gold sets.**
  - SWORDS and TSAR are news and fiction.
  - Annotators judged whether a substitute fits, not strict meaning
    equivalence.
  - The ±6 window that motivated this study was first seen on dev swaps.
    That is why W was chosen on gold only.
- **NLI precision.** The veto's NLI uses float32 weights; the evaluator
  uses float16 (within 0.002, November 17 fidelity check).
- **Selection.** The dev passages already chose F2. Nothing here touches
  the locked test split, and no default changed.

## Provenance

- **Code:**
  - `models/src/cleartext/swap_veto.py`;
  - `evals/swap_veto.py` (`gold`, `calibrate`, `freeze`, `run`, `judge`,
    `page`).
- **Commits:** the protocol and design (`373c226`) came before any window
  score; the choice (`de7d61f`) came before any dev output.
- **Queue:** Yoga, personal queue at `later` priority, one job at a time.
  - Gold window NLI, 4 threads: `0356c0640477`, `0615f230f70f`,
    `32a6a09a9785`.
  - Dev generation and scoring, 1 torch thread, 6 of the 12 budgeted
    chunks: `3f0b062b0be9`, `7a05f797e87c`, `5d3f832ce177`,
    `1ef57e3568f4`, `899370e4a22a`, `e910591c2aa6`.
  - Judge and page: `b75af43d2563`, then `1674334e4c92` after fixing how
    applied swaps are counted, then `a030208c100f` after the independent
    review's fixes. No verdict or delta changed.
  - The swap-matching fix, the diagnostics and the `judged` flag were
    written after the dev results were read. None of them feeds the rule.
- **Verify:** `scripts/verify` passes (600 tests, jobs `4e51546af255` and `8c42bb2dfbed` after the review fixes).
- **Dev state signatures:**
  - Reused sweep state: generation `223acb6f0e3f…`, scoring
    `b818b07ac341…`.
  - This state's scoring signature is identical, so metrics and
    sentence-NLI memos were reused.
- **Rerun:**
  1. `python -m evals.swap_veto gold --state S`, then
     `calibrate --state S --out calibration.json --choice veto-choice.json`.
  2. Commit the choice.
  3. `freeze --state S/dev`, then `run --plan S/dev/plan.json --state S/dev
     --design design-dev.json` until COMPLETE.
  4. `judge --state S/dev --design design-dev.json --out judge-dev.json`,
     then `page`.
