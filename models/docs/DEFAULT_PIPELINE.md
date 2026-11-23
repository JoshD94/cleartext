# Default pipeline (on since November 23, 2026)

ClearText's default system runs three stages in order. No stage uses an LLM.

| Stage | What it does | Evidence it passed |
|---|---|---|
| 1. **F2, the safe split** | Authored phrase rules, then T5 + dependency compound splitter with the five hard checks and a 0.5 small-NLI floor per split | Locked test (200 passages): ΔFK −1.20 [−1.48, −0.95] vs the previous incumbent ([report](../../evals/reports/20261116-locked-test/README.md)) |
| 2. **Structure v3** (`KEPT_V3`) | passive_v5 (by-phrase passive → active), passive_recovered ("we" from context), appositive_v4, fronted_split; each edit behind F2's hard checks and the 0.5 entailment floor | Fresh text, three blind raters: 94.1%, 91.4%, 93.8%, 92.9% Good ([report](../../evals/reports/20261122-structure-v2/README.md)) |
| 3. **Learned words, arm p50** | Trained eligibility, candidates, ranker and gate, then round 19's frozen p50 filter at its pre-committed threshold; at most one swap per sentence, three per passage | Round 26 (fresh): 96.3% Good; rounds 24–26 frozen: 92.1% (174/189); adoption page: 93.3% (56/60) ([report](../../evals/reports/20261118-learned-words/README.md)) |

```sh
# Stage 1 is generated like every finalist (ensemble_study / param_sweeps, config F2
# in evals/default_pipeline.py). Stages 2-3 on F2 outputs ({id: text}):
python -m evals.default_pipeline run --state ~/.local/state/cleartext/<dir> \
  --f2 f2.json --out ~/.local/state/cleartext/<dir>/default.json
```

```python
from evals.default_pipeline import DefaultAfterF2
stage = DefaultAfterF2(state_dir)        # loads structure v3, small NLI, the word stage
text, edits = stage(f2_output)           # edits = {"structure": [...], "words": [...]}
```

## Dev check of the combined system

Test set v3 dev, 100 passages, on top of F2's cached dev outputs (locked
test split not read). Metrics: Week 7 readability/preservation and passage
NLI (DeBERTa), F2's cached metrics reproduced 87/87. Paired deltas vs F2
with 95% CIs ([evals/reports/20261123-default-on/](../../evals/reports/20261123-default-on/README.md)).

| System | Passages changed | ΔFK grade | Δ passage NLI | Below 0.80 NLI | Dev gates |
|---|---:|---:|---:|---:|---|
| F2 + structure v3 | 5 | −0.031 [−0.067, −0.003] | +0.0003 [+0.0001, +0.0005] | 2 (F2: 2) | all pass |
| F2 + words p50 | 31 | −0.070 [−0.096, −0.046] | −0.0049 [−0.021, +0.004] | 3 | all pass |
| **Default (F2 + v3 + p50)** | **35** | **−0.100 [−0.142, −0.064]** | −0.0047 [−0.020, +0.004] | 3 | **all pass** |

On the 35 changed passages, FK drops from 14.38 to 14.10 and passage NLI
from 0.965 to 0.951; one passage newly falls below 0.80. Numbers, negations
and entities are all preserved; the over-editing control gates pass. The
word stage made 34 swaps on dev (for example utilizes → uses, purchased →
bought, numerous → many). `default_pipeline check` reproduced round 26's
recorded p50 picks on 4 of 4 passages.

## Caveats

- Each stage was rated on its own. Stages 2 and 3 were not rated together on
  top of F2; the combined system has the dev gates above, not a joint blind
  rating. The word stage's ratings used raw fresh passages, not F2 output.
- `passive_v5` is 89.8% Good outside Federal Register text; it fires rarely
  (about 0.4 sentences per 100).
- p50 keeps about half of the word stage's candidate swaps; on news and
  encyclopedic text it swaps rarely.
- The adoption page was marked by Claude at Joshua's request (one AI rater).
- The v7 sentence demo (`models/demo/`) is a separate process and is
  unchanged. The locked test split was not reopened.
- Resources: BERT-base, cross-encoder/nli-deberta-v3-small, `en_core_web_sm`
  and `_md`, the learned-word state (`~/.local/state/cleartext/learned-words-20261118/`,
  including the 160 MB sense table), all offline. About 20–90 s per
  4-sentence passage on Yoga at 4 threads; peak RAM about 2.3 GB.
