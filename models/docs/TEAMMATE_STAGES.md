# Chhoeng's passive and phrase stages in the pipeline

**Credit:** the passive-to-active converter (`passive/passive_to_active.py`,
PR #14) and the phrase finder with its CRF detector, SimplePPDB list and
BERT/fluency/NLI phrase replacer (`pos_tag/`, PRs #14 and #15) are
Chhoeng's work. This page only describes how they plug into
`EnsembleClearText`. His code is unchanged; it needed no fixes to run.

## Wiring (opt-in, defaults unchanged)

`models/src/cleartext/teammate_stages.py` loads his modules by file path and
returns the `(text, edits)` shape `analyze()` uses:

| Adapter | `analyze()` argument | What it calls |
|---|---|---|
| `PassiveStage(placeholder="auto")` | `passive_converter=` | `PassiveToActive(placeholder=...).transform` per sentence; `placeholder=None` converts only passives with a "by" phrase |
| `PhraseFinderStage(sources)` | `phrase_finder=` | `phrase_finder.find_complex_phrases` with his `PHRASE_LISTS` sources; replacements applied as in his `test_phrase_finder.apply_replacements` |
| `EntailmentVeto()` | `word_veto=` | his `phrase_replacer.entailment` (min of forward/backward NLI) with his tuned `min_entailment` 0.575, as a meaning floor for word edits |

Stage order inside `analyze()`: grammar → negation → passive → structure →
authored phrase rules → his phrase finder → words. Term protection and the
code-literal guard wrap the new stages like the old ones. `structure_rewriter`
may also be a list of learned backends applied in order (T5, then the
compound splitter).

```python
from cleartext.ensemble_pipeline import EnsembleClearText
from cleartext.teammate_stages import PassiveStage, PhraseFinderStage

pipe = EnsembleClearText.load(run)
pipe.analyze(text, passive_converter=PassiveStage(placeholder=None),
             phrase_finder=PhraseFinderStage(("manual", "verified", "model")))
```

`PhraseFinderStage(("simpleppdb",))` uses his 38,949-phrase SimplePPDB list
alone; his default (`manual`, `verified`, `model`) adds the CRF detector and
the replacer, which needs `bert-base-uncased`, `all-MiniLM-L6-v2` and
`cross-encoder/nli-deberta-v3-small` locally (pass `models=` to pin paths).

Landmine: never put `pos_tag/` on `sys.path`. Its `evaluate.py` would shadow
Hugging Face `evaluate`; the adapters register his modules under their own
import names instead.

## Evidence

The November 10 ensemble study measured every level on the 90 sourced
development passages: [evals/reports/20261110-ensemble-v2](../../evals/reports/20261110-ensemble-v2/README.md).
Tests: `models/tests/test_teammate_stages.py`.
