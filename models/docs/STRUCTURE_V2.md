# Structure v2: sentence operations beyond splitting

F2 (the shipped default) simplifies structure only by splitting long
sentences (T5 plus the compound splitter). Structure v2 adds rewrites that
change how a sentence is built, not how many sentences there are. Each one
is rule-based on the pipeline's spaCy parse, opt-in, applied after F2, and
measured on its own (evidence:
[evals/reports/20261122-structure-v2](../../evals/reports/20261122-structure-v2/README.md)).
No LLM is used.

```python
from cleartext.structure_v2 import KEPT, StructureV2

stage = StructureV2(ops=KEPT)            # the two operations that passed
text, edits = stage.rewrite(f2_output)   # (text, edits), like the other stages
```

## Results (fresh text, three blind Opus raters, majority Good)

| Op | Good | Edits / 100 sentences | Status |
|---|---|---:|---|
| `fronted_split` | 92.9% [84–97] (rounds 2+3, 70 items) | ~0.4 | in `KEPT` |
| `passive_recovered` | 91.4% [83–96] (70; Federal Register text) | ~0.05 | in `KEPT` |
| `passive` (rounds 1, 2), `passive_v3`, `passive_v4` (by-phrase) | 72%, 61%, 78%; v4 78.9% [70–86] (109 items, rounds 4+5) | ~1 | not kept |
| `nominal` / `nominal_v2` | 90%, 88%, 77% / 72% (39, rounds 4+5) | ~0.1 | dropped |
| `appositive` / `_v2` / `_v3` | 75% / 75% / 71% | 0.3–1.7 | not kept |
| `frames` / `frames_v2` | 43% / 77% (13 items) | ~0.1 | too few |
| `fronted_move` | 0% (not simpler) | 0.3 | dropped |

Chhoeng's converter on the same text: 42% and 37% (by-phrase only), 3%
("someone/people"). After F2 on the test set v3 dev split, `KEPT` changes 3
of 100 passages (ΔFK −0.011, Δ passage NLI +0.0002, all gates pass).

## Operations (`models/src/cleartext/structure_v2/`)

| Op | Example | Module |
|---|---|---|
| `passive` (`_v3`, `_v4`) | "The UN has launched an investigation" from "An investigation has been launched by the UN" | `passive.py` |
| `passive_recovered` | "We analyzed the samples" from "The samples were analyzed", only when the preceding sentences speak as "we" and the verb is an author action | `passive.py` |
| `appositive` | "Amman is the capital of Jordan." split off from "…based in Amman, the capital of Jordan." | `appositive.py` |
| `fronted_split` | "Although X, Y." to "X. But Y." (because → So, after → Then) | `fronted.py` |
| `fronted_move` | "If X, Y." to "Y if X." | `fronted.py` |
| `frames` | "There are many people who live here." to "Many people live here."; "It was the board that approved it." to "The board approved it." | `frames.py` |
| `nominal` | "conduct an evaluation of the program" to "evaluate the program" | `nominal.py` |

Every edit must pass the F2 hard checks (numbers, negation, entities,
technical phrases, code literals) or it is dropped. An optional
`entailment`/`min_entailment` floor works like F2's split floor.

## Credit and relation to the teammates' stages

- **Passive to active** follows the approach of Chhoeng's converter
  (`passive/passive_to_active.py`, PR #14): spaCy's passive labels find the
  clause, the by-phrase becomes the subject and the passive "be" picks the
  active verb form. `structure_v2/passive.py` is a separate implementation;
  his code is unchanged and still runs through `teammate_stages.PassiveStage`.
  What it adds: realization from the original tokens (no "anti - tank" or
  moved quotes), skip rules for parses where a flip changes meaning
  (coordinated predicates sharing the subject, means agents like "by
  ambulance", dangling fronted modifiers, time phrases, relative clauses
  after the agent, non-projective parses), and "we" recovery from context.
- **Double negation** is Kea-Roy's rule (`rewrites.negation_rewrite`,
  [NEGATION.md](NEGATION.md)); it is not re-implemented here.

Versioned ops (`passive_v3`, `appositive_v2`, `nominal_v2`, ...) add the
skips each rating round motivated without changing the earlier versions'
outputs; `gate.py` holds the synthetic active→passive round trip and its
gradient-boosting gate (it did not raise human Good).

## Landmines

- Repeating an anchor noun phrase (appositive split) duplicates any number
  or negation in it, so the hard checks drop that edit. This is deliberate.
- `passive_recovered` never guesses a doer: no "we" in the three preceding
  sentences, a verb outside `AUTHOR_VERBS_V2`, a modal, a subordinate clause or
  a quote means no edit.
- The operations parse each sentence with `en_core_web_sm`; misparses are
  the main source of errors that the skip rules do not catch. Replaying
  rated items through rules written from them overstates quality (passive
  v4: 90–96% on replay, 76% and 82% on two fresh windows).
- The modules live under `models/src/cleartext/`, so they change
  `ensemble_study.generation_sources()` for any newly frozen plan.
