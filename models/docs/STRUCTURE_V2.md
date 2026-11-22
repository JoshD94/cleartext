# Structure v2: sentence operations beyond splitting

F2 (the shipped default) simplifies structure only by splitting long
sentences (T5 plus the compound splitter). Structure v2 adds rewrites that
change how a sentence is built, not how many sentences there are. Each one
is rule-based on the pipeline's spaCy parse, opt-in, applied after F2, and
measured on its own (evidence:
[evals/reports/20261122-structure-v2](../../evals/reports/20261122-structure-v2/README.md)).
No LLM is used.

```python
from cleartext.structure_v2 import StructureV2

stage = StructureV2(ops=("passive", "appositive"))  # any subset of OPS, in order
text, edits = stage.rewrite(f2_output)
```

## Operations (`models/src/cleartext/structure_v2/`)

| Op | Example | Module |
|---|---|---|
| `passive` | "The UN has launched an investigation" from "An investigation has been launched by the UN" | `passive.py` |
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

## Landmines

- Repeating an anchor noun phrase (appositive split) duplicates any number
  or negation in it, so the hard checks drop that edit. This is deliberate.
- `passive_recovered` never guesses a doer: no "we" in the three preceding
  sentences, a verb outside `AUTHOR_VERBS`, a modal, a subordinate clause or
  a quote means no edit.
- The operations parse each sentence with `en_core_web_sm`; misparses are
  the main source of errors that the skip rules do not catch.
