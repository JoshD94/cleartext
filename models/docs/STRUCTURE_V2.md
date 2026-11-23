# Structure v2: sentence operations beyond splitting

F2 (the safe split) simplifies structure only by splitting long
sentences (T5 plus the compound splitter). Since November 23, `KEPT_V3`
runs after F2 by default ([DEFAULT_PIPELINE.md](DEFAULT_PIPELINE.md)). Structure v2 adds rewrites that
change how a sentence is built, not how many sentences there are. Each one
is rule-based on the pipeline's spaCy parse, applied after F2, and
measured on its own (evidence:
[evals/reports/20261122-structure-v2](../../evals/reports/20261122-structure-v2/README.md)).
No LLM is used.

```python
from cleartext.structure_v2 import KEPT, KEPT_V3, KEPT_V4, StructureV2
from cleartext.teammate_stages import EntailmentVeto

stage = StructureV2(ops=KEPT)            # the two v2 operations that passed (Good >= 85%)
text, edits = stage.rewrite(f2_output)   # (text, edits), like the other stages

# Structure v3: adds by-phrase passive -> active and appositive extraction,
# which passed Good >= 90% / meaning <= 8% with the 0.5 entailment floor.
stage = StructureV2(ops=KEPT_V3, entailment=EntailmentVeto().score, min_entailment=0.5)

# Structure v4: KEPT_V4 = KEPT_V3 + nominal_v3 (nominalization -> verb)
# + passive_v7 (coordinated passives; v6 without the shared-participle split).
stage = StructureV2(ops=KEPT_V4, entailment=EntailmentVeto().score, min_entailment=0.5)
```

## Results (fresh text, three blind Opus raters, majority Good)

| Op | Good | Edits / 100 sentences | Status |
|---|---|---:|---|
| **`passive_v7`** (v4: v6 + shared-participle skip) | **97.0% [92–99]**, meaning 0% [0–3.7] (round 10, fresh window, 100 items, no 8-gram repeats) | ~0.7 (v6 0.77 on the same pool) | in `KEPT_V4` |
| `passive_v6` (v4: v5 + coordination) | 90.7% [86–94], meaning 3.6% (rounds 8+9 without 8-gram repeats, 194 items); round 8 alone 84/94 = 89.4%, under its pre-registered 90% | ~0.5 (v5 0.39 on the same pools) | not kept |
| **`nominal_v3`** (v4) | **100% [95–100]**, meaning 0% (rounds 8+9, 78 items without repeats, 74 Federal Register) | ~0.07 in Federal Register text, ~0 elsewhere | in `KEPT_V4` |
| **`passive_v5`** (by-phrase, v3) | **94.1% [88–97]**, meaning 4.2% (rounds 6+7, 118 items) | ~0.4 | in `KEPT_V3` |
| **`appositive_v4`** (v3) | **93.8% [83–98]**, meaning 0% (rounds 6+7, 48 items) | ~0.3 | in `KEPT_V3` |
| `fronted_split` | 92.9% [84–97] (rounds 2+3, 70 items) | ~0.4 | in `KEPT` and `KEPT_V3` |
| `passive_recovered` | 91.4% [83–96] (70; Federal Register text) | ~0.05 | in `KEPT` and `KEPT_V3` |
| `passive` (rounds 1, 2), `passive_v3`, `passive_v4` (by-phrase) | 72%, 61%, 78%; v4 78.9% [70–86] (109 items, rounds 4+5) | ~1 | not kept |
| `nominal` / `nominal_v2` | 90%, 88%, 77% / 72% (39, rounds 4+5) | ~0.1 | dropped |
| `appositive` / `_v2` / `_v3` | 75% / 75% / 71% | 0.3–1.7 | not kept |
| `frames` / `frames_v2` | 43% / 77% (13 items) | ~0.1 | too few |
| `fronted_move` | 0% (not simpler) | 0.3 | dropped |

Chhoeng's converter on the same text: 42% and 37% (by-phrase only), 3%
("someone/people"). After F2 on the test set v3 dev split, `KEPT` changes 3
of 100 passages (ΔFK −0.011, Δ passage NLI +0.0002, all gates pass) and
`KEPT_V3` 5 (ΔFK −0.031, Δ NLI +0.0003, all gates pass; `passive_v5` fires
on none of them).

**Structure v4** ([evals/reports/20261123-structure-v4](../../evals/reports/20261123-structure-v4/README.md)):
`passive_v6` keeps every v5 rule and check and adds the coordination shapes
v2–v5 skipped: a second clause with its own subject flipped in place, and
"S was V-ed by A and VP2" split into "A V-ed S. S VP2." (each half flipped
under the v5 rules; the split shape alone scored 89.1%, and its known failure
is two participles sharing one by-phrase, "written and produced by X"). Six of
its round-8 items repeated text from earlier pools; without them round 8
missed its bar, so `passive_v6` is not kept. `passive_v7` is `passive_v6`
plus one skip: no split when a by-phrase after coordinated participles reads
as shared ("written and produced by X"); it passed round 10 on a fresh
window (97/100). On F2's dev outputs `KEPT_V4` still equals `KEPT_V3`.
`nominal_v3` uses curated pairs only, whitelisted light verbs, second-parser
agreement, an output parse check and the fluency floor. Neither fires on F2's
dev outputs, so `KEPT_V4` equals `KEPT_V3` there.

## Operations (`models/src/cleartext/structure_v2/`)

| Op | Example | Module |
|---|---|---|
| `passive` (`_v3`, `_v4`, `_v5`) | "The UN has launched an investigation" from "An investigation has been launched by the UN" | `passive.py` |
| `passive_recovered` | "We analyzed the samples" from "The samples were analyzed", only when the preceding sentences speak as "we" and the verb is an author action | `passive.py` |
| `appositive` (`_v2`–`_v4`) | "Amman is the capital of Jordan." split off from "…based in Amman, the capital of Jordan." | `appositive.py` |
| `fronted_split` | "Although X, Y." to "X. But Y." (because → So, after → Then) | `fronted.py` |
| `fronted_move` | "If X, Y." to "Y if X." | `fronted.py` |
| `frames` | "There are many people who live here." to "Many people live here."; "It was the board that approved it." to "The board approved it." | `frames.py` |
| `nominal` (`_v2`, `_v3`) | "conduct an evaluation of the program" to "evaluate the program" | `nominal.py` |
| `passive_v6`/`_v7` split | "The episode was written by X and directed by Y." to "X wrote the episode. Y directed the episode." (v7 skips "written and produced by X") | `passive.py` |

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

**Structure v3 (`passive_v5`, `appositive_v4`)** converts only a whitelisted
clause shape (a plain agent noun phrase that ends its clause or is followed
by a verb-attached phrase, nothing scoping over it, an action verb) and
requires four generic checks: `en_core_web_md` agrees with
`en_core_web_sm` on the subject, agent and attachments; the rewrite parses
with the intended subject and object (`output_check_v5`); the BERT-base
pseudo-log-likelihood of the rewrite drops by at most 16 nats
(`fluency.py`, `FLUENCY_FLOOR_V5`); and, as rated, the 0.5 entailment floor.
`v5_reasons(...)` lists every rule a clause fails.

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
- `passive_v5` and `appositive_v4` load `en_core_web_md` (installed in the
  venv; F2 does not use it) and `passive_v5` loads BERT-base; on a new host
  both must be present offline. Their measured configuration includes the
  0.5 entailment floor, which `StructureV2` applies only when given a scorer.
- Coverage is the price of the v3 precision: `passive_v5` converts about a
  third of what v4 converts (0.39 vs 1.06 sentences per 100), and on F2's
  dev outputs it found nothing to convert.
- The modules live under `models/src/cleartext/`, so they change
  `ensemble_study.generation_sources()` for any newly frozen plan.
