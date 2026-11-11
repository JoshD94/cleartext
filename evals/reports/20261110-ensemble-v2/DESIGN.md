# Ensemble v2 design (pre-registered before any new score was read)

Written 2026-11-10 on branch `josh/ensemble-v2-ablations-20261110`, before
the v2 plan was frozen and before any v2 cell was generated or scored. Git
history for this file is the evidence. It extends the November 9 study
([design](../20261109-ensemble-study/DESIGN.md)); the
[frozen selection rule](../20261109-ensemble-study/SELECTION_RULE.md) applies
unchanged.

## New stages (Chhoeng's code, wired as opt-in stages)

| Factor | New levels | What runs |
|---|---|---|
| passive | `on`, `agentive` (Nov 9 cells are `off`) | `passive/passive_to_active.py` (PR #14), `PassiveToActive(placeholder="auto")` for `on`, `placeholder=None` (only passives with a "by" phrase) for `agentive` |
| phrases | `simpleppdb`, `authored+simpleppdb`, `crf`, `authored+crf` | `pos_tag/phrase_finder.py` (PRs #14/#15). `simpleppdb`: his SimplePPDB list alone (`PHRASE_LISTS=simpleppdb`). `crf`: his default sources (manual list, verified list, CRF detector with `phrase_replacer.py`'s learned table/BERT and fluency + NLI checks). `authored+X`: our authored phrase rules first, then his finder |
| structure | `t5+compound_tagger` | the screened T5 splitter, then the PR #12 compound splitter on its output |
| structure (ablation) | `t5_raw`, `t5_unchecked` | T5 without its per-sentence generation checks; and without those or the passage structure checks |
| words (ablation) | `contextual_veto` | PR #8 contextual words; an accepted word edit is dropped when Chhoeng's `phrase_replacer.entailment` (min of forward/backward, `cross-encoder/nli-deberta-v3-small`) is below his tuned 0.575 (`pos_tag/replacer_settings.json`); the next accepted word may take the slot. Not the evaluation NLI model, and no threshold is tuned on this panel |

Stage order: grammar → negation → passive → structure → authored phrases →
Chhoeng's finder → words. Guardrails and term protection wrap the new stages
like the existing ones. Every default stays off.

## Phase C: OFAT for the new levels ([design-ofat.json](design-ofat.json))

From two bases: **W** = the November 9 winner (`words=off, phrases=authored,
structure=t5`) and **V** = the v7 baseline. Each of the seven new
passive/phrases/structure levels is changed alone from each base (14 configs).
W and V are regenerated under the new source tree; `report` must find their
outputs identical to the November 9 cells (90/90) or it refuses to merge.

**Activity test** (unchanged from November 9): a level is active if, against
the base it was changed from, mean FK moves by ≥ 0.02, mean passage or
sentence NLI by ≥ 0.005, or ≥ 3 passage outputs differ. A level is active if
it is active from either base.

## Phase C2: interactions the activity test calls for

Derived mechanically from the phase C activity results, committed before
running, with phrases=authored and the other factors at base level unless
listed:

1. **passive × structure** (if a passive level is active): passive {off,
   active levels} × structure {rules, t5, compound_tagger, plus
   t5+compound_tagger if active} × words {v7, off}.
2. **phrases × words** (if a new phrase level is active): phrases {authored,
   active new levels} × words {v7, contextual, off}, structure=t5.
3. **t5+compound_tagger × words** (if active): words {v7, contextual, off}.

If the union exceeds 40 new configs, words is restricted to {v7, off}.

## Best ensemble and phase D ablations

**Best** = the frozen rule's winner over every config run in November 9 and
phases C/C2. Ablations then remove or replace each component of the best one
at a time: each other structure level (including `t5_raw`, `t5_unchecked`
when the best uses T5), each other phrases level, each other passive level,
words → v7 / contextual / contextual_veto (and off if the best uses words),
negation / grammar / terms on. A stage chain (all stages off, then the best's
stages added in pipeline order) gives the stage-by-stage FK. If the best is W,
[design-ablation-W.json](design-ablation-W.json) is the ablation design; the
W ablations may run alongside phase C because every config counts in the
final selection anyway.

For each ablation, against the best: paired per-passage deltas of FK and
passage NLI, mean with a percentile bootstrap 95% CI (10,000 resamples of the
90 passages, seed 20261110), and the passages whose output differs. A
component **matters** if removing or replacing it moves FK by ≥ 0.02 or NLI by
≥ 0.005 with a 95% CI that excludes zero.

## Selection, finalists and reporting

- **Final winner:** the frozen rule over every config run (November 9, C, C2
  and D). Tie-break: ASSET test SARI with `evaluate`'s 0/0 = 1 convention,
  the one pre-registered in `31f2334`; the Xu 0/0 = 0 ranking is reported
  alongside, with the disclosure that the November 9 report switched
  conventions after scoring.
- **ASSET finalists:** the top five passing configs, filled with the
  FK/passage-NLI Pareto configs nearest the NLI gate if fewer than five pass,
  plus every config tied with the winner. Configs already scored on ASSET on
  November 9 are reused.
- Labeled alternatives as on November 9: gate 4 on non-sentence-initial
  entities, and gates 1–2 on sentence-level NLI.
- Raw text and predictions stay in `~/.local/state/cleartext/ensemble-v2-20261110`;
  Git gets aggregates (per-passage metric values by passage ID, no text).

## Caching

New plan and state directory. The plan reuses the November 9 T5 sentence,
grammar and lexical-decision memos and its metric cache, only after
`reuse_compatibility` checks that the frozen artifacts, runtime and every
source file are unchanged except `EnsembleClearText.analyze` (compared by
AST with that method removed) and new files. Outputs are never reused.
