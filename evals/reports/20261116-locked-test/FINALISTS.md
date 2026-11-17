# Locked-test finalists and judging rule (committed before the test split is unlocked)

Chosen from the test set v3 **dev** split only
([param sweeps](../20261116-param-sweeps/SWEEP_PROTOCOL.md), amendments 1–3).
Joshua approved this one-shot locked-test run on 2026-11-16. Nothing in this
file changes after test scores are read.

## Systems ([finalists.json](finalists.json))

| Label | Config | Why | Dev FK | Dev Kea S | Dev passage NLI | Dev NLI < 0.8 |
|---|---|---|---:|---:|---:|---:|
| `original` | source text | reference | 14.627 | 0.2502 | 0.9861 | 0 |
| `rule_based` | dictionary baseline | gate-4 reference | 14.529 | 0.2520 | 0.9773 | 1 |
| `nov9_t5` | `words=off, phrases=authored, structure=t5` | November 9 winner | 14.433 | 0.2519 | 0.9837 | 1 |
| `incumbent` | `words=off, phrases=authored, structure=t5+compound_tagger` | November 10 (v2) winner; the dev rule's pick (**no change**) | 14.377 | 0.2527 | 0.9838 | 1 |
| `F1` | incumbent with `phrases=simpleppdb`, `finder.ppdb_min=4.3`, T5 with every soft check waived (hard checks kept) | rule 5: lowest-FK config passing every dev gate | 12.790 | 0.2715 | 0.9548 | 4 |
| `F2` | incumbent with the "safe split": soft checks waived, passage-aware check, small-NLI floor 0.5 per split | amendment 3 (post hoc): lowest-FK config passing the full adoption test | 12.851 | 0.2700 | 0.9746 | 2 |

The v7 word baseline is not run: it is too slow on 200 passages, and the
gates already reject it on dev. Word configs did not complete on dev
(amendment 2), so no finalist uses word substitution.

## Panels

The plan is frozen with `param_sweeps freeze --unlock-test finalists.json
--dev-plan <dev plan>`. That refuses unless this design is committed and
unmodified, and unless the generation and scoring sources, artifacts and
signatures are identical to the dev plan's. The panels are:

- `test`: the 200 locked test passages (`test-LOCKED.json`, 168 standard + 32
  simple controls);
- `onestop_test`: 363 OneStopEnglish test sentences;
- `asset_test`: 359 ASSET test sentences (used before; not fresh).

Each system runs once. There is no rerun after scores are read, except to
complete missing cells of the same plan.

## Judging (from [PROTOCOL.md](../20261116-testset-v3/PROTOCOL.md), applied to each system)

- **R (readability)**, on the 168 standard test passages: paired ΔFK vs
  `original` with 95% CI entirely below 0, **and** paired ΔS (Kea-Roy) vs
  `original` with 95% CI entirely above 0.
- **M (meaning)**: gates 1–4 on all 200 test passages (fixed entity check;
  gate 4 vs `rule_based` on the same passages).
- **O (over-edit)**: on the 32 simple controls, mean changed-token rate ≤ 0.10
  and no control passage with NLI < 0.80.
- **Verdict**: *confirmed* if R, M and O all pass, otherwise *not confirmed*,
  naming each failed part.
- CIs come from the bootstrap stratified by domain (`testset_v3.paired`,
  10,000 resamples, seed 20261116).

## Head-to-head with the incumbent (this study)

A finalist **beats the incumbent on test** only if all of these hold:

1. its own verdict is *confirmed*;
2. paired ΔFK vs the incumbent ≤ −0.10, with CI entirely below 0;
3. paired Δ passage NLI vs the incumbent ≥ −0.010 (the dev adoption rule);
4. it passes the over-edit gate.

The v3 protocol's "no worse in meaning than X" (Δ NLI CI lower bound ≥ −0.005)
and ΔS vs the incumbent are reported alongside. If no finalist beats the
incumbent, the report says the incumbent stands. If the incumbent itself
fails a gate on test, the report says so.

Reported for every system, without a gate: OneStopEnglish test and ASSET test
SARI (Xu 0/0 = 0 primary, `evaluate` 0/0 = 1 alongside) and output FK.
