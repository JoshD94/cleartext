# Structure v4 on top of F2: dev plan (written November 23, 2026, before the dev outputs)

Descriptive, not a selection step: round 8 judges the new operations on
fresh text. This applies the combined stages to F2's cached outputs on the
test set v3 **dev** split (100 passages; `structure_v2 dev` refuses locked
test IDs) and reports paired deltas against F2, as DEV_V3.md did for
`KEPT_V3`. The dev outputs may be produced while round 8 is rated; which
row becomes `KEPT_V4` is decided by round 8 alone.

- **Arms** (all with the 0.5 entailment floor on every edit, operations in
  this order):
  - `combo-v4-p6`: `KEPT_V3` + `passive_v6`
  - `combo-v4-n3`: `KEPT_V3` + `nominal_v3`
  - `combo-v4-all`: `KEPT_V3` + `passive_v6` + `nominal_v3`
  - `KEPT_V3` = `passive_v5`, `passive_recovered`, `appositive_v4`,
    `fronted_split` (the `combo-v3` row of DEV_V3.md, reused from its
    cached outputs for comparison).
- **Reported:** passages changed, ΔFK, Δ passage NLI and Δ sentence NLI with
  the domain-stratified bootstrap, the selection gates and the over-edit gate.
  **Dev rule** (as ROUND1.md and DEV_V3.md): every gate passes, the lower 95%
  bound of Δ passage NLI ≥ −0.005, and ΔFK ≤ 0. The row matching `KEPT_V4`
  (`KEPT_V3` + the round-8 passing ops) must meet it.
- Nothing ships; F2 stays the default.
