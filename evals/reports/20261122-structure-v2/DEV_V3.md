# Structure v3 on top of F2: dev plan (written November 23, 2026, before the dev outputs)

Descriptive, not a selection step: rounds 6–7 already judged the operations
on fresh text. This applies the combined stage to F2's cached outputs on the
test set v3 **dev** split (100 passages; `structure_v2 dev` refuses locked
test IDs) and reports paired deltas against F2, as round 1 did for `KEPT`.

- **Arm `combo-v3`:** `passive_v5`, `passive_recovered`, `appositive_v4`,
  `fronted_split`, in that order, with the 0.5 entailment floor on every
  edit (round 6–7 configuration for the two new operations; the floor is
  stricter than the rated configuration of the two `KEPT` operations).
- **Reported:** passages changed, ΔFK, Δ passage NLI and Δ sentence NLI with
  the domain-stratified bootstrap, the selection gates and the over-edit
  gate (ROUND1.md's dev rule: all gates pass, Δ passage NLI lower 95% bound
  ≥ −0.005, ΔFK ≤ 0), next to the earlier `combo-kept2` (`KEPT`) row.
- Nothing ships; F2 stays the default.
