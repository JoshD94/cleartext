# Round 9 protocol (structure v4; written November 23, 2026, before any round-9 output)

A confirmation round. No operation code changes: `passive_v6`, `nominal_v3`
and `passive_v5` run exactly as in round 8 (operation sources byte-identical
to [r8-precommit.json](r8-precommit.json); later commits change only the
pool builder's exclusion list for pools J/K, reports and this protocol).

## Round 8 result (read before this file)

[round8.json](round8.json), majority of three blind raters, 132 cards (all
three agreed on 89%):

| Arm | n | Good | Meaning | Per 100 standard sentences (H + I, 6,452) |
|---|---:|---|---:|---:|
| `passive_v6` | 100 | 90 = 90.0% [83–94] | 2 = 2.0% [0.6–7.0] | 0.50 (outside the Federal Register 0.62) |
| `nominal_v3` | 20 | 20 = 100% [84–100] | 0 | 0.00 (18 of the 20 items are Federal Register) |
| `passive_v5` (reference) | 50 | 46 = 92.0% [81–97] | 0 | 0.40 (outside the Federal Register 0.49) |

`passive_v6` met the round-8 bar (n ≥ 40, Good ≥ 90%, meaning ≤ 8%) at the
boundary; by kind, in-place v5 shapes 63/71, coordination split 19/20,
clause coordination 8/9; outside the Federal Register 55/61 (90.2%).
`nominal_v3` had n = 20 < 40, so ROUND8.md defers it to rounds 8 + 9
pooled. The ten `passive_v6` failures (all read): v5 realization errors in
the in-place shape (a sentence-initial common noun or name left as "fish" /
"The Parliament", "does no longer", "States is", a "that"-relative left
after the new object, "respectively" with two agents, a quantifier scope
change "Multiple organizations … signed many submittals"), one split
starting with a number ("8.07 million viewers watched the episode"), one
clause coordination where "that availability" was read as a complementizer.
None is fixed here.

## Fresh windows (pools J and K, unread)

- **Pool J** (`structure_v2_fetch.py --round 9`, `download9`, fetched
  during round 8): Wikinews (211 documents, stopped at the time limit) and
  Good-article titles (800) not in any earlier window, Federal Register
  significant rules 2018-07-01..12-31 (20), openFDA labels effective in 2016
  (33 after exclusions).
- **Pool K** (`--round 11`, `download11`): Federal Register significant rules
  2016-07-01..2017-12-31 (400 documents), for nominalization items.
- Pool builder `pool-c --name J|K`: as round 8, and now also excluding pools
  C–E (and H, I, and J for K) by 8-gram; `prefilter` as in round 8 (pool J
  for every rated op, pool K for `nominal_v3` only). Sizes: J 3,895 standard
  and 13,068 wide sentences (prefiltered 2,310: Good articles 1,332, Federal
  Register 539, Wikinews 355, openFDA 84); K 1,500 and 244,384 (prefiltered
  14,870). The builder labels pool J's sources `*10` and pool K's `fedreg11`
  (its suffixes for J and K), whatever the download number. Hashes are in
  [r9-precommit.json](r9-precommit.json). The pool files were built before
  this file is committed; no sentence has been read or rewritten. The locked
  test split is never read.

## Arms, panel and rule

- As round 8 (`r9_apply.sh`, `r9_apply_wide.sh`, `r9_panel.sh` in the state
  directory): `passive_v6`, `nominal_v3`, `passive_v5`, each with the 0.5
  entailment floor; `nominal_v3` items from the prefiltered J + K wide pools,
  the passive items from the prefiltered J wide pool; caps 100 / 60 / 50,
  round robin over sources, seed 20261122; three new blind raters, brief and
  wrapper verbatim, majority rule.
- **`nominal_v3`** is judged on rounds 8 + 9 pooled (identical code): pass if
  pooled n ≥ 40, Good ≥ 90% and meaning ≤ 8%; round 9 alone is reported.
- **`passive_v6`** passed in round 8; it is **confirmed** if rounds 8 + 9
  pooled also meet the bar. If the pooled result misses, the report says so
  and `passive_v6` is not added to `KEPT_V4`.
- Nothing ships; F2 stays the default.
