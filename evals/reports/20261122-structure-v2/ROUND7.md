# Round 7 protocol (structure v3; written November 23, 2026, before any round-7 output)

A confirmation round. No code changes: `passive_v5`, `appositive_v4` and
`passive_v4` run exactly as in round 6 (operation code at `803eefd`; later
commits add only reports, the results page and this protocol).

## Round 6 result (read before this file)

[round6.json](round6.json), majority of three blind raters, 120 cards
(all three raters agreed on 96%):

| Arm | n | Good | Meaning | Edits / 100 pool-F sentences |
|---|---:|---|---:|---:|
| `passive_v5` | 63 | 58 = 92.1% [83–97] | 3 = 4.8% | 0.45 |
| `appositive_v4` | 22 | 21 = 95.5% [78–99] | 0 | 0.25 |
| `passive_v4` (reference) | 60 | 49 = 81.7% [70–89] | 4 = 6.7% | 1.18 |

`passive_v5` met the round-6 bar (n ≥ 40, Good ≥ 90%, meaning ≤ 8%).
`appositive_v4` was inconclusive (n < 40). By source, `passive_v5` was
34/34 Good on Federal Register items and 24/29 (82.8%) on the rest
(Wikinews 12/14, Good articles 6/7, openFDA 6/8): the pass leans on
Federal Register text, so this round checks it on a new window.

The five round-6 failures (all read): a trailing "whilst working" that now
attaches to the new subject ("a tree hit a man whilst working …"),
"applause … met Pence's words" (not simpler), "their oxidations" now after a
plural agent that could be its antecedent, "MTA Bridges and Tunnels operate
it" (a single organization made plural), "repeated inversion of the
diluent bag must mix the solution" (a procedure as the doer); appositive:
"Negritos is the archipelago's earliest inhabitants" (agreement). None is
fixed here.

## Fresh window (pool G)

`structure_v2_fetch.py --round 7` (fetched during round 6, unread):
Wikinews and Good-article titles not in any earlier window; Federal
Register significant rules 2019-07-01..12-31; openFDA prescription labels
effective in 2018, excluding every earlier label document and every test set
v3 generic. Pool builder `pool-c --name G` (exclusions as in round 6, plus
pools F and F-wide): 5,731 standard and 28,678 wide sentences. The pool
files were built before this file; no sentence has been read.

## Arms, panel and rule

- As in round 6 (`r7_apply.sh`, `r7_apply_wide.sh`, `r7_panel.sh` in the
  state directory): `passive_v5` and `appositive_v4` with the 0.5
  entailment floor from pool G-wide (caps 80 and 60), `passive_v4` from pool
  G (cap 60), round robin over sources, seed 20261122, three new blind
  raters, brief and wrapper verbatim, majority rule.
- **`passive_v5` is confirmed** if rounds 6 + 7 pooled (identical code)
  meet the bar: Good ≥ 90% and meaning ≤ 8%. Round 7 alone is also reported
  against the bar, with per-source results.
- **`appositive_v4`** is judged on rounds 6 + 7 pooled by the same bar,
  if pooled n ≥ 40; otherwise it stays inconclusive.
- Nothing ships; F2 stays the default.
