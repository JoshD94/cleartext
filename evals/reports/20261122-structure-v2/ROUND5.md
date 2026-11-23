# Round 5 protocol (written November 22, 2026, before any round-5 output)

A confirmation round. No code changes: `passive_v4` and `nominal_v2` run
exactly as in round 4 (code at `e9e6f24`; later commits added only the
`KEPT` preset, tests and reports). Round 4 gave `passive_v4` 37/49 = 76%
[62–85] and `nominal_v2` 12/15 = 80% (too few to judge).

## What had been read before this file

- Every round 1–4 card, mark and failure.
- **Rating-trained gate (development only, not tested here).** The 146
  rated items whose output `passive_v4` reproduces exactly (128 Good, 18
  not; 12 of the 18 from round 4) were used to fit a standardized logistic
  regression on the converter's features plus five sentence features. Fit
  on rounds 1–3 and scored on round 4, AUC 0.44; 5-fold CV over all rows,
  AUC 0.58. There is no usable signal, so no gate is pre-registered.
- No pool-E sentence has been read or rewritten.

## Fresh window (pool E)

`structure_v2_fetch.py --round 5`: Wikinews and Good-article titles not in
any earlier window; Federal Register significant rules 2020-07..12; openFDA
labels effective in 2020, excluding every earlier label document and test
set v3 generic. Same pool builder and exclusions (pools A–D added).

## Arms and rule

- `passive_v4` up to 60 items, `nominal_v2` up to 40 (pool E-wide if rare),
  three new blind raters, brief and wrapper verbatim, majority rule.
- **Decision:** each is judged on rounds 4 + 5 pooled (identical code):
  kept if n ≥ 20, majority Good ≥ 85% and Changes meaning ≤ 10%. Round 5 is
  also reported alone.
- If either is kept, the combined stage adds it (order passive_v4,
  passive_recovered, nominal_v2, fronted_split) and the dev rule of
  ROUND1.md applies (dev outputs `combo-final-p4`, `combo-final-n2`,
  `combo-final-p4-n2` were already scored before this file).
