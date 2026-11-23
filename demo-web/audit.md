# UI self-audit (erp-ui, redesign mode, 2026-11-23)

Live site, headless Chromium through the job queue (`tools/shoot.mjs`), 1440×900 and
390×844 (3×). States: empty (E1, E8), cold start (E2), live progress (E3, E9, E10),
all three steps changing (E4, E12), steps that change nothing (E5, E11, E13), dark (E6),
offline (E7, E14). Text is the authored examples only.

## Never
| # | Item | Result |
|---|---|---|
| 1 | Slop captions (eyebrows, restating subtitles, meta lines, corner text) | Pass. Title, then the input card. The footer holds one plain privacy line. |
| 2 | Number + title + caption triple | Pass. Steps are icon + title + status only. |
| 3 | Dividing lines | Pass. Space and soft cards only. |
| 4 | Sharp shapes | Pass. Cards 16 px, controls 12 px, `corner-shape: squircle` where supported. |
| 5 | Glued to the edge | Pass. 16 px gutter on phones, floating cards. |
| 6 | More than one typeface | Pass. System stack, tabular figures. |
| 7 | Random colour | Pass. Ink, greys, one blue accent (Simplify, change dots, inserted-word tint). Removed words are grey strikethrough, not red. |
| 8 | Text acting as a button | Pass. Examples and Copy are real buttons. |
| 9 | Inconsistent buttons | Pass after fix: every button 36 px (40 px on phones), one radius. |
| 10–13 | Avatars, clutter, logo, mockups | Pass. Plain wordmark; nothing else. |

## Always (the ones that apply)
- Empty/offline state: one 40 px Feather icon + short title (E7, E14). Pass.
- Skeletons, no spinners: pulsing step skeletons. Partial: a step that ends with
  no change collapses to its header, so the page gets shorter on arrival (E3 → E5).
- Labels next to controls, no help text under fields; focus is a fill, no ring. Pass.
- Plain language: "Split long sentences", "Passive to active", "No change". Pass.
- Phone: one column, full-width primary button in the thumb zone. Pass.
- Dark mode: neutral near-black, softer accent (E6). Pass.

## Fixed during the audit
- `[hidden]` was overridden by `display: flex`: the offline block showed on load (r1).
- Re-rendering replayed the entry motion each poll, so cards flashed mid-run (r2 E3).
- Passive/appositive rewrites diffed word by word read as confetti; changed runs now
  fall back to "old sentence struck, new sentence marked" (E4).
- Op labels fell back to "Sentence rewrite" (edits carry the op family, not `_v4`).
- Sentence-opening words ("Applicants") were listed as names under "Facts kept".

## Results tab (2026-11-23, edit mode)

New header switch (Try it / Results) and a Results view for Chhoeng's 200 research-paper
sentences (`site/public/results.json`, built by `tools/build_results.py` from a local
`evals/sentence_set.py` run). Headless shots, 1440×900 and 390×844 (3×), light and dark;
before = live site, after = local files.

- Never list: no eyebrows or captions under the title (source and caveats sit in one
  quiet note at the end), no row lines (table uses spacing), one accent (selected
  pill is a white card on the grey track; "Reads well" dot is the accent, other verdicts grey),
  every control 30–36 px with the shared radius, Feather icons, external-source link is a
  real icon button with a hover fill.
- Always list: segmented pill slides (220 ms); verdict counts are the list's filters;
  phone keeps metric tiles two-up then one, the system switch full width, and drops
  the domain tag so each card head stays on one line.
- Placeholder: none. Verdicts are one reviewer's reading, stated in the note.

## Results tab as one text (2026-11-23, Joshua: "one big text piece, with highlights")

The per-sentence cards and verdict chips are replaced by one card holding all 200
sentences in order (paragraphs of five, a heading per field), with the same strike +
insert diff as the Try tab. Inserted text is tinted by the step that made it, with a
legend on top: split (blue), passive to active (violet), simpler word (amber). Three
hues break the one-accent rule at Joshua's request; tints stay soft and have dark-mode
values. ClearText changes always diff word by word; the converter's rewrites keep the
whole-sentence strike. Hovering a sentence shows its paper.
