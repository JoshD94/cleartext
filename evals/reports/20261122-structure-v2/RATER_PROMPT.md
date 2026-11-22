# Rater task wrapper (sent to each rater subagent with the brief)

Each rater is a separate general-purpose Opus subagent. It receives this
text with `{BRIEF}`, `{CARDS}` and `{MARKS}` filled in, and nothing else
about the study (no arm names, hypotheses or other raters' marks).

---

You are one of several independent raters. Read the rating instructions
below, then rate every card in the JSON file `{CARDS}`. Each card has
`card` (an ID), `context` (a list of up to three earlier sentences), `before`
and `after`.

Work through the file in order, in chunks (for example 40 cards at a time),
and append your marks to the CSV file `{MARKS}` as you go, so nothing is
lost. The CSV starts with the header line `card,mark`. Use only the marks
`good`, `notsimpler`, `meaning`, `ungrammatical`. Rate every card exactly
once. Do not run any program that rates or rewrites the sentences for you;
read each card yourself. Do not open any other file in that directory.

When you have rated every card, check that the CSV has one row per card,
then reply with only the number of cards rated.

Rating instructions:

{BRIEF}
