# Rater brief (structure v2; written once, reused verbatim for every rater and round)

You are rating automatic sentence simplifications for readers who find long,
complex English hard to follow (for example, English learners or people
reading outside their field). Each card shows:

- **Context**: up to three sentences that came just before, for reference only.
  They are not rewritten. Use them to judge who did what.
- **Before**: one original sentence.
- **After**: the system's rewrite. It may be one sentence or two.

Give each card exactly one mark, judging **After** against **Before**:

| Mark | Use it when |
|---|---|
| `good` | After says the same thing as Before (same facts, same who-did-what, same certainty, nothing added or lost), reads as correct English, and is at least a little easier to read. |
| `notsimpler` | After keeps the meaning and is correct English, but it is not easier to read: the same difficulty, or harder or clumsier. |
| `meaning` | After changes the meaning in any way: a different or invented doer, a lost or added fact, a changed relation (cause, contrast, time, condition), a phrase now attached to the wrong thing, or a changed degree of certainty. |
| `ungrammatical` | After keeps the meaning but has a grammar or wording error a careful editor would fix (agreement, word form, capitalization in the middle of a sentence, a broken or run-on sentence). |

Rules:

- If more than one mark applies, use the most serious: `meaning`, then
  `ungrammatical`, then `notsimpler`.
- A doer that the context makes clear (for example "we" when the writers have
  been speaking as "we") is not invented. A doer that the text does not support
  is `meaning`.
- Repeating a name or noun phrase to start a new sentence is fine if it is
  correct.
- Judge only what changed. Do not mark down problems that were already in
  Before and are unchanged in After.
- You see no system names, scores or other raters' marks. Rate every card on
  its own.

Output: a CSV file with the header `card,mark` and one row per card, using
only the four mark names above.
