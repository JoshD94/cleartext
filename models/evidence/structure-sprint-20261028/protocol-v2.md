# Revision after the complete v1 replay

V1 on both reused development panels recovered three baseline copies but lost31 baseline edits. It is rejected as a general replacement. All rows and code are archived in measured-v1. No threshold changes follow.

V2 preserves any model_order output that differs from the source after common.normalize. It applies the same fixed source_anchored_split eligibility and ranking ONLY when model_order outputs a copy. Existing raw checks, guards and repairs remain. Extra split constraints govern only newly introduced edits, never certify inherited baseline edits. Default stays model_order, opt-in only. No reference influences decisions.

Before replay: run native gates and a positive test that a valid baseline punctuation repair or edit survives. Report all differences, inherited edits and copy-reference edits, all gains/losses on both panels, and raw v1/v2 source hashes. All256 rows are reused development. No model weights or new decoding are used. This revision was designed after inspecting v1 aggregates, so it is a development iteration, never a fresh test result.

V2 first replay showed two additional splits, zero overlap losses, and a third raw-string difference that only undid baseline whitespace cleanup. Its aggregate misnamed that normalized-copy difference a lost edit. Before final validation, retain the exact baseline output when no extra split passes and correct counters to require a normalized edit for recovery/loss. Save all rejected split traces in split_candidates. This is an implementation/reporting correction, not another threshold or ranking variant. Archive initial v2 before refreezing.
