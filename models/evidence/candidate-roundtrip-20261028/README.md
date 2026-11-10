# Candidate verb round-trip experiment

The model-free audit retains all 929 reused development inputs and changes two
verb analyses. Both gain dictionary-supported listed candidates. This does not
measure contextual correctness or accepted edits.

`audit-*` files record the completed dictionary audit. `protocol.json` fixes the
native comparison and adoption bar. `native-freeze.json` fingerprints all model
components and inference resources, including implicit word and context models.
Pinned reproduction and post-run validation reject drift. The historical cache
has no equivalent artifact hashes, so exact historical model-byte equality is
unproven. Three fixed native controls remain required.

The 31 candidate tests and nine subtests pass. Two independent review rounds
closed the missing-artifact finding. The native comparison completed on the Mac,
coordinated by Yoga tmux `cleartext-candidate-roundtrip-20261028`. It verified all
929 repair decisions, three unchanged controls and baseline probabilities within
the fixed tolerance. Default selection adds two listed replacements per repeat;
strict selection adds one. Neither adds unlisted edits. The measured net gain of
two fails the adoption requirement of five. The repair stays opt-in.

`native-summary.json` records all three repeats and post-run hash validation.
`native-freeze-mac.json` identifies the actual native resources and runtime.
`mac-stage-validation.json` records copied-file parity, all 929 audit records and
the fresh freeze. The original Yoga freeze remains `native-freeze.json`.
These are reused development results, without a full-sentence accuracy claim.

Raw text, candidate rows, source snapshots and detailed reviews remain outside
Git in `~/.local/state/cleartext/improvement-sprint-20261028/candidate-coverage/`
and its parent. See the [report](../../docs/coverage_morphology.md).
