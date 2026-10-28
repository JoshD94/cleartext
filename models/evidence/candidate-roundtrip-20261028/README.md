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
closed the missing-artifact finding. Native measurement is queued for suitable
Yoga resources. It must complete before native parity or selection changes can
be reported. The repair stays opt-in and has no automatic promotion path.

Raw text, candidate rows, source snapshots and detailed reviews remain outside
Git in `~/.local/state/cleartext/improvement-sprint-20261028/candidate-coverage/`
and its parent. See the [report](../../docs/coverage_morphology.md).
