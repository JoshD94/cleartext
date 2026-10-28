# Code preservation evidence

The [report](../../docs/CODE_PRESERVATION.md) describes supported delimiter formats
and their limits.

- `protocol.json` records the fixed replay and delimiter corrections.
- `summary.json` retains all 96 development inputs and 17 authored diagnostics.
- `manifest.json` fingerprints saved inputs and replay sources.
- `verification.json` records 35 passing checks within the combined suite, source hashes and Mac placement.

The replay rolls back one authored code-damage case in each saved condition.
No development-panel output changes. These are formatting checks and a saved-output
replay, without new inference, semantic labels, fitted weights or accuracy claims.
Raw text and outputs stay outside Git.
