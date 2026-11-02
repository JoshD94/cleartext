# Sentence split development replay

The first selector is rejected. It recovered three preferred copies but lost 31 baseline edits. Its full aggregate and frozen manifest remain here.

The corrected opt-in policy retains baseline outputs and only exposes a strictly checked split behind a baseline copy. Final reused development replay: 256 inputs, 72 baseline edits versus 74 opt-in edits, two additional shorter splits, zero baseline edits lost, zero changed copy annotations among 63 copy references. Both changed rows gain reference overlap; zero losses. Overlap and coverage are diagnostics, not semantic accuracy. Both panels are inspected development.

Mac native parser checks 11 passed; existing search/review 37 passed; pure selection 21 passed. Memory pressure remained normal. Yoga coordinated named tmux cleartext-structure-sprint-20261028. Raw predictions, all changes and parser logs remain outside Git under ~/.local/state/cleartext/improvement-sprint-20261028/structure/. Manifests describe the exact frozen source snapshots; later comment-only edits are not part of those snapshots. No weights changed or default promoted.

The current step 906 checkpoint was subsequently decoded on the complete 128-input
development panel. One four-beam pool feeds both selectors. Baseline 54 edits become
55, with one new shorter split, zero baseline edits lost and all 32 copy references
unchanged. Mean reference overlap rises 0.491 points. `step906-*` files retain
generation and replay receipts, hashes and versions. `opportunities-*` files count
where saved proposals stop passing checks. Raw rows and all attempts stay outside Git.
