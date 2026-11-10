# Source-anchored split fallback

October 28, 2026. `source_anchored_split` is an opt-in selector for a split hidden
behind the model's preferred copy. The default remains `model_order`. No weights,
training data, demo settings or services change.

The selector first runs the existing policy. It retains every baseline edit and
punctuation repair. If the baseline copies the source, the selector checks all raw
beams for an added sentence and a strictly shorter longest sentence. It chooses
the shortest longest sentence, with model rank breaking ties. If none qualifies,
it retains the exact baseline output, including spacing cleanup.

Every newly exposed split must pass all existing generation and boundary checks.
It must also retain source words in order. Only punctuation, boundary `and`
removal and exact subject or shared-auxiliary repetition are allowed. Event
records bind each predicate occurrence to its participants, finite tense,
operators and embedded-clause ancestors. Counters retain multiplicity. These
extra constraints govern new edits only and do not certify inherited baseline
edits. No reference influences selection.

Newly exposed splits abstain on quote or bracket delimiters. Shared generation
checks also protect detected code snippets byte for byte. Parser attachments,
pronoun references and adjunct meaning remain imperfect. For example, spaCy
misparses `Alice praised Bob and Carol praised Dave.`. The policy rejects its
otherwise reasonable split. A native test records that conservative limitation.

```python
from cleartext.structure.search import CandidateGenerator

generator = CandidateGenerator(
    checkpoint, device="cpu", selection_policy="source_anchored_split"
)
result = generator.predict(text)
```

## Measured development replay

The first version applied the extra constraints to every output. It recovered
three copies but lost 31 baseline edits across 256 inputs, so it was rejected.
The revised fallback preserves baseline edits. Its first replay also exposed
an unnecessary whitespace rollback and mislabeled that normalized-copy
difference as a lost edit. The implementation now retains baseline spacing
cleanup and the counters require a real normalized edit. All attempts remain
saved locally and the final replay used a fresh source/input freeze.

On the same two 128-input archived panels, the final fallback makes 74 edits versus
72 for the baseline. It adds two shorter splits, loses zero baseline edits and
keeps all 63 copy annotations unchanged. Both changed rows gain single-reference
overlap, with zero overlap losses. Mean overlap deltas are +0.491 and +0.273 points
on the separate panels. These are development diagnostics, not semantic accuracy.
The old historical test panel has been inspected and is development now.

One recovered example is `We came in and changed their way of life for their
goods.` It becomes `We came in. We changed their way of life for their goods.`,
matching its reference. The other retains `But` at the new sentence boundary.

Pure selection tests 21 passed. Native parser tests 11 passed and existing search
and review tests 37 passed on the Mac with normal memory pressure. Yoga coordinated
`cleartext-structure-sprint-20261028`. The frozen replay loads the local parser,
without transformer inference or fitting. The archived pools use an older
checkpoint. Their gains cannot establish a gain for aligned step 906.

## Current step 906 comparison

The complete 128-input reused development panel was then decoded once with the
current step 906 checkpoint, four deterministic CPU beams and unchanged weights.
Both selectors replayed the exact same saved beam pool. The fallback makes 55
edits versus 54, adding one shorter split and losing zero baseline edits. All 32
copy references stay unchanged. The one changed row gains reference overlap;
mean overlap rises 0.491 points across the complete panel. This is the same
`We came in` example above. It is one recovered edit, not an accuracy estimate.

The current pool also shows why selection alone has limited reach. Of 74 baseline
copies, 17 have no shorter split in the four beams, 45 have shorter proposals that
all fail existing generation checks, and 11 pass those checks but fail the added
source-word or event checks. One yields the new split. These are mechanism counts;
they do not establish that every rejected proposal is wrong.

Generation completed in 52.63 seconds on the Mac with normal memory pressure.
The first attempt stopped before inference because Mac Transformers 4.57.6 could
not read the tokenizer saved by 5.17.0. Matching Transformers 5.17.0 and Tokenizers
0.23.2 resolved that failure without editing the checkpoint. Mac Torch was 2.14.1;
Yoga's training runtime was 2.14.0 CPU. Package versions, exact weight and source
hashes, completion receipt and paired replay are saved. The combined focused suite
passed 242 tests and four subtests after the package update. No default promotion
follows this reused development result.

[Aggregate evidence](../evidence/structure-sprint-20261028/) includes the rejected
and retained policy manifests and results. Full raw rows and every difference
remain under `~/.local/state/cleartext/improvement-sprint-20261028/structure/`.

To reproduce with existing saved pools, use a fresh directory after code changes:

```sh
models/.venv/bin/python models/scripts/structure_sprint_replay.py freeze \
  --input /path/to/dev.jsonl --input /path/to/inspected-historical-test.jsonl \
  --run /path/to/new-frozen-run
models/.venv/bin/python models/scripts/structure_sprint_replay.py replay \
  --run /path/to/new-frozen-run
```

A current-checkpoint comparison must first decode its complete development panel
once and save all four beams. `structure_sprint_generate.py` prepares that pool
with code/input/weight fingerprints. Replay both policies on that same pool.
Neither coverage nor reference overlap alone justifies default promotion.
