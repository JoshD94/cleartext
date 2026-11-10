# Preserving code during simplification

The earlier grammar trial found that the sentence splitter could remove inline
backticks even though input preprocessing had protected them. Input protection
did not cover later stages. The pipeline now checks code separately.

The lexical and sentence-generation guards require the same detected code
snippets, byte for byte and with the same multiplicity. The ensemble also excludes their tokens from word
selection. Phrase matching skips code spans. A final ensemble check restores the
preprocessed source if a later stage damages code. Its trace records the failed
attempt and rollback. This check does not establish meaning preservation in the
surrounding prose. The existing guard-disabled ablation bypasses the lexical and
final checks and word-target masking; phrase matching always skips code.

The detector supports single-backtick spans, unpadded multibacktick spans, and
backtick or tilde fenced blocks. Multibacktick spans containing literal backticks
are supported even with padding. Unfinished fenced blocks are protected to the
end of the input. Unmatched inline markers are plain text. Escaped inline markers
are ignored.

Tokenized corpora use backticks as quotation marks. Unmatched backtick/apostrophe
pairs are prose. A matching backtick run takes priority over an apostrophe, which
can occur inside code. Multiple single-backtick prose quotations may therefore
be ambiguous and receive code protection. Space-padded multibacktick pairs without a literal
backtick inside are ambiguous and are also excluded. This sacrifices support for
that Markdown spelling to avoid freezing ordinary tokenized quotations. The
detector is not a complete Markdown parser. Natural-language quotation protection
remains a separate problem.

## Checks and saved-output replay

All 35 checks pass, including byte-level extraction and preservation fixtures,
lexical rejection, phrase matching, word-target masking, final rollback and an
unguarded control. A positive fixture simplifies prose beside unchanged code.
These checks ran within a combined 242-test suite, with four additional subtests,
on the Mac in 19.95 seconds because Yoga's 15-minute load exceeded
the placement limit. Mac memory pressure stayed normal. Yoga coordinated it in
`cleartext-structure-906-20261028`. The demo trace also retains discarded phrase
proposals as attempts and does not label them as applied after rollback.

The fixed replay keeps every row of the existing 96-input grammar development
panel and all 17 authored diagnostics. It does not run a model or select new
thresholds. The 96 inputs contain no detected code, and their saved outputs are
unchanged. The authored panel contains one code input. Both saved conditions had
damaged its backticks; both are rolled back to their respective source. That
prevents the logged formatting loss and also removes the attempted edit. It is
one diagnostic, not an accuracy estimate.

All 929 BenchLS source strings were checked for detected code. None matches the
final detector. Earlier extraction attempts mistook tokenized quotations for
code; those attempts remain saved locally. The format corrections used source
text and delimiter patterns, without joining substitute labels. The final
candidate-pool result does not measure native candidate selection or a new model
score.

[Aggregate evidence](../evidence/code-preservation-20261028/) records the protocol,
input and source hashes, counts, tests and host. Raw source strings, replayed
outputs and prior attempts stay outside Git under
`~/.local/state/cleartext/improvement-sprint-20261028/`. Existing weights and the
running demo are preserved.

Run from the repository root with the existing local outputs and a new directory:

```sh
PYTHONPATH=models/src models/.venv/bin/python models/scripts/experiments/audit_literal_preservation.py \
  --saved-panel /path/to/grammar-comparison/outputs.json \
  --diagnostics /path/to/grammar-comparison/diagnostics.json \
  --candidate-input /path/to/all-candidates.json \
  --output /path/to/new-literal-replay
```
