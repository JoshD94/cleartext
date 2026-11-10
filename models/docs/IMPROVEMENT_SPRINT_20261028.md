# Cleartext improvement sprint, October 28

The clearest measured gain is small. On the current sentence model's complete
128-input development panel, the optional split selector makes 55 edits instead
of 54. It keeps every baseline edit and all 32 copy references unchanged. The
new example is:

> We came in and changed their way of life for their goods.
>
> We came in. We changed their way of life for their goods.

The selectors share one frozen pool of four beams per input. Mean reference
overlap rises 0.491 points. This reused development result does not measure
semantic accuracy or justify a new default.

## Changes and evidence

- [Code preservation](CODE_PRESERVATION.md) now reaches candidate guards, phrase
  matching, word targeting and the final ensemble output. Damaged snippets cause
  rollback. The demo describes discarded edits as attempts. The saved replay
  prevents one known authored backtick-loss case in each condition and changes
  no output on the 96 code-free development inputs.
- [Grammar correction](GRAMMAR_PIPELINE_SPRINT.md) permits a bounded local checker
  follow-up after adding a contraction apostrophe. The native authored example
  changes `He dont` through `He don't` to `He doesn't`. The 96-input development
  panel has no preprocessing difference. Three checker timeouts remain failures
  in the records. This is a narrow mechanism fix, without a general accuracy gain.
- [Sentence selection](structure_sprint_selection.md) adds an opt-in fallback
  behind a preferred copy. An earlier version lost 31 baseline edits and was
  rejected. The revised policy keeps baseline edits. It recovers two shorter
  splits on 256 archived inputs and one on the current 128-input panel.
- [Candidate morphology](coverage_morphology.md) repairs two dictionary-supported
  verb analyses, `taping/tap` and `scraping/scrap`, among 929 reused development
  cases. The native comparison adds two listed replacements at the default
  threshold in each of three repeats, with no additional unlisted edits. Strict
  selection adds one listed replacement per repeat. The gain falls below the
  fixed adoption bar, so the repair stays opt-in.
- [Benchmark validation](PIPELINE_BENCHMARK.md) now requires the independent
  passage file and expected system list before scoring. All 1,900 existing
  predictions match the 190-input, ten-system matrix. Shared omissions and a
  missing system fail before metric computation.

The focused suites pass 249 distinct tests and 13 subtests across the Mac and
Yoga. This includes seven new tests for the inference artifact manifest. Independent
reviews covered literal protection, rollback reporting, selector routing,
grammar composition and evaluation completeness. The candidate manifest review
and native comparison are recorded separately with their exact freeze.

No new model variant was trained or saved in this sprint. The candidate
comparison reconstructs baseline classifiers from their original folds only;
repaired candidates never enter fitting. The current step 906 checkpoint was
evaluated with unchanged bytes. The running demo was preserved.
Raw inputs, outputs, earlier failed attempts and source snapshots remain outside
Git in `~/.local/state/cleartext/improvement-sprint-20261028/`. Aggregate evidence
is under `models/evidence/`.

## What still limits improvement

Of the current sentence model's 74 baseline copies, 17 have no shorter split in
four beams. Another 45 have shorter proposals that all fail existing checks.
Eleven pass those checks but fail the added source-word or event checks, and one
yields the recovered split. This points to proposal quality and parser limits,
beyond which beam the selector prefers. The counts do not establish that every
rejection is correct.

Dictionary lemma repair reaches only two lexical inputs. Its measured paired
net gain is two, below the fixed adoption requirement of five. Broader gains need
better meaning-preserving proposals and reliable
measurements of accepted edits. The project still has no verified 90% full-pipeline
semantic accuracy result.
