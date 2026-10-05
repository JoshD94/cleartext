# Contextual replacement similarity

This experiment adds one feature to the 51-feature BERT support decision model. It compares the original target's contextual BERT vector with the complete replacement's vector in the same sentence. Each vector averages the last four encoder layers over every piece in its target span, then normalizes to unit length. The feature is their cosine similarity.

The scorer accepts multiword replacements, retains every replacement piece and centers long contexts around the target. Inputs have at most 256 pieces, including special tokens. Each input has its own bounded window, so different replacement lengths can retain slightly different context in long sentences. It builds attention masks from input lengths, including when a literal padding token occurs in the sentence.

Cosine similarity is a ranking signal. Antonyms and broader terms can have high similarity, so it cannot certify meaning preservation or serve as a substitute for the existing meaning floor.

## Development protocol

The candidate pool stays fixed at 14,617 rows across all 929 BenchLS cases. All BenchLS is reused development data, including the historical holdout. No TSAR gold, new human ratings or generated labels enter this experiment.

Both models use three target-word-grouped 10-fold repeats, decision threshold 0.45 and meaning floor 0.10. The new feature table must preserve all prior 51 columns exactly. The predeclared acceptance bar remains a mean net gain of at least five edits, or a candidate log-loss decrease of at least 0.001. Passing this bar permits a separate opt-in run, followed by the ungraded domain audit. The existing v7 model and demo remain the default.

The cache script saves scores after every case and supports resuming only when input, source and package fingerprints match. Native checks use three fixed cache positions and compare identity substitutions plus batch-one versus batched scores. Labels do not influence scoring or the choice of native checks.

## Current verification

Four scorer tests pass, including full-span pooling, attention masks, cache feature alignment, capitalization and wrong target offsets. Lint passes. The first cache attempt stopped after 99 cases because a case-sensitive assertion rejected the sentence's `Hurricane-force` against BenchLS's lowercase target. That attempt remains in `outputs/context-similarity-cache-20261002`. The corrected cache uses `outputs/context-similarity-cache-review-20261002` and accepts case differences while retaining exact offsets.

The job runs on Yoga in `cleartext-context-similarity-review-20261002`, with four CPU threads and a 3 GB memory cap. Yoga had about 12 GB available and no other heavy job at launch. The Mac was checked on AC power; it was not needed.

The complete cache contains all 929 cases and 14,617 candidate rows. Three native checks passed. Identity similarities were within 1.2e-7 of one, and the maximum single-target versus batched score difference was 1.2e-7. Scoring took 900 seconds, including CPU contention from another authorized job.

## Development result

| Mean across three repeats | BERT support | BERT support plus similarity |
| --- | ---: | ---: |
| Edits matching a listed substitute | 510.0 | 512.3 |
| Edits absent from the substitute list | 136.0 | 129.3 |
| Listed-substitute edit precision | 78.9% | 79.9% |
| Edit coverage | 69.5% | 69.1% |
| Correct minus incorrect edits | 374.0 | 383.0 |
| Candidate log loss | 0.221769 | 0.213589 |

Net gains were 12, 8 and 7 edits. The mean gain of nine and log-loss decrease of 0.00818 both pass the predeclared acceptance bar. These are dependent comparisons on reused development data, not fresh test results. An absent substitute counts as incorrect in this benchmark but does not establish a semantic error.

The saved opt-in run is `runs/ensemble-context-similarity-20261002`, loaded through `decision_features: prompted_similarity`. The prior 51 columns match exactly in the comparison. Results and fold probabilities are in `outputs/context-similarity-dev-20261002`. The default and running demo remain on v7.

All 80 tests pass after loader integration, and lint passes. Source snapshots and input hashes match the scored cache and fitted run. The original uncommitted source, v7 artifacts and prior model artifacts are preserved, and the running demo responds with HTTP 200.

## Technical diagnostic limits

Twelve of the 13 ungraded diagnostic outputs match the prior BERT support run. The bank example changes from `She put the money at the bank.` to `She placed the money at the bank.` This still omits the action of depositing funds; it does not establish a semantic improvement.

Other known detail-loss choices remain, including `encodes` to `converts`, `classify` to `separate`, `dividend` to `profit`, `constitution` to `law`, and `filibuster` to `delay`. The model still preserves `database` and `harsh sentence`, and keeps `convoluted` to `complex`.

The audit changed 10 of 13 sentences. Its observed median latency was 2.19 seconds, with a maximum of 8.88 seconds. These small-sample timings are not a controlled speed comparison. No human accuracy was measured.

The development gain supports retaining this opt-in model for further work. It does not resolve technical detail preservation or justify changing the demo. Novel BERT proposals still lack a validated meaning score and remain outside the candidate pool. Next work should examine context-specific sense compatibility and broader replacements, using existing data and the technical corpus when it becomes available locally.

## Reproduction

Run from `models`, using new output paths and a named tmux job after resource checks:

```sh
.venv/bin/python scripts/experiments/cache_replacement_similarity.py --output outputs/context-similarity-review
.venv/bin/python scripts/experiments/refine_context_similarity.py --cache outputs/context-similarity-review --output outputs/context-similarity-dev-review --save-run runs/ensemble-context-similarity-review
```

A completed cache requires a new output path for a fresh run. Use `--resume` only for an incomplete cache with unchanged source and inputs.
