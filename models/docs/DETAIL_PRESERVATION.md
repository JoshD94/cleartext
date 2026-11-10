# Detail preservation and technical terms

This work adds two opt-in blocks to the lexical pipeline: context-weighted detail evidence and technical-term spans. Existing v7 artifacts and the running demo remain unchanged.

## Context-weighted detail evidence

The existing sense-fit member adds the contextual probability of every target sense that produced a candidate. Hypernyms inherit that probability even when they omit distinctions. The new scorer examines the candidate's relationship to each target sense, weighted by the full contextual sense distribution.

It appends five columns to the 52-feature contextual-similarity model:

1. Probability mass where the candidate is a lemma of the same synset.
2. Probability mass where the candidate is a direct hypernym.
3. Probability mass with no matching generator edge.
4. Expected loss of specificity, using the nonnegative difference in log descendant counts for hypernym edges.
5. Whether a contextual sense distribution is available.

Verb-group and similar-adjective edges are reported separately in candidate evidence. They are not counted as hypernym broadening. All nonempty sense distributions must be finite, nonnegative and sum to one. The new breadth feature clamps descendant counts to 20,000.

These are WordNet graph features weighted by a sense model. They do not measure retained facts or certify meaning. The ranker learns their usefulness from existing BenchLS candidate labels. No blanket hypernym ban or example-specific word ban is added.

## Technical-term spans

`detect_terms(doc)` returns exact character offsets, original text, evidence and a protection flag. It protects the existing phrase inventory, including plural and hyphenated forms. It also protects explicitly defined acronyms and their long forms when their initials match, in either `long form (ABC)` or `ABC (long form)` order. Repeated exact acronym occurrences are included.

Defined acronyms are case-sensitive during preservation checks, so the pronoun `us` cannot satisfy a missing `US` occurrence. Other noun compounds and adjective-noun spans are proposals with `protected: false`. This prevents ordinary phrases such as `school bus` from becoming automatic bans. The detector is not a trained domain-term classifier and does not identify every technical term.

With `technical_terms: true`, the pipeline excludes targets inside protected spans and rejects their replacement in direct target probes. It checks protected occurrence counts after syntax, phrase and combined word edits. This catches removal of one of two repeated terms, which a presence-only check can miss. Rejected stages are recorded in `term_preservation.blocked_stages`; term spans appear in the analysis result.

The term guard preserves surface forms and occurrence counts. It cannot verify their referents, relationships or surrounding claims.

## Development protocol

The comparison keeps all 929 BenchLS cases and 14,617 candidates, with three target-word-grouped 10-fold repeats, threshold 0.45 and meaning floor 0.10. All BenchLS is reused development data, including the historical holdout. No TSAR gold, new human labels or generated correctness labels enter the work.

The cached full sense distributions must reproduce each saved sense-fit score within 1e-5. Paired feature tables preserve the prior 52 columns exactly and are saved for replay. The acceptance bar remains a mean net gain of at least five edits, or a candidate log-loss decrease of at least 0.001.

The existing breadth traversal can exceed its 20,000 cap by a few siblings, with counts depending on traversal order. This caused the first strict historical-probability reproduction check to fail. The original failed attempt is preserved. The final comparison reports historical predictions separately and compares both models on the same frozen baseline columns. `--frozen-features` can replay the saved paired table exactly. Existing runs' breadth features are not changed.

BenchLS has no protected targets in this detector's scan. Its guard ablation therefore measures no technical-term benefit. Separate authored mechanism diagnostics cover acronym definitions, repeated terms, plurals and an ordinary noun phrase. Tests check that a proposed `convoluted` to `complex` edit outside a protected span remains eligible for term preservation. These are ungraded examples, not an in-domain accuracy benchmark.

## Result

The detail-feature ranker missed the acceptance bar. Its net changes were +2, -2 and -2 edits across the three repeats: mean -0.67. Mean candidate log loss increased by 0.0000784. No fitted detail-feature run was retained. The scorer remains available as an inspection block; the contextual-similarity ranker remains the best development model.

The final frozen term-only run is `runs/ensemble-technical-terms-final-20261002`. It shares the previous decision weights and enables `technical_terms: true`. Term guards made no difference on BenchLS because none of its targets lay inside a protected span. Development results are saved in `outputs/detail-preservation-final-20261002`.

Full sense caching covered all 929 cases. Maximum disagreement with the saved sense-fit member was 4.94e-8. A separate saved-table replay reproduced all fold probabilities exactly. All 90 tests and fatal/undefined-name lint pass. The original source changes and v7 artifacts remain preserved.

The six term diagnostics retained every protected occurrence. All five protected target probes rejected replacement; three sentences still received surrounding edits. The term variant's outputs on the original 13 domain diagnostics matched the previous best model exactly, including `convoluted` to `complex`. Known detail-loss choices such as `encodes` to `converts` and `deposited` to `placed` remain. No human correctness score is assigned to these examples.

Final jobs ran on Yoga in `cleartext-terms-final-20261002`, with a 3 GB memory cap and four CPU threads. The cache, comparison and both audits exited successfully. All 82 final source snapshots and their input/artifact fingerprints were verified. The existing demo still responds with HTTP 200.

The reusable inspection command exposes the detail scorer independently of ranker adoption:

```sh
.venv/bin/python scripts/inspect_preservation.py --text 'The database stores data.' --target database --start 4
```

It reports the full contextual sense distribution, candidate edge masses, expected specificity loss and term spans. It does not rewrite the input.

## Reproduction

Run from `models`, after resource checks, inside a named Yoga tmux job with a 3 GB memory cap and four CPU threads:

```sh
.venv/bin/python scripts/experiments/cache_detail_senses.py --output outputs/detail-senses-review
.venv/bin/python scripts/experiments/refine_detail_preservation.py --cache outputs/detail-senses-review --output outputs/detail-development-review --save-run runs/ensemble-detail-review --term-run runs/ensemble-terms-review
.venv/bin/python scripts/audit_domains.py --run runs/ensemble-terms-review --input diagnostics/term-spans.jsonl --output outputs/term-audit-review --limit 6
```

Use new output paths. A completed cache cannot be resumed. An incomplete cache permits `--resume` only when the source, inputs and artifacts still match its manifest. For an exact feature-table replay, add `--frozen-features outputs/detail-development-review/feature-tables.npz` to a new comparison invocation.
