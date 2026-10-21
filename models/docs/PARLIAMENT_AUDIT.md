# Parliamentary-text audit

This is a bounded check on real text already available locally. It uses the first 12 distinct sentences marked `europarl` in `data/raw/complex_test.tsv`. Selection follows source order. The importer excludes the annotated target and complexity rating.

CompLex test data was already used for component evaluation. These inputs therefore do not form a fresh pipeline test. They have no labels for the correctness of ClearText's replacements. This is an ungraded robustness check, not a semantic accuracy benchmark or a sample of the team's GovInfo corpus. Computer-science and corporate project corpora are still missing locally.

The source receipt records CompLex commit `4b31c2bd34da4c1265f9db3d727d95ee23099f14`, the raw TSV hash and record indices. `outputs/parliament-input-20261003` contains the exact inputs and selection manifest. `outputs/parliament-audit-20261003` contains source/model fingerprints, before/after text, candidate traces, six-block reports and identity/dictionary baselines.

## Result

The existing six-block run changed seven of 12 sentences, with one word edit per changed sentence. The dictionary baseline changed two; identity changed none. All three passed the differential grammar and document-consistency checks. No full-output rollback occurred. Passing these checks does not establish retained meaning.

| Mean per sentence | Six-block pipeline | Dictionary | Identity |
| --- | ---: | ---: | ---: |
| Predicted word-complexity reduction | 0.00336 | 0.00109 | 0 |
| Predicted CEFR-level reduction | 0.06685 | 0.00476 | 0 |

The explainer produced 42 annotations: 25 available, 17 ambiguous and none unavailable. These are availability counts, not verified explanation accuracy. Median lexical-analysis latency was 5.99 seconds and maximum latency was 41.44 seconds. These timings exclude loading, target probes and baseline work. They show the cost of long inputs; they are not a controlled comparison with the short authored diagnostics.

Examples needing review include `invested` to `spent` and `gentlemen` to `men`. Rule checks passed for both. The source/trace reports preserve the context for review, with `human_correctness: null`. No correctness labels from these examples enter training.

The audit covers lexical edits with sentence-structure and phrase rewriting disabled, as the existing audit command specifies. The wrapper still returns all six reports. Full-document comprehension, grammar beyond the supported rules and preservation of claims remain unmeasured.

## Reproduction

Use new output paths. Run inference in a named Yoga tmux job after resource checks, with four threads and a 3 GB memory cap:

```sh
.venv/bin/python scripts/build_parliament_audit.py --output outputs/parliament-input-review --limit 12
.venv/bin/python scripts/audit_domains.py --building-blocks --baselines --run runs/ensemble-building-blocks-20261002 --input outputs/parliament-input-review/input.jsonl --limit 12 --output outputs/parliament-audit-review
```

The completed job was `cleartext-parliament-audit-20261003` on Yoga. No new labels, data downloads, paid calls, TSAR gold, agents, pushes or PR updates were used. The default v7 model and demo remain unchanged.
