# Text-review packet: cases needing eyes on Yoga

The benchmark evidence stores output hashes, not texts, so the 11 cases
below are signal-coded only. Each needs a human (or LLM rubric pass)
reading the actual output on Yoga, where the predictions live
(`~/.local/state/cleartext/pipeline-eval-20261027/`).

## How to review

For each case, read original vs output and answer:

1. What did the output change? (quote the changed span)
2. Is anything in the original missing, altered, or contradicted?
3. Code it with the taxonomy in `error_analysis.md`:
   E1 detail loss, E2 meaning shift, E3 predicate-role error,
   E4 mechanical check failure, E5 grammar breakage, E0 check artifact.
4. If E1/E2/E3: name the lost or altered detail in one sentence.

`score_predictions.py` (same directory) builds the checklist from a
predictions file so the reading is mechanical. Input: JSONL, one object
per line, `{"id": ..., "system": ..., "output": ...}`, plus `--originals`
(a JSON file mapping id to original text, or `evals/data/week7_passages.json`
for the authored-smoke ids). It prints explicit-check results, TF-IDF,
readability, and the stored benchmark signals for each id, and writes
`review_checklist.md`.

## Cases

Signals are the `full_pipeline` (guarded) arm unless noted. NLI fwd =
P(output entailed by original); NLI bwd = P(original entailed by output).

### Both directions low: read first (8)

| ID | Panel | NLI fwd / bwd | Question |
| --- | --- | --- | --- |
| quic-rfc9000-1 | domain (CS) | 0.41 / 0.49 | Output diverges broadly. E1, E2, or a rewrite that drops the spec condition? |
| asset-valid-663 | ASSET | 0.13 / 0.07 | Severe divergence in all arms. What did the output become? |
| asset-valid-278 | ASSET | 0.38 / 0.003 | Near-zero backward. Which detail vanished? |
| asset-valid-1782 | ASSET | 0.68 / 0.006 | Near-zero backward despite high forward. Dropped qualifier? |
| asset-valid-790 | ASSET | 0.99 / 0.014 | Forward intact, backward collapsed. E1: name the lost detail. |
| desse-train-8123 | DeSSE | 0.96 / 0.010 | Same pattern. Name the lost detail. |
| asset-valid-778 | ASSET | 0.95 / 0.37 | Moderate backward loss. Which clause or modifier went? |
| asset-valid-1222 | ASSET | 0.97 / 0.37 | Same question. Compare with 778: same mechanism? |

### Documented detail loss (2)

| ID | Panel | NLI fwd / bwd | Question |
| --- | --- | --- | --- |
| asset-valid-565 | ASSET | 0.85 / 0.08 | Benchmark note: geographic directional detail lost in all guarded combos. Confirm from text; quote the direction phrase. |
| asset-valid-219 | ASSET | 0.85 / 0.57 guarded, 0.72 / 0.47 unguarded | Same documented loss, but guarded NLI sits above the 0.50 flag line. Confirm the loss is present in the guarded output too. |

### Mechanical + improvement cases (2)

| ID | Panel | Signal | Question |
| --- | --- | --- | --- |
| biden-inaugural2021-0 | domain (polsci) | `tenses_preserved` fails, NLI 0.97/0.97, all arms | What tense shifted? Quote original and output verb phrases. |
| polsci-01 | authored smoke | bwd 0.165 lexical-only, 0.374 combined | What did the combined output change vs lexical-only, and what loss remains? (Original: the legislation/reforms passage in `evals/data/sample_passages.json`.) |

## Also flagged, already coded E0 (no review needed unless disputed)

Seven entity-check failures with NLI ~0.99 in both directions
(desse-train-5012/3664/7215/5487, asset-valid-781/1723, plus the
verified corp-05 case in the local run). The entity check is a
capitalized-span heuristic; these are check artifacts, coded E0 in
`error_analysis.md`.
