# Target-centered BERT meaning windows

The original BERT word-vector function truncates input to the first 256 wordpieces. If the requested word occurs later, it returns an all-zero vector. A target split across the cutoff can also lose some of its pieces. The BERT sense scorer then loses evidence about the intended word. Its other sense-model components still run.

The opt-in window implementation keeps the complete target within a 256-piece input, including BERT's two special tokens. It spends the remaining space on nearby context and shifts the window toward the available text at either end. Each requested word uses its own window, so repeated words and several requested targets retain their exact positions. Targets that cannot fit completely raise an explicit error.

The implementation reuses the cached encoder, frozen sense vectors, temperature and learned decision model. It keeps the original last-four-layer averaging and normalization. The original `contextual.py`, existing configurations and demo default are untouched. Set `bert_sense.target_window` to `true` in a separate run configuration to select the new sense class. The round-trip sense model and BERT slot-fit members keep their existing behavior.

Validation uses short inputs, a target after 350 padding words and a word split into several pieces at the original truncation boundary. Padding examples are ungraded boundary tests, not a technical corpus or synthetic semantic labels. The check also scans the existing BenchLS input text without using substitute labels. No training or TSAR evaluation is involved.

Run from `models/` with the existing environment. Use new paths and a named Yoga tmux job for model work.

```sh
.venv/bin/python scripts/experiments/check_context_windows.py \
  --output outputs/context-window-another-check \
  --base-run runs/ensemble-relations-20261002 \
  --save-run runs/ensemble-windowed-another-run

.venv/bin/python scripts/audit_domains.py --baselines \
  --run runs/ensemble-windowed-another-run \
  --output outputs/domain-audit-windowed-another-run
```

The validation saves exact diagnostic inputs, vector arrays, window boundaries, BenchLS target-visibility counts, input and source hashes, and source snapshots. The saved run shares its base run's decision model and other weights. It copies the fit stacker and writes a separate configuration. Existing run directories cannot be overwritten.

This corrects target visibility. It does not establish better replacement accuracy on long technical text, and it does not address the known broader replacements such as `database -> information`. Those require separate evaluation on the existing team corpus when it becomes available locally.

## Verified result

The final encoder check reproduced both truncation failures. The late `encodes` target had zero of its two pieces visible to the old encoder and an all-zero vector. The windowed encoder retained both pieces and returned a unit vector. At the cutoff, `unaffordable` had only one of four pieces visible; the new window retained all four. These are input-alignment checks, not judgments about the words' meanings.

Five short-input vectors matched exactly, including inputs containing a literal `[PAD]` token. The implementation builds attention masks from sequence lengths so a literal token is not mistaken for padding added by batching. Tests also cover requests across the 16-item batch boundary, empty requests, repeated words and invalid target positions.

All 929 BenchLS contexts fit within the original limit. The longest has 121 wordpieces, and the latest target ends at piece 71. No target windows change on this data, so this check reports no benchmark improvement. It explains why the previous development comparisons could not catch this long-input failure.

The full 13-input diagnostic audit completed with the separate windowed configuration. Every output matched the prior relation model, including its known failures. It changed 12 sentences and retained the same 349 candidates. No human accuracy was computed. All 60 tests and fatal/undefined-name lint passed.

The saved configuration is `runs/ensemble-windowed-20261002`. It reuses the decision weights from `runs/ensemble-relations-20261002`; no model was retrained. Use it explicitly with `scripts/predict_ensemble.py --run runs/ensemble-windowed-20261002` or the audit command. The current v7 default is unchanged.

Review `outputs/context-window-check-review-20261002/` for encoder evidence and `outputs/domain-audit-windowed-review-20261002/` for full pipeline outputs, source snapshots, frozen artifact hashes, local BERT cache hashes and dependency versions. The earlier boundary check completed, but its following audit was stopped after a host-resource queue result. Its partial outputs remain separate from these completed runs.
