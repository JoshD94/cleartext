# Six simplification blocks

All six blocks are available through `BuildingBlockClearText`. The opt-in run is `runs/ensemble-building-blocks-20261002`. It uses the existing 52-feature contextual-similarity ranker and saved weights. The default v7 run and demo are unchanged.

| Block | What it does | Limit |
| --- | --- | --- |
| Detail preservation | Reports context-weighted WordNet sense relationships and expected specificity loss for candidates. | Graph evidence cannot certify retained meaning. The learned extension missed the development acceptance bar, so its weights were not adopted. |
| Technical-term detection | Protects the existing phrase inventory, validated acronym definitions and repeated exact acronym occurrences. Proposes other noun compounds for inspection. | This is a rule detector, not a trained domain-term classifier. Ordinary compounds are not automatically protected. |
| Grammar validation | Rejects candidates that increase detected article, demonstrative or simple subject-verb agreement errors. Tries the next eligible candidate. Checks the combined output too. | Parser errors remain possible. Counts avoid penalizing existing errors but can miss an error that moves elsewhere. Unknown pronunciations and unsupported constructions get no sound/agreement verdict. |
| Jargon explanations | Adds source-attributed dictionary glosses or document acronym expansions alongside the final text. | Ambiguous senses return alternatives. Missing entries stay unavailable. Model confidence is heuristic, and glosses never rewrite the text. |
| Document consistency | Checks protected occurrences, repeated entities, numbers, quantities, negation, modality and introduced acronym-definition conflicts. | Surface checks cannot establish that referents or claims are unchanged. Different replacements of a repeated word need an explicit shared sense ID to cause rejection; otherwise they are review suggestions. |
| Edit quality | Reports predicted content-word complexity, reading level, token change ratio, grammar, consistency and detail evidence separately. | There is no calibrated overall quality score or human rating. Readability predictions do not establish semantic preservation. |

The wrapper rolls back the combined output when grammar or document checks fail. It records the rejected output, edits and reports, then recomputes checks for the returned original text. Candidate detail evidence is explicitly marked as inspection evidence because the candidate may not appear in the final output.

## Local use

Run from `models`. Model inference belongs in a named Yoga tmux job after resource checks, with a 3 GB memory cap and four CPU threads. The CLI runs offline, accepts at most 20,000 characters and refuses to overwrite an output file.

```sh
.venv/bin/python scripts/analyze_blocks.py --text 'Although this sentence is convoluted, the idea is simple.' --output /tmp/cleartext-blocks-example.json
.venv/bin/python scripts/analyze_blocks.py --input /path/to/document.txt --output /tmp/cleartext-document.json
```

The output includes `term_spans`, `term_preservation`, `grammar_validation`, `document_consistency`, `jargon_explanations`, `edit_quality` and `block_rollbacks`, alongside the existing output, edits and trace.

The attributed audit CLI also supports the wrapper. `--building-blocks` activates the flags in the selected run and records the choice in its manifest:

```sh
.venv/bin/python scripts/audit_domains.py --building-blocks --run runs/ensemble-building-blocks-20261002 --input diagnostics/domain-audit.jsonl --limit 13 --output outputs/blocks-domain-review
```

For direct Python use, load a `Path` pointing to the opt-in run:

```python
from pathlib import Path
from cleartext.building_blocks import BuildingBlockClearText

pipe = BuildingBlockClearText.load(Path('runs/ensemble-building-blocks-20261002'))
result = pipe.analyze('Although this sentence is convoluted, the idea is simple.')
```

The run config enables `technical_terms` and five boolean `building_blocks` flags: `detail_evidence`, `grammar_validation`, `jargon_explanations`, `document_consistency` and `edit_quality`. Unsupported names or nonboolean values fail on load. The original `EnsembleClearText` loader does not activate the wrapper.

After rebuilding the 52-feature ranker using [CONTEXT_SIMILARITY.md](CONTEXT_SIMILARITY.md) and its linked prerequisite experiments, create the wrapper run locally. This copies the fit stacker and points at the already fitted decision weights. It does not retrain them or replace the default. Run once from `models`; an existing destination causes an error:

```python
import json
import shutil
from pathlib import Path

base = Path('runs/ensemble-context-similarity-20261002')
run = Path('runs/ensemble-building-blocks-20261002')
config = json.loads((base / 'config.json').read_text())
config['technical_terms'] = True
config['building_blocks'] = {
    'detail_evidence': True,
    'grammar_validation': True,
    'jargon_explanations': True,
    'document_consistency': True,
    'edit_quality': True,
}
run.mkdir()
shutil.copy2(base / 'stacker.pkl', run / 'stacker.pkl')
(run / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
```

## Verification

The full suite has 100 passing tests. Tests cover new grammar errors, existing errors, coordination, passive agreement, unknown acronym pronunciation, candidate fallback, repeated protected occurrences, conflicting acronym expansions, homonym review, uncertain glosses, empty quality inputs and output rollback.

The final experiment checks grammar against frozen probabilities for 929 reused BenchLS development cases over three target-word-grouped 10-fold repeats. It does not refit the ranker. It also runs all six blocks on 19 ungraded native diagnostics. Inputs, source snapshots and model artifacts are fingerprinted in `outputs/six-blocks-final-20261002/manifest.json`; results are in `summary.json` and `records.jsonl` there.

The final grammar ablation checked 688 eligible candidates and rejected none. Scores and selections matched the frozen baseline in all three repeats, with mean net change zero. This establishes no benchmark gain. An initial version rejected five candidates and lost three net edits per repeat because of parser coordination errors and existing source agreement errors. The corrected rules address those mechanisms and retain the failed attempt in `outputs/six-blocks-check-20261002`.

All 19 native outputs matched the preceding term-only run and required no combined rollback. The explainer returned 50 annotations: 24 available, 23 ambiguous and three unavailable. These counts describe availability, not correctness. The CLI smoke example simplifies `convoluted` to `complex` and emits the six blocks' reports. The final job ran on Yoga in `cleartext-six-blocks-final-20261002` with four threads and a 3 GB cap.

All BenchLS, including the historical holdout, is development data. The authored examples check mechanisms, not domain accuracy. No TSAR gold, new human labels, generated correctness labels or paid calls enter this work. See [detail preservation results](DETAIL_PRESERVATION.md) for the first two blocks' separate experiment.

To repeat the integrated check, use a new output path inside a named Yoga job:

```sh
.venv/bin/python scripts/experiments/check_building_blocks.py --run runs/ensemble-building-blocks-20261002 --output outputs/six-blocks-review
```

The next useful improvement is an independent domain evaluation with sentence-level meaning and grammar judgments, when collection of those labels is authorized. Reused lexical substitution labels cannot validate these six blocks' document behavior. The [parliamentary-text audit](PARLIAMENT_AUDIT.md) now covers a bounded sample of existing real text with no correctness labels. Two [replacement-sense experiments](SENSE_CONTRAST.md) missed the development acceptance bar, so the ranker remains unchanged.
