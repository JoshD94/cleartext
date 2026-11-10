# Sentence splitter repairs and human review

The splitter copies a subject only for a finite verb coordinated with the main
verb. A second clause with its own subject keeps it. Noun lists, embedded clauses,
shared auxiliaries and uncertain logical scope abstain. Semicolon splits need
two complete clauses. Original character slices preserve technical spelling.

For example:

```
The server checks each request and sends a response to the client.
The server checks each request. The server sends a response to the client.
```

For a second clause starting with "the client", it adds no "the server".
"The researcher visited her father and mother" stays together. Checks also
reject multiple direct subjects, extra nominal dependents, inverted fragments
and newly finite predicates absent from the source parse. These address specific
failures. Parser errors, ambiguous pronouns and modifier scope remain concerns.

## Compatibility and results

Changed candidates/check features require ranker feature version 2 and a refit.
Older ranker artifacts fail with an actionable error. To reproduce historical
experiments, use the frozen source recorded alongside those results.

The repaired ranker uses the same 2,000 training and 128 development pairs.
It sees 2,153 candidates including identities, versus 10,293 previously.
Development SARI is 34.98 versus the previous 40.46. This loses rewrite coverage
and reference overlap; it is not a measured overall quality improvement.
Its purpose is to remove known bad proposals. No new test score or human quality
rating exists. The fit uses source `c6ffeaa`, feature version 2 and threshold 10.
Later compatible source preserves those numerical feature meanings.

The [repair record](../evidence/structure-repair-20261026/README.md) contains
aggregate evidence, provenance and the fitted bundle receipt. Original v7,
CRF/T5 and earlier ranker artifacts remain intact. T5 has not been trained
further or promoted into the demo. Rules stay preferred; the ranker is experimental.

## Source collection

The [configuration](../structure-review-sources.json) pins URLs and document
hashes. Each domain/split has a different document, assigned before predictions:

| Domain | Training | Development | Reserved test |
|---|---|---|---|
| Technical | RFC 8259, JSON | RFC 9000, QUIC | RFC 9113, HTTP/2 |
| Corporate | Microsoft 2025 report | Apple 2025 report | NVIDIA 2025 report |
| Political | Obama 2009 inaugural | Biden 2021 inaugural | Obama 2013 inaugural |

Eight inputs per document give 72 total. Selection takes the first eligible
20..70-word sentences after any declared section anchor, with an initial capital,
terminal period and parser-complete main clause. Exact normalized overlap with
previous selected inputs and references is excluded. Test inputs stay reserved.

The initial extraction included filing forms and fragments. Its inputs and
development predictions are preserved as an extraction diagnostic. Revised
corporate inputs start in business prose. PDF headings and parser errors can
remain, so reviewers must reject unsuitable inputs. This small convenience
sample covers internet standards, three technology companies and US inaugural
speeches. Document separation does not exclude shared authors, paraphrases or
pretraining overlap. It does not establish broad domain performance.

Preparation needs the structural Python environment and `pdftotext`:

```sh
PYTHONPATH=models/src /path/to/venv/bin/python -m cleartext.structure.review prepare \
  --config models/structure-review-sources.json \
  --cache /path/to/cache --output /path/to/new-domain --count 8 \
  --exclude /path/to/pilot/data.json --exclude /path/to/followup/data.json \
  --exclude /path/to/earlier-technical-inputs.json
```

Raw documents, source rows, predictions and review sheets stay outside Git.
Public receipts contain URLs, hashes and aggregates. Source availability does
not grant redistribution rights.

## Human reference sheet

The prepared `references.csv` contains 48 training/development inputs, surrounding
paragraphs and blank review fields. Test inputs do not enter this sheet.

1. An author checks the extraction and writes a simpler version in `reference`.
   Focus on sentence structure. Keep facts, actors, quantities, technical names,
   tense, conditions, negation and uncertainty. Avoid ambiguous pronouns.
2. A different person compares the source, context and reference. Enter both
   names in `author` and `reviewer`. Mark `meaning_preserved`, `grammar_ok` and
   `simpler` as `yes` only after checking each one. Set `decision` to `approve`.
3. For already-clear inputs, copy the source unchanged, mark `simpler` as `no`
   and use `keep`. Use `reject` for unsuitable inputs or extraction errors.
   Leave unfinished rows blank. Retain every row and its original metadata.

Import rejects changed sources/splits, missing/duplicate ids, self-review and
unapproved labels. It cannot authenticate names; people must perform the review.

```sh
PYTHONPATH=models/src /path/to/venv/bin/python -m cleartext.structure.review import \
  --inputs /path/to/review_inputs.json --sheet /path/to/references.csv \
  --output /path/to/new-reviewed-data.json
```

Only approved training/development rows enter the export. Blank sheets export
zero examples. Keep development rows out of training and expand this starter
collection before claiming useful domain training.

## Blind output comparison

Development predictions compare rules, the repaired ranker and unchanged T5.
The `blind` command shuffles system order per input and writes the key separately:

```sh
PYTHONPATH=models/src /path/to/venv/bin/python -m cleartext.structure.review blind \
  --inputs /path/to/development_outputs.json --output /path/to/blind-review.csv
```

Give each reviewer a copy of the CSV, without its `.key.json` file. Fill in
`simpler`, `grammar_ok`, `meaning_preserved`, `reviewer` and `notes`. Use `yes`,
`no` or `unsure`. Explain lost facts or changed conditions. An unchanged sentence
is not a successful simplification simply because it preserves meaning. Keep
both reviewers' ratings and resolve disagreements without dropping failures.
Identical output strings share one variant and one judgment per reviewer. The
private key maps that variant to every system which produced it; it does not
create multiple independent judgments from the same rating.

## Next training round

After people approve enough examples, make a separate T5 training run mixing
reviewed training pairs with original general-domain training material. Choose
the mix, checkpoint and stopping point using development data only. Include
copy examples for already-clear inputs.

The trainer now accepts `--initial-checkpoint /path/to/original/seq2seq` to start
from our selected weights. It loads local files only, records their hashes,
starts a new optimizer and writes to a separate directory. It does not resume
the old optimizer or step count. The training record counts domains after length
filtering. Without this option, training starts from the pinned pretrained model.

Put the approved data mixture in a new run's `data.json`. A continuation command
can then use the explicit checkpoint and learning rate:

```sh
python models/scripts/structure_experiment.py train-seq2seq \
  --run /path/to/new-run --initial-checkpoint /path/to/original/seq2seq \
  --learning-rate 0.0001 --max-steps 100 --batch-size 2 \
  --accumulation-steps 2 --device cpu
```

Those are example settings, not settings selected by a new experiment. Follow
the host/resource rules before running. Freeze code, weights and settings before
reserved testing. Obtain independent human judgments and measure serving latency
before changing the demo default. Automatic scores alone do not meet that
requirement. No further project T5 training has run; local tiny-model regression
tests verify the continuation code without using project weights or data.
