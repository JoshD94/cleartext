# Input grammar correction

An optional local LanguageTool step now corrects the input before either the
ensemble pipeline or the experimental T5 splitter sees it. It preserves the
original text and records applied and rejected suggestions. It is disabled by
default, and the running demo has not changed.

The correction stage uses rules, not a language model. The evaluated server was
LanguageTool 6.6 with an empty server configuration, fixed `en-US`, and no n-gram
or neural-rule data. LanguageTool documents that its
[self-hosted server excludes cloud AI rules](https://dev.languagetool.org/http-server).
The existing spaCy parser helps identify protected spans and verb types.

## What it corrects

The policy accepts a/an corrections, missing apostrophes in selected negative
contractions, and verb-agreement suggestions that preserve tense and lemma.
For example, LanguageTool offers both `goes` and `went` for `She go`. Only `goes`
is eligible. Ambiguous eligible suggestions are left unchanged.

Detected names, protected technical terms, digits, quoted text and inline code
are protected from correction. Date and time spans permit only a/an changes;
the first policy incorrectly blocked `A hour` because the parser marked it as a
date. The second policy fixes that case. Name recognition can still miss names.

The client accepts only a loopback HTTP endpoint, disables redirects and proxy
use, and never falls back to a vendor API. A timeout or malformed response keeps
the original input and reports the failure. Corrections run once, with at most
eight edits. This is a limited grammar cleanup step, not a grammar certificate.

## Measured trial, October 27

The frozen step906 checkpoint from [the training-alignment pass](STRUCTURE_ALIGNMENT.md)
was compared with and without preprocessing. The trial reused all 96 sentences
from the already inspected broad development panel. Unchanged inputs reused
saved predictions; two unchanged controls and all five affected baselines were
reproduced exactly before comparison. No weights, references or training data
changed. This was not a fresh test or a checkpoint-selection run.

- Five inputs received five corrections, all missing contraction apostrophes.
- Five returned outputs changed. Shorter sentence splits stayed at 13 of 96.
- Mean correction latency was about 0.10 seconds per input on Yoga.
- One corrected input still had a checker match. Zero matches on the others
  does not prove correct grammar.

Seventeen authored mechanism checks were also run. Fourteen had explicit
expected corrected strings, including four inputs expected to stay unchanged.
All 14 expectations matched the grammar stage. After the frozen splitter,
13 of 14 matched, compared with 4 of 14 without preprocessing. These small,
authored checks are regression diagnostics, not an accuracy benchmark.

Examples from those diagnostics:

- `She go to school every day.` became `She goes to school every day.`
- `She were ready yesterday.` became `She was ready yesterday.`
- `They has 25 neural networks.` became `They have 25 neural networks.`
- `This is an useful tool.` became `This is a useful tool.`

There are clear limits. LanguageTool missed `There are several schools offer
evening classes.` and the wrong `where` in `The houses were painted and where
ready for sale.` A single pass changed `He dont know the result.` to `He don't
know the result.`, leaving disagreement that a later check detected. Separately,
the splitter removed backticks from an inline-code diagnostic in both conditions.
The grammar stage's protection therefore does not guarantee that later stages
preserve code. Preprocessing also does not fix lost causal links or resolve
pronouns after a sentence split.

The initial run is preserved locally. Its two public illustrative sentences were
replaced with authored diagnostics in the final run so corpus text stays outside
Git. Aggregate evidence is in `models/evidence/grammar-20261027/`. Raw suggestions,
predictions, input hashes and both run receipts are local under
`~/.local/state/cleartext/grammar-20261027/`. The trial ran on Yoga in
`cleartext-grammar-20261027`, capped at 3 GiB for the server and evaluation together.
The trial server stopped when evaluation finished.

## Review of correction accuracy

I inspected all five applied development edits in context, the 17 authored
diagnostics, and the 96 development inputs with their saved warnings. The five
applied edits look correct, but all five only add contraction apostrophes. Some
sentences still have other errors after that edit. This is a small qualitative
review by the coding assistant, not independent human validation or an estimate
of general correction accuracy.

The checker missed a missing noun or extra article, a malformed reflexive,
sentence fragments and possessive apostrophes. It also treated a word-boundary
error as a contraction in one raw suggestion. Our restricted correction policy
rejected that suggestion. A grammatical sentence received a vocabulary-strength
suggestion, and names received spelling warnings. Rejecting every checker warning
would therefore discard valid inputs while retaining some malformed ones.

The earlier 14-of-14 result covered ten intended repairs and four unchanged
controls. It excluded three report-only examples. Those exposed two missed
errors and one partial repair, `He dont` becoming `He don't`. Passing the 14
checks does not mean the checker fixes every sentence.

## Screen candidate data without changing the benchmark

`cleartext.grammar_screening` reads saved grammar results and creates a manifest.
It keeps every row and its original order, records input hashes, and never
rewrites text. It classifies warnings about the original input, before correction.
No inference, training or running grammar server is required.

- `flagged`: a selected grammar, punctuation or spacing rule found a suspected
  problem. This can be used to quarantine candidate data for a separate panel.
- `review`: spelling or an unknown rule needs inspection. Names and technical
  terms can trigger these warnings.
- `no_covered_issue`: no selected problem was found. This does not certify
  grammaticality. Known style preferences and warnings in protected spans are
  ignored for this screen.
- `unavailable`: the saved check failed or has malformed results. These inputs
  must never count as passing a grammar check.

On the 96 development inputs, the screen flagged 16, marked one for review and
found no covered issue in 79. Several of those 79 have clear grammar errors in
the qualitative review. All 96 remain in the manifest and benchmark. These
counts describe rule coverage, not a measured accuracy or recall.

From the repository root:

```sh
PYTHONPATH=models/src models/.venv/bin/python -m cleartext.grammar_screening \
  --saved-outputs /path/to/grammar-comparison/outputs.json \
  --output /path/to/new-screening-manifest.json
```

The command refuses to overwrite an existing manifest and rejects checks whose
original text differs from the saved source. Aggregate review evidence and the
manifest are in `models/evidence/grammar-review-20261027/`. Detailed judgments
with source text remain outside Git under the local grammar trial directory.

Use this screen to flag future training candidates or define a separately
reported diagnostic subset. A subset selected this way still needs its own
grammar assessment. Keep the original full benchmark, report its scores and
denominator, and disclose the screen policy and counts for any subset. Removing
flagged cases alone is not a model improvement. This change does not exclude
any data or enable grammar correction in the demo.

## Try it locally

Use Java 17 and the existing model environment. Download and extract
[LanguageTool 6.6](https://languagetool.org/download/LanguageTool-6.6.zip) outside
the repository. The downloaded archive's observed SHA256 was
`53600506b399bb5ffe1e4c8dec794fd378212f14aaf38ccef9b6f89314d11631`.

From its extracted directory, start the local server:

```sh
java -Xmx768m -XX:ActiveProcessorCount=2 -cp languagetool-server.jar \
  org.languagetool.server.HTTPServer --port 8786
```

From the repository root:

```sh
PYTHONPATH=models/src models/.venv/bin/python -m cleartext.grammar_preprocessing \
  --text 'She go home and we was ready.'
```

To include it in inference:

```python
from cleartext.grammar_preprocessing import GrammarCorrector

grammar = GrammarCorrector()
result = ensemble.analyze(text, grammar_corrector=grammar)
# Or, with a loaded SentenceGenerator:
result = generator.predict(text, grammar_corrector=grammar)
```

`original` stays raw; `preprocessed` is the corrected model input. `grammar`
contains the correction history. The splitter's `raw_checks` compare its proposal
with the corrected input. Passing those checks is not an end-to-end meaning
certificate.

The comparison command requires saved development outputs and an existing local
checkpoint, then creates a fresh output directory:

```sh
PYTHONPATH=models/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  models/.venv/bin/python -m cleartext.grammar_experiment \
  --saved-outputs /path/to/dev-comparison/outputs.json \
  --model /path/to/seq2seq --output /path/to/new-comparison
```
