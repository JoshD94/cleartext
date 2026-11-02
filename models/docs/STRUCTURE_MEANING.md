# Learned meaning check

[Completed results](../evidence/structure-meaning-20261027/README.md) show a PAWS
classification gain, with frequent remaining errors and no Cleartext
simplification gain. Keep this experiment opt-in.

This experiment adds a small NLI model and a trained paraphrase classifier to
the optional sentence simplification tools. The demo still uses its existing
defaults. Scores are model predictions, not proof that a rewrite is correct.

The previous candidate-search experiment failed to improve held-out SARI and
missed a knowledge/how scope change. Its results and failure notes remain in
`evidence/structure-search-20261027`. Those examples are now development inputs.

## Protocol, fixed before scoring

Use [MiniLM NLI](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768),
revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`. Its published model card
lists SNLI and MultiNLI training. Freeze its weights. Score each text pair in
both directions without truncation. Abstain if either direction exceeds 512
tokens. Keep contradiction, entailment and neutral probabilities in the trace.

Train one logistic regression on 2,048 official PAWS-Wiki labeled-final training
pairs. Use C=1, balanced class weights, lbfgs and at most 500 iterations.
Inputs are the minimum and maximum of each NLI class probability across the two
directions. This makes the classifier independent of pair order. No synthetic
or authored training labels are allowed.

Use 256 official validation pairs to choose a probability threshold from
0.05 through 0.95 in steps of 0.05. Maximize balanced accuracy, then precision,
then threshold to break ties. Freeze the classifier and threshold before
scoring 512 official test pairs. Compare it with minimum directional entailment
at least 0.5, and with always predicting the training majority class.

Select each sample by SHA256 of seed 4701, split and row ID, without consulting
labels. Reserve all validation/test sentence strings before selecting training;
exclude exact normalized sentence overlap across selected splits and within
each split. Do not filter by grammaticality, NLI confidence or reference quality.
Record shortages rather than silently changing sample sizes. Public labels are
existing dataset annotations, not new Cleartext human ratings.

The [publisher's PAWS mirror](https://huggingface.co/datasets/google-research-datasets/paws)
supplies the official split files. The original archive returned HTTP 403.
`meaning-sources.json` pins file sizes, repository revisions and hashes. The
original [dataset license](https://github.com/google-research-datasets/paws/blob/02b29f3af1143620d1b7f352247e65c0bdd2ec18/LICENSE)
permits use and asks that Google be acknowledged. PAWS and the NLI model may
share Wikipedia or pretraining content. Exact exclusions do not remove that
possibility or all near duplicates.

After freezing, apply the scorer to saved Cleartext development outputs and
authored semantic diagnostics. Report those separately. PAWS classification
accuracy does not measure simplification quality, grammaticality or CEFR level.
Do not predict the reserved domain test inputs or promote a demo default.

## Run it

Install `requirements-meaning.txt` in the optional model environment. From the
repository root, set `PYTHONPATH=models/src`. Use a fresh directory for each
stage. Downloads, all pretrained weights and raw examples stay outside Git.

```sh
python -m cleartext.structure.meaning_experiment download --cache "$CACHE"
python -m cleartext.structure.meaning_experiment prepare --cache "$CACHE" --output "$RUN"
python -m cleartext.structure.meaning_experiment score --cache "$CACHE" --run "$RUN" --phase train --output "$TRAIN"
python -m cleartext.structure.meaning_experiment score --cache "$CACHE" --run "$RUN" --phase dev --output "$DEV"
python -m cleartext.structure.meaning_experiment fit --run "$RUN" --train-scores "$TRAIN" --dev-scores "$DEV" --output "$FIT"
python -m cleartext.structure.meaning_experiment score --cache "$CACHE" --run "$RUN" --phase test --classifier "$FIT/classifier.json" --output "$TEST"
```

Set each variable above to a separate local path. The runner saves input,
source and model hashes before inference. It rejects changed data, mismatched
NLI weights, changed scorer code and incomplete scored splits. Classifier
weights are a small JSON document, loaded without pickle or remote code.

The optional rollback compares only the already selected search output with the
source. A rejection or token-budget abstention returns the source and keeps the
rejected candidate and its checks in the trace. It does not search additional
candidates or certify the accepted text.

```sh
python models/scripts/predict_structure_search.py \
  --checkpoint "$T5_CHECKPOINT" --device cpu --details \
  --meaning-model "$CACHE/model" --meaning-classifier "$FIT/classifier.json" \
  'Alice reviewed the report and Bob checked the figures.'
```

To reproduce the separate development diagnostics, use the previous completed
search run and a fresh output directory:

```sh
python models/scripts/evaluate_meaning_diagnostics.py \
  --cache "$CACHE" --classifier "$FIT/classifier.json" \
  --authored models/evidence/structure-meaning-20261027/authored_inputs.json \
  --prior-run "$PRIOR_SEARCH_RUN" --output "$DIAGNOSTICS"
```
