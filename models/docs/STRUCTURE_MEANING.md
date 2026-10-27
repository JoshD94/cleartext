# Learned meaning check

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
