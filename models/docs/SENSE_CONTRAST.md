# Replacement sense contrast

The current ranker models the original word's intended sense and the replacement's contextual similarity. Its round-trip member uses the smaller SemCor/gloss model. This experiment adds BERT-based evidence about competing senses of the replacement.

The scorer compares the original target's contextual BERT vector with every WordNet sense vector for the candidate. It uses the existing frozen SemCor BERT vectors, dictionary vectors for unseen senses and temperature 0.02. Dictionary vectors use the existing first-lemma-word plus definition recipe, with batched encoding. A bounded input window retains all pieces of the target.

The candidate posterior uses the original target's vector. It does not read the rewritten sentence or establish what a human reader would infer. Word identity can affect the embedding. These are experimental ranking features, not semantic correctness scores.

For each original sense and candidate sense, the scorer identifies a shared synset, direct hypernym, verb-group/adjective connection or no connection. It appends seven features:

1. Joint original/candidate probability mass on shared senses.
2. Joint mass on direct broader senses.
3. Joint mass on verb-group or adjective connections.
4. Joint mass on unlinked senses.
5. Cosine gap between the best linked candidate sense and the best candidate sense.
6. Joint-mass-weighted cosine change from the original sense to a linked candidate sense.
7. Whether both sense inventories are available.

The four masses sum to one when both distributions exist. Missing evidence returns zeros with availability zero. Broader senses remain separate from exact matches. There are no word-specific bans or hard rejection thresholds.

## Fixed development comparison

The baseline is the saved 52-column feature table from `outputs/detail-preservation-final-20261002`. The candidate pool remains 14,617 rows across all 929 BenchLS cases. Original columns must match exactly. The scorer uses no candidate labels. Three target-word-grouped 10-fold repeats reuse the saved fold assignments, decision threshold 0.45 and meaning floor 0.10. Refitting the frozen baseline must reproduce its saved probabilities within 1e-12.

The adoption bar is predeclared: a mean gain of at least five net edits or a candidate log-loss reduction of at least 0.001. Only a passing experiment creates a new opt-in run. The default and demo stay on v7. A retained run must also pass loader/native feature checks and the existing domain diagnostics before further use.

All BenchLS, including its historical holdout, is reused development data. Listed substitutions do not measure full-sentence detail preservation. No TSAR gold, new human ratings, generated labels, downloads or paid calls enter this work.

The experiment runs on Yoga in a named tmux job with four CPU threads and a 3 GB memory cap:

```sh
.venv/bin/python scripts/experiments/refine_sense_contrast.py --output outputs/sense-contrast-dev-20261003 --save-run runs/ensemble-sense-contrast-20261003
```

Use new paths. The script saves source snapshots, fingerprints, per-case feature records, paired feature tables, fold probabilities and selected edits. It also compares batched scoring with fresh native sense distributions and single-target vectors at three fixed positions.

## Original-slot result and follow-up

The original-slot experiment missed the bar. Net changes across the repeats were -4, zero and +2, with mean -0.67. Mean candidate log loss increased by 0.0000787. No fitted run was retained. Baseline replay and three native feature checks passed. Results are in `outputs/sense-contrast-dev-review-20261003`. The first attempt stopped at an empty-candidate case and remains in `outputs/sense-contrast-dev-20261003`.

The follow-up uses each replacement's own BERT vector in the rewritten sentence. It pools all replacement pieces, including multiword candidates, with the existing complete-slot input builder and encoder. It limits scoring to the five candidates with the highest frozen fit scores. Selection uses no BenchLS labels or decision probabilities. Unscored candidates get zero features and availability zero. The original contextual sense distribution still weights the connection masses.

This bounded variant uses the same seven features and fixed comparison protocol. It introduces no new classifier settings or threshold search. Its gap features compare sense-vector similarities in the replacement's context. It is another development attempt, so a passing result would still need independent evaluation.

```sh
.venv/bin/python scripts/experiments/refine_sense_contrast.py --rewritten --output outputs/rewritten-sense-contrast-dev-20261003 --save-run runs/ensemble-rewritten-sense-contrast-20261003
```

The rewritten-slot variant also missed the bar. Net changes were -3, -6 and +3, with mean -2.0. Mean candidate log loss increased by 0.000531. No fitted run was retained. Its complete scoring and comparison took 248 seconds on Yoga. The saved baseline probabilities reproduced within 1e-12 in both variants.

These two attempts do not support replacing the existing contextual-similarity ranker. The scorer and saved feature records remain available for investigating candidate senses. No learned detail or sense-contrast model is active in the default, demo or six-block run.
