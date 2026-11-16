#!/usr/bin/env bash
# One bounded queue batch for the paper run (run from the repo root as the job cwd).
# Freezes the plan on first use, then generates and scores paragraphs
# (ensemble_study run), then scores sentences (paper_run score). Everything
# resumes from per-paragraph checkpoints in $ST, which stays outside Git.
set -euo pipefail
ST=${PAPER_STATE:-$HOME/.local/state/cleartext/paper-run-20261116}
HUB=$HOME/.cache/huggingface/hub
export CLEARTEXT_EVAL_DEPS="$HOME/.local/state/cleartext/status-benchmark-pr-20261101/deps"
export PYTHONPATH="$CLEARTEXT_EVAL_DEPS:models/src:."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${CLEARTEXT_PYTHON:-$HOME/Documents/projects/cleartext/models/.venv/bin/python}
BUDGET=${PAPER_BATCH_SECONDS:-1440}
start=$(date +%s)
if [ ! -f "$ST/run1/plan.json" ]; then
  $PY -m evals.ensemble_study freeze --state "$ST/run1" \
    --t5 "$HOME/.local/state/cleartext/structure-quality-v2-20261027/run/seq2seq" \
    --tagger "$HOME/.local/state/cleartext/compound-coverage-20261109/boundary-tagger.joblib" \
    --encoder "$HUB/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594" \
    --nltk-data "$HOME/nltk_data" \
    --languagetool "$HOME/.local/state/cleartext/grammar-20261027/LanguageTool-6.6" \
    --java "$HOME/.local/share/meramu-android-toolchain/jdk-17.0.20.1+1/bin/java" \
    --embedding "$HUB/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41" \
    --nli "$HUB/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb" \
    --panel "paper=$ST/paragraphs.json"
fi
# The winner and T5 unchecked on every paragraph (finished work is skipped),
# then sentence scores: those two everywhere, v7 words wherever its output
# exists. v7 generation (design-paper.json) was stopped after 18 paragraphs:
# about 30 min per paragraph on a loaded Yoga; see README.
run_design() {
  left=$(( BUDGET - ($(date +%s) - start) ))
  [ "$left" -gt 60 ] || exit 0
  /usr/bin/time -v $PY -m evals.ensemble_study run --plan "$ST/run1/plan.json" --state "$ST/run1" \
    --design "$1" --panel paper --max-seconds "$left" 2>&1 | tee -a "$ST/run.log"
}
run_design "$ST/design-paper-fast.json"
left=$(( BUDGET - ($(date +%s) - start) ))
/usr/bin/time -v $PY -m evals.paper_run score --plan "$ST/run1/plan.json" --state "$ST/run1" \
  --configs winner,t5_unchecked --optional v7_words --max-seconds "$left" 2>&1 | tee -a "$ST/run.log"
