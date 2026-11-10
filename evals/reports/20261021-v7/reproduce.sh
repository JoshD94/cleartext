#!/usr/bin/env bash
# Run from the repository root using an environment with evals/requirements.txt,
# sentence-transformers, transformers and CPU torch installed.
# The demo endpoint must have the saved v7 artifacts already loaded.
set -euo pipefail
EVAL_PYTHON=${EVAL_PYTHON:-python3}
EVAL_MODEL_URL=${EVAL_MODEL_URL:-http://100.83.223.83:8770}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
"$EVAL_PYTHON" evals/datasets.py --fetch
"$EVAL_PYTHON" evals/run_evals.py \
  --systems original,rule_based,full_pipeline \
  --embedding-model all-MiniLM-L6-v2 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --model-url "$EVAL_MODEL_URL"
"$EVAL_PYTHON" evals/detection_baseline.py
