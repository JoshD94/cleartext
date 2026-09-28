#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p .tools
curl -fLsS --retry 2 https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz -o .tools/uv.tar.gz
tar -xzf .tools/uv.tar.gz -C .tools
.tools/uv-x86_64-unknown-linux-gnu/uv venv --python 3.12 .venv
.tools/uv-x86_64-unknown-linux-gnu/uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn nltk wordfreq spacy lemminflect pytest playwright
.tools/uv-x86_64-unknown-linux-gnu/uv pip install --python .venv/bin/python https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
.tools/uv-x86_64-unknown-linux-gnu/uv pip install --python .venv/bin/python https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
.venv/bin/python -m nltk.downloader wordnet omw-1.4 cmudict brown stopwords semcor
.tools/uv-x86_64-unknown-linux-gnu/uv pip freeze --python .venv/bin/python > requirements-lock.txt
printf 'SETUP_COMPLETE\n'
