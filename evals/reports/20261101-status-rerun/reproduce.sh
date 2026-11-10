#!/usr/bin/env bash
# Run from a named, resource-capped job after the host/resource checks.
# All outputs and historical source archives stay outside the checkout.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:-six}
python_bin=${CLEARTEXT_PYTHON:-models/.venv/bin/python}
cleartext_work=${CLEARTEXT_REPLAY_OUT:?Set CLEARTEXT_REPLAY_OUT to a fresh directory outside Git}
structure_run=${CLEARTEXT_STRUCTURE_RUN:-$HOME/.local/state/cleartext/structure-quality-v2-20261027/run/seq2seq}
embedding=${CLEARTEXT_EMBEDDING_MODEL:-$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41}
nli=${CLEARTEXT_NLI_MODEL:-$HOME/.cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb}
export PYTHONPATH="${CLEARTEXT_EVAL_DEPS:+$CLEARTEXT_EVAL_DEPS:}models/src:."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$python_bin" - "$cleartext_work" <<'PYGUARD'
from pathlib import Path
import sys
out, repo = Path(sys.argv[1]).resolve(), Path.cwd().resolve()
if out == repo or repo in out.parents:
    raise SystemExit("CLEARTEXT_REPLAY_OUT must be outside the checkout; raw predictions stay local.")
PYGUARD
mkdir -p "$cleartext_work"
if [[ "$mode" == detector ]]; then
  "$python_bin" evals/reports/20261101-status-rerun/rerun_detector.py \
    --out "$cleartext_work/detection_baseline.json"
  exit
fi
for path in "$structure_run/model.safetensors" "$embedding/model.safetensors" "$nli/model.safetensors"; do
  if [[ ! -f "$path" ]]; then printf 'Required local artifact missing: %s\n' "$path" >&2; exit 1; fi
done
if [[ "$mode" == six ]]; then
  mkdir -p "$cleartext_work/before-source"
  git archive 0a17b49 models/src | tar -x -C "$cleartext_work/before-source"
  "$python_bin" - "$cleartext_work" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
source = pathlib.Path('evals/data/sample_passages.json')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '79e55b1118e1c333ca4984b0faa12a22f893de17c6fd5a4fa68871d284f9a70b'
rows = json.loads(source.read_text())
for row in rows:
    row.update(panel='status_report_authored_smoke', references=[])
(out / 'passages.json').write_text(json.dumps(rows, indent=2) + '\n')
PY
  "$python_bin" evals/status_report_benchmark.py --passages "$cleartext_work/passages.json" \
    --out "$cleartext_work/before" --model-src "$cleartext_work/before-source/models/src" \
    --lexical-run models/runs/ensemble-v7-20260928 --systems original,rule_based,lexical_only
  "$python_bin" evals/status_report_benchmark.py --passages "$cleartext_work/passages.json" \
    --out "$cleartext_work/after-v7" --lexical-run models/runs/ensemble-v7-20260928 \
    --structure-run "$structure_run" --systems original,rule_based,lexical_only,full_pipeline,full_rules
  "$python_bin" evals/status_report_benchmark.py --passages "$cleartext_work/passages.json" \
    --out "$cleartext_work/after-context" --lexical-run models/runs/ensemble-context-similarity-20261002 \
    --structure-run "$structure_run" --systems full_pipeline,full_precision
elif [[ "$mode" == matrix ]]; then
  : "${CLEARTEXT_PASSAGES:?Set CLEARTEXT_PASSAGES to the frozen 190-input file}"
  "$python_bin" - "$cleartext_work" "$CLEARTEXT_PASSAGES" "${CLEARTEXT_WEEK7_PASSAGES:-}" <<'PY'
import hashlib, json, pathlib, sys
out, source = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
assert hashlib.sha256(source.read_bytes()).hexdigest() == '2d39da8438317166a24e69d6a822e29039c5ae3031b8a23da95b912244ae2d4f'
rows = json.loads(source.read_text())
if sys.argv[3]:
    week7 = json.loads(pathlib.Path(sys.argv[3]).read_text())
    assert len(week7) == 18
    canonical = [{k: row[k] for k in ['id', 'domain', 'original']} for row in week7]
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == '0219f34211a1b097616ae0ba0bfd7537d453c2d2712279c6457f05fdfe323fd1'
    existing = {r['original'] for r in rows}
    extra = [dict(r, panel='week7_authored_smoke', references=[]) for r in week7 if r['original'] not in existing]
    assert len(extra) == 12
    rows += extra
(out / 'passages.json').write_text(json.dumps(rows, indent=2) + '\n')
PY
  "$python_bin" evals/pipeline_benchmark.py generate --passages "$cleartext_work/passages.json" \
    --out "$cleartext_work/after-v7" --lexical-run models/runs/ensemble-v7-20260928 \
    --structure-run "$structure_run" \
    --systems original,rule_based,lexical_only,structural_only,lexical_structural,full_no_guardrails,full_pipeline,full_rules
  "$python_bin" evals/pipeline_benchmark.py generate --passages "$cleartext_work/passages.json" \
    --out "$cleartext_work/after-context" --lexical-run models/runs/ensemble-context-similarity-20261002 \
    --structure-run "$structure_run" --systems full_pipeline,full_precision --threshold .80
else
  printf 'Usage: %s [six|matrix|detector]\n' "$0" >&2
  exit 2
fi
"$python_bin" - "$cleartext_work" "$mode" <<'PY'
import json, pathlib, sys
out, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
parts = [('before', 'before_')] if mode == 'six' else []
parts += [('after-v7', 'after_v7_'), ('after-context', 'after_context_')]
rows = []
for directory, prefix in parts:
    for row in map(json.loads, (out / directory / 'predictions.jsonl').read_text().splitlines()):
        row['system'] = prefix + row['system']
        rows.append(row)
(out / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
(out / 'systems.txt').write_text(','.join(dict.fromkeys(r['system'] for r in rows)))
PY
"$python_bin" evals/pipeline_benchmark.py score --passages "$cleartext_work/passages.json" \
  --predictions "$cleartext_work/predictions.jsonl" --systems "$(cat "$cleartext_work/systems.txt")" \
  --out "$cleartext_work/scoring" --embedding-model "$embedding" --nli-model "$nli"
