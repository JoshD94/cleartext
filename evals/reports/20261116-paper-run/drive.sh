#!/usr/bin/env bash
# Queue bounded batches one at a time (job wait inside a worker script) until
# sentence scoring reports complete. Usage: drive.sh OWNER_THREAD_ID [RAM_GB]
set -uo pipefail
ST=${PAPER_STATE:-$HOME/.local/state/cleartext/paper-run-20261116}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
owner=$1 ram=${2:-4.5}
for attempt in $(seq 1 30); do
  # Resume: wait on a batch that is already queued or running (one job at a time).
  id=$(~/.local/bin/job status 2>/dev/null | awk '$NF=="cleartext-paper-run" && ($2=="running" || $2=="queued") {print $1; exit}')
  [ -n "$id" ] || id=$(~/.local/bin/job submit --kind script --name cleartext-paper-run --ram "$ram" --minutes 30 --needs yoga \
    --owner "$owner" --priority later --restartable --notify none --cwd "$repo" \
    --cmd "$repo/evals/reports/20261116-paper-run/batch.sh" 2>&1 | tail -1)
  echo "$(date -Is) attempt $attempt job $id" >> "$ST/drive.log"
  result=$(~/.local/bin/job wait "$id" 2>&1); echo "$result" >> "$ST/drive.log"
  if echo "$result" | grep -q " failed\| cancelled"; then failures=$(( ${failures:-0} + 1 )); else failures=0; fi
  if [ "$failures" -ge 2 ]; then echo "$(date -Is) STOPPED after two failed batches" >> "$ST/drive.log"; exit 1; fi
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$ST/run1/sentences/receipt-t5_unchecked+v7_words+winner.json'))['complete'] else 1)" 2>/dev/null; then
    echo "$(date -Is) COMPLETE" >> "$ST/drive.log"; exit 0
  fi
done
echo "$(date -Is) GAVE UP" >> "$ST/drive.log"; exit 1
