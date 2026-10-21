#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
job_name="$1"
shift
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
"$@" > "logs/${job_name}.log" 2>&1
job_exit_code=$?
printf '%s\n' "$job_exit_code" > "logs/${job_name}.exit"
exit "$job_exit_code"
