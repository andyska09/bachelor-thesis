#!/bin/bash
# PBS job script for NetTS-eval evaluation.
# Do not submit directly — use submit.py which sets resources per model type.
#
# Required env vars (passed via qsub -v):
#   MODEL       - model name: sarima, prophet, timesfm, ttm, chronos, moirai
#   BENCHMARKS  - space-separated benchmark types, e.g. "drift seasonal" or "all"
# Optional env vars:
#   FREQ        - h/d/m (default: h)
#   LOOKBACK    - context window size
#   HORIZON     - forecast horizon
#   N_RUNS      - number of repeated runs (default: 1)
#   SAVE_PREDICTIONS - if set, save prediction CSVs

#PBS -N ntfbench
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

echo "============================="
echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname -f)"
echo "Model:   $MODEL"
echo "Benchmarks: $BENCHMARKS"
echo "Started: $(date)"
echo "============================="

cd "$PBS_O_WORKDIR"

mkdir -p logs
conda activate "$PBS_O_HOME/.conda/envs/ntfbench"

ARGS="--model $MODEL"
ARGS="$ARGS --benchmark $BENCHMARKS"
ARGS="$ARGS --freq ${FREQ:-h}"
[ -n "${LOOKBACK:-}" ] && ARGS="$ARGS --lookback $LOOKBACK"
[ -n "${HORIZON:-}" ]  && ARGS="$ARGS --horizon $HORIZON"
[ -n "${N_RUNS:-}" ]   && ARGS="$ARGS --n-runs $N_RUNS"
[ -n "${SAVE_PREDICTIONS:-}" ] && ARGS="$ARGS --save-predictions"

echo "Running: python run_evaluation.py $ARGS"
python run_evaluation.py $ARGS # -u for output streaming

echo "============================="
echo "Done: $(date)"
echo "============================="
