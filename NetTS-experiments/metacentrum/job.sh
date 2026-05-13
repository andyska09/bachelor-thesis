#!/bin/bash
#PBS -N scoring
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname -f)"
echo "Started: $(date)"

cd "$PBS_O_WORKDIR"
conda activate "$PBS_O_HOME/.conda/envs/netmts-env"

ARGS="--aggregation ${AGGREGATION:-hourly}"
ARGS="$ARGS --benchmark ${BENCHMARK:-SEASON}"
ARGS="$ARGS --source ${SOURCE:-ips_full}"
[ -n "${DATA_ROOT:-}" ] && ARGS="$ARGS --data-root $DATA_ROOT"
[ -n "${SHARD:-}" ] && ARGS="$ARGS --shard $SHARD --num-shards $NUM_SHARDS"
[ -n "${WORKERS:-}" ] && ARGS="$ARGS --workers $WORKERS"

echo "Running: python -m scoring.runner $ARGS"
python -m scoring.runner $ARGS

echo "Done: $(date)"
