#!/usr/bin/env bash
# Q4 for 2, 4 and 8 heads per layer, seeds 0..N-1 (8-head models reuse Q1 checkpoints;
# others are trained if missing), then aggregate.
#   bash experiments/run_q4.sh [n_seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-5}
OUT=results/q4
mkdir -p "$OUT/logs"
for h in 2 4 8; do
  for s in $(seq 0 $((N - 1))); do
    python -u experiments/q4_redundancy.py --heads "$h" --seed "$s" --out "$OUT" --threads 4 > "$OUT/logs/h${h}_seed$s.log" 2>&1
  done
done
python experiments/q4_analyze.py "$OUT"
