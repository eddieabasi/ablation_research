#!/usr/bin/env bash
# Run Q2 for seeds 0..N-1 (reusing Q1 checkpoints), one at a time, then aggregate.
#   bash experiments/run_q2.sh [n_seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-5}
OUT=results/q2
mkdir -p "$OUT/logs"
for s in $(seq 0 $((N - 1))); do
  python -u experiments/q2_mean_shift.py --seed "$s" --out "$OUT" --threads 4 > "$OUT/logs/seed$s.log" 2>&1
done
python experiments/q2_analyze.py "$OUT"
