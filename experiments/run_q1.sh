#!/usr/bin/env bash
# Run Q1 for several seeds in parallel (one CPU thread each), then aggregate.
#   bash experiments/run_q1.sh [n_seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-5}
OUT=results/q1
mkdir -p "$OUT/logs"
for s in $(seq 0 $((N - 1))); do
  python -u experiments/q1_method_agreement.py --seed "$s" --out "$OUT" > "$OUT/logs/seed$s.log" 2>&1 &
done
wait
python experiments/q1_analyze.py "$OUT"
