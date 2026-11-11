#!/usr/bin/env bash
# Q4 for 2, 4 and 8 heads per layer, seeds 0..N-1 (8-head models reuse Q1 checkpoints;
# others are trained to criterion if missing), then extra 2-head seeds, then aggregate.
#   bash experiments/run_q4.sh [n_seeds]
#
# Only 1 of the first 5 two-head models learned induction within 10,000 steps, so seeds
# 5-34 are added for 2 heads/layer, capped at 1,500 steps: in the main grid every model that
# met the criterion did so at the first check (1,500 steps), and none that missed it then
# recovered by 10,000. Models that miss the criterion are excluded by q4_analyze.py.
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
for s in $(seq 5 34); do
  python -u experiments/q4_redundancy.py --heads 2 --seed "$s" --max-steps 1500 --out "$OUT" --threads 4 > "$OUT/logs/h2_seed$s.log" 2>&1
done
python experiments/q4_analyze.py "$OUT"
