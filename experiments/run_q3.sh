#!/usr/bin/env bash
# Q3 on every usable 2-layer model (Q1: 8 heads; Q4: 2 and 4 heads) and on 3-layer models
# with 4 and 8 heads per layer (trained to criterion if missing), then aggregate.
#   bash experiments/run_q3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/q3
mkdir -p "$OUT/logs"
run() { python -u experiments/q3_self_repair.py --layers "$1" --heads "$2" --seed "$3" --out "$OUT" > "$OUT/logs/L$1_h$2_seed$3.log" 2>&1; }
for s in 0 14 16 17 27 29; do run 2 2 $s; done   # the 2-head models that met the Q4 criterion
for s in 1 2 3 4; do run 2 4 $s; done            # 4-head seed 0 missed it
for s in 0 1 2 3 4; do run 2 8 $s; done
for h in 4 8; do for s in 0 1 2 3 4; do run 3 $h $s; done; done
python experiments/q3_analyze.py "$OUT"
