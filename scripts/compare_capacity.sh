#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source configs/env.sh
python compare_capacity.py "$RESULTS/capacity/base" "$RESULTS/capacity/pd" --out "$RESULTS/capacity/compare"
