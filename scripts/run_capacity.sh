#!/usr/bin/env bash
# 동시 사용자 수용량 탐색 (pd-bench run_capacity.py).
#   bash scripts/run_capacity.sh base [추가 옵션...]
#   bash scripts/run_capacity.sh pd   [추가 옵션...]
#   둘 다 끝나면: bash scripts/compare_capacity.sh
set -euo pipefail
cd "$(dirname "$0")/.."; source configs/env.sh
TARGET=${1:-base}; shift || true
if [ "$TARGET" = "base" ]; then URL=$BASE_URL; METRICS=($BASE_METRICS); NGPU=$BASE_NUM_GPUS
else URL=$PD_URL; METRICS=($PD_METRICS); NGPU=$PD_NUM_GPUS; fi
MARGS=(); for m in "${METRICS[@]}"; do MARGS+=(--metrics-url "$m"); done

python run_capacity.py --url "$URL" --model "$SERVED_NAME" --tokenizer "$TOKENIZER" \
  --num-gpus "$NGPU" "${MARGS[@]}" --reset-cache \
  --system-prompt-len "$CAP_SYSTEM_LEN" --first-input-len "$CAP_FIRST_INPUT" \
  --turn-input-len "$CAP_TURN_INPUT" --output-len "$CAP_OUTPUT" --turns "$CAP_TURNS" \
  --think-time "$CAP_THINK" --max-context "$CAP_MAX_CTX" \
  --slo-ttft-ms "$SLO_TTFT_MS" --slo-tpot-ms "$SLO_TPOT_MS" --slo-percentile "$SLO_PCT" \
  --warmup-s "$CAP_WARMUP" --duration-s "$CAP_DURATION" \
  --search --search-start "$CAP_START" --repeats "$CAP_REPEATS" --search-runs "$CAP_SEARCH_RUNS" \
  --label "$TARGET" --out "$RESULTS/capacity/$TARGET" "$@" 2>&1 | tee -a "$RESULTS/capacity_$TARGET.log"
