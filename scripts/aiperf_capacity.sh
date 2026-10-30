#!/usr/bin/env bash
# AIPerf 로 같은 질문(최대 동시 세션 수 under SLO)을 교차 검증.
# AIPerf multi-turn 에서는 think time(turn delay) 동안에도 세션이 concurrency 슬롯을 차지하고,
# 실제 응답이 history 에 누적된다 → --concurrency ≈ 동시 사용자 수.
#   bash scripts/aiperf_capacity.sh base|pd search   # max-concurrency-under-sla 레시피 (자동 탐색)
#   bash scripts/aiperf_capacity.sh base|pd sweep    # 고정 concurrency 목록 × 반복 → aiperf plot
# AIPerf v0.13.0 에서 플래그 확인함. 다른 버전이면 `aiperf profile --help` 로 재확인.
set -euo pipefail
cd "$(dirname "$0")/.."; source configs/env.sh
TARGET=${1:-base}; MODE=${2:-search}
if [ "$TARGET" = "base" ]; then URL=$BASE_URL; METRICS=($BASE_METRICS); else URL=$PD_URL; METRICS=($PD_METRICS); fi
SM=(); for m in "${METRICS[@]}"; do SM+=(--server-metrics "$m/metrics"); done
OUT=$RESULTS/aiperf/$TARGET-$MODE
TURNS_MEAN=${AIPERF_TURNS_MEAN:-6}
CONTEXT=$(( CAP_FIRST_INPUT - CAP_TURN_INPUT ))   # 첫 턴 = 사용자별 고유 context + 턴 입력

COMMON=(--model "$SERVED_NAME" --url "$URL" --endpoint-type chat --streaming --tokenizer "$TOKENIZER"
  --num-conversations 4000 --num-dataset-entries 4000
  --session-turns-mean "$TURNS_MEAN" --session-turns-stddev 1
  --session-turn-delay-mean $(( CAP_THINK * 1000 )) --session-turn-delay-stddev $(( CAP_THINK * 300 ))
  --user-context-prompt-length "$CONTEXT"
  --synthetic-input-tokens-mean "$CAP_TURN_INPUT" --osl "$CAP_OUTPUT"
  --extra-inputs ignore_eos:true --use-server-token-count
  --warmup-duration "$CAP_WARMUP" --benchmark-duration "$CAP_DURATION"
  --goodput "time_to_first_token:$SLO_TTFT_MS inter_token_latency:$SLO_TPOT_MS"
  "${SM[@]}" --ui simple --artifact-dir "$OUT")

if [ "$MODE" = "search" ]; then
  aiperf profile "${COMMON[@]}" \
    --search-recipe max-concurrency-under-sla --search-style monotonic \
    --ttft-sla-ms "$SLO_TTFT_MS" --tpot-sla-ms "$SLO_TPOT_MS" --error-rate-sla 0.01 \
    --num-profile-runs "${AIPERF_RUNS:-2}"
else
  aiperf profile "${COMMON[@]}" \
    --concurrency "${AIPERF_LEVELS:-16,32,64,96,128,192,256}" \
    --num-profile-runs "${AIPERF_RUNS:-3}" --parameter-sweep-cooldown-seconds 10
  aiperf plot "$OUT" || true
fi
echo "-> $OUT"
