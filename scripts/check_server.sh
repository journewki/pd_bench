#!/usr/bin/env bash
# 벤치마크 전에 서버가 pd-bench 요구사항을 만족하는지 확인.
#   bash scripts/check_server.sh base
#   bash scripts/check_server.sh pd
#   bash scripts/check_server.sh <요청 URL> [metrics URL ...]
set -uo pipefail
cd "$(dirname "$0")/.."; source configs/env.sh
case "${1:-base}" in
  base) URL=$BASE_URL; METRICS=($BASE_METRICS) ;;
  pd)   URL=$PD_URL;   METRICS=($PD_METRICS) ;;
  *)    URL=$1; shift; METRICS=("$@"); [ ${#METRICS[@]} -eq 0 ] && METRICS=("$URL") ;;
esac
ok(){ echo "  [OK]   $*"; }; ng(){ echo "  [FAIL] $*"; }; wn(){ echo "  [WARN] $*"; }

echo "== 요청 엔드포인트 $URL"
curl -sf "$URL/v1/models" > /dev/null && ok "/v1/models" || ng "/v1/models 응답 없음"
RESP=$(curl -sN "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED_NAME\", \"messages\": [{\"role\":\"user\",\"content\":\"hello\"}],
  \"max_tokens\": 8, \"stream\": true, \"stream_options\": {\"include_usage\": true}, \"ignore_eos\": true}")
echo "$RESP" | grep -q '"content"' && ok "streaming chat 응답" || ng "streaming 응답 이상: $(echo "$RESP" | head -c 300)"
echo "$RESP" | grep -q '"usage"' && ok "usage 반환" || ng "usage 없음 → 출력 토큰 수를 청크 수로 근사하게 됨"
echo "$RESP" | grep -q 'cached_tokens' && ok "cached_tokens 반환" || wn "cached_tokens 없음 → --enable-prompt-tokens-details 권장"

for M in "${METRICS[@]}"; do
  echo "== metrics $M"
  MT=$(curl -sf "$M/metrics")
  [ -n "$MT" ] || { ng "/metrics 응답 없음"; continue; }
  echo "$MT" | grep -q 'prefix_cache_queries' && ok "prefix cache 카운터" || ng "prefix_cache_queries 없음"
  echo "$MT" | grep -qE 'kv_cache_usage_perc|gpu_cache_usage_perc' && ok "KV 사용률 gauge" || wn "KV 사용률 gauge 없음"
  echo "$MT" | grep -q 'num_requests_waiting' && ok "waiting gauge" || wn "waiting gauge 없음"
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$M/reset_prefix_cache")
  [ "$CODE" = "200" ] && ok "/reset_prefix_cache" || ng "/reset_prefix_cache HTTP $CODE → VLLM_SERVER_DEV_MODE=1 로 재시작 필요"
done
