#!/usr/bin/env bash
# PD 참고 구성: vLLM NixlConnector 1P1D + vLLM 저장소의 toy proxy.
# 사내 PD 시스템을 쓰는 경우 이 스크립트 대신 그 시스템을 띄우고 configs/env.sh 의 PD_URL / PD_METRICS 만 맞추면 된다.
# 주의: vLLM 버전에 따라 플래그/프록시 경로가 다를 수 있음 → vLLM 문서의 NixlConnector 가이드 확인.
set -euo pipefail
cd "$(dirname "$0")/.."; source configs/env.sh
mkdir -p logs
export VLLM_SERVER_DEV_MODE=1
KV_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
COMMON=(--served-model-name "$SERVED_NAME" --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL" --enable-prefix-caching --enable-prompt-tokens-details
        --kv-transfer-config "$KV_CFG")

CUDA_VISIBLE_DEVICES=$P_GPUS VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  vllm serve "$MODEL" --port "$P_PORT" "${COMMON[@]}" > logs/prefill.log 2>&1 &
echo "prefill pid $!  (logs/prefill.log)"
CUDA_VISIBLE_DEVICES=$D_GPUS VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  vllm serve "$MODEL" --port "$D_PORT" "${COMMON[@]}" > logs/decode.log 2>&1 &
echo "decode pid $!  (logs/decode.log)"

echo "waiting for P/D servers..."
for port in "$P_PORT" "$D_PORT"; do
  until curl -sf "http://localhost:$port/health" > /dev/null; do sleep 5; done
done
python "$VLLM_REPO/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port "$PROXY_PORT" \
  --prefiller-hosts localhost --prefiller-ports "$P_PORT" \
  --decoder-hosts localhost --decoder-ports "$D_PORT" > logs/proxy.log 2>&1 &
echo "proxy pid $!  -> $PD_URL  (logs/proxy.log)"
wait
