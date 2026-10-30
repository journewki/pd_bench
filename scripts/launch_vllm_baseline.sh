#!/usr/bin/env bash
# 베이스라인: 일반 vLLM (P/D 한 인스턴스). GPU 수를 PD 와 맞추기 위해 기본 DP=2.
#   bash scripts/launch_vllm_baseline.sh [추가 vllm 옵션...]
set -euo pipefail
cd "$(dirname "$0")/.."; source configs/env.sh

export VLLM_SERVER_DEV_MODE=1      # /reset_prefix_cache 사용
CUDA_VISIBLE_DEVICES=$BASE_GPUS vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$BASE_PORT" \
  --tensor-parallel-size "$BASE_TP" \
  --data-parallel-size "$BASE_DP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  "$@"
