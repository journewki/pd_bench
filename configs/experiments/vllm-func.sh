# 기능 확인 (Phase 3-4): vLLM v0.32.0, GPU 0-3 만 사용
export VLLM_REPO=/workspace/vllm-src
export BASE_GPUS=0,1 BASE_TP=1 BASE_DP=2 BASE_NUM_GPUS=2
export P_GPUS=2 D_GPUS=3 PD_NUM_GPUS=2
