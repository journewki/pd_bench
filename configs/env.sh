# pd-bench 공통 설정. 모든 scripts/*.sh 가 이 파일을 source 한다.
# 서버 환경에 맞게 수정해서 사용하세요. (환경변수로 덮어쓰기 가능: MODEL=... bash scripts/xxx.sh)

# ---- 모델 ----
export MODEL=${MODEL:-Qwen/Qwen3-8B}            # vllm serve 에 넘길 모델 (HF id 또는 로컬 경로)
export SERVED_NAME=${SERVED_NAME:-$MODEL}       # API 의 "model" 값 (베이스라인과 PD 가 같아야 함)
export TOKENIZER=${TOKENIZER:-$MODEL}           # 길이 제어/토큰 비교용 토크나이저
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}

# ---- 베이스라인 (일반 vLLM) ----
export BASE_GPUS=${BASE_GPUS:-0,1}               # 사용할 GPU
export BASE_TP=${BASE_TP:-1}
export BASE_DP=${BASE_DP:-2}                     # GPU 수 = TP x DP (PD 의 P+D 합과 맞추는 것을 권장)
export BASE_PORT=${BASE_PORT:-8000}
export BASE_URL=${BASE_URL:-http://localhost:$BASE_PORT}
export BASE_METRICS=${BASE_METRICS:-$BASE_URL}   # 공백 구분 여러 개 가능
export BASE_NUM_GPUS=${BASE_NUM_GPUS:-2}

# ---- PD (vLLM NixlConnector 참고 구성. 사내 PD 시스템이면 PD_URL / PD_METRICS 만 바꿔 쓰면 됨) ----
export P_GPUS=${P_GPUS:-0}
export D_GPUS=${D_GPUS:-1}
export P_PORT=${P_PORT:-8100}
export D_PORT=${D_PORT:-8200}
export PROXY_PORT=${PROXY_PORT:-9000}
export PD_URL=${PD_URL:-http://localhost:$PROXY_PORT}
export PD_METRICS=${PD_METRICS:-"http://localhost:$P_PORT http://localhost:$D_PORT"}   # 공백 구분
export PD_NUM_GPUS=${PD_NUM_GPUS:-2}
export VLLM_REPO=${VLLM_REPO:-$HOME/vllm}       # toy_proxy_server.py 가 있는 vLLM 소스 경로

# ---- capacity 워크로드 (사용자 1명의 대화) ----
export CAP_SYSTEM_LEN=${CAP_SYSTEM_LEN:-0}       # 모든 사용자 공통 system prompt
export CAP_FIRST_INPUT=${CAP_FIRST_INPUT:-2000}  # 첫 턴 입력
export CAP_TURN_INPUT=${CAP_TURN_INPUT:-200}     # 이후 턴마다 추가 입력
export CAP_OUTPUT=${CAP_OUTPUT:-300}             # 턴당 출력
export CAP_TURNS=${CAP_TURNS:-4-8}               # 대화당 턴 수
export CAP_THINK=${CAP_THINK:-10}                # 턴 사이 사용자 대기(초)
export CAP_MAX_CTX=${CAP_MAX_CTX:-30000}         # 넘으면 대화 종료 후 새 대화

# ---- SLO ----
export SLO_TTFT_MS=${SLO_TTFT_MS:-5000}
export SLO_TPOT_MS=${SLO_TPOT_MS:-70}
export SLO_PCT=${SLO_PCT:-95}              # AIPerf max-concurrency-under-sla 레시피도 p95 로 판정

# ---- 측정 시간 / 반복 ----
export CAP_WARMUP=${CAP_WARMUP:-60}
export CAP_DURATION=${CAP_DURATION:-180}
export CAP_START=${CAP_START:-16}
export CAP_REPEATS=${CAP_REPEATS:-2}
export CAP_SEARCH_RUNS=${CAP_SEARCH_RUNS:-3}

export RESULTS=${RESULTS:-results}
