# pd-bench

vLLM(colocated)을 베이스라인으로 PD(Prefill/Decode disaggregation) 시스템을 같은 조건에서 비교하는 벤치마크 세트입니다.

| 측정 대상 | 도구 | 핵심 지표 |
|---|---|---|
| 속도 | `run_perf.py` | TTFT, TPOT, ITL, E2E, 턴별 TTFT, goodput, 목표 vs 실측 hit rate |
| 정확성 | `run_accuracy.py`, `compare_outputs.py` | GSM8K·NIAH 점수, 베이스라인 대비 출력 일치율, McNemar 검정 |
| **동시 사용자 수용량** | `run_capacity.py`, `compare_capacity.py` | SLO(TTFT/TPOT)를 지키는 최대 동시 multi-turn 사용자 수, tok/s/gpu, tok/s/user |

모든 측정은 클라이언트 기준입니다. 내부 telemetry(KV 전송 시간 등)는 범위에서 제외했습니다.

## 디렉토리 구조

```
pd-bench/
├── configs/
│   ├── env.sh                 모델·GPU·포트·워크로드·SLO 설정 (scripts/*.sh 가 source)
│   └── sweep_example.yaml     속도/정확성 실험 매트릭스 예시 (run_sweep.py)
├── scripts/
│   ├── smoke_test.sh          GPU 없이 전체 파이프라인 점검 (mock 서버)
│   ├── launch_vllm_baseline.sh    베이스라인 vLLM 실행
│   ├── launch_vllm_pd_nixl.sh     PD 참고 구성 (vLLM NixlConnector 1P1D + toy proxy)
│   ├── check_server.sh        서버가 요구사항을 만족하는지 점검
│   ├── run_capacity.sh        수용량 탐색 (base / pd)
│   ├── compare_capacity.sh    베이스라인 vs PD 수용량 비교
│   └── aiperf_capacity.sh     AIPerf 교차 검증 (search / sweep)
├── docs/
│   └── CAPACITY.md            수용량 벤치마크 방법론·시간 예산·해석 가이드
├── common.py                  공통 유틸 (스트리밍 클라이언트, metrics 스크랩, 통계)
├── gen_workload.py            워크로드 생성 (synth / mooncake 트레이스 변환)
├── run_perf.py                속도 측정
├── run_accuracy.py            정확성 측정 (gsm8k / niah × single / multi)
├── compare_outputs.py         출력 일치 비교
├── run_capacity.py            동시 사용자 수용량 탐색
├── compare_capacity.py        수용량 비교 리포트
├── report.py                  속도/정확성 결과 리포트
├── run_sweep.py               속도/정확성 매트릭스 일괄 실행
└── mock_server.py             가짜 OpenAI 호환 서버 (colocated / pd 동작 흉내)
```

python 스크립트는 모두 저장소 루트에서 실행합니다.

## 설치

```bash
pip install -r requirements.txt
pip install aiperf            # 선택: AIPerf 교차 검증
```

## GPU 서버 실행 순서

### 0. 설정

`configs/env.sh` 에서 모델, GPU, 포트, 워크로드, SLO 를 수정합니다. 환경변수로 덮어쓸 수도 있습니다.

```bash
MODEL=Qwen/Qwen3-8B CAP_THINK=15 bash scripts/run_capacity.sh base
```

### 1. 동작 확인 (GPU 불필요, 약 5분)

```bash
bash scripts/smoke_test.sh
```

### 2. 서버 실행

```bash
# 베이스라인 (기본: GPU 0,1 / DP=2 → PD 1P1D 와 GPU 수를 맞춤)
bash scripts/launch_vllm_baseline.sh

# PD: 사내 PD 시스템이면 그것을 띄우고 configs/env.sh 의 PD_URL / PD_METRICS 만 맞추면 됩니다.
# vLLM NixlConnector 참고 구성을 쓰려면 (VLLM_REPO 에 vLLM 소스 경로 필요):
bash scripts/launch_vllm_pd_nixl.sh
```

서버 요구사항:
- `VLLM_SERVER_DEV_MODE=1`: `/reset_prefix_cache` 를 열어 실행마다 캐시를 초기화합니다.
- `--enable-prompt-tokens-details`: 응답 usage 에 `cached_tokens` 가 들어와 요청 단위 hit rate 를 검증할 수 있습니다.
- 베이스라인과 PD 의 `--served-model-name` 이 같아야 합니다.

두 실행 스크립트 모두 이 설정을 이미 넣어 두었습니다. vLLM 버전에 따라 플래그나 proxy 경로가 다를 수 있으니 처음에는 로그를 확인하세요.

### 3. 서버 점검

```bash
bash scripts/check_server.sh base
bash scripts/check_server.sh pd
```

`[FAIL]` 항목이 있으면 벤치마크 전에 고쳐야 합니다. `[WARN]` 은 일부 지표가 빠진다는 뜻입니다.

### 4. 동시 사용자 수용량

```bash
bash scripts/run_capacity.sh base
bash scripts/run_capacity.sh pd
bash scripts/compare_capacity.sh        # → results/capacity/compare/capacity_compare.md
```

방법론, 결과 해석, 시간 예산은 [docs/CAPACITY.md](docs/CAPACITY.md) 를 참고하세요. 요약하면:
- N 명의 사용자가 각자 multi-turn 대화를 진행하는 closed-loop 부하이며, think time 동안에도 사용자는 "동시 사용자"로 존재합니다.
- p95 TTFT, p95 TPOT, 에러율로 레벨을 판정합니다(AIPerf 레시피와 같은 p95).
- 확장 → 이분 탐색을 여러 번 반복하고, 모든 실행을 모은 통과확률 곡선에서 **보수적 수용량(통과확률 ≥ 80%)** 을 보고합니다.
- 기본 설정 기준으로 시스템당 약 3시간 걸립니다. 빠르게 감만 잡으려면 `CAP_WARMUP=45 CAP_DURATION=120 CAP_SEARCH_RUNS=2` 로 줄이세요.

AIPerf 로 같은 질문을 교차 검증할 수 있습니다.

```bash
bash scripts/aiperf_capacity.sh base search   # max-concurrency-under-sla 레시피
bash scripts/aiperf_capacity.sh base sweep    # 고정 concurrency 목록 × 반복 + aiperf plot
```

### 5. 속도 / 정확성 매트릭스

```bash
# configs/sweep_example.yaml 에서 model / tokenizer / targets URL 을 수정한 뒤
python run_sweep.py --config configs/sweep_example.yaml

# 서버를 번갈아 띄워야 하는 경우 (워크로드 파일은 재사용되어 입력이 완전히 동일)
python run_sweep.py --config configs/sweep_example.yaml --targets vllm --stages gen,perf,acc
python run_sweep.py --config configs/sweep_example.yaml --targets pd   --stages perf,acc
python run_sweep.py --config configs/sweep_example.yaml --stages cmp,report
```

결과는 `results/report/report.md` 와 workload 별 `*_latency_vs_qps.png` 로 나옵니다.

## 개별 도구

### 워크로드 생성 (`gen_workload.py`)

| 옵션 | 의미 |
|---|---|
| `--prefix-ratio` | 첫 턴 입력 중 공유 prefix 비율 (0~1) |
| `--num-prefix-groups` | 서로 다른 공유 prefix 개수 |
| `--turns` | 대화 턴 수 (`5` 또는 `3-8`) |
| `--input-len` / `--turn-input-len` | 첫 턴 입력 / 이후 턴마다 추가 입력 |
| `--output-len` | 출력 길이 |
| `--think-time` | 턴 사이 대기 평균 (초) |

생성기는 `<out>.meta.json` 에 이론상 hit rate 두 가지를 기록합니다.
- `target_hit_rate_colocated`: 이전 턴의 prompt 와 output 이 모두 캐시에 있다고 가정
- `target_hit_rate_pd_prefill_only`: P 캐시에는 이전 prompt 만 있다고 가정

multi-turn 에서 두 값의 차이가 PD 의 구조적 불리함을 나타냅니다. `mooncake` 모드는 Mooncake 형식 트레이스를 hash_ids 구조를 보존한 텍스트로 변환합니다.

### 정확성 (`run_accuracy.py`, `compare_outputs.py`)

- **GSM8K (decode-heavy):** lm-eval 과 같은 few-shot 포맷, strict / flexible 채점. 오프라인이면 `--data-path` 로 지정합니다.
- **NIAH (prefill-heavy):** 로컬에서 합성하며, 깊이별 정확도를 따로 집계합니다.
- **모드:** `single` 은 요청 간 prefix 공유를, `multi` 는 이전 응답을 다시 prefill 하는 PD 경로를 검증합니다.
- **출력 비교:** greedy 기준 완전 일치율, 첫 토큰 일치율, 처음 갈라지는 위치, McNemar 검정을 봅니다.

**자연 변동 기준선을 먼저 재세요.** 베이스라인을 두 번 돌려 `compare_outputs.py` 로 비교하면 배치 구성 차이만으로 생기는 불일치 수준을 알 수 있습니다. PD 결과는 이 기준선과 비교해야 버그인지 판단할 수 있습니다.

## 신뢰성 체크리스트

- [ ] `check_server.sh` 통과 (cache reset, cached_tokens, metrics)
- [ ] 클라이언트를 서버와 같은 노드/망에서 실행
- [ ] 베이스라인 GPU 수 = PD 의 P+D 합계 (또는 사용자/GPU 로 비교)
- [ ] 실측 hit rate ≈ 목표 hit rate (속도 실험)
- [ ] 조건당 여러 번 반복 (속도: `perf.repeats`, 수용량: `CAP_SEARCH_RUNS`)
- [ ] 에러 0건 또는 허용치 이내
- [ ] 베이스라인 vs 베이스라인 자연 변동을 먼저 측정 (정확성)
- [ ] 수용량 보고 시 워크로드 가정(think time, 입출력 길이, 턴 수, SLO 백분위)을 함께 기재

## 한계

- TTFT 는 클라이언트 기준 합산값이라 PD 에서 P prefill, KV 전송, D 첫 스텝을 나눠 볼 수 없습니다.
- generated history 는 응답 텍스트를 다시 토크나이즈하므로 hit 가 이론값보다 조금 낮을 수 있습니다.
- Mooncake 변환 텍스트는 합성이라 트레이스 재생은 속도 측정 전용입니다.
- mock 서버는 구조만 흉내 내며 수치에는 의미가 없습니다.
