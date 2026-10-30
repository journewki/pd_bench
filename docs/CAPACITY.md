# 동시 사용자 수용량(capacity) 벤치마크

## 질문

> multi-turn chat 사용자(턴당 수천 토큰 입력 / 수백 토큰 출력)를 한 서버에서
> TTFT ≤ 5s, TPOT ≤ 60–80ms 를 지키며 **최대 몇 명까지 동시에** 받을 수 있는가?
> 실험마다 값이 흔들리므로, 여러 번 돌려서 대략적인 범위를 파악한다.

## 먼저 합의해야 할 정의

수용량 숫자는 아래 가정에 따라 크게 달라집니다. 결과를 보고할 때 반드시 함께 적어야 합니다.

| 항목 | 기본값 | 영향 |
|---|---|---|
| **think time** (턴 사이 사용자 대기) | 10초 (지수분포) | 가장 큰 영향. 사용자가 오래 생각할수록 같은 서버가 더 많은 사용자를 받음 |
| 첫 턴 입력 / 이후 턴 추가 입력 | 2000 / 200 토큰 | 컨텍스트가 턴마다 커지므로 KV 메모리와 prefill 비용을 결정 |
| 턴당 출력 | 300 토큰 (`ignore_eos` 로 고정) | decode 부하 |
| 대화당 턴 수 | 4–8 | 평균 컨텍스트 길이 |
| SLO 판정 백분위 | **p95** | AIPerf `max-concurrency-under-sla` 레시피와 동일. p90 으로 바꾸면 수용량이 커짐 |
| TPOT SLO | 70ms | 60 / 80 으로 바꿔 민감도를 같이 보는 것을 권장 |
| 에러율 상한 | 1% | |

**동시 사용자 ≠ 동시 요청.** 사용자는 대부분의 시간을 think time 으로 보내므로, 실제로 서버에서 처리 중인 요청 수는 사용자 수보다 훨씬 적습니다. 결과의 `활성 요청`, `active_fraction` 열이 이 비율을 보여줍니다. 대략 다음 관계가 성립합니다.

```
활성 요청 수 ≈ 사용자 수 × 응답시간 / (응답시간 + think time)
```

반면 think time 동안에도 사용자의 대화 KV 는 prefix cache 에 남아 있어야 다음 턴이 빠릅니다. 사용자가 많아지면 KV 메모리가 부족해져 eviction 이 일어나고, 다음 턴에서 history 전체를 다시 prefill 하게 됩니다. 그래서 multi-turn 수용량은 보통 **compute(TPOT)** 와 **KV 메모리(hit rate 하락 → TTFT 증가)** 중 먼저 닿는 쪽에서 결정됩니다. 결과의 `KV 최대`, `hit`, `대기 최대` 열로 어느 쪽이 병목인지 판단할 수 있습니다.

## 부하 모델 (`run_capacity.py`)

- **closed-loop:** N 명의 사용자가 각자 "턴 전송 → 응답 수신 → think time → 다음 턴"을 반복합니다. 대화가 끝나면 새 대화를 시작하므로 사용자 수는 항상 N 으로 유지됩니다.
- **실제 응답 누적:** 이전 턴의 실제 응답이 history 에 붙어 컨텍스트가 턴마다 커집니다.
- **정상상태에서 출발:** 사용자는 대화 중간 턴에서 출발합니다(앞선 턴은 합성 history). 시작부터 짧은 대화와 긴 대화가 섞인 상태를 만들기 위함입니다. 사용자 시작 시각은 warmup 앞 절반에 고르게 퍼뜨립니다.
- **측정 구간:** warmup 이후 `duration` 동안 '시작된' 요청만 집계합니다. 구간이 끝난 뒤 완료되지 못한 요청은 에러로 셉니다.

## 판정과 탐색

**레벨 판정:** 한 번의 실행(사용자 수 N)은 다음을 모두 만족하면 통과입니다.
- p95(TTFT) ≤ SLO
- p95(TPOT) ≤ SLO
- 에러율 ≤ 1%
- 집계 요청 수 ≥ 30

**탐색 절차:**
1. **확장:** 시작값에서 ×2 씩 늘리며 처음 실패하는 지점을 찾습니다. 이 단계는 반복 1회입니다.
2. **이분 탐색:** 마지막 통과 지점과 첫 실패 지점 사이를 좁힙니다. 이 단계는 레벨마다 2회 반복하고, 모두 통과해야 통과로 봅니다.
3. **반복:** 위 탐색 전체를 `--search-runs` 번 반복합니다.

**최종 수용량은 두 가지 방식으로 보고합니다.**
- **통과확률 곡선 (권장):** 모든 실행의 pass/fail 을 사용자 수별로 모아 "사용자가 많을수록 통과확률은 줄어든다"는 단조성을 가정해 보정합니다(isotonic regression). 보정 통과확률이 80% 이상인 최대 사용자 수가 **보수적 수용량**, 50% 가 되는 지점이 **중앙 수용량**입니다.
- **이분 탐색 결과의 분포:** 탐색마다 나온 수용량의 중앙값과 최소–최대 범위입니다.

통과확률 곡선을 권장하는 이유가 있습니다. 경계 근처에서는 같은 N 이 어떤 실행에서는 통과하고 어떤 실행에서는 실패합니다. 이분 탐색은 그런 한 번의 우연한 실패에 휘둘리지만, 곡선은 모든 실행 결과를 함께 반영합니다.

## 실행

```bash
# 서버 확인 (cached_tokens, /metrics, /reset_prefix_cache)
bash scripts/check_server.sh base
bash scripts/check_server.sh pd

# 수용량 탐색 (configs/env.sh 의 워크로드/SLO 사용)
bash scripts/run_capacity.sh base
bash scripts/run_capacity.sh pd
bash scripts/compare_capacity.sh          # → results/capacity/compare/capacity_compare.md

# 레벨을 직접 지정 (탐색 대신): 곡선을 고르게 그리고 싶을 때
bash scripts/run_capacity.sh base --users 32,64,96,128,160,192 --search-runs 2

# 리포트만 다시 생성
python run_capacity.py --report-only --out results/capacity/base
```

### 시간 예산

레벨 1회 ≈ warmup 60s + 측정 180s + drain + cooldown 10s ≈ **4–5분**입니다.
- 탐색 1회는 보통 확장 4–5레벨 + 이분 탐색 3–4레벨 × 2회 ≈ 12레벨, 즉 약 **1시간**입니다.
- `CAP_SEARCH_RUNS=3` 이면 시스템당 약 3시간, 베이스라인과 PD 를 합치면 약 6시간입니다.
- **빠르게 감만 잡으려면:** `CAP_WARMUP=45 CAP_DURATION=120 CAP_SEARCH_RUNS=2`
- **정식 보고용:** 기본값 또는 `CAP_DURATION=300`

## 결과 해석

`capacity_report.md` 의 열은 다음과 같습니다.

| 열 | 의미 |
|---|---|
| 통과율 / 보정 통과확률 | 해당 사용자 수에서 SLO 를 만족한 실행 비율 / 단조 보정값 |
| TTFT·TPOT p95 (중앙/최대) | 반복 실행들의 중앙값과 최댓값 |
| SLO 달성률 | TTFT·TPOT 를 동시에 만족한 요청 비율 (DistServe 의 attainment) |
| tok/s/gpu | 출력 토큰 처리량 ÷ GPU 수 (Dynamo 문서의 throughput per GPU) |
| tok/s/user | 요청별 1/TPOT 의 중앙값 (Dynamo 문서의 interactivity) |
| 활성 요청 | 실제로 서버에서 처리 중이던 요청 수 평균 |
| KV 최대 / 대기 최대 | vLLM `kv_cache_usage_perc` 최댓값 / `num_requests_waiting` 최댓값 |
| hit | 응답 usage 의 cached_tokens 로 계산한 prefix hit rate |

**병목 판단:**
- **TPOT 먼저 실패, KV 여유:** decode compute 병목입니다. 이 경우 PD 의 이점(prefill 간섭 제거)이 가장 크게 나타날 수 있습니다.
- **TTFT 먼저 실패, KV 가 1.0 근처, hit 하락:** KV 메모리 병목입니다. 사용자들의 history 가 캐시에서 밀려나 재-prefill 되고 있다는 뜻입니다.
- **TTFT 먼저 실패, 대기열 증가:** prefill 처리량 또는 스케줄러 병목입니다.

**그래프:**
- `capacity_latency.png`: 사용자 수 대비 TTFT·TPOT p95 와 SLO 선
- `capacity_pareto.png`: tok/s/user 대비 tok/s/gpu (Dynamo 문서의 Pareto 그림과 같은 축)

## AIPerf 로 교차 검증

AIPerf(v0.13 확인)의 multi-turn 모드에서는 think time(`--session-turn-delay-mean`) 동안에도 세션이 concurrency 슬롯을 차지하고, 실제 응답이 history 에 누적됩니다. 그래서 `--concurrency` 를 동시 사용자 수로 해석할 수 있습니다.

```bash
bash scripts/aiperf_capacity.sh base search   # max-concurrency-under-sla 레시피 (p95 판정)
bash scripts/aiperf_capacity.sh base sweep    # 고정 concurrency 목록 × 반복 + aiperf plot
```

**두 도구의 차이:**
- 워크로드 형태: AIPerf 는 사용자별 고유 context(`--user-context-prompt-length`) + 매 턴 같은 길이 입력으로 첫 턴 2000 / 이후 200 을 근사합니다. 턴 수와 think time 분포도 정규분포라 pd-bench 와 약간 다릅니다.
- 반복: AIPerf 의 반복(`--num-profile-runs`)은 같은 seed 로 같은 입력을 다시 돌려 runtime 변동을 측정합니다. pd-bench 의 `--search-runs` 는 seed 를 바꿔 입력 변동까지 포함합니다.

두 결과가 같은 범위에 들어오면 측정을 신뢰할 수 있다는 근거가 됩니다. 크게 다르면 워크로드 가정 차이부터 확인합니다.

## 주의사항

- **클라이언트 위치:** 클라이언트는 서버와 같은 노드나 같은 망에서 실행해야 합니다. Dynamo 문서도 고부하 측정에서는 port-forwarding 같은 클라이언트 측 오버헤드를 피하라고 권장합니다. 결과에 `client event loop 지연` 경고가 나오면 클라이언트가 병목입니다.
- **사용자 수가 많을 때의 warmup:** 사용자가 수백 명이면 시작 직후 cold prefill 이 몰립니다. `KV 최대`나 `대기 최대`가 측정 구간 초반에 튀면 `CAP_WARMUP` 을 늘리세요.
- **공정한 GPU 비교:** 베이스라인의 GPU 수를 PD 의 P+D 합과 맞추세요(`BASE_DP`). 그래도 절대 수용량보다 **사용자/GPU** 로 비교하는 것이 안전합니다.
- **민감도 분석:** 가정 하나를 바꾸면 수용량이 크게 달라질 수 있습니다. think time 은 5 / 10 / 30초, TPOT SLO 는 60 / 80ms 정도로 바꿔 보며 결론이 유지되는지 확인하세요.
