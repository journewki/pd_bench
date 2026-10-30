#!/usr/bin/env bash
# GPU 없이 전체 파이프라인 점검 (mock 서버 2개: colocated / pd). 약 5–8분.
#   bash scripts/smoke_test.sh
# 실서버에 붙이기 전에 한 번 돌려서 의존성/경로 문제가 없는지 확인하는 용도.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-/tmp/pdbench_smoke}; rm -rf "$OUT"; mkdir -p "$OUT"
MOCK=(--max-num-seqs 24 --load-factor 0.04 --decode-ms 6 --prefill-ms-per-1k 30)
python mock_server.py --port 18000 --mode colocated "${MOCK[@]}" > "$OUT/mock_co.log" 2>&1 & M1=$!
python mock_server.py --port 19000 --mode pd "${MOCK[@]}" > "$OUT/mock_pd.log" 2>&1 & M2=$!
trap 'kill $M1 $M2 2>/dev/null || true' EXIT
sleep 2

echo "## 1) workload + perf"
python gen_workload.py synth --out "$OUT/wl/mt" --num-sessions 20 --turns 3 --input-len 400 --turn-input-len 50 \
  --output-len 24 --prefix-ratio 0.5 --num-prefix-groups 2 > /dev/null
for t in "co 18000" "pd 19000"; do set -- $t
  python run_perf.py --workload "$OUT/wl/mt" --url http://localhost:$2 --model mock --api chat --qps 10 \
    --max-concurrency 8 --metrics-url http://localhost:$2 --reset-cache --warmup 2 --label $1 \
    --out "$OUT/perf/$1.json" | grep -E "===|ttft_ms|hit rate target"
done

echo "## 2) accuracy + compare"
for t in "co 18000" "pd 19000"; do set -- $t
  python run_accuracy.py --task niah --mode multi --url http://localhost:$2 --model mock --context-len 2000 \
    --num-contexts 3 --needles-per-context 4 --label $1 --out "$OUT/acc/$1__niah_multi" | grep "accuracy"
done
python compare_outputs.py --a "$OUT/acc/co__niah_multi.records.jsonl" --b "$OUT/acc/pd__niah_multi.records.jsonl" \
  | grep -E "exact match|accuracy A/B"

echo "## 3) capacity"
for t in "co 18000" "pd 19000"; do set -- $t
  python run_capacity.py --url http://localhost:$2 --model mock --num-gpus 2 --metrics-url http://localhost:$2 \
    --reset-cache --first-input-len 400 --turn-input-len 50 --output-len 40 --turns 3-5 --think-time 1 \
    --max-context 4000 --slo-ttft-ms 800 --slo-tpot-ms 12 --warmup-s 4 --duration-s 8 --drain-timeout 10 \
    --cooldown-s 0 --poll-interval 1 --search --search-start 8 --repeats 1 --min-requests 10 \
    --label $1 --out "$OUT/capacity/$1" | tail -3
done
python compare_capacity.py "$OUT/capacity/co" "$OUT/capacity/pd" --out "$OUT/capacity/compare" | grep -E "^\| (co|pd) "

echo "smoke test OK -> $OUT"
