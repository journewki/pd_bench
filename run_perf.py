"""속도 벤치마크 러너 (TTFT / TPOT / ITL / E2E).

같은 워크로드 파일을 vLLM 베이스라인과 PD proxy 에 각각 돌려서 결과 JSON 을 비교한다.

  - single-turn / multi-turn 모두 지원 (multi-turn 은 이전 턴 응답을 history 에 붙여 다음 턴 전송)
  - 도착 패턴: Poisson(--qps) 또는 트레이스 timestamp(--use-trace-timestamps)
  - --max-concurrency: 동시에 활성화된 세션 수 상한 (multi-turn 은 think time 동안에도 세션 유지)
  - 측정 전 warmup → (옵션) prefix cache reset → /metrics 스냅샷 → 실행 → 스냅샷
  - 목표(이론상) hit rate 와 실측 hit rate 를 함께 기록 → 워크로드가 의도대로 동작했는지 검증

예시:
  # 베이스라인 (일반 vLLM)
  python run_perf.py --workload wl/st_p50 --url http://localhost:8000 --model Qwen/Qwen3-8B \\
      --qps 4 --metrics-url http://localhost:8000 --reset-cache --label vllm --out results/vllm_st_p50_q4.json

  # PD (proxy 로 요청, metrics 는 P/D 각각)
  python run_perf.py --workload wl/st_p50 --url http://localhost:9000 --model Qwen/Qwen3-8B \\
      --qps 4 --metrics-url http://p-node:8100 --metrics-url http://d-node:8200 --reset-cache \\
      --label pd --out results/pd_st_p50_q4.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time

from common import (
    build_payload,
    hit_rates,
    make_session,
    metric_delta,
    read_jsonl,
    reset_prefix_cache,
    scrape_prefix_metrics,
    stats,
    stream_request,
    tpot_of,
    write_json,
    Tok,
)


def assign_arrivals(sessions: list[dict], qps: float, use_trace: bool, seed: int) -> list[float]:
    if use_trace and all("arrival_s" in s and s["arrival_s"] is not None for s in sessions):
        return [float(s["arrival_s"]) for s in sessions]
    if math.isinf(qps) or qps <= 0:
        return [0.0] * len(sessions)
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in sessions:
        out.append(t)
        t += rng.expovariate(qps)
    return out


async def warmup(session, a, n: int) -> None:
    if n <= 0:
        return
    rng = random.Random(12345)
    tok = Tok(None)
    coros = []
    for i in range(n):
        text = tok.text_of_len(rng, 64, lead=f"[warmup {i} {rng.getrandbits(32):x}]")
        path, payload = build_payload(a.api, a.model, None, [{"role": "user", "content": text}], 16)
        coros.append(stream_request(session, a.url, path, payload, timeout=a.timeout))
    res = await asyncio.gather(*coros)
    bad = [r["error"] for r in res if not r["ok"]]
    if bad:
        print(f"[warmup] {len(bad)}/{n} 실패: {bad[0]}")


async def run(a: argparse.Namespace) -> dict:
    sessions = read_jsonl(a.workload + ".jsonl")
    meta_path = a.workload + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if a.max_sessions:
        sessions = sessions[: a.max_sessions]
    arrivals = assign_arrivals(sessions, a.qps, a.use_trace_timestamps, a.seed)
    metrics_urls = a.metrics_url or []

    records: list[dict] = []
    all_itl: list[float] = []
    async with make_session() as http:
        await warmup(http, a, a.warmup)
        reset_info = await reset_prefix_cache(http, metrics_urls) if a.reset_cache else {}
        m_before = await scrape_prefix_metrics(http, metrics_urls) if metrics_urls else {}

        sem = asyncio.Semaphore(a.max_concurrency) if a.max_concurrency > 0 else None
        t0 = time.perf_counter()
        done = {"n": 0}
        total_req = sum(len(s["turns"]) for s in sessions)

        async def run_session(s: dict, arr: float) -> None:
            delay = t0 + arr - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if sem:
                await sem.acquire()
            try:
                history: list[dict] = []
                for k, turn in enumerate(s["turns"]):
                    history.append({"role": "user", "content": turn["user"]})
                    path, payload = build_payload(
                        a.api, a.model, s.get("system") or None, history, turn["max_tokens"],
                        temperature=a.temperature, ignore_eos=a.ignore_eos, sep="\n",
                    )
                    rec = await stream_request(http, a.url, path, payload, timeout=a.timeout)
                    itl = rec.pop("itl")
                    all_itl.extend(itl)
                    text = rec.pop("text")
                    rec.update(
                        session_id=s["session_id"],
                        group=s.get("group"),
                        turn=k,
                        t_rel=rec["t_start"] - t0,
                        t_end_rel=rec["t_end"] - t0,
                        tpot=tpot_of(rec),
                        itl_mean=(sum(itl) / len(itl)) if itl else None,
                        max_tokens=turn["max_tokens"],
                    )
                    rec.pop("t_start", None)
                    rec.pop("t_end", None)
                    records.append(rec)
                    done["n"] += 1
                    if a.progress and done["n"] % max(1, total_req // 10) == 0:
                        print(f"  {done['n']}/{total_req} requests")
                    if not rec["ok"]:
                        break
                    reply = text if a.history_mode == "generated" else turn.get("assistant_ref", "")
                    history.append({"role": "assistant", "content": reply})
                    if k < len(s["turns"]) - 1 and turn.get("think_s", 0) > 0 and a.think_scale > 0:
                        await asyncio.sleep(turn["think_s"] * a.think_scale)
            finally:
                if sem:
                    sem.release()

        await asyncio.gather(*(run_session(s, arr) for s, arr in zip(sessions, arrivals)))
        m_after = await scrape_prefix_metrics(http, metrics_urls) if metrics_urls else {}

    return summarize(a, meta, records, all_itl, m_before, m_after, reset_info)


def summarize(a, meta, records, all_itl, m_before, m_after, reset_info) -> dict:
    ok = [r for r in records if r["ok"]]
    errs = [r for r in records if not r["ok"]]
    if ok:
        dur = max(r["t_end_rel"] for r in ok) - min(r["t_rel"] for r in ok)
    else:
        dur = 0.0
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)

    by_turn: dict[str, list] = {}
    for r in ok:
        by_turn.setdefault(str(r["turn"]), []).append(r)
    later = [r for r in ok if r["turn"] >= 1]

    # 서버가 usage.prompt_tokens_details.cached_tokens 를 주면 요청 단위 hit 도 계산
    # (vLLM: --enable-prompt-tokens-details)
    with_cached = [r for r in ok if r.get("cached_tokens") is not None and r.get("prompt_tokens")]
    usage_hit = (
        sum(r["cached_tokens"] for r in with_cached) / sum(r["prompt_tokens"] for r in with_cached)
        if with_cached else None
    )

    goodput = None
    if a.slo_ttft_ms or a.slo_tpot_ms:
        def meets(r):
            if a.slo_ttft_ms and (r["ttft"] * 1000 > a.slo_ttft_ms):
                return False
            if a.slo_tpot_ms and r["tpot"] is not None and r["tpot"] * 1000 > a.slo_tpot_ms:
                return False
            return True
        n_ok = sum(1 for r in ok if meets(r))
        goodput = {
            "slo_ttft_ms": a.slo_ttft_ms,
            "slo_tpot_ms": a.slo_tpot_ms,
            "attainment": n_ok / len(records) if records else 0.0,
            "goodput_req_per_s": n_ok / dur if dur else 0.0,
        }

    delta = metric_delta(m_before, m_after) if m_after else {}
    summary = {
        "label": a.label,
        "workload": os.path.basename(a.workload),
        "url": a.url,
        "api": a.api,
        "qps": a.qps,
        "max_concurrency": a.max_concurrency,
        "history_mode": a.history_mode,
        "ignore_eos": a.ignore_eos,
        "num_requests": len(records),
        "num_ok": len(ok),
        "num_errors": len(errs),
        "error_samples": [r["error"] for r in errs[:5]],
        "duration_s": dur,
        "request_throughput": len(ok) / dur if dur else 0.0,
        "output_token_throughput": out_tok / dur if dur else 0.0,
        "total_token_throughput": (out_tok + in_tok) / dur if dur else 0.0,
        "prompt_tokens": stats([r["prompt_tokens"] for r in ok if r["prompt_tokens"]]),
        "ttft_ms": stats([r["ttft"] for r in ok], 1000),
        "tpot_ms": stats([r["tpot"] for r in ok if r["tpot"] is not None], 1000),
        "itl_ms": stats(all_itl, 1000),
        "e2e_ms": stats([r["e2e"] for r in ok], 1000),
        "ttft_ms_by_turn": {k: stats([r["ttft"] for r in v], 1000) for k, v in sorted(by_turn.items(), key=lambda x: int(x[0]))},
        "tpot_ms_by_turn": {k: stats([r["tpot"] for r in v if r["tpot"] is not None], 1000) for k, v in sorted(by_turn.items(), key=lambda x: int(x[0]))},
        "ttft_ms_later_turns": stats([r["ttft"] for r in later], 1000),
        "goodput": goodput,
        "cache": {
            "target_hit_rate_colocated": meta.get("target_hit_rate_colocated"),
            "target_hit_rate_pd_prefill_only": meta.get("target_hit_rate_pd_prefill_only"),
            "measured_usage_hit_rate": usage_hit,
            "measured_server": hit_rates(delta) if delta else None,
            "reset": reset_info,
        },
        "workload_meta": {k: v for k, v in meta.items() if k != "params"},
        "workload_params": meta.get("params"),
    }
    return {"summary": summary, "records": records}


def print_summary(s: dict) -> None:
    def f(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"

    print(f"\n=== {s['label']} | {s['workload']} | qps={s['qps']} conc={s['max_concurrency']} ===")
    print(f"requests ok/err : {s['num_ok']}/{s['num_errors']}   duration {f(s['duration_s'])}s")
    print(f"throughput      : {f(s['request_throughput'], 2)} req/s, {f(s['output_token_throughput'])} out tok/s")
    for key in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        st = s[key]
        if st.get("n"):
            print(f"{key:15s} : mean {f(st['mean'])}  p50 {f(st['p50'])}  p90 {f(st['p90'])}  p99 {f(st['p99'])}")
    if len(s["ttft_ms_by_turn"]) > 1:
        line = "  ".join(f"t{k}:{f(v.get('p50'))}" for k, v in s["ttft_ms_by_turn"].items())
        print(f"TTFT p50 by turn: {line}")
    c = s["cache"]
    print(f"hit rate target : colocated {f(c['target_hit_rate_colocated'], 3)}  pd(P only) {f(c['target_hit_rate_pd_prefill_only'], 3)}")
    if c["measured_usage_hit_rate"] is not None:
        print(f"hit rate usage  : {f(c['measured_usage_hit_rate'], 3)}")
    if c["measured_server"]:
        for u, rates in c["measured_server"].items():
            if "__error__" in rates:
                print(f"hit rate server : {u} ERROR {rates['__error__']}")
                continue
            for name, v in rates.items():
                print(f"hit rate server : {u} {name} = {v['hit_rate']:.3f} ({int(v['hits'])}/{int(v['queries'])})")
    if s["goodput"]:
        g = s["goodput"]
        print(f"SLO attainment  : {g['attainment']:.3f}  goodput {g['goodput_req_per_s']:.2f} req/s")
    if s["error_samples"]:
        print(f"errors          : {s['error_samples'][:2]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", required=True, help="gen_workload.py 의 --out 경로 (확장자 제외)")
    p.add_argument("--url", required=True, help="요청 보낼 서버/프록시 base URL")
    p.add_argument("--model", required=True)
    p.add_argument("--api", choices=["completions", "chat"], default="completions")
    p.add_argument("--qps", type=float, default=float("inf"), help="세션 도착률 (inf = 한 번에 전부)")
    p.add_argument("--use-trace-timestamps", action="store_true", help="워크로드의 arrival_s 사용 (mooncake)")
    p.add_argument("--max-concurrency", type=int, default=0, help="동시 활성 세션 상한 (0=무제한)")
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--history-mode", choices=["generated", "fixed"], default="generated",
                   help="generated: 실제 응답을 history 에 사용(현실적) / fixed: 미리 만든 텍스트(재현성)")
    p.add_argument("--think-scale", type=float, default=1.0, help="think time 배율 (0 = 대기 없음)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="기본은 ignore_eos=True (출력 길이 고정). 끄면 모델이 자연 종료")
    p.add_argument("--metrics-url", action="append", help="/metrics 스크랩 대상 (여러 번 지정 가능, PD 는 P/D 각각)")
    p.add_argument("--reset-cache", action="store_true", help="측정 전 /reset_prefix_cache 호출 (VLLM_SERVER_DEV_MODE=1 필요)")
    p.add_argument("--warmup", type=int, default=8, help="워밍업 요청 수 (캐시 reset 전에 보냄)")
    p.add_argument("--slo-ttft-ms", type=float, default=0.0)
    p.add_argument("--slo-tpot-ms", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None, help="결과 JSON 경로")
    p.add_argument("--no-records", action="store_true", help="요청별 레코드 저장 안 함")
    p.add_argument("--progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict:
    a = build_parser().parse_args(argv)
    res = asyncio.run(run(a))
    print_summary(res["summary"])
    if a.out:
        if a.no_records:
            res.pop("records")
        write_json(a.out, res)
        print(f"-> {a.out}")
    return res


if __name__ == "__main__":
    main()
