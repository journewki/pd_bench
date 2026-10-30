"""동시 사용자 수용량(capacity) 벤치마크.

질문: "multi-turn chat 사용자(턴당 수천 토큰 입력 / 수백 토큰 출력)를 한 서버에서
       TTFT / TPOT SLO 를 지키며 최대 몇 명까지 동시에 받을 수 있는가?"

부하 모델 (closed-loop, 사용자 수 고정):
  - N 명의 가상 사용자가 각자 대화를 진행한다.
    turn 전송 → 응답 수신 → think time 대기 → 다음 turn ... → 대화 종료 → 새 대화 시작
  - 이전 턴의 실제 응답이 history 에 누적된다 (컨텍스트가 턴마다 커짐).
  - 사용자는 think time 동안에도 "동시 사용자"로 계속 존재한다.
    (요청 동시성 ≠ 사용자 수. 실제 in-flight 요청 수는 결과에 따로 기록)
  - warmup 구간 이후 측정 구간(duration)에 '시작된' 요청만 집계한다.
  - 기본적으로 사용자는 대화 중간 턴에서 출발한다(--start-mid-conversation):
    시작부터 짧은/긴 컨텍스트가 섞인 정상상태(steady state)에 가깝게 만들기 위함.

판정 (레벨 = 사용자 수 N 한 번 실행):
  pass ⇔ p{P}(TTFT) ≤ SLO_TTFT  and  p{P}(TPOT) ≤ SLO_TPOT  and  error_rate ≤ max  (P 기본 90)
  같은 N 을 여러 번 반복해서 --pass-rule (all / majority) 로 최종 판정.

탐색:
  1) 시작값에서 ×growth 로 늘려가며 처음 실패하는 지점을 찾고 (coarse, 반복 적게)
  2) 마지막 통과 ~ 첫 실패 구간을 이분 탐색 (반복 많게)
  3) 이 전체 탐색을 --search-runs 번 반복 → 수용량의 분포(최소/중앙/최대) 보고
  또는 --users 8,16,32 처럼 레벨을 직접 지정하는 sweep 모드.

출력 (--out 디렉토리):
  levels/u<N>_s<run>_r<rep>.json   레벨별 상세 결과
  capacity.json                    탐색 결과 요약
  capacity_report.md               표
  capacity_latency.png / capacity_pareto.png   (matplotlib 있을 때)

예시:
  python run_capacity.py --url http://localhost:8000 --model Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B \\
      --num-gpus 2 --metrics-url http://localhost:8000 --reset-cache \\
      --first-input-len 2000 --turn-input-len 200 --output-len 300 --turns 4-8 --think-time 10 \\
      --slo-ttft-ms 5000 --slo-tpot-ms 70 --slo-percentile 90 \\
      --search --search-start 16 --repeats 2 --search-runs 3 \\
      --warmup-s 60 --duration-s 180 --label vllm --out results/capacity/vllm
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time

from common import (
    Tok,
    build_payload,
    make_session,
    percentile,
    reset_prefix_cache,
    scrape_gauges,
    stats,
    stream_request,
    tpot_of,
    write_json,
)


# --------------------------------------------------------------------------- text pool
class TextPool:
    """런타임 토크나이즈 비용을 없애기 위해 길이별 텍스트를 미리 만들어 둔다.
    실제 요청에는 사용자/대화/턴 고유 마커를 앞에 붙여 사용자 간 캐시 공유를 차단한다."""

    def __init__(self, tok: Tok, seed: int, size: int = 64):
        self.tok, self.rng, self.size = tok, random.Random(seed), size
        self.cache: dict[int, list[str]] = {}

    def get(self, n: int, rng: random.Random) -> str:
        b = max(1, int(round(n / 16.0)) * 16)  # 16 토큰 단위로 버킷팅
        if b not in self.cache:
            self.cache[b] = [self.tok.text_of_len(self.rng, b) for _ in range(self.size)]
        return rng.choice(self.cache[b])


def jit(rng: random.Random, base: int, frac: float) -> int:
    return max(1, int(round(base * rng.uniform(1 - frac, 1 + frac)))) if frac > 0 else max(1, base)


def think(rng: random.Random, a) -> float:
    if a.think_time <= 0:
        return 0.0
    if a.think_dist == "fixed":
        return a.think_time
    if a.think_dist == "uniform":
        return rng.uniform(0.5 * a.think_time, 1.5 * a.think_time)
    return min(rng.expovariate(1.0 / a.think_time), a.think_time * 6)  # exp (꼬리 캡)


# --------------------------------------------------------------------------- one level
async def run_level(a, users: int, seed: int, pool: TextPool, system_text: str) -> dict:
    t_start = time.perf_counter()
    w0 = t_start + a.warmup_s
    w1 = w0 + a.duration_s
    stop = asyncio.Event()
    recs: list[dict] = []
    state = {"active": 0, "inflight_window": 0}
    samples: list[dict] = []
    tmin, tmax = (int(x) for x in (a.turns.split("-") if "-" in a.turns else (a.turns, a.turns)))

    async with make_session() as http:
        if a.reset_cache and a.metrics_url:
            await reset_prefix_cache(http, a.metrics_url)
        g_before = await scrape_gauges(http, a.metrics_url or [])

        async def send(uid, conv, k, history, max_tokens, scheduled):
            path, payload = build_payload(a.api, a.model, system_text or None, history, max_tokens,
                                          temperature=a.temperature, ignore_eos=a.ignore_eos, sep="\n")
            now = time.perf_counter()
            in_window = w0 <= now < w1
            state["active"] += 1
            if in_window:
                state["inflight_window"] += 1
            rec = await stream_request(http, a.url, path, payload, timeout=a.request_timeout)
            state["active"] -= 1
            if in_window:
                state["inflight_window"] -= 1
                text = rec.pop("text")
                rec.pop("itl", None)
                rec.update(uid=uid, conv=conv, turn=k, t_rel=rec["t_start"] - t_start,
                           tpot=tpot_of(rec), lag=max(0.0, now - scheduled))
                rec.pop("t_start", None)
                rec.pop("t_end", None)
                recs.append(rec)
                return rec, text
            return rec, rec.get("text", "")

        async def user_loop(uid: int):
            rng = random.Random(seed * 1_000_003 + uid)
            # 시작 시각을 warmup 구간 앞부분에 고르게 분산 (동시 폭주 방지)
            spread = max(1.0, a.warmup_s * 0.5)  # 사용자 시작을 warmup 앞 절반에 분산 (cold prefill 폭주 방지)
            await asyncio.sleep(rng.uniform(0, spread))
            conv, first = 0, True
            while not stop.is_set():
                nturns = rng.randint(tmin, tmax)
                history: list[dict] = []
                ctx = a.system_prompt_len
                k0 = rng.randrange(nturns) if (first and a.start_mid_conversation) else 0
                # 대화 중간 출발: 앞선 턴들을 합성 history 로 채움 (첫 요청은 cold prefill)
                for k in range(k0):
                    ulen = jit(rng, a.first_input_len if k == 0 else a.turn_input_len, a.len_jitter)
                    olen = jit(rng, a.output_len, a.len_jitter)
                    if ctx + ulen + olen + a.output_len > a.max_context:
                        break
                    history.append({"role": "user", "content": f"[u{uid} c{conv} t{k}] " + pool.get(ulen, rng)})
                    history.append({"role": "assistant", "content": pool.get(olen, rng)})
                    ctx += ulen + olen
                first = False
                next_sched = time.perf_counter()
                for k in range(len(history) // 2, nturns):
                    if stop.is_set():
                        return
                    ulen = jit(rng, a.first_input_len if k == 0 else a.turn_input_len, a.len_jitter)
                    olen = jit(rng, a.output_len, a.len_jitter)
                    if ctx + ulen + olen > a.max_context:  # 컨텍스트 한도 → 대화 종료
                        break
                    history.append({"role": "user", "content": f"[u{uid} c{conv} t{k}] " + pool.get(ulen, rng)})
                    rec, text = await send(uid, conv, k, history, olen, scheduled=next_sched)
                    if not rec["ok"]:
                        await asyncio.sleep(1.0)
                        break
                    history.append({"role": "assistant", "content": text})
                    if rec.get("prompt_tokens"):
                        ctx = rec["prompt_tokens"] + (rec.get("completion_tokens") or olen)
                    else:
                        ctx += ulen + olen
                    if k < nturns - 1:
                        d = think(rng, a)
                        next_sched = time.perf_counter() + d  # 깨어나야 할 시각 → 실제와의 차이 = client lag
                        await asyncio.sleep(d)
                conv += 1
                await asyncio.sleep(think(rng, a) if a.conversation_gap < 0 else a.conversation_gap)

        async def sampler():
            while not stop.is_set():
                now = time.perf_counter()
                if w0 <= now < w1:
                    g = await scrape_gauges(http, a.metrics_url or []) if a.metrics_url else {}
                    samples.append({"t": now - t_start, "active": state["active"], "gauges": g})
                await asyncio.sleep(a.poll_interval)

        tasks = [asyncio.create_task(user_loop(u)) for u in range(users)]
        samp = asyncio.create_task(sampler())
        await asyncio.sleep(max(0.0, w1 - time.perf_counter()))
        stop.set()
        done, pending = await asyncio.wait(tasks, timeout=a.drain_timeout)
        for t in pending:
            t.cancel()
        unfinished = state["inflight_window"]
        await asyncio.gather(*pending, return_exceptions=True)
        samp.cancel()
        await asyncio.gather(samp, return_exceptions=True)
        g_after = await scrape_gauges(http, a.metrics_url or [])

    return summarize_level(a, users, seed, recs, unfinished, samples, g_before, g_after, state)


def summarize_level(a, users, seed, recs, unfinished, samples, g_before, g_after, state) -> dict:
    ok = [r for r in recs if r["ok"]]
    n_err = len(recs) - len(ok) + unfinished
    n_total = len(recs) + unfinished
    P = a.slo_percentile
    ttft = [r["ttft"] * 1000 for r in ok]
    tpot = [r["tpot"] * 1000 for r in ok if r["tpot"] is not None]
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    dur = a.duration_s
    tps_user = [1000.0 / x for x in tpot if x > 0]
    attain = (sum(1 for r in ok if r["ttft"] * 1000 <= a.slo_ttft_ms and (r["tpot"] is None or r["tpot"] * 1000 <= a.slo_tpot_ms))
              / n_total) if n_total else 0.0
    with_cached = [r for r in ok if r.get("cached_tokens") is not None and r.get("prompt_tokens")]
    hit = (sum(r["cached_tokens"] for r in with_cached) / sum(r["prompt_tokens"] for r in with_cached)) if with_cached else None

    def gauge_series(name):
        vals = []
        for s in samples:
            per = [d.get(name) for d in s["gauges"].values() if d.get(name) is not None]
            if per:
                vals.append(max(per) if name.endswith("_perc") else sum(per))
        return vals

    kv = gauge_series("vllm:kv_cache_usage_perc") or gauge_series("vllm:gpu_cache_usage_perc")
    waiting = gauge_series("vllm:num_requests_waiting")
    running = gauge_series("vllm:num_requests_running")
    pre_b = sum(d.get("vllm:num_preemptions_total", 0) for d in g_before.values())
    pre_a = sum(d.get("vllm:num_preemptions_total", 0) for d in g_after.values())
    active = [s["active"] for s in samples]

    p_ttft, p_tpot = percentile(ttft, P), percentile(tpot, P)
    err_rate = n_err / n_total if n_total else 1.0
    reasons = []
    if len(ok) < a.min_requests:
        reasons.append(f"too few requests ({len(ok)} < {a.min_requests})")
    if p_ttft is None or p_ttft > a.slo_ttft_ms:
        reasons.append(f"TTFT p{P} {p_ttft and round(p_ttft)}ms > {a.slo_ttft_ms}")
    if p_tpot is None or p_tpot > a.slo_tpot_ms:
        reasons.append(f"TPOT p{P} {p_tpot and round(p_tpot, 1)}ms > {a.slo_tpot_ms}")
    if err_rate > a.max_error_rate:
        reasons.append(f"error rate {err_rate:.3f} > {a.max_error_rate}")
    if a.min_attainment and attain < a.min_attainment:
        reasons.append(f"attainment {attain:.3f} < {a.min_attainment}")
    lag_p99 = percentile([r["lag"] for r in recs], 99)
    warnings = []
    if lag_p99 is not None and lag_p99 > a.max_client_lag_s:
        warnings.append("client event loop 지연 감지: 클라이언트가 병목일 수 있음 (서버와 같은 노드/망에서 실행 권장)")

    return {
        "users": users, "seed": seed, "label": a.label,
        "pass": not reasons, "fail_reasons": reasons, "warnings": warnings,
        "num_requests": n_total, "num_ok": len(ok), "num_errors": n_err, "unfinished": unfinished,
        "error_rate": err_rate,
        "error_samples": [r["error"] for r in recs if not r["ok"]][:3],
        f"ttft_p{P}_ms": p_ttft, f"tpot_p{P}_ms": p_tpot,
        "ttft_ms": stats(ttft), "tpot_ms": stats(tpot), "e2e_ms": stats([r["e2e"] * 1000 for r in ok]),
        "slo_attainment": attain,
        "request_throughput": len(ok) / dur,
        "output_tok_per_s": out_tok / dur,
        "output_tok_per_s_per_gpu": out_tok / dur / max(1, a.num_gpus),
        "output_tok_per_s_per_user": stats(tps_user),
        "prompt_tokens": stats([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]),
        "usage_hit_rate": hit,
        "active_requests": stats(active),
        "active_fraction": (sum(active) / len(active) / users) if active else None,
        "kv_usage": stats(kv), "waiting": stats(waiting), "running": stats(running),
        "preemptions": pre_a - pre_b,
        "client_lag_p99_s": lag_p99,
        "ttft_ms_by_turn": {str(k): stats([r["ttft"] * 1000 for r in ok if r["turn"] == k])
                            for k in sorted({r["turn"] for r in ok})},
    }


# --------------------------------------------------------------------------- search
class Runner:
    def __init__(self, a):
        self.a = a
        self.tok = Tok(a.tokenizer)
        self.pool = TextPool(self.tok, a.seed)
        self.system_text = (self.tok.text_of_len(random.Random(a.seed + 7), a.system_prompt_len,
                                                 lead="[system prompt shared by all users]")
                            if a.system_prompt_len > 0 else "")
        self.results: list[dict] = []
        self.full_pass: set[tuple[int, int]] = set()  # (run, users) 를 repeats 번 돌려 통과한 레벨
        os.makedirs(os.path.join(a.out, "levels"), exist_ok=True)

    def level(self, users: int, run: int, rep: int) -> dict:
        a = self.a
        seed = a.seed + run * 1000 + rep
        print(f"[run {run} rep {rep}] users={users} ... ", end="", flush=True)
        r = asyncio.run(run_level(a, users, seed, self.pool, self.system_text))
        r.update(run=run, rep=rep)
        P = a.slo_percentile
        print(f"{'PASS' if r['pass'] else 'FAIL'}  TTFT p{P}={fmt(r[f'ttft_p{P}_ms'])}ms "
              f"TPOT p{P}={fmt(r[f'tpot_p{P}_ms'], 1)}ms  err={r['error_rate']:.3f}  "
              f"tok/s/gpu={r['output_tok_per_s_per_gpu']:.0f}  active={fmt(r['active_requests'].get('mean'), 1)}"
              + (f"  kv={r['kv_usage']['max']:.2f}" if r['kv_usage'].get('n') else "")
              + (f"  ({'; '.join(r['fail_reasons'])})" if r["fail_reasons"] else ""))
        for w in r["warnings"]:
            print(f"    ! {w}")
        write_json(os.path.join(a.out, "levels", f"u{users}_s{run}_r{rep}.json"), r)
        self.results.append(r)
        if a.cooldown_s > 0:
            time.sleep(a.cooldown_s)
        return r

    def evaluate(self, users: int, run: int, reps: int) -> bool:
        res = [self.level(users, run, i) for i in range(reps)]
        n_pass = sum(r["pass"] for r in res)
        ok = n_pass == reps if self.a.pass_rule == "all" else n_pass * 2 > reps
        if ok and reps >= self.a.repeats:
            self.full_pass.add((run, users))
        return ok

    def search(self, run: int) -> dict:
        a = self.a
        lo = hi = None
        n = a.search_start
        while n <= a.search_max:  # 1) 확장 단계
            if self.evaluate(n, run, a.coarse_repeats):
                lo = n
                n = max(n + 1, int(math.ceil(n * a.search_growth)))
            else:
                hi = n
                break
        if lo is None:  # 시작값부터 실패 → 줄여가며 통과점 찾기
            n = a.search_start // 2
            while n >= 1:
                if self.evaluate(n, run, a.coarse_repeats):
                    lo = n
                    break
                hi = n
                n //= 2
            if lo is None:
                return {"run": run, "capacity": 0, "upper_fail": hi, "note": "1명도 SLO 를 만족하지 못함"}
        if hi is None:
            return {"run": run, "capacity": lo, "upper_fail": None, "note": f"search_max({a.search_max})까지 모두 통과"}
        # 2) 이분 탐색 (반복 많게)
        while hi - lo > max(a.min_step, int(lo * a.resolution)):
            mid = (lo + hi) // 2
            if self.evaluate(mid, run, a.repeats):
                lo = mid
            else:
                hi = mid
        # 3) 최종 확인: lo 를 repeats 번으로 재확인 (확장 단계에서 coarse 로만 봤을 수 있음)
        tries = 0
        while a.repeats > a.coarse_repeats and tries < 3 and (run, lo) not in self.full_pass:
            if self.evaluate(lo, run, a.repeats):
                break
            hi, lo = lo, max(1, lo - max(a.min_step, int(lo * a.resolution)))
            tries += 1
        return {"run": run, "capacity": lo, "upper_fail": hi}

    def sweep(self, levels: list[int], run: int) -> dict:
        passed = {}
        for n in levels:
            passed[n] = self.evaluate(n, run, self.a.repeats)
        cap = 0
        for n in sorted(levels):  # 단조 가정: 처음 실패 전까지의 최대 통과 레벨
            if passed[n]:
                cap = n
            else:
                break
        return {"run": run, "capacity": cap, "level_pass": passed}


def isotonic_pass_curve(results: list[dict]) -> list[tuple[int, float, int]]:
    """모든 실행의 pass/fail 을 사용자 수별로 모아, 사용자 수가 늘수록 통과확률이
    줄어든다(단조감소)는 가정으로 PAVA 보정. 반환: [(users, 보정 통과확률, 실행 수)]"""
    by: dict[int, list[int]] = {}
    for r in results:
        by.setdefault(r["users"], []).append(1 if r["pass"] else 0)
    blocks = []  # [sum, weight, [users...]]
    for u in sorted(by):
        blocks.append([sum(by[u]), len(by[u]), [u]])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = []
    for sm, w, us in blocks:
        for u in us:
            out.append((u, sm / w, len(by[u])))
    return out


def capacity_at(curve: list[tuple[int, float, int]], threshold: float) -> int:
    """보정 통과확률이 threshold 이상인 최대 사용자 수."""
    ok = [u for u, p, _ in curve if p >= threshold]
    return max(ok) if ok else 0


def fmt(x, nd=0):
    return "-" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- report
def per_level_table(results: list[dict], P: int) -> list[dict]:
    by = {}
    for r in results:
        by.setdefault(r["users"], []).append(r)
    rows = []
    for u in sorted(by):
        rs = by[u]

        def med(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return statistics.median(vals) if vals else None
        rows.append({
            "users": u, "runs": len(rs), "pass_rate": sum(r["pass"] for r in rs) / len(rs),
            "ttft_p": med(f"ttft_p{P}_ms"), "tpot_p": med(f"tpot_p{P}_ms"),
            "ttft_p_max": max((r[f"ttft_p{P}_ms"] or 0) for r in rs),
            "tpot_p_max": max((r[f"tpot_p{P}_ms"] or 0) for r in rs),
            "err": med("error_rate"), "attain": med("slo_attainment"),
            "tps_gpu": med("output_tok_per_s_per_gpu"),
            "tps_user": statistics.median([r["output_tok_per_s_per_user"].get("p50") or 0 for r in rs]),
            "active": statistics.median([r["active_requests"].get("mean") or 0 for r in rs]),
            "kv_max": max((r["kv_usage"].get("max") or 0) for r in rs) if any(r["kv_usage"].get("n") for r in rs) else None,
            "waiting": max((r["waiting"].get("max") or 0) for r in rs) if any(r["waiting"].get("n") for r in rs) else None,
            "hit": med("usage_hit_rate"),
        })
    return rows


def write_report(out_dir: str, summary: dict, rows: list[dict], P: int) -> str:
    lines = [f"# Capacity report: {summary['label']}\n"]
    caps = summary["capacities"]
    lines.append(f"- SLO: TTFT p{P} ≤ {summary['slo']['ttft_ms']} ms, TPOT p{P} ≤ {summary['slo']['tpot_ms']} ms, "
                 f"error ≤ {summary['slo']['max_error_rate']}")
    lines.append(f"- 워크로드: {summary['workload_desc']}")
    if caps:
        lines.append(f"- **보수적 수용량 (통과확률 ≥ {int(summary['stable_pass_rate']*100)}%): {summary['capacity_stable']}명**")
        lines.append(f"- 통과확률 50% 지점: {summary['capacity_p50']}명")
        lines.append(f"- 이분 탐색 결과: 중앙값 {summary['capacity_median']} (범위 {min(caps)}–{max(caps)}, 탐색 {len(caps)}회)")
    for n in summary.get("notes", []):
        lines.append(f"- 참고: {n}")
    lines.append("")
    pc = {d["users"]: d["pass_prob"] for d in summary.get("pass_curve", [])}
    lines.append(f"| users | 실행 | 통과율 | 보정 통과확률 | TTFT p{P} (중앙/최대) | TPOT p{P} (중앙/최대) | err | SLO 달성률 | tok/s/gpu | tok/s/user | 활성 요청 | KV 최대 | 대기 최대 | hit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['users']} | {r['runs']} | {r['pass_rate']:.2f} | {fmt(pc.get(r['users']), 2)} | {fmt(r['ttft_p'])} / {fmt(r['ttft_p_max'])} | "
            f"{fmt(r['tpot_p'], 1)} / {fmt(r['tpot_p_max'], 1)} | {fmt(r['err'], 3)} | {fmt(r['attain'], 3)} | "
            f"{fmt(r['tps_gpu'])} | {fmt(r['tps_user'], 1)} | {fmt(r['active'], 1)} | {fmt(r['kv_max'], 2)} | "
            f"{fmt(r['waiting'])} | {fmt(r['hit'], 3)} |")
    fn = os.path.join(out_dir, "capacity_report.md")
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return fn


def plot_runs(runs: list[tuple[str, list[dict], dict]], out_dir: str, P: int, slo_ttft: float, slo_tpot: float) -> list[str]:
    """runs: [(label, results, summary)] — 여러 시스템을 한 그래프에 겹쳐 그림."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, results, _ in runs:
        xs = [r["users"] for r in results]
        axes[0].scatter(xs, [r[f"ttft_p{P}_ms"] or float("nan") for r in results], s=14, alpha=0.6, label=label)
        axes[1].scatter(xs, [r[f"tpot_p{P}_ms"] or float("nan") for r in results], s=14, alpha=0.6, label=label)
        rows = per_level_table(results, P)
        axes[0].plot([r["users"] for r in rows], [r["ttft_p"] for r in rows], alpha=0.7)
        axes[1].plot([r["users"] for r in rows], [r["tpot_p"] for r in rows], alpha=0.7)
    axes[0].axhline(slo_ttft, color="red", ls="--", lw=1, label="SLO")
    axes[1].axhline(slo_tpot, color="red", ls="--", lw=1, label="SLO")
    axes[0].set_ylabel(f"TTFT p{P} (ms)")
    axes[1].set_ylabel(f"TPOT p{P} (ms)")
    for ax in axes:
        ax.set_xlabel("concurrent users")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(out_dir, "capacity_latency.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    files.append(fn)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, results, _ in runs:
        rows = per_level_table(results, P)
        ax.plot([r["tps_user"] for r in rows], [r["tps_gpu"] for r in rows], marker="o", label=label)
        for r in rows:
            ax.annotate(str(r["users"]), (r["tps_user"], r["tps_gpu"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("output tok/s/user (p50)")
    ax.set_ylabel("output tok/s/gpu")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("throughput vs interactivity (labels = users)")
    fig.tight_layout()
    fn = os.path.join(out_dir, "capacity_pareto.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    files.append(fn)
    return files


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("server")
    g.add_argument("--url")
    g.add_argument("--model")
    g.add_argument("--api", choices=["chat", "completions"], default="chat")
    g.add_argument("--tokenizer", default=None)
    g.add_argument("--num-gpus", type=int, default=1, help="tok/s/gpu 계산용 (PD 는 P+D 합계)")
    g.add_argument("--metrics-url", action="append", help="/metrics 대상 (PD 는 P/D 각각)")
    g.add_argument("--reset-cache", action="store_true", help="레벨마다 prefix cache 초기화 (VLLM_SERVER_DEV_MODE=1)")
    g.add_argument("--request-timeout", type=float, default=600)
    w = p.add_argument_group("workload (사용자 1명의 대화)")
    w.add_argument("--system-prompt-len", type=int, default=0, help="모든 사용자 공통 system prompt 토큰")
    w.add_argument("--first-input-len", type=int, default=2000, help="대화 첫 턴 user 입력 토큰")
    w.add_argument("--turn-input-len", type=int, default=200, help="2턴부터 매 턴 새 user 입력 토큰")
    w.add_argument("--output-len", type=int, default=300, help="턴당 출력 토큰 (ignore_eos 로 고정)")
    w.add_argument("--len-jitter", type=float, default=0.2)
    w.add_argument("--turns", default="4-8", help="대화당 턴 수 (고정 또는 범위)")
    w.add_argument("--max-context", type=int, default=32000, help="이 길이를 넘으면 대화 종료 후 새 대화")
    w.add_argument("--think-time", type=float, default=10.0, help="턴 사이 사용자 대기 평균(초)")
    w.add_argument("--think-dist", choices=["exp", "uniform", "fixed"], default="exp")
    w.add_argument("--conversation-gap", type=float, default=-1, help="대화 사이 대기(초). 음수면 think time 분포 사용")
    w.add_argument("--no-start-mid-conversation", dest="start_mid_conversation", action="store_false")
    w.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    w.add_argument("--temperature", type=float, default=0.0)
    t = p.add_argument_group("timing")
    t.add_argument("--warmup-s", type=float, default=60)
    t.add_argument("--duration-s", type=float, default=180)
    t.add_argument("--drain-timeout", type=float, default=120)
    t.add_argument("--cooldown-s", type=float, default=10, help="레벨 사이 대기")
    t.add_argument("--poll-interval", type=float, default=2.0)
    t.add_argument("--max-client-lag-s", type=float, default=0.5, help="client lag p99 가 이 값을 넘으면 경고")
    s = p.add_argument_group("SLO")
    s.add_argument("--slo-ttft-ms", type=float, default=5000)
    s.add_argument("--slo-tpot-ms", type=float, default=70)
    s.add_argument("--slo-percentile", type=int, default=95, help="AIPerf max-concurrency-under-sla 레시피와 같은 p95 가 기본")
    s.add_argument("--max-error-rate", type=float, default=0.01)
    s.add_argument("--min-attainment", type=float, default=0.0, help="(선택) SLO 동시 만족 요청 비율 하한")
    s.add_argument("--min-requests", type=int, default=30)
    m = p.add_argument_group("search")
    m.add_argument("--users", default="", help="직접 지정: '8,16,32,64' (지정 시 sweep 모드)")
    m.add_argument("--search", action="store_true", help="자동 탐색 모드")
    m.add_argument("--search-start", type=int, default=16)
    m.add_argument("--search-max", type=int, default=2048)
    m.add_argument("--search-growth", type=float, default=2.0)
    m.add_argument("--resolution", type=float, default=0.1, help="이분 탐색 종료 폭 (상대값, 0.1 = 10%)")
    m.add_argument("--min-step", type=int, default=2)
    m.add_argument("--coarse-repeats", type=int, default=1)
    m.add_argument("--repeats", type=int, default=2)
    m.add_argument("--pass-rule", choices=["all", "majority"], default="all")
    m.add_argument("--search-runs", type=int, default=1, help="전체 탐색 반복 횟수 → 수용량 분포")
    m.add_argument("--stable-pass-rate", type=float, default=0.8,
                   help="보수적 수용량 = 보정 통과확률이 이 값 이상인 최대 사용자 수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="run")
    p.add_argument("--out", required=True)
    p.add_argument("--report-only", action="store_true",
                   help="새로 실행하지 않고 --out/levels/*.json 으로 리포트만 다시 생성")
    return p


def main(argv: list[str] | None = None) -> dict:
    a = build_parser().parse_args(argv)
    if a.report_only:
        return report_only(a)
    if not a.url or not a.model:
        raise SystemExit("--url 과 --model 이 필요합니다")
    if not a.search and not a.users:
        raise SystemExit("--search 또는 --users 중 하나를 지정하세요")
    runner = Runner(a)
    outcomes = []
    for run in range(a.search_runs):
        if a.users:
            outcomes.append(runner.sweep([int(x) for x in a.users.split(",")], run))
        else:
            outcomes.append(runner.search(run))
        print(f"==> run {run}: capacity = {outcomes[-1]['capacity']} users"
              + (f"  ({outcomes[-1].get('note')})" if outcomes[-1].get("note") else ""))
    return finalize(a, runner.results, outcomes)


def report_only(a) -> dict:
    import glob
    results = []
    for fn in sorted(glob.glob(os.path.join(a.out, "levels", "*.json"))):
        with open(fn, encoding="utf-8") as f:
            results.append(json.load(f))
    old = {}
    if os.path.exists(os.path.join(a.out, "capacity.json")):
        with open(os.path.join(a.out, "capacity.json"), encoding="utf-8") as f:
            old = json.load(f)
        prm = old.get("params", {})
        for k in ("label", "slo_ttft_ms", "slo_tpot_ms", "slo_percentile", "max_error_rate", "pass_rule",
                  "system_prompt_len", "first_input_len", "turn_input_len", "output_len", "turns",
                  "think_time", "think_dist", "max_context", "num_gpus"):
            if k in prm:
                setattr(a, k, prm[k])
    return finalize(a, results, old.get("outcomes", []))


def finalize(a, results: list[dict], outcomes: list[dict]) -> dict:
    P = a.slo_percentile
    caps = [o["capacity"] for o in outcomes]
    curve = isotonic_pass_curve(results)
    summary = {
        "label": a.label,
        "capacities": caps,
        "capacity_median": statistics.median_low(caps) if caps else None,
        "capacity_p50": capacity_at(curve, 0.5),
        "capacity_stable": capacity_at(curve, a.stable_pass_rate),
        "stable_pass_rate": a.stable_pass_rate,
        "pass_curve": [{"users": u, "pass_prob": p, "runs": n} for u, p, n in curve],
        "capacity_min": min(caps) if caps else None,
        "capacity_max": max(caps) if caps else None,
        "outcomes": outcomes,
        "notes": [o["note"] for o in outcomes if o.get("note")],
        "slo": {"ttft_ms": a.slo_ttft_ms, "tpot_ms": a.slo_tpot_ms, "percentile": P,
                "max_error_rate": a.max_error_rate, "pass_rule": a.pass_rule},
        "workload_desc": (f"system {a.system_prompt_len} + 첫 턴 {a.first_input_len} / 이후 턴 +{a.turn_input_len} 입력, "
                          f"출력 {a.output_len}, 턴 {a.turns}, think {a.think_time}s({a.think_dist}), "
                          f"max ctx {a.max_context}"),
        "params": vars(a),
        "per_level": per_level_table(results, P),
    }
    write_json(os.path.join(a.out, "capacity.json"), summary)
    fn = write_report(a.out, summary, summary["per_level"], P)
    plot_runs([(a.label, results, summary)], a.out, P, a.slo_ttft_ms, a.slo_tpot_ms)
    print(f"\n수용 가능 동시 사용자 ({a.label})")
    print(f"  탐색 결과      : 중앙값 {summary['capacity_median']}  범위 {summary['capacity_min']}–{summary['capacity_max']}  (탐색 {len(caps)}회)")
    print(f"  통과확률 곡선  : 50% 지점 {summary['capacity_p50']}명 / {int(a.stable_pass_rate*100)}% 이상 {summary['capacity_stable']}명 (보수적)")
    print(f"-> {fn}")
    return summary


if __name__ == "__main__":
    main()
