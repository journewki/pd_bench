"""워크로드 생성기.

두 가지 모드:
  synth    : prefix 비율 / prefix 그룹 수 / 턴 수 / 길이 / think time 을 손잡이로 조절하는 합성 워크로드
  mooncake : Mooncake 형식 트레이스(timestamp, input_length, output_length, hash_ids)를
             hash_ids 구조가 보존되도록 텍스트로 복원 (실제 도착 패턴 + 실제 prefix 재사용 구조)

출력: <out>.jsonl (세션 1개 = 1줄) + <out>.meta.json (파라미터, 이론상 hit rate)

세션 포맷:
  {"session_id", "group", "arrival_s"(trace 만), "system"(공유 prefix), "system_tokens",
   "turns": [{"user", "user_tokens", "max_tokens", "think_s", "assistant_ref"}]}

예시:
  # single-turn, 입력 2048 중 50% 가 8개 그룹 중 하나와 공유되는 prefix
  python gen_workload.py synth --out wl/st_p50 --tokenizer Qwen/Qwen3-8B \\
      --num-sessions 500 --turns 1 --input-len 2048 --output-len 256 \\
      --prefix-ratio 0.5 --num-prefix-groups 8

  # multi-turn 5턴, 턴당 신규 입력 256, think time 평균 2초
  python gen_workload.py synth --out wl/mt5 --tokenizer Qwen/Qwen3-8B \\
      --num-sessions 100 --turns 5 --input-len 1024 --turn-input-len 256 \\
      --output-len 256 --prefix-ratio 0.3 --think-time 2

  # Mooncake 트레이스 재생용 변환
  python gen_workload.py mooncake --trace conversation_trace.jsonl --out wl/mooncake \\
      --tokenizer Qwen/Qwen3-8B --block-size 512 --max-requests 2000 --time-scale 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import random

from common import Tok, read_jsonl, write_json, write_jsonl


def _jitter(rng: random.Random, base: int, frac: float) -> int:
    if frac <= 0:
        return max(1, base)
    return max(1, int(round(base * rng.uniform(1 - frac, 1 + frac))))


def _parse_turns(spec: str) -> tuple[int, int]:
    if "-" in str(spec):
        a, b = str(spec).split("-")
        return int(a), int(b)
    return int(spec), int(spec)


def compute_targets(sessions: list[dict]) -> dict:
    """무한 캐시 + 순차 도착 가정에서의 이론상 prefix hit rate (상한).

    colocated       : 이전 턴의 prompt + output 전부 캐시에 있다고 가정 (일반 vLLM)
    pd_prefill_only : P 노드 캐시에는 이전 턴 prompt 만 있고, D 가 만든 output KV 는 없다고 가정
                      (vLLM P/D 기본 동작에 가까움. 다음 턴에서 이전 output 은 P 가 다시 계산)
    """
    seen_groups: set = set()
    tot = hit_co = hit_pd = 0
    n_req = 0
    for s in sessions:
        sys_t = s.get("system_tokens", 0)
        prev_prompt = 0
        prev_out = 0
        for k, t in enumerate(s["turns"]):
            if k == 0:
                prompt = sys_t + t["user_tokens"]
                cached = sys_t if (s.get("group") in seen_groups and sys_t > 0) else 0
                c_co = c_pd = cached
            else:
                prompt = prev_prompt + prev_out + t["user_tokens"]
                c_co = prev_prompt + prev_out
                c_pd = prev_prompt
            tot += prompt
            hit_co += c_co
            hit_pd += c_pd
            n_req += 1
            prev_prompt, prev_out = prompt, t["max_tokens"]
        seen_groups.add(s.get("group"))
    return {
        "num_requests": n_req,
        "total_prompt_tokens_est": tot,
        "target_hit_rate_colocated": hit_co / tot if tot else 0.0,
        "target_hit_rate_pd_prefill_only": hit_pd / tot if tot else 0.0,
    }


def gen_synth(a: argparse.Namespace) -> tuple[list[dict], dict]:
    rng = random.Random(a.seed)
    tok = Tok(a.tokenizer)
    G = max(1, a.num_prefix_groups)
    prefix_len = int(round(a.prefix_ratio * a.input_len))
    if prefix_len >= a.input_len:
        prefix_len = a.input_len - 1
    prefixes = [
        tok.text_of_len(rng, prefix_len, lead=f"[shared context group {g}]") if prefix_len > 0 else ""
        for g in range(G)
    ]
    tmin, tmax = _parse_turns(a.turns)
    sessions = []
    for i in range(a.num_sessions):
        g = i % G
        nturns = rng.randint(tmin, tmax)
        turns = []
        for k in range(nturns):
            if k == 0:
                ulen = max(1, _jitter(rng, a.input_len, a.len_jitter) - prefix_len)
            else:
                ulen = _jitter(rng, a.turn_input_len, a.len_jitter)
            olen = _jitter(rng, a.output_len, a.len_jitter)
            # 고유 마커로 시작 → 공유 prefix 이후로는 캐시 공유가 끊기도록
            user = tok.text_of_len(rng, ulen, lead=f"[session {i} turn {k} id {rng.getrandbits(32):08x}]")
            think = rng.expovariate(1.0 / a.think_time) if a.think_time > 0 else 0.0
            turns.append(
                {
                    "user": user,
                    "user_tokens": tok.count(user) if tok.exact else ulen,
                    "max_tokens": olen,
                    "think_s": round(think, 3),
                    # history-mode=fixed 에서 assistant 응답 대신 사용 (재현성 우선 실험용)
                    "assistant_ref": tok.text_of_len(rng, olen),
                }
            )
        sessions.append(
            {
                "session_id": f"s{i:05d}",
                "group": g,
                "system": prefixes[g],
                "system_tokens": tok.count(prefixes[g]) if (tok.exact and prefixes[g]) else prefix_len,
                "turns": turns,
            }
        )
    meta = {
        "generator": "synth",
        "params": vars(a),
        "tokenizer_exact": tok.exact,
        "num_sessions": len(sessions),
        **compute_targets(sessions),
    }
    return sessions, meta


def gen_mooncake(a: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = read_jsonl(a.trace)
    if a.max_requests:
        rows = rows[: a.max_requests]
    tok = Tok(a.tokenizer)
    bs = a.block_size
    block_text: dict = {}

    def full_block(h) -> str:
        if h not in block_text:
            block_text[h] = tok.text_of_len(random.Random(f"{a.seed}-{h}"), bs, lead=f"<blk {h}>")
        return block_text[h]

    t_first = rows[0].get("timestamp", 0) if rows else 0
    seen: set = set()
    sessions = []
    tot = hit = 0
    rng = random.Random(a.seed)
    for i, row in enumerate(rows):
        L = int(row["input_length"])
        hids = row.get("hash_ids") or []
        parts, remaining = [], L
        cached = 0
        full_ids = []
        for h in hids:
            n = min(bs, remaining)
            if n <= 0:
                break
            if n == bs:
                parts.append(full_block(h))
                full_ids.append(h)
                if h in seen:
                    cached += bs
            else:  # 마지막 부분 블록: 재사용 대상 아님
                parts.append(tok.text_of_len(random.Random(f"{a.seed}-{h}-p{n}"), n, lead=f"<blk {h}p>"))
            remaining -= n
        if remaining > 0:
            parts.append(tok.text_of_len(rng, remaining, lead=f"<tail {i}>"))
        seen.update(full_ids)  # 꽉 찬 블록만 캐시 재사용 대상
        tot += L
        hit += min(cached, L)
        out_len = max(1, int(row.get("output_length", 1)))
        sessions.append(
            {
                "session_id": f"m{i:06d}",
                "group": None,
                "arrival_s": (row.get("timestamp", 0) - t_first) / 1000.0 * a.time_scale,
                "system": "",
                "system_tokens": 0,
                "turns": [
                    {
                        "user": " ".join(parts),
                        "user_tokens": L,
                        "max_tokens": out_len,
                        "think_s": 0.0,
                        "assistant_ref": "",
                    }
                ],
            }
        )
    rate = hit / tot if tot else 0.0
    meta = {
        "generator": "mooncake",
        "params": vars(a),
        "tokenizer_exact": tok.exact,
        "num_sessions": len(sessions),
        "num_requests": len(sessions),
        "total_prompt_tokens_est": tot,
        "target_hit_rate_colocated": rate,
        "target_hit_rate_pd_prefill_only": rate,  # single-turn 트레이스라 동일
        "trace_duration_s": sessions[-1]["arrival_s"] if sessions else 0,
    }
    return sessions, meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("synth", help="합성 워크로드")
    s.add_argument("--out", required=True, help="출력 경로 prefix (확장자 제외)")
    s.add_argument("--tokenizer", default=None, help="HF tokenizer id/경로 (없으면 단어 수 근사)")
    s.add_argument("--num-sessions", type=int, default=200)
    s.add_argument("--turns", default="1", help="턴 수. 고정 '5' 또는 범위 '3-8'")
    s.add_argument("--input-len", type=int, default=2048, help="첫 턴 입력 길이 (공유 prefix 포함)")
    s.add_argument("--turn-input-len", type=int, default=256, help="2번째 턴부터 매 턴 새로 추가되는 user 토큰")
    s.add_argument("--output-len", type=int, default=256)
    s.add_argument("--prefix-ratio", type=float, default=0.0, help="첫 턴 입력 중 공유 prefix 비율 (0~1)")
    s.add_argument("--num-prefix-groups", type=int, default=1, help="서로 다른 공유 prefix 개수")
    s.add_argument("--think-time", type=float, default=0.0, help="턴 사이 대기 평균(초, 지수분포)")
    s.add_argument("--len-jitter", type=float, default=0.0, help="길이 ±비율 균등 변동 (예: 0.2)")
    s.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("mooncake", help="Mooncake 형식 트레이스 변환")
    m.add_argument("--trace", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--tokenizer", default=None)
    m.add_argument("--block-size", type=int, default=512, help="hash_id 하나가 나타내는 토큰 수 (Mooncake 공개 트레이스=512)")
    m.add_argument("--max-requests", type=int, default=0)
    m.add_argument("--time-scale", type=float, default=1.0, help="도착 시각 배율 (0.5 = 2배 빠르게)")
    m.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> dict:
    a = build_parser().parse_args(argv)
    sessions, meta = gen_synth(a) if a.mode == "synth" else gen_mooncake(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    write_jsonl(a.out + ".jsonl", sessions)
    write_json(a.out + ".meta.json", meta)
    print(
        json.dumps(
            {k: meta[k] for k in ("num_sessions", "num_requests", "target_hit_rate_colocated", "target_hit_rate_pd_prefill_only")},
            indent=2,
        )
    )
    print(f"-> {a.out}.jsonl")
    return meta


if __name__ == "__main__":
    main()
