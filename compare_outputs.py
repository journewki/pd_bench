"""베이스라인 vs PD 출력 일치 비교 (run_accuracy.py 의 records 두 개를 비교).

점수(accuracy)가 비슷해도 출력이 미묘하게 다를 수 있고, 반대로 KV 전송 버그가 있으면
특정 조건(긴 context, 2턴 이후 등)에서만 출력이 틀어진다. 그래서 다음을 함께 본다.

  - accuracy A/B 와 McNemar exact test (두 시스템의 정답 여부 차이가 우연인지)
  - exact match rate          : 출력 문자열 완전 일치 비율 (greedy 기준)
  - first token match rate    : 첫 토큰 일치 (PD 에서 첫 토큰 경로는 P→KV 전송에 가장 민감)
  - first divergence position : 처음 갈라지는 토큰(또는 문자) 위치 분포
  - multi-turn 은 '이전 턴이 모두 일치한 턴'만 비교 가능(comparable)으로 집계
    (앞 턴 출력이 다르면 다음 턴 prompt 자체가 달라지므로)

주의: greedy 라도 배치 구성/커널 차이로 긴 출력 뒷부분이 조금 달라지는 건 흔하다.
      같은 베이스라인을 두 번 돌려서(A vs A') 자연 변동 수준을 먼저 재두는 것을 권장.

예시:
  python compare_outputs.py --a results/acc/vllm_gsm8k_single.records.jsonl \\
      --b results/acc/pd_gsm8k_single.records.jsonl --tokenizer Qwen/Qwen3-8B \\
      --out results/cmp/gsm8k_single.json
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from math import comb

from common import Tok, percentile, read_jsonl, write_json


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, float(Fraction(2 * tail, 2 ** n)))


def first_divergence(x: list, y: list) -> int | None:
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    if len(x) != len(y):
        return min(len(x), len(y))
    return None


def compare(a_rows: list[dict], b_rows: list[dict], tok: Tok | None) -> dict:
    A = {(r["id"], r["turn"]): r for r in a_rows}
    B = {(r["id"], r["turn"]): r for r in b_rows}
    keys = sorted(set(A) & set(B), key=lambda k: (A[k]["unit"], k[1]))

    def toks(text: str) -> list:
        if tok and tok.exact:
            return tok.encode(text)
        return list(text)  # 토크나이저 없으면 문자 단위

    # unit 별로 이전 턴 완전 일치 여부를 추적
    unit_ok: dict[str, bool] = {}
    rows = []
    for k in keys:
        ra, rb = A[k], B[k]
        unit = ra["unit"]
        comparable = unit_ok.get(unit, True)
        ta, tb = toks(ra["output"]), toks(rb["output"])
        div = first_divergence(ta, tb)
        exact = div is None
        rows.append({
            "key": k, "turn": k[1], "comparable": comparable, "exact": exact,
            "first_token_match": bool(ta[:1] == tb[:1]), "div": div,
            "a_correct": bool(ra.get("correct")), "b_correct": bool(rb.get("correct")),
            "a_out": ra["output"], "b_out": rb["output"],
        })
        unit_ok[unit] = comparable and exact

    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            return {"n": 0}
        divs = [r["div"] for r in rs if r["div"] is not None]
        b_ = sum(1 for r in rs if r["a_correct"] and not r["b_correct"])
        c_ = sum(1 for r in rs if not r["a_correct"] and r["b_correct"])
        return {
            "n": n,
            "acc_a": sum(r["a_correct"] for r in rs) / n,
            "acc_b": sum(r["b_correct"] for r in rs) / n,
            "a_only_correct": b_,
            "b_only_correct": c_,
            "mcnemar_p": mcnemar_exact(b_, c_),
            "exact_match_rate": sum(r["exact"] for r in rs) / n,
            "first_token_match_rate": sum(r["first_token_match"] for r in rs) / n,
            "divergence_pos_p50": percentile(divs, 50),
            "divergence_pos_p10": percentile(divs, 10),
            "num_diverged": len(divs),
        }

    by_turn: dict[int, list] = {}
    for r in rows:
        by_turn.setdefault(r["turn"], []).append(r)
    comp = [r for r in rows if r["comparable"]]
    worst = sorted([r for r in comp if r["div"] is not None], key=lambda r: r["div"])[:5]
    return {
        "unit": "token" if (tok and tok.exact) else "char",
        "num_common": len(keys),
        "only_in_a": len(set(A) - set(B)),
        "only_in_b": len(set(B) - set(A)),
        "all": agg(rows),
        "comparable": agg(comp),
        "comparable_by_turn": {str(t): agg([r for r in v if r["comparable"]]) for t, v in sorted(by_turn.items())},
        "earliest_divergences": [
            {"key": list(r["key"]), "div": r["div"], "a": r["a_out"][:200], "b": r["b_out"][:200]} for r in worst
        ],
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="베이스라인 records.jsonl")
    p.add_argument("--b", required=True, help="비교 대상(PD) records.jsonl")
    p.add_argument("--tokenizer", default=None, help="토큰 단위 비교용 (없으면 문자 단위)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    tok = Tok(a.tokenizer) if a.tokenizer else None
    res = compare(read_jsonl(a.a), read_jsonl(a.b), tok)
    res["a"], res["b"] = a.a, a.b
    al, cp = res["all"], res["comparable"]
    print(f"\n=== compare ({res['unit']}) n={res['num_common']} ===")
    if al.get("n"):
        print(f"accuracy A/B      : {al['acc_a']:.4f} / {al['acc_b']:.4f}   McNemar p={al['mcnemar_p']:.4f} "
              f"(A만 정답 {al['a_only_correct']}, B만 정답 {al['b_only_correct']})")
        print(f"exact match       : {al['exact_match_rate']:.4f}   first token match {al['first_token_match_rate']:.4f}")
    if cp.get("n") and cp["n"] != al.get("n"):
        print(f"comparable only   : n={cp['n']} exact {cp['exact_match_rate']:.4f}")
    if len(res["comparable_by_turn"]) > 1:
        print("exact by turn     : " + "  ".join(
            f"t{k}:{v['exact_match_rate']:.3f}(n={v['n']})" for k, v in res["comparable_by_turn"].items() if v.get("n")))
    if a.out:
        write_json(a.out, res)
        print(f"-> {a.out}")
    return res


if __name__ == "__main__":
    main()
