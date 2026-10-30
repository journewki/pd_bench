"""여러 시스템의 capacity 결과 비교 (예: vLLM 베이스라인 vs PD).

  python compare_capacity.py results/capacity/vllm results/capacity/pd --out results/capacity/compare

각 디렉토리는 run_capacity.py 의 --out 결과. 같은 워크로드/SLO 로 측정한 것끼리만 비교해야 한다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from run_capacity import fmt, per_level_table, plot_runs


def load(d: str):
    with open(os.path.join(d, "capacity.json"), encoding="utf-8") as f:
        summary = json.load(f)
    results = []
    for fn in sorted(glob.glob(os.path.join(d, "levels", "*.json"))):
        with open(fn, encoding="utf-8") as f:
            results.append(json.load(f))
    return summary["label"], results, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs = [load(d) for d in a.dirs]
    s0 = runs[0][2]
    P = s0["slo"]["percentile"]
    warn = []
    for _, _, s in runs[1:]:
        if s["workload_desc"] != s0["workload_desc"] or s["slo"] != s0["slo"]:
            warn.append(f"'{s['label']}' 의 워크로드/SLO 가 '{s0['label']}' 와 다름 → 직접 비교 주의")

    lines = ["# Capacity 비교\n",
             f"- SLO: TTFT p{P} ≤ {s0['slo']['ttft_ms']} ms, TPOT p{P} ≤ {s0['slo']['tpot_ms']} ms",
             f"- 워크로드: {s0['workload_desc']}"]
    lines += [f"- ⚠ {w}" for w in warn]
    thr = int(s0.get("stable_pass_rate", 0.8) * 100)
    lines += ["", f"| 시스템 | GPU 수 | 보수적 수용량 (통과확률≥{thr}%) | 50% 지점 | 탐색 결과 (중앙, 범위) | 사용자/GPU (보수적) | 그 지점 tok/s/gpu | 그 지점 TTFT p | 그 지점 TPOT p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for label, results, s in runs:
        ngpu = s["params"]["num_gpus"]
        cap = s.get("capacity_stable", s["capacity_median"])
        rows = {r["users"]: r for r in per_level_table(results, P)}
        near = rows.get(cap)
        lines.append(
            f"| {label} | {ngpu} | {cap} | {s.get('capacity_p50')} | {s['capacity_median']} ({s['capacity_min']}–{s['capacity_max']}) | "
            f"{fmt(cap / ngpu if cap else None, 1)} | {fmt(near and near['tps_gpu'])} | "
            f"{fmt(near and near['ttft_p'])} | {fmt(near and near['tpot_p'], 1)} |")
    figs = plot_runs([(l, r, s) for l, r, s in runs], a.out, P, s0["slo"]["ttft_ms"], s0["slo"]["tpot_ms"])
    if figs:
        lines += ["", "## 그래프", ""] + [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figs]
    fn = os.path.join(a.out, "capacity_compare.md")
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len(runs) + len(warn)]))
    print(f"-> {fn}")


if __name__ == "__main__":
    main()
