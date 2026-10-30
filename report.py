"""결과 집계 리포트 (Markdown 표 + 그래프).

  python report.py --perf "results/perf/*.json" --acc "results/acc/*.summary.json" \\
      --cmp "results/cmp/*.json" --out results/report

같은 (label, workload, qps) 의 반복 실행은 평균 ± 표준편차로 묶는다.
matplotlib 이 있으면 workload 별 TTFT / TPOT vs QPS 그래프를 그린다.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def _load(pattern: str) -> list[dict]:
    out = []
    for fn in sorted(glob.glob(pattern)) if pattern else []:
        with open(fn, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _ms(x):
    return "-" if x is None else f"{x:.1f}"


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def _fmt(ms):
    m, sd = ms
    if m is None:
        return "-"
    return f"{m:.1f}" if not sd else f"{m:.1f}±{sd:.1f}"


def aggregate_perf(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        s = r["summary"]
        groups[(s["workload"], s["label"], s["qps"])].append(s)
    rows = []
    for (wl, label, qps), ss in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        def g(path):
            vals = []
            for s in ss:
                v = s
                for p in path:
                    v = v.get(p) if isinstance(v, dict) else None
                vals.append(v)
            return _mean_sd(vals)

        def srv_hit(s):
            ms = (s.get("cache") or {}).get("measured_server") or {}
            parts = []
            for u, rates in ms.items():
                for name, v in (rates or {}).items():
                    if isinstance(v, dict) and "hit_rate" in v and "external" not in name:
                        parts.append(f"{v['hit_rate']:.2f}")
            return "/".join(parts) if parts else "-"

        c0 = ss[0].get("cache") or {}
        turns = sorted({k for s in ss for k in s.get("ttft_ms_by_turn", {})}, key=int)
        rows.append({
            "workload": wl, "label": label, "qps": qps, "runs": len(ss),
            "ttft_p50": g(["ttft_ms", "p50"]), "ttft_p99": g(["ttft_ms", "p99"]),
            "tpot_p50": g(["tpot_ms", "p50"]), "tpot_p99": g(["tpot_ms", "p99"]),
            "itl_p99": g(["itl_ms", "p99"]),
            "out_tps": g(["output_token_throughput"]), "req_s": g(["request_throughput"]),
            "errors": sum(s["num_errors"] for s in ss),
            "target_co": c0.get("target_hit_rate_colocated"), "target_pd": c0.get("target_hit_rate_pd_prefill_only"),
            "usage_hit": _mean_sd([(s.get("cache") or {}).get("measured_usage_hit_rate") for s in ss]),
            "server_hit": srv_hit(ss[0]),
            "ttft_by_turn": {t: _mean_sd([s["ttft_ms_by_turn"].get(t, {}).get("p50") for s in ss]) for t in turns},
            "attain": _mean_sd([(s.get("goodput") or {}).get("attainment") for s in ss]),
        })
    return rows


def perf_markdown(rows: list[dict]) -> str:
    out = ["## 속도 (TTFT / TPOT, ms)\n"]
    by_wl = defaultdict(list)
    for r in rows:
        by_wl[r["workload"]].append(r)
    for wl, rs in by_wl.items():
        t = rs[0]
        out.append(f"### {wl}\n")
        out.append(f"이론상 hit rate: colocated {t['target_co']:.3f}, PD(P only) {t['target_pd']:.3f}\n"
                   if t["target_co"] is not None else "")
        out.append("| label | qps | runs | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 | out tok/s | hit(usage) | hit(server) | SLO | err |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            uh = r["usage_hit"][0]
            at = r["attain"][0]
            out.append(
                f"| {r['label']} | {r['qps']} | {r['runs']} | {_fmt(r['ttft_p50'])} | {_fmt(r['ttft_p99'])} | "
                f"{_fmt(r['tpot_p50'])} | {_fmt(r['tpot_p99'])} | {_fmt(r['itl_p99'])} | {_fmt(r['out_tps'])} | "
                f"{'-' if uh is None else f'{uh:.3f}'} | {r['server_hit']} | {'-' if at is None else f'{at:.3f}'} | {r['errors']} |"
            )
        if any(len(r["ttft_by_turn"]) > 1 for r in rs):
            turns = sorted({k for r in rs for k in r["ttft_by_turn"]}, key=int)
            out.append("\n턴별 TTFT p50 (ms)\n")
            out.append("| label | qps | " + " | ".join(f"turn {t}" for t in turns) + " |")
            out.append("|---|---|" + "---|" * len(turns))
            for r in rs:
                out.append(f"| {r['label']} | {r['qps']} | " + " | ".join(_fmt(r['ttft_by_turn'].get(t, (None, None))) for t in turns) + " |")
        out.append("")
    return "\n".join(out)


def acc_markdown(accs: list[dict], cmps: list[dict]) -> str:
    out = []
    if accs:
        out.append("## 정확성 (문제 풀이 점수)\n")
        out.append("| task | mode | label | acc | ±stderr | n | by turn | err |")
        out.append("|---|---|---|---|---|---|---|---|")
        for s in sorted(accs, key=lambda s: (s["task"], s["mode"], s["label"])):
            a = s["accuracy"]
            bt = " ".join(f"t{k}:{v['acc']:.3f}" for k, v in s.get("accuracy_by_turn", {}).items() if v.get("n")) \
                if len(s.get("accuracy_by_turn", {})) > 1 else ""
            out.append(f"| {s['task']} | {s['mode']} | {s['label']} | {a.get('acc', 0):.4f} | {a.get('stderr', 0):.4f} | "
                       f"{a.get('n')} | {bt} | {s['num_errors']} |")
        out.append("")
    if cmps:
        out.append("## 베이스라인 대비 출력 일치\n")
        out.append("| A | B | n | acc A | acc B | McNemar p | exact | first tok | exact by turn (comparable) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for c in cmps:
            al = c["all"]
            if not al.get("n"):
                continue
            bt = " ".join(f"t{k}:{v['exact_match_rate']:.3f}" for k, v in c["comparable_by_turn"].items() if v.get("n")) \
                if len(c["comparable_by_turn"]) > 1 else ""
            out.append(f"| {os.path.basename(c['a'])} | {os.path.basename(c['b'])} | {al['n']} | {al['acc_a']:.4f} | "
                       f"{al['acc_b']:.4f} | {al['mcnemar_p']:.3f} | {al['exact_match_rate']:.3f} | "
                       f"{al['first_token_match_rate']:.3f} | {bt} |")
        out.append("")
    return "\n".join(out)


def plot(rows: list[dict], outdir: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    by_wl = defaultdict(list)
    for r in rows:
        by_wl[r["workload"]].append(r)
    for wl, rs in by_wl.items():
        qs = sorted({r["qps"] for r in rs})
        if len(qs) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label in sorted({r["label"] for r in rs}):
            lr = sorted([r for r in rs if r["label"] == label], key=lambda r: r["qps"])
            x = [r["qps"] for r in lr]
            for ax, key in ((axes[0], "ttft"), (axes[1], "tpot")):
                ax.plot(x, [r[f"{key}_p50"][0] for r in lr], marker="o", label=f"{label} p50")
                ax.plot(x, [r[f"{key}_p99"][0] for r in lr], marker="x", linestyle="--", label=f"{label} p99")
        for ax, name in ((axes[0], "TTFT (ms)"), (axes[1], "TPOT (ms)")):
            ax.set_xlabel("QPS")
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(wl)
        fig.tight_layout()
        fn = os.path.join(outdir, f"{wl}_latency_vs_qps.png")
        fig.savefig(fn, dpi=120)
        plt.close(fig)
        files.append(fn)
    return files


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perf", default="", help="run_perf 결과 glob")
    p.add_argument("--acc", default="", help="run_accuracy *.summary.json glob")
    p.add_argument("--cmp", default="", help="compare_outputs 결과 glob")
    p.add_argument("--out", required=True, help="출력 디렉토리")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = aggregate_perf(_load(a.perf))
    md = ["# PD Benchmark Report\n"]
    if rows:
        md.append(perf_markdown(rows))
    md.append(acc_markdown(_load(a.acc), _load(a.cmp)))
    figs = plot(rows, a.out)
    if figs:
        md.append("## 그래프\n")
        md += [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figs]
    fn = os.path.join(a.out, "report.md")
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"-> {fn}")
    return fn


if __name__ == "__main__":
    main()
