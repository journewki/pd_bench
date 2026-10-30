"""실험 매트릭스 일괄 실행: (target × workload × qps × repeat) 속도 + 정확성 + 비교 + 리포트.

  python run_sweep.py --config configs/sweep_example.yaml
  python run_sweep.py --config configs/sweep_example.yaml --targets pd          # 특정 타겟만 (서버를 번갈아 띄울 때)
  python run_sweep.py --config configs/sweep_example.yaml --stages perf,report  # 일부 단계만

결과 디렉토리 구조:
  <out>/workloads/*.jsonl        생성된 워크로드 (이미 있으면 재사용 → 타겟 간 완전히 같은 입력)
  <out>/perf/<target>__<wl>__q<qps>__r<i>.json
  <out>/acc/<target>__<task>_<mode>.summary.json / .records.jsonl
  <out>/cmp/<baseline>_vs_<target>__<task>_<mode>.json
  <out>/report/report.md
"""
from __future__ import annotations

import argparse
import json
import os

import compare_outputs
import gen_workload
import report
import run_accuracy
import run_perf


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def to_argv(d: dict) -> list[str]:
    argv = []
    for k, v in d.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, list):
            for x in v:
                argv += [flag, str(x)]
        elif v is not None:
            argv += [flag, str(v)]
    return argv


def qtag(q) -> str:
    return "inf" if str(q) in ("inf", "Infinity") else str(q).replace(".", "p")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--targets", default="", help="쉼표로 구분된 타겟 이름 (기본: 전부)")
    p.add_argument("--stages", default="gen,perf,acc,cmp,report")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    stages = set(a.stages.split(","))
    out = cfg.get("out_dir", "results")
    targets = cfg["targets"]
    if a.targets:
        want = set(a.targets.split(","))
        targets = [t for t in targets if t["name"] in want]
    model, api = cfg["model"], cfg.get("api", "completions")
    tokenizer = cfg.get("tokenizer")

    # 1) 워크로드 생성
    wl_paths = {}
    for w in cfg.get("workloads", []):
        path = os.path.join(out, "workloads", w["name"])
        wl_paths[w["name"]] = path
        if "gen" in stages and not os.path.exists(path + ".jsonl"):
            args = dict(w.get("args", {}))
            mode = args.pop("mode", "synth")
            if tokenizer and "tokenizer" not in args:
                args["tokenizer"] = tokenizer
            print(f"\n[gen] {w['name']}")
            gen_workload.main([mode, "--out", path] + to_argv(args))

    # 2) 속도
    if "perf" in stages:
        perf = cfg.get("perf", {})
        for t in targets:
            for w in cfg.get("workloads", []):
                qps_list = w.get("qps", perf.get("qps", ["inf"]))
                for q in qps_list:
                    for r in range(perf.get("repeats", 1)):
                        fn = os.path.join(out, "perf", f"{t['name']}__{w['name']}__q{qtag(q)}__r{r}.json")
                        if os.path.exists(fn) and not cfg.get("overwrite", False):
                            print(f"[perf] skip (exists) {fn}")
                            continue
                        args = {
                            "workload": wl_paths[w["name"]], "url": t["url"], "model": model, "api": api,
                            "qps": q, "metrics_url": t.get("metrics_urls", []), "label": t["name"],
                            "out": fn, "seed": r, "reset_cache": perf.get("reset_cache", False),
                            "warmup": perf.get("warmup", 8),
                            "max_concurrency": w.get("max_concurrency", perf.get("max_concurrency", 0)),
                            "history_mode": perf.get("history_mode", "generated"),
                            "think_scale": perf.get("think_scale", 1.0),
                            "slo_ttft_ms": perf.get("slo_ttft_ms", 0), "slo_tpot_ms": perf.get("slo_tpot_ms", 0),
                            "use_trace_timestamps": w.get("use_trace_timestamps", False),
                        }
                        print(f"\n[perf] {t['name']} {w['name']} qps={q} rep={r}")
                        run_perf.main(to_argv(args))

    # 3) 정확성
    acc_cfgs = cfg.get("accuracy", [])
    if "acc" in stages:
        for t in targets:
            for ac in acc_cfgs:
                ac = dict(ac)
                tag = f"{ac['task']}_{ac.get('mode', 'single')}"
                fn = os.path.join(out, "acc", f"{t['name']}__{tag}")
                if os.path.exists(fn + ".summary.json") and not cfg.get("overwrite", False):
                    print(f"[acc] skip (exists) {fn}")
                    continue
                if ac["task"] == "niah" and tokenizer and "tokenizer" not in ac:
                    ac["tokenizer"] = tokenizer
                args = {"url": t["url"], "model": model, "api": api, "label": t["name"], "out": fn, **ac}
                print(f"\n[acc] {t['name']} {tag}")
                run_accuracy.main(to_argv(args))

    # 4) 베이스라인 대비 출력 비교 (전체 타겟 기준, --targets 필터와 무관)
    if "cmp" in stages:
        base = cfg.get("baseline", cfg["targets"][0]["name"])
        for t in cfg["targets"]:
            if t["name"] == base:
                continue
            for ac in acc_cfgs:
                tag = f"{ac['task']}_{ac.get('mode', 'single')}"
                fa = os.path.join(out, "acc", f"{base}__{tag}.records.jsonl")
                fb = os.path.join(out, "acc", f"{t['name']}__{tag}.records.jsonl")
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    continue
                print(f"\n[cmp] {base} vs {t['name']} {tag}")
                argv_c = ["--a", fa, "--b", fb, "--out", os.path.join(out, "cmp", f"{base}_vs_{t['name']}__{tag}.json")]
                if tokenizer:
                    argv_c += ["--tokenizer", tokenizer]
                compare_outputs.main(argv_c)

    # 5) 리포트
    if "report" in stages:
        report.main([
            "--perf", os.path.join(out, "perf", "*.json"),
            "--acc", os.path.join(out, "acc", "*.summary.json"),
            "--cmp", os.path.join(out, "cmp", "*.json"),
            "--out", os.path.join(out, "report"),
        ])


if __name__ == "__main__":
    main()
