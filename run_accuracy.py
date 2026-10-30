"""정확성 벤치마크 러너 (문제 풀이 점수).

기존 벤치마크들의 방식을 따른다:
  - gsm8k : decode-heavy. lm-eval-harness 와 같은 few-shot 포맷, '####' 뒤 숫자로 채점
            (strict = '#### N' 형식, flexible = 마지막 숫자).
            few-shot 예제가 모든 요청에 공통 prefix → prefix cache 경로도 같이 검증된다.
  - niah  : prefill-heavy (RULER 의 needle-in-a-haystack 계열을 로컬에서 합성, 다운로드 불필요).
            긴 문서(context) 하나에 needle(비밀 코드) 여러 개를 숨기고 질문한다.
            같은 문서에 여러 질문 → 문서가 공유 prefix 가 된다.

모드:
  single : 질문마다 독립 요청 (같은 prefix 를 여러 요청이 공유 = SCBench 의 multi-request 모드와 유사)
  multi  : 하나의 대화 안에서 여러 질문을 연속으로 (SCBench 의 multi-turn 모드와 유사).
           이전 턴 응답(decode 에서 생성된 토큰)이 다음 턴 prompt 에 들어가므로,
           PD 에서 'P 노드가 이전 응답을 다시 prefill' 하는 경로가 검증된다.

결과: <out>.summary.json, <out>.records.jsonl
compare_outputs.py 로 베이스라인 vs PD 의 records 를 비교하면 출력 일치율까지 볼 수 있다.

예시:
  python run_accuracy.py --task gsm8k --mode single --url http://localhost:8000 --model Qwen/Qwen3-8B \\
      --num-samples 500 --concurrency 64 --label vllm --out results/acc/vllm_gsm8k_single
  python run_accuracy.py --task niah --mode multi --url http://localhost:9000 --model Qwen/Qwen3-8B \\
      --tokenizer Qwen/Qwen3-8B --context-len 16000 --num-contexts 20 --needles-per-context 6 \\
      --label pd --out results/acc/pd_niah_multi
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re

from common import Tok, WORDS, build_payload, make_session, stream_request, write_json, write_jsonl

# ----------------------------------------------------------------------------- GSM8K
_NUM = r"-?[\d,]*\.?\d+"


def load_gsm8k(path: str | None) -> tuple[list[dict], list[dict]]:
    """path 가 있으면 {path}/train.jsonl, {path}/test.jsonl (필드: question, answer) 사용.
    없으면 HF datasets 로 openai/gsm8k 다운로드."""
    if path:
        def rd(fn):
            with open(os.path.join(path, fn), encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]
        return rd("train.jsonl"), rd("test.jsonl")
    from datasets import load_dataset  # noqa: WPS433

    ds = load_dataset("openai/gsm8k", "main")
    return list(ds["train"]), list(ds["test"])


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_extract(text: str) -> tuple[str | None, str | None]:
    m = re.search(r"####\s*(" + _NUM + ")", text)
    strict = m.group(1).replace(",", "") if m else None
    nums = re.findall(_NUM, text)
    flexible = nums[-1].replace(",", "") if nums else None
    return strict, flexible


def num_eq(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return math.isclose(float(a.rstrip(".")), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False


def build_gsm8k_units(a) -> tuple[str, list[dict]]:
    train, test = load_gsm8k(a.data_path)
    shots = train[: a.num_fewshot]
    fewshot = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    rng = random.Random(a.seed)
    idx = list(range(len(test)))
    rng.shuffle(idx)
    qs = [test[i] for i in idx[: a.num_samples]]
    if a.api == "chat":
        system = (
            "Solve the grade-school math problem step by step. "
            "End your answer with '#### <number>'.\n\nExamples:\n\n" + fewshot
        )
    else:
        system = fewshot
    items = [{"id": f"gsm8k-{idx[i]}", "question": q["question"], "gold": gsm8k_gold(q["answer"]),
              "gold_full": q["answer"]} for i, q in enumerate(qs)]
    if a.mode == "single":
        units = [{"id": it["id"], "items": [it]} for it in items]
    else:
        T = a.turns
        units = [{"id": f"conv{c}", "items": items[c * T:(c + 1) * T]} for c in range(len(items) // T)]
    return system, units


# ----------------------------------------------------------------------------- NIAH
def build_niah_units(a) -> tuple[None, list[dict]]:
    tok = Tok(a.tokenizer)
    rng = random.Random(a.seed)
    units = []
    for c in range(a.num_contexts):
        needles = []
        used = set()
        while len(needles) < a.needles_per_context:
            key = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randrange(100)}"
            if key in used:
                continue
            used.add(key)
            needles.append((key, str(rng.randrange(1_000_000, 10_000_000))))
        sentences = [f" The secret code for {k} is {v}. " for k, v in needles]
        header = (
            "Below is a long document. Some sentences contain secret codes. "
            "Read carefully and answer questions about the codes.\n\n<document>\n"
        )
        footer = "\n</document>\n"
        budget = a.context_len - sum(tok.count(s) for s in sentences) - tok.count(header + footer)
        budget = max(budget, 16 * (len(needles) + 1))
        # 필러를 needle 수+1 조각으로 랜덤 분할 → needle 깊이가 문서 전반에 분포
        cuts = sorted(rng.sample(range(1, budget), len(needles)))
        bounds = [0] + cuts + [budget]
        order = list(range(len(needles)))
        rng.shuffle(order)
        body = ""
        for j in range(len(needles) + 1):
            seg = tok.text_of_len(rng, bounds[j + 1] - bounds[j])
            body += seg
            if j < len(needles):
                body += sentences[order[j]]
        context = header + body + footer
        items = []
        qorder = list(range(len(needles)))
        rng.shuffle(qorder)
        for qi in qorder:
            k, v = needles[qi]
            depth = order.index(qi) / max(1, len(needles) - 1)
            items.append({"id": f"niah-c{c}-{k}", "key": k, "gold": v, "depth": round(depth, 3),
                          "question": f"What is the secret code for {k}? Answer with only the number."})
        if a.mode == "single":
            units += [{"id": it["id"], "context": context, "items": [it]} for it in items]
        else:
            units.append({"id": f"ctx{c}", "context": context, "items": items})
    return None, units


# ----------------------------------------------------------------------------- run
def user_msg(task: str, api: str, it: dict, first: bool, unit: dict) -> str:
    if task == "gsm8k":
        return f"Question: {it['question']}" if api == "chat" else f"Question: {it['question']}\nAnswer:"
    q = it["question"]
    if api == "chat":
        return (unit["context"] + "\n" + q) if first else q
    return (unit["context"] + f"\nQuestion: {q}\nAnswer:") if first else f"\nQuestion: {q}\nAnswer:"


def score(task: str, text: str, it: dict) -> dict:
    if task == "gsm8k":
        strict, flex = gsm8k_extract(text)
        return {"pred_strict": strict, "pred_flexible": flex,
                "correct_strict": num_eq(strict, it["gold"]), "correct": num_eq(flex, it["gold"])}
    m = re.search(r"\d{7}", text)
    pred = m.group(0) if m else None
    return {"pred": pred, "correct": pred == it["gold"]}


async def run_unit(http, a, system, unit, records, progress):
    history: list[dict] = []
    for t, it in enumerate(unit["items"]):
        history.append({"role": "user", "content": user_msg(a.task, a.api, it, t == 0, unit)})
        if a.task == "gsm8k":
            stop = ["Question:", "\n\n\n"] if a.api == "completions" else ["Question:"]
            max_tokens = a.max_tokens or 384
        else:
            stop = ["\n", "Question:"] if a.api == "completions" else None
            max_tokens = a.max_tokens or 24
        path, payload = build_payload(a.api, a.model, system, history, max_tokens,
                                      temperature=a.temperature, stop=stop, sep="")
        rec = await stream_request(http, a.url, path, payload, timeout=a.timeout)
        text = rec["text"]
        row = {"id": it["id"], "unit": unit["id"], "turn": t, "task": a.task, "mode": a.mode,
               "gold": it["gold"], "output": text, "ok": rec["ok"], "error": rec["error"],
               "ttft": rec["ttft"], "e2e": rec["e2e"], "prompt_tokens": rec["prompt_tokens"],
               "cached_tokens": rec["cached_tokens"], "completion_tokens": rec["completion_tokens"]}
        if "depth" in it:
            row["depth"] = it["depth"]
        row.update(score(a.task, text, it) if rec["ok"] else {"correct": False})
        records.append(row)
        progress["n"] += 1
        if not rec["ok"]:
            break
        # history 구성: generated(실제 응답, PD 경로 검증) / reference(정답 사용, 턴 간 오류 전파 차단)
        if a.history_mode == "reference":
            reply = it["gold_full"] if a.task == "gsm8k" else it["gold"]
        else:
            reply = text
        if a.api == "completions":
            reply = (" " + reply.strip() + ("\n\n" if a.task == "gsm8k" else "\n"))
        history.append({"role": "assistant", "content": reply})


def summarize(a, records) -> dict:
    def acc(rows, key="correct"):
        n = len(rows)
        if n == 0:
            return {"n": 0}
        p = sum(1 for r in rows if r.get(key)) / n
        return {"n": n, "acc": p, "stderr": math.sqrt(p * (1 - p) / n)}

    by_turn: dict[int, list] = {}
    for r in records:
        by_turn.setdefault(r["turn"], []).append(r)
    s = {
        "label": a.label, "task": a.task, "mode": a.mode, "api": a.api, "url": a.url, "model": a.model,
        "history_mode": a.history_mode, "concurrency": a.concurrency,
        "num_records": len(records), "num_errors": sum(1 for r in records if not r["ok"]),
        "error_samples": [r["error"] for r in records if not r["ok"]][:5],
        "accuracy": acc(records),
        "accuracy_by_turn": {str(k): acc(v) for k, v in sorted(by_turn.items())},
    }
    if a.task == "gsm8k":
        s["accuracy_strict"] = acc(records, "correct_strict")
    else:
        bins: dict[str, list] = {}
        for r in records:
            if "depth" in r:
                b = f"{min(int(r['depth'] * 4), 3) * 25}-{min(int(r['depth'] * 4), 3) * 25 + 25}%"
                bins.setdefault(b, []).append(r)
        s["accuracy_by_depth"] = {k: acc(v) for k, v in sorted(bins.items())}
        s["params"] = {"context_len": a.context_len, "num_contexts": a.num_contexts,
                       "needles_per_context": a.needles_per_context}
    return s


async def run(a) -> dict:
    system, units = build_gsm8k_units(a) if a.task == "gsm8k" else build_niah_units(a)
    records: list[dict] = []
    progress = {"n": 0}
    sem = asyncio.Semaphore(a.concurrency)
    async with make_session() as http:
        async def guarded(u):
            async with sem:
                await run_unit(http, a, system, u, records, progress)
        await asyncio.gather(*(guarded(u) for u in units))
    records.sort(key=lambda r: (r["unit"], r["turn"]))
    return {"summary": summarize(a, records), "records": records}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["gsm8k", "niah"], required=True)
    p.add_argument("--mode", choices=["single", "multi"], default="single")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api", choices=["completions", "chat"], default="completions",
                   help="base 모델/lm-eval 과 맞추려면 completions, instruct 모델 실사용 조건이면 chat")
    p.add_argument("--concurrency", type=int, default=32, help="동시 요청(대화) 수. PD 버그는 부하가 있을 때 잘 드러난다")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0, help="0 = 태스크 기본값")
    p.add_argument("--history-mode", choices=["generated", "reference"], default="generated")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=0)
    # gsm8k
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--turns", type=int, default=4, help="multi 모드에서 대화당 문제 수 (gsm8k)")
    p.add_argument("--data-path", default=None, help="오프라인 환경: train.jsonl/test.jsonl 이 있는 디렉토리")
    # niah
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--context-len", type=int, default=8000)
    p.add_argument("--num-contexts", type=int, default=20)
    p.add_argument("--needles-per-context", type=int, default=5)
    p.add_argument("--label", default="run")
    p.add_argument("--out", required=True, help="출력 prefix (.summary.json / .records.jsonl)")
    return p


def main(argv: list[str] | None = None) -> dict:
    a = build_parser().parse_args(argv)
    res = asyncio.run(run(a))
    s = res["summary"]
    write_json(a.out + ".summary.json", s)
    write_jsonl(a.out + ".records.jsonl", res["records"])
    acc = s["accuracy"]
    print(f"\n=== {s['label']} | {s['task']}/{s['mode']} ===")
    print(f"accuracy : {acc.get('acc', 0):.4f} ± {acc.get('stderr', 0):.4f}  (n={acc.get('n')}, errors={s['num_errors']})")
    if "accuracy_strict" in s:
        print(f"strict   : {s['accuracy_strict'].get('acc', 0):.4f}")
    if len(s["accuracy_by_turn"]) > 1:
        print("by turn  : " + "  ".join(f"t{k}:{v['acc']:.3f}" for k, v in s["accuracy_by_turn"].items()))
    if s.get("accuracy_by_depth"):
        print("by depth : " + "  ".join(f"{k}:{v['acc']:.3f}" for k, v in s["accuracy_by_depth"].items()))
    if s["error_samples"]:
        print(f"errors   : {s['error_samples'][:2]}")
    print(f"-> {a.out}.summary.json")
    return res


if __name__ == "__main__":
    main()
