"""pd-bench 공통 유틸리티.

- Tok: 토큰 길이 제어용 토크나이저 래퍼 (transformers 없으면 단어 수로 근사)
- build_payload / stream_request: OpenAI 호환 스트리밍 클라이언트 (TTFT/ITL/usage 측정)
- scrape_prefix_metrics / reset_prefix_cache: vLLM /metrics, /reset_prefix_cache 연동
- stats: 분위수 통계
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from typing import Any

import aiohttp

# 대부분의 BPE 토크나이저에서 "공백+단어"가 1토큰이 되는 흔한 영어 단어들
WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community "
    "name president team minute idea kid body information back parent face others level "
    "office door health person art war history party result change morning reason "
    "research girl guy moment air teacher force education foot boy age policy music "
    "market sense nation plan college interest death experience effect class control "
    "care field development role effort rate heart show leader light voice wife police "
    "mind price report decision son view relationship town road arm difference value "
    "building action model season society tax director position player record paper "
    "space ground form event official matter center couple site project activity star "
    "table need court oil situation cost industry figure street image phone data picture "
    "practice piece land product doctor wall patient worker news test movie north love "
    "support technology step baby computer type attention film tree source organization "
    "hair window evidence population truth song energy career"
).split()


# --------------------------------------------------------------------------- tokens
class Tok:
    """토큰 수 계산/지정 길이 텍스트 생성.

    tokenizer 이름(HF id 또는 로컬 경로)을 주면 정확히 맞추고,
    없으면 '단어 1개 ≈ 1토큰'으로 근사한다 (mock 서버 테스트용).
    """

    def __init__(self, name: str | None = None):
        self.tok = None
        self.name = name
        if name:
            from transformers import AutoTokenizer  # noqa: WPS433

            self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    @property
    def exact(self) -> bool:
        return self.tok is not None

    def encode(self, text: str) -> list[int]:
        if self.tok is not None:
            return self.tok.encode(text, add_special_tokens=False)
        return text.split()  # type: ignore[return-value]

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def text_of_len(self, rng: random.Random, n: int, lead: str = "") -> str:
        """대략 n 토큰 길이의 텍스트. lead는 앞에 붙는 고유 마커(캐시 공유 차단용)."""
        if n <= 0:
            return ""
        words = [rng.choice(WORDS) for _ in range(int(n * 1.15) + 8)]
        # 12단어마다 마침표를 찍어 문장 형태 유지
        body = " ".join(w + ("." if (i + 1) % 12 == 0 else "") for i, w in enumerate(words))
        text = (lead + " " + body).strip() if lead else body
        if self.tok is not None:
            ids = self.tok.encode(text, add_special_tokens=False)[:n]
            return self.tok.decode(ids)
        return " ".join(text.split()[:n])


# --------------------------------------------------------------------------- stats
def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(xs: list[float], scale: float = 1.0) -> dict[str, Any]:
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p99": percentile(xs, 99),
        "min": min(xs),
        "max": max(xs),
    }


# --------------------------------------------------------------------------- requests
def render_completion_prompt(system: str | None, msgs: list[dict], sep: str = "\n") -> str:
    """/v1/completions 용 프롬프트. role 태그 없이 내용을 그대로 이어 붙인다.

    (태스크 쪽에서 'Question: ...\\nAnswer:' 같은 포맷을 content에 직접 넣는다.)
    """
    parts = [system] if system else []
    parts += [m["content"] for m in msgs]
    return sep.join(parts)


def build_payload(
    api: str,
    model: str,
    system: str | None,
    msgs: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    ignore_eos: bool = False,
    stop: list[str] | None = None,
    sep: str = "\n",
    extra: dict | None = None,
) -> tuple[str, dict]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # vLLM 확장 파라미터
    if stop:
        payload["stop"] = stop
    if extra:
        payload.update(extra)
    if api == "chat":
        messages = ([{"role": "system", "content": system}] if system else []) + list(msgs)
        payload["messages"] = messages
        return "/v1/chat/completions", payload
    payload["prompt"] = render_completion_prompt(system, msgs, sep=sep)
    return "/v1/completions", payload


async def stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    path: str,
    payload: dict,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """스트리밍 요청 1건. TTFT = 첫 토큰 청크 수신 시각 - 요청 시작 시각 (클라이언트 기준).

    PD 구성에서는 proxy 를 거치므로 TTFT 에 P prefill + KV 전송 + D 첫 스텝이 모두 포함된다.
    """
    rec: dict[str, Any] = {
        "ok": False,
        "ttft": None,
        "e2e": None,
        "itl": [],
        "text": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "error": None,
    }
    t0 = time.perf_counter()
    rec["t_start"] = t0
    last = t0
    chunks = 0
    try:
        async with session.post(
            base_url.rstrip("/") + path,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                rec["t_end"] = time.perf_counter()
                return rec
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage:
                    rec["prompt_tokens"] = usage.get("prompt_tokens")
                    rec["completion_tokens"] = usage.get("completion_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    if details.get("cached_tokens") is not None:
                        rec["cached_tokens"] = details["cached_tokens"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                c = choices[0]
                content = c.get("text")
                reasoning = ""
                if content is None:
                    delta = c.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if content or reasoning:
                    now = time.perf_counter()
                    if rec["ttft"] is None:
                        rec["ttft"] = now - t0
                    else:
                        rec["itl"].append(now - last)
                    last = now
                    chunks += 1
                    rec["text"] += content
        rec["t_end"] = time.perf_counter()
        rec["e2e"] = rec["t_end"] - t0
        if rec["completion_tokens"] is None:
            rec["completion_tokens"] = chunks  # usage 미지원 서버: 청크 수로 근사
        rec["ok"] = rec["ttft"] is not None
        if not rec["ok"]:
            rec["error"] = "no tokens received"
    except Exception as e:  # noqa: BLE001
        rec["t_end"] = time.perf_counter()
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def tpot_of(rec: dict) -> float | None:
    """TPOT = (E2E - TTFT) / (출력 토큰 수 - 1)."""
    n = rec.get("completion_tokens") or 0
    if not rec.get("ok") or n <= 1 or rec.get("ttft") is None:
        return None
    return (rec["e2e"] - rec["ttft"]) / (n - 1)


# --------------------------------------------------------------------------- metrics
_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


async def scrape_prefix_metrics(session: aiohttp.ClientSession, urls: list[str]) -> dict[str, dict]:
    """각 서버 /metrics 에서 이름에 'prefix_cache'가 들어간 카운터를 라벨 합산해서 가져온다.

    vLLM 버전에 따라 이름이 다르다:
      vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total (V1)
      vllm:gpu_prefix_cache_*  (구버전), vllm:external_prefix_cache_* (KV connector)
    """
    out: dict[str, dict] = {}
    for u in urls:
        try:
            async with session.get(u.rstrip("/") + "/metrics", timeout=aiohttp.ClientTimeout(total=15)) as r:
                text = await r.text()
        except Exception as e:  # noqa: BLE001
            out[u] = {"__error__": f"{type(e).__name__}: {e}"}
            continue
        d: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = _METRIC_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            if "prefix_cache" not in name or name.endswith("_created"):
                continue
            try:
                d[name] = d.get(name, 0.0) + float(m.group(3))
            except ValueError:
                pass
        out[u] = d
    return out


def metric_delta(before: dict, after: dict) -> dict[str, dict]:
    res = {}
    for u, a in after.items():
        b = before.get(u, {})
        if "__error__" in a or "__error__" in b:
            res[u] = {"__error__": a.get("__error__") or b.get("__error__")}
            continue
        res[u] = {k: v - b.get(k, 0.0) for k, v in a.items()}
    return res


def hit_rates(delta: dict[str, dict]) -> dict[str, Any]:
    """서버별 hit rate (hits/queries). PD 에서는 P/D 각각 따로 보는 것이 의미 있다."""
    res: dict[str, Any] = {}
    for u, d in delta.items():
        if "__error__" in d:
            res[u] = d
            continue
        rates = {}
        for name, v in d.items():
            if "hits" in name:
                q = d.get(name.replace("hits", "queries"))
                if q:
                    rates[name.replace("_hits", "").replace("_total", "")] = {
                        "hits": v,
                        "queries": q,
                        "hit_rate": v / q,
                    }
        res[u] = rates
    return res


GAUGE_NAMES = (
    "vllm:kv_cache_usage_perc",      # V1
    "vllm:gpu_cache_usage_perc",     # 구버전
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",    # counter (delta 로 사용)
)


async def scrape_gauges(session: aiohttp.ClientSession, urls: list[str], names=GAUGE_NAMES) -> dict[str, dict]:
    """서버 상태 지표 (KV 사용률, running/waiting 요청 수, preemption).

    *_perc 는 라벨(엔진) 간 최댓값, 나머지는 합계. 실패하면 해당 URL 은 빈 dict.
    """
    out: dict[str, dict] = {}
    for u in urls:
        d: dict[str, float] = {}
        try:
            async with session.get(u.rstrip("/") + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                text = await r.text()
        except Exception:  # noqa: BLE001
            out[u] = d
            continue
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = _METRIC_RE.match(line)
            if not m or m.group(1) not in names:
                continue
            try:
                v = float(m.group(3))
            except ValueError:
                continue
            name = m.group(1)
            d[name] = max(d.get(name, v), v) if name.endswith("_perc") else d.get(name, 0.0) + v
        out[u] = d
    return out


async def reset_prefix_cache(session: aiohttp.ClientSession, urls: list[str]) -> dict[str, str]:
    """vLLM 은 VLLM_SERVER_DEV_MODE=1 로 띄워야 /reset_prefix_cache 가 열린다."""
    out = {}
    for u in urls:
        try:
            async with session.post(u.rstrip("/") + "/reset_prefix_cache", timeout=aiohttp.ClientTimeout(total=60)) as r:
                out[u] = f"HTTP {r.status}"
        except Exception as e:  # noqa: BLE001
            out[u] = f"{type(e).__name__}: {e}"
    return out


def make_session(limit: int = 0) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=limit, force_close=False))


def write_json(path: str, obj: Any) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: list[dict]) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_async(coro):
    return asyncio.run(coro)
