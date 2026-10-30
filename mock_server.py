"""GPU 없이 파이프라인을 점검하기 위한 가짜 OpenAI 호환 서버.

실제 성능 수치와는 무관하며, 스크립트가 끝까지 도는지 / 지표가 기대한 방향으로 나오는지 확인하는 용도.
토큰 = 공백 단위 단어로 근사한다.

  --mode colocated : prefill 이 decode 스텝을 막음(간섭) + 생성 토큰도 prefix cache 에 저장
  --mode pd        : prefill/decode 분리(간섭 없음) + KV 전송 지연 + 캐시에는 prompt 만 저장

포화 동작(capacity 테스트용):
  --max-num-seqs   동시에 decode 할 수 있는 요청 수 (넘으면 대기열 → TTFT 증가)
  --kv-blocks      prefix cache 블록 수 상한 (LRU eviction → 재-prefill 증가)
  decode 스텝 시간 = decode_ms * (1 + running * load_factor)

  python mock_server.py --port 8000 --mode colocated
  python mock_server.py --port 9000 --mode pd --kv-transfer-ms 20
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from collections import OrderedDict

from aiohttp import web


class State:
    def __init__(self, a):
        self.a = a
        self.blocks: OrderedDict[str, None] = OrderedDict()
        self.queries = 0.0
        self.hits = 0.0
        self.prefill_lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(a.max_num_seqs)
        self.running = 0
        self.waiting = 0

    def touch(self, hashes: list[str]) -> None:
        for h in hashes:
            self.blocks[h] = None
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.a.kv_blocks:
            self.blocks.popitem(last=False)


def block_hashes(tokens: list[str], bs: int) -> list[str]:
    hs, h = [], hashlib.sha1()
    for i in range(len(tokens) // bs):
        h.update(" ".join(tokens[i * bs:(i + 1) * bs]).encode())
        hs.append(h.hexdigest())
    return hs


def answer_for(prompt: str) -> list[str] | None:
    qs = re.findall(r"secret code for ([\w-]+)\?", prompt)
    if qs:
        m = re.search(r"secret code for " + re.escape(qs[-1]) + r" is (\d+)", prompt)
        return [m.group(1)] if m else ["unknown"]
    if prompt.rstrip().endswith("Answer:") or "####" in prompt:
        return "Let me think step by step. The answer is 42 . #### 42".split()
    return None


async def handle(request: web.Request, chat: bool) -> web.StreamResponse:
    st: State = request.app["state"]
    a = st.a
    body = await request.json()
    if chat:
        prompt = "\n".join(m["content"] if isinstance(m["content"], str) else "x" for m in body["messages"])
    else:
        prompt = body["prompt"]
    toks = prompt.split()
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
    ans = answer_for(prompt)
    if body.get("ignore_eos") or ans is None:
        rng = random.Random(hash(prompt) & 0xFFFFFFFF)
        out = ((ans or []) + [rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(max_tokens)])[:max_tokens]
    else:
        out = ans[:max_tokens]

    st.waiting += 1
    await st.slots.acquire()  # 스케줄러 슬롯 (max-num-seqs)
    st.waiting -= 1
    st.running += 1
    try:
        hs = block_hashes(toks, a.block)
        hit_blocks = 0
        for h in hs:
            if h in st.blocks:
                hit_blocks += 1
            else:
                break
        cached = hit_blocks * a.block
        st.queries += len(toks)
        st.hits += cached
        st.touch(hs)

        prefill_s = (len(toks) - cached) / 1000.0 * a.prefill_ms_per_1k / 1000.0
        async with st.prefill_lock:  # prefill 은 직렬 (compute 경쟁)
            await asyncio.sleep(prefill_s)
        if a.mode == "pd":
            await asyncio.sleep(a.kv_transfer_ms / 1000.0)

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        rid = f"mock-{random.getrandbits(48):x}"
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"
        base = {"id": rid, "object": obj, "created": created, "model": a.model}
        for i, w in enumerate(out):
            if i > 0:
                step = a.decode_ms / 1000.0 * (1 + st.running * a.load_factor)
                if a.mode == "colocated":
                    async with st.prefill_lock:  # prefill 진행 중이면 decode 스텝이 밀림
                        pass
                await asyncio.sleep(step)
            piece = " " + w
            last = i == len(out) - 1
            fr = "length" if last else None
            chunk = dict(base, choices=[{"index": 0, "delta": ({"role": "assistant", "content": piece} if i == 0 else {"content": piece}),
                                         "finish_reason": fr}] if chat
                         else [{"index": 0, "text": piece, "finish_reason": fr}])
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if a.mode == "colocated":
            st.touch(block_hashes(toks + out, a.block))
        usage = {"prompt_tokens": len(toks), "completion_tokens": len(out),
                 "prompt_tokens_details": {"cached_tokens": cached}}
        await resp.write(f"data: {json.dumps(dict(base, choices=[], usage=usage))}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp
    finally:
        st.running -= 1
        st.slots.release()


async def metrics(request: web.Request) -> web.Response:
    st: State = request.app["state"]
    a = st.a
    txt = (
        f'vllm:prefix_cache_queries_total{{model_name="mock"}} {st.queries}\n'
        f'vllm:prefix_cache_hits_total{{model_name="mock"}} {st.hits}\n'
        f'vllm:num_requests_running{{model_name="mock"}} {st.running}\n'
        f'vllm:num_requests_waiting{{model_name="mock"}} {st.waiting}\n'
        f'vllm:kv_cache_usage_perc{{model_name="mock"}} {len(st.blocks) / a.kv_blocks}\n'
        f'vllm:num_preemptions_total{{model_name="mock"}} 0\n'
    )
    return web.Response(text=txt)


async def reset(request: web.Request) -> web.Response:
    request.app["state"].blocks.clear()
    return web.Response(text="ok")


async def models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": request.app["state"].a.model, "object": "model"}]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="mock")
    p.add_argument("--mode", choices=["colocated", "pd"], default="colocated")
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--prefill-ms-per-1k", type=float, default=40.0)
    p.add_argument("--decode-ms", type=float, default=8.0)
    p.add_argument("--kv-transfer-ms", type=float, default=15.0)
    p.add_argument("--load-factor", type=float, default=0.005)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--kv-blocks", type=int, default=1_000_000)
    a = p.parse_args()
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["state"] = State(a)
    app.router.add_post("/v1/completions", lambda r: handle(r, False))
    app.router.add_post("/v1/chat/completions", lambda r: handle(r, True))
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/reset_prefix_cache", reset)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    web.run_app(app, port=a.port, print=lambda *_: print(f"mock server ({a.mode}) on :{a.port}"))


if __name__ == "__main__":
    main()
