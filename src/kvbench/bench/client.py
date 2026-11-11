"""Streaming benchmark client and closed-loop session driver.

Metric definitions (per request, measured at the client):

* **TTFT**  time from sending the request to receiving the first non-empty token.
* **E2E**   time from sending the request to receiving the final chunk.
* **TPOT**  (E2E - TTFT) / (completion_tokens - 1): mean time per output token
            after the first. Uses the server-reported token count, so it stays
            correct when a chunk carries more than one token.
* **ITL**   gaps between consecutive token chunks (we keep mean / p95 / max).

Closed loop: ``concurrency`` sessions each take the next *conversation* from a
shared queue and replay its turns sequentially (turn t+1 is sent only after
turn t completes), so at most ``concurrency`` requests are ever in flight.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import httpx
import numpy as np

from kvbench.workload.trace import TraceRequest


@dataclass
class RequestResult:
    request_id: str
    conversation_id: str
    turn: int
    session_id: int = -1
    status: str = "ok"
    error: str | None = None
    http_status: int | None = None
    start_ts: float = 0.0          # seconds since run start
    end_ts: float = 0.0
    ttft_s: float | None = None
    e2e_s: float | None = None
    tpot_s: float | None = None
    itl_mean_s: float | None = None
    itl_p95_s: float | None = None
    itl_max_s: float | None = None
    prompt_chars: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    max_tokens: int = 0
    worker: str | None = None
    prefill_worker: str | None = None
    decode_worker: str | None = None
    route_reason: str | None = None
    matched_chunks: int | None = None
    prompt_chunks: int | None = None
    prefill_prompt_tokens: int | None = None
    prefill_cached_tokens: int | None = None
    prefill_ms: float | None = None
    queue_ms: float | None = None      # disagg: time held in the router waiting for a decode slot

    @property
    def prefill_computed_tokens(self) -> int | None:
        """Prompt tokens actually computed (not served from cache)."""
        if self.prefill_prompt_tokens is not None and self.prefill_cached_tokens is not None:
            return self.prefill_prompt_tokens - self.prefill_cached_tokens   # disagg: prefill worker's view
        if self.prompt_tokens is not None and self.cached_tokens is not None:
            return self.prompt_tokens - self.cached_tokens
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prefill_computed_tokens"] = self.prefill_computed_tokens
        return d


def _int_header(h: httpx.Headers, name: str) -> int | None:
    v = h.get(name)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    req: TraceRequest,
    run_t0: float,
    ignore_eos: bool = True,
    temperature: float = 0.0,
    extra_body: dict | None = None,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }
    if extra_body:
        payload.update(extra_body)
    res = RequestResult(
        request_id=req.request_id,
        conversation_id=req.conversation_id,
        turn=req.turn,
        prompt_chars=len(req.prompt),
        max_tokens=req.max_tokens,
    )
    t0 = time.perf_counter()
    res.start_ts = t0 - run_t0
    itls: list[float] = []
    last = None
    n_chunks = 0
    usage = None
    try:
        async with client.stream("POST", base_url.rstrip("/") + "/v1/completions", json=payload) as resp:
            res.http_status = resp.status_code
            h = resp.headers
            res.worker = h.get("x-kvbench-worker")
            res.prefill_worker = h.get("x-kvbench-prefill-worker")
            res.decode_worker = h.get("x-kvbench-decode-worker")
            res.route_reason = h.get("x-kvbench-reason")
            res.matched_chunks = _int_header(h, "x-kvbench-matched-chunks")
            res.prompt_chunks = _int_header(h, "x-kvbench-prompt-chunks")
            res.prefill_prompt_tokens = _int_header(h, "x-kvbench-prefill-prompt-tokens")
            res.prefill_cached_tokens = _int_header(h, "x-kvbench-prefill-cached-tokens")
            pm = h.get("x-kvbench-prefill-ms")
            res.prefill_ms = float(pm) if pm else None
            qm = h.get("x-kvbench-queue-ms")
            res.queue_ms = float(qm) if qm else None
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                res.status, res.error = "error", f"HTTP {resp.status_code}: {body[:300]}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        c0 = choices[0]
                        text = c0.get("text")
                        if text is None:
                            text = (c0.get("delta") or {}).get("content")
                        if text:
                            now = time.perf_counter()
                            if res.ttft_s is None:
                                res.ttft_s = now - t0
                            elif last is not None:
                                itls.append(now - last)
                            last = now
                            n_chunks += 1
    except Exception as e:  # noqa: BLE001
        res.status, res.error = "error", f"{type(e).__name__}: {e}"
    t1 = time.perf_counter()
    res.end_ts = t1 - run_t0
    res.e2e_s = t1 - t0
    if usage:
        res.prompt_tokens = usage.get("prompt_tokens")
        res.completion_tokens = usage.get("completion_tokens")
        res.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if res.completion_tokens is None and n_chunks:
        res.completion_tokens = n_chunks
    if res.status == "ok" and res.ttft_s is None:
        res.status, res.error = "error", "no tokens received"
    if res.status == "ok" and res.completion_tokens and res.completion_tokens > 1:
        res.tpot_s = (res.e2e_s - res.ttft_s) / (res.completion_tokens - 1)
    if itls:
        arr = np.asarray(itls)
        res.itl_mean_s = float(arr.mean())
        res.itl_p95_s = float(np.percentile(arr, 95))
        res.itl_max_s = float(arr.max())
    return res


async def run_closed_loop(
    conversations: list[list[TraceRequest]],
    base_url: str,
    model: str,
    concurrency: int,
    ignore_eos: bool = True,
    temperature: float = 0.0,
    think_time_s: float = 0.0,
    timeout_s: float = 900.0,
    extra_body: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[RequestResult], float]:
    """Replay conversations with ``concurrency`` closed-loop sessions.

    Returns (results, wall_time_s).
    """
    queue = deque(conversations)
    total = sum(len(c) for c in conversations)
    results: list[RequestResult] = []
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        run_t0 = time.perf_counter()

        async def session(sid: int) -> None:
            while queue:
                conv = queue.popleft()
                for req in conv:
                    r = await send_request(client, base_url, model, req, run_t0, ignore_eos, temperature, extra_body)
                    r.session_id = sid
                    results.append(r)
                    if progress:
                        progress(len(results), total)
                    if think_time_s > 0:
                        await asyncio.sleep(think_time_s)

        await asyncio.gather(*(session(i) for i in range(concurrency)))
        wall = time.perf_counter() - run_t0
    return results, wall
