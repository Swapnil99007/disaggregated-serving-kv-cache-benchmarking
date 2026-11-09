"""OpenAI/vLLM-compatible HTTP server around :class:`SimEngine`.

Endpoints (subset of vLLM's surface that the router and benchmark rely on):

* ``POST /v1/completions``, ``POST /v1/chat/completions`` (streaming + non-streaming,
  ``stream_options.include_usage``, ``usage.prompt_tokens_details.cached_tokens``,
  ``kv_transfer_params`` for P/D disaggregation)
* ``GET /metrics`` (Prometheus text with vLLM metric names + ``sim:*`` extras)
* ``POST /reset_prefix_cache`` (vLLM dev-mode endpoint), ``POST /flush_cache`` (SGLang name)
* ``GET /health``, ``GET /v1/models``, ``GET /sim/state``
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from kvbench.sim.config import SimConfig
from kvbench.sim.engine import RequestTooLarge, SimEngine, SimRequest
from kvbench.sim.tokenizer import block_hashes, tokenize
from kvbench.workload.words import VOCAB


def _chat_to_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"<|{m.get('role', 'user')}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def create_app(cfg: SimConfig, advertised_host: str = "localhost", advertised_port: int = 9000) -> FastAPI:
    engine = SimEngine(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        yield
        await engine.stop()

    app = FastAPI(title=f"kvbench-sim {cfg.engine_id}", lifespan=lifespan)
    app.state.engine = engine

    def _build_request(body: dict, prompt: str) -> SimRequest:
        toks = tokenize(prompt)
        prompt_len = max(1, len(toks))
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
        kv = body.get("kv_transfer_params") or {}
        remote_decode = bool(kv.get("do_remote_decode"))
        remote_prefill = bool(kv.get("do_remote_prefill"))
        if remote_decode:
            max_tokens = 1
        return SimRequest(
            rid=uuid.uuid4().hex[:16],
            prompt_len=prompt_len,
            block_hashes=block_hashes(toks, cfg.block_size, cfg.model_name),
            max_tokens=max(1, max_tokens),
            remote_prefill=remote_prefill,
            remote_decode=remote_decode,
        )

    def _usage(req: SimRequest) -> dict:
        return {
            "prompt_tokens": req.prompt_len,
            "completion_tokens": req.generated,
            "total_tokens": req.prompt_len + req.generated,
            "prompt_tokens_details": {"cached_tokens": req.prompt_len - req.prefill_computed},
        }

    def _kv_params(req: SimRequest) -> dict | None:
        if not req.remote_decode:
            return None
        return {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": cfg.engine_id,
            "remote_block_ids": list(range(len(req.block_hashes))),
            "remote_host": advertised_host,
            "remote_port": advertised_port,
            "tp_size": 1,
        }

    async def _handle(request: Request, chat: bool):
        body = await request.json()
        prompt = _chat_to_prompt(body.get("messages") or []) if chat else body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(map(str, prompt))
        req = _build_request(body, str(prompt))
        try:
            engine.submit(req)
        except RequestTooLarge as e:
            return JSONResponse({"error": {"message": str(e), "type": "BadRequestError"}}, status_code=400)

        model = body.get("model") or cfg.model_name
        created = int(time.time())
        cid = ("chatcmpl-" if chat else "cmpl-") + req.rid
        obj_type = "chat.completion.chunk" if chat else "text_completion"
        rng = random.Random(hash(req.rid) ^ cfg.seed)

        def choice(text: str, finish: str | None) -> dict:
            if chat:
                return {"index": 0, "delta": {"content": text} if text else {}, "logprobs": None, "finish_reason": finish}
            return {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def gen():
                try:
                    while True:
                        kind, _ = await req.queue.get()
                        if kind == "token":
                            text = rng.choice(VOCAB) + " "
                            yield _sse({"id": cid, "object": obj_type, "created": created, "model": model,
                                        "choices": [choice(text, None)]})
                        else:
                            yield _sse({"id": cid, "object": obj_type, "created": created, "model": model,
                                        "choices": [choice("", "length")]})
                            if include_usage:
                                yield _sse({"id": cid, "object": obj_type, "created": created, "model": model,
                                            "choices": [], "usage": _usage(req)})
                            yield b"data: [DONE]\n\n"
                            return
                finally:
                    engine.abort(req)   # no-op when finished; frees KV if the client disconnected

            return StreamingResponse(gen(), media_type="text/event-stream")

        # non-streaming
        pieces = []
        try:
            while True:
                kind, _ = await req.queue.get()
                if kind == "token":
                    pieces.append(rng.choice(VOCAB) + " ")
                else:
                    break
        except asyncio.CancelledError:
            engine.abort(req)
            raise
        text = "".join(pieces)
        if chat:
            choices = [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}]
        else:
            choices = [{"index": 0, "text": text, "logprobs": None, "finish_reason": "length"}]
        out = {"id": cid, "object": "chat.completion" if chat else "text_completion", "created": created,
               "model": model, "choices": choices, "usage": _usage(req)}
        kvp = _kv_params(req)
        if kvp:
            out["kv_transfer_params"] = kvp
        return JSONResponse(out)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _handle(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle(request, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": cfg.model_name, "object": "model", "owned_by": "kvbench-sim"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache():
        return {"dropped_blocks": engine.reset_prefix_cache()}

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def flush_cache():
        return {"dropped_blocks": engine.reset_prefix_cache()}

    @app.get("/sim/state")
    async def state():
        return {"config": cfg.__dict__, "state": engine.snapshot()}

    @app.get("/metrics")
    async def metrics():
        s = engine.snapshot()
        lbl = f'{{model_name="{cfg.model_name}",engine="{cfg.engine_id}"}}'
        lines = [
            ("vllm:num_requests_running", "gauge", s["num_running"]),
            ("vllm:num_requests_waiting", "gauge", s["num_waiting"]),
            ("vllm:kv_cache_usage_perc", "gauge", s["kv_usage"]),
            ("vllm:gpu_cache_usage_perc", "gauge", s["kv_usage"]),
            ("vllm:prefix_cache_queries_total", "counter", s["prefix_queries"]),
            ("vllm:prefix_cache_hits_total", "counter", s["prefix_hits"]),
            ("vllm:prompt_tokens_total", "counter", s["prompt_tokens"]),
            ("vllm:generation_tokens_total", "counter", s["generation_tokens"]),
            ("vllm:num_preemptions_total", "counter", 0),
            ("sim:prefill_tokens_computed_total", "counter", s["prefill_tokens_computed"]),
            ("sim:kv_tokens_transferred_total", "counter", s["kv_tokens_transferred"]),
            ("sim:kv_blocks_cached", "gauge", s["kv_blocks_cached"]),
            ("sim:evictions_total", "counter", s["evictions"]),
            ("sim:gpu_memory_used_bytes", "gauge", s["gpu_mem_used_bytes"]),
            ("sim:gpu_memory_total_bytes", "gauge", s["gpu_mem_total_bytes"]),
            ("sim:busy_seconds_total", "counter", s["busy_s"]),
            ("sim:num_transferring", "gauge", s["num_transferring"]),
        ]
        out = []
        for name, typ, val in lines:
            out.append(f"# TYPE {name} {typ}")
            out.append(f"{name}{lbl} {float(val)}")
        return PlainTextResponse("\n".join(out) + "\n", media_type="text/plain; version=0.0.4")

    return app


def run_sim_worker(cfg: SimConfig, host: str = "0.0.0.0", port: int = 9000, advertised_host: str | None = None) -> None:
    import uvicorn

    app = create_app(cfg, advertised_host or "localhost", port)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
