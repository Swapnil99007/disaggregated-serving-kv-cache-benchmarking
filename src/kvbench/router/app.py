"""Router HTTP app: an OpenAI-compatible reverse proxy with pluggable routing.

Colocated mode: pick one worker per request (policy), stream its response back.

Disaggregated mode (vLLM NixlConnector-style protocol):
  1. Pick a *prefill* worker (policy) and send the request with ``max_tokens=1``,
     ``stream=false`` and ``kv_transfer_params.do_remote_decode=true``.
  2. The prefill worker computes the prompt KV and returns ``kv_transfer_params``
     telling a decoder where to pull it from.
  3. Pick a *decode* worker (decode_policy) and send the original request plus
     those ``kv_transfer_params``; the decoder pulls the KV and streams tokens.

  Both legs carry the same ``X-Request-Id`` header, so both engines derive the same internal
  request id. vLLM's NixlConnector relies on that: after pulling the KV, the decoder notifies
  the prefiller *by request id*, and the prefiller only frees the blocks of an id it knows.
  Mismatched ids leave every request's KV pinned on the prefiller until
  VLLM_NIXL_ABORT_REQUEST_TIMEOUT (120 s), which exhausts its KV pool within seconds.

Every response carries ``x-kvbench-*`` headers (chosen worker, routing reason,
predicted prefix match, and in disagg mode the prefill-side cached-token count)
so the benchmark client can attribute results without parsing logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from kvbench.common.prefix import chunk_hashes, extract_prompt_text
from kvbench.router.config import RouterConfig
from kvbench.router.index import PrefixIndex
from kvbench.router.policies import POLICIES, Decision, RouteRequest, make_policy
from kvbench.router.workers import WorkerState, flush_worker, scrape_worker, telemetry_loop

log = logging.getLogger("kvbench.router")

_DISAGG_PREFILL_KV_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


class Lease:
    """Router-side accounting for one request on one worker."""

    def __init__(self, router: "Router", w: WorkerState, rr: RouteRequest, matched: int, est_prefill: float | None = None):
        self.router = router
        self.w = w
        self.hashes = rr.hashes
        self.pred_frac = (matched / len(rr.hashes)) if rr.hashes else 0.0
        self.est_prefill = rr.est_prompt_tokens * (1.0 - self.pred_frac) if est_prefill is None else est_prefill
        self._first = False
        self._released = False
        self._fed_back = False
        w.inflight += 1
        w.prefill_inflight += 1
        w.pending_prefill_tokens += self.est_prefill
        w.routed_total += 1

    def accepted(self) -> None:
        """The worker accepted the request: record that it will hold this prefix after prefill.

        Done at acceptance (not at completion) so concurrent requests sharing the prefix
        are steered to the same worker while the first one is still prefilling.
        """
        self.w.index.insert(self.hashes)

    def first_token(self) -> None:
        if not self._first:
            self._first = True
            self.w.prefill_inflight -= 1
            self.w.pending_prefill_tokens = max(0.0, self.w.pending_prefill_tokens - self.est_prefill)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.first_token()
        self.w.inflight -= 1

    def feedback(self, usage: dict | None) -> None:
        """Compare predicted vs reported cached fraction (prediction-quality telemetry)."""
        if self._fed_back or not usage:
            return
        pt = usage.get("prompt_tokens")
        ct = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if ct is not None:
            self.w.reports_cached = True
        elif self.w.reports_cached:
            ct = 0          # vLLM omits prompt_tokens_details when nothing was cached
        if not pt or ct is None:
            return
        self._fed_back = True
        actual = ct / pt
        w = self.w
        w.pred_samples += 1
        w.pred_frac_sum += self.pred_frac
        w.actual_frac_sum += actual
        if self.pred_frac - actual > 0.25:
            w.overpredictions += 1
            if self.router.cfg.adaptive_capacity:
                w.index.shrink()
        elif self.router.cfg.adaptive_capacity:
            w.index.grow(8)


class Router:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.workers: list[WorkerState] = []
        for spec in cfg.workers:
            cap_tokens = spec.kv_capacity_tokens or cfg.index_capacity_tokens
            cap_chunks = int(cap_tokens * cfg.chars_per_token / cfg.chunk_chars)
            self.workers.append(WorkerState(spec=spec, index=PrefixIndex(cap_chunks, cfg.index_ttl_s), healthy=False))
        self.policy = make_policy(cfg.policy, cfg.policy_params)
        self.decode_policy = make_policy(cfg.decode_policy, cfg.policy_params)
        self.client: httpx.AsyncClient | None = None
        self._telemetry_task: asyncio.Task | None = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self.requests_total = 0
        self.errors_total = 0
        self.reasons: Counter[str] = Counter()
        self.route_time_sum = 0.0
        self.matched_chunks_sum = 0
        self.prompt_chunks_sum = 0
        for w in self.workers:
            w.routed_total = 0
            w.errors_total = 0
            w.pred_samples = 0
            w.pred_frac_sum = 0.0
            w.actual_frac_sum = 0.0
            w.overpredictions = 0

    # ---- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.request_timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=2000, max_keepalive_connections=512),
        )
        await asyncio.gather(*(scrape_worker(self.client, w) for w in self.workers))
        for w in self.workers:
            w.healthy = w.consecutive_failures == 0
        self._telemetry_task = asyncio.create_task(
            telemetry_loop(self.client, self.workers, self.cfg.scrape_interval_s, self.cfg.unhealthy_after_failures)
        )

    async def stop(self) -> None:
        if self._telemetry_task:
            self._telemetry_task.cancel()
        if self.client:
            await self.client.aclose()

    # ---- helpers -------------------------------------------------------------
    def pool(self, role: str) -> list[WorkerState]:
        if role == "both":
            ws = [w for w in self.workers if w.healthy and w.spec.role == "both"]
            return ws or [w for w in self.workers if w.healthy]
        return [w for w in self.workers if w.healthy and w.spec.role == role]

    def make_route_request(self, body: dict) -> RouteRequest:
        text = extract_prompt_text(body)
        return RouteRequest(
            hashes=chunk_hashes(text, self.cfg.chunk_chars, self.cfg.model),
            est_prompt_tokens=len(text) / self.cfg.chars_per_token,
            prompt_chars=len(text),
        )

    def _record_decision(self, d: Decision, rr: RouteRequest, t0: float) -> None:
        self.reasons[d.reason] += 1
        self.route_time_sum += time.perf_counter() - t0
        self.matched_chunks_sum += d.matched_chunks
        self.prompt_chunks_sum += len(rr.hashes)

    async def _send(self, w: WorkerState, path: str, body: dict, headers: dict | None = None) -> httpx.Response:
        assert self.client is not None
        req = self.client.build_request("POST", w.spec.url.rstrip("/") + path, json=body, headers=headers)
        return await self.client.send(req, stream=True)

    async def _relay(self, resp: httpx.Response, lease: Lease, headers: dict, stream: bool) -> Response:
        if resp.status_code >= 400:
            data = await resp.aread()
            await resp.aclose()
            lease.release()
            lease.w.errors_total += 1
            self.errors_total += 1
            return Response(content=data, status_code=resp.status_code, media_type="application/json", headers=headers)

        if not stream:
            data = await resp.aread()
            await resp.aclose()
            lease.first_token()
            try:
                lease.feedback(json.loads(data).get("usage"))
            except Exception:  # noqa: BLE001
                pass
            lease.release()
            return Response(content=data, status_code=resp.status_code, media_type="application/json", headers=headers)

        async def gen():
            buf = b""
            try:
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    lease.first_token()
                    buf += chunk
                    if b"\n" in buf:
                        lines = buf.split(b"\n")
                        buf = lines.pop()
                        for ln in lines:
                            if ln.startswith(b"data:") and b'"usage"' in ln:
                                try:
                                    obj = json.loads(ln[5:])
                                    if obj.get("usage"):
                                        lease.feedback(obj["usage"])
                                except Exception:  # noqa: BLE001
                                    pass
                    yield chunk
            finally:
                await resp.aclose()
                lease.release()

        return StreamingResponse(gen(), status_code=resp.status_code, media_type="text/event-stream", headers=headers)

    # ---- request handling ----------------------------------------------------
    async def handle(self, body: dict, path: str) -> Response:
        t0 = time.perf_counter()
        self.requests_total += 1
        rr = self.make_route_request(body)
        if self.cfg.mode == "disagg":
            return await self._handle_disagg(body, path, rr, t0)

        excluded: set[str] = set()
        last_err = "no healthy workers"
        for _ in range(self.cfg.max_retries + 1):
            pool = [w for w in self.pool("both") if w.name not in excluded]
            if not pool:
                break
            d = self.policy.select(rr, pool)
            self._record_decision(d, rr, t0)
            lease = Lease(self, d.worker, rr, d.matched_chunks)
            try:
                resp = await self._send(d.worker, path, body)
            except httpx.TransportError as e:
                lease.release()
                d.worker.errors_total += 1
                d.worker.consecutive_failures += 1
                excluded.add(d.worker.name)
                last_err = f"{d.worker.name}: {e!r}"
                log.warning("dispatch to %s failed (%s); retrying", d.worker.name, e)
                continue
            lease.accepted()
            headers = {
                "x-kvbench-worker": d.worker.name,
                "x-kvbench-policy": self.policy.name,
                "x-kvbench-reason": d.reason,
                "x-kvbench-matched-chunks": str(d.matched_chunks),
                "x-kvbench-prompt-chunks": str(len(rr.hashes)),
            }
            return await self._relay(resp, lease, headers, bool(body.get("stream")))
        self.errors_total += 1
        return JSONResponse({"error": {"message": f"routing failed: {last_err}"}}, status_code=503)

    async def _handle_disagg(self, body: dict, path: str, rr: RouteRequest, t0: float) -> Response:
        assert self.client is not None
        pre_pool, dec_pool = self.pool("prefill"), self.pool("decode")
        if not pre_pool or not dec_pool:
            self.errors_total += 1
            return JSONResponse({"error": {"message": "no healthy prefill/decode workers"}}, status_code=503)

        # one id for both legs (see module docstring: NixlConnector frees prefill KV by request id)
        req_headers = {"X-Request-Id": f"kvb-{uuid.uuid4().hex}"}

        # 1) prefill
        pd = self.policy.select(rr, pre_pool)
        self._record_decision(pd, rr, t0)
        lease_p = Lease(self, pd.worker, rr, pd.matched_chunks)
        pbody = dict(body)
        pbody["stream"] = False
        pbody["max_tokens"] = 1
        if "max_completion_tokens" in pbody:
            pbody["max_completion_tokens"] = 1
        pbody.pop("stream_options", None)
        pbody["kv_transfer_params"] = dict(_DISAGG_PREFILL_KV_PARAMS)
        tp = time.perf_counter()
        lease_p.accepted()   # mark before the (blocking) prefill call so concurrent same-prefix requests follow
        try:
            presp = await self.client.post(pd.worker.spec.url.rstrip("/") + path, json=pbody, headers=req_headers)
        except httpx.TransportError as e:
            lease_p.release()
            pd.worker.errors_total += 1
            self.errors_total += 1
            return JSONResponse({"error": {"message": f"prefill failed: {e!r}"}}, status_code=502)
        lease_p.first_token()
        if presp.status_code >= 400:
            lease_p.release()
            pd.worker.errors_total += 1
            self.errors_total += 1
            return Response(content=presp.content, status_code=presp.status_code, media_type="application/json")
        pj = presp.json()
        lease_p.feedback(pj.get("usage"))
        lease_p.release()
        prefill_ms = (time.perf_counter() - tp) * 1000.0
        usage = pj.get("usage") or {}
        p_prompt = usage.get("prompt_tokens")
        p_cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        kvp = pj.get("kv_transfer_params")

        # 2) decode
        dd = self.decode_policy.select(rr, dec_pool)
        lease_d = Lease(self, dd.worker, rr, dd.matched_chunks, est_prefill=0.0)
        dbody = dict(body)
        if kvp:
            dbody["kv_transfer_params"] = kvp
        try:
            resp = await self._send(dd.worker, path, dbody, headers=req_headers)
        except httpx.TransportError as e:
            lease_d.release()
            dd.worker.errors_total += 1
            self.errors_total += 1
            return JSONResponse({"error": {"message": f"decode failed: {e!r}"}}, status_code=502)
        lease_d.accepted()
        headers = {
            "x-kvbench-worker": dd.worker.name,
            "x-kvbench-prefill-worker": pd.worker.name,
            "x-kvbench-decode-worker": dd.worker.name,
            "x-kvbench-policy": self.policy.name,
            "x-kvbench-reason": pd.reason,
            "x-kvbench-matched-chunks": str(pd.matched_chunks),
            "x-kvbench-prompt-chunks": str(len(rr.hashes)),
            "x-kvbench-prefill-ms": f"{prefill_ms:.3f}",
            "x-kvbench-request-id": req_headers["X-Request-Id"],
        }
        if p_prompt is not None:
            headers["x-kvbench-prefill-prompt-tokens"] = str(p_prompt)
        if p_cached is not None:
            headers["x-kvbench-prefill-cached-tokens"] = str(p_cached)
        if not kvp:
            headers["x-kvbench-warning"] = "prefill-returned-no-kv_transfer_params"
        return await self._relay(resp, lease_d, headers, bool(body.get("stream")))

    # ---- admin ---------------------------------------------------------------
    def state(self) -> dict:
        n = max(1, self.requests_total)
        return {
            "mode": self.cfg.mode,
            "policy": self.policy.name,
            "decode_policy": self.decode_policy.name if self.cfg.mode == "disagg" else None,
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "reasons": dict(self.reasons),
            "route_time_ms_mean": 1000.0 * self.route_time_sum / n,
            "predicted_match_frac_mean": self.matched_chunks_sum / max(1, self.prompt_chunks_sum),
            "workers": [w.to_dict() for w in self.workers],
        }

    def set_policy(self, name: str, params: dict | None = None, decode_policy: str | None = None) -> None:
        if params:
            self.cfg.policy_params.setdefault(name, {}).update(params)
        self.policy = make_policy(name, self.cfg.policy_params)
        if decode_policy:
            self.decode_policy = make_policy(decode_policy, self.cfg.policy_params)

    async def reset(self, flush_workers: bool, clear_index: bool = True) -> dict:
        """Reset router stats; optionally clear the prefix index and flush worker caches.

        ``clear_index=False, flush_workers=False`` keeps caches warm (stats-only reset).
        """
        assert self.client is not None
        if clear_index or flush_workers:
            for w in self.workers:
                w.index.clear()
                w.index.capacity = w.index.max_capacity
        self.reset_stats()
        results = {}
        if flush_workers:
            healthy = [w for w in self.workers if w.healthy]
            outs = await asyncio.gather(*(flush_worker(self.client, w) for w in healthy))
            results = {w.name: {"ok": ok, "detail": msg} for w, (ok, msg) in zip(healthy, outs)}
        return {"flushed": results}

    def prometheus(self) -> str:
        lines = [
            "# TYPE kvbench_router_requests_total counter",
            f"kvbench_router_requests_total {self.requests_total}",
            "# TYPE kvbench_router_errors_total counter",
            f"kvbench_router_errors_total {self.errors_total}",
            "# TYPE kvbench_router_routing_seconds_sum counter",
            f"kvbench_router_routing_seconds_sum {self.route_time_sum}",
        ]
        for reason, c in self.reasons.items():
            lines.append(f'kvbench_router_decisions_total{{reason="{reason}"}} {c}')
        for w in self.workers:
            lbl = f'{{worker="{w.name}",role="{w.spec.role}"}}'
            lines += [
                f"kvbench_router_worker_routed_total{lbl} {w.routed_total}",
                f"kvbench_router_worker_inflight{lbl} {w.inflight}",
                f"kvbench_router_worker_pending_prefill_tokens{lbl} {w.pending_prefill_tokens}",
                f"kvbench_router_worker_index_chunks{lbl} {len(w.index)}",
                f"kvbench_router_worker_healthy{lbl} {int(w.healthy)}",
            ]
            if w.kv_usage is not None:
                lines.append(f"kvbench_router_worker_kv_usage{lbl} {w.kv_usage}")
        return "\n".join(lines) + "\n"


def create_app(cfg: RouterConfig) -> FastAPI:
    router = Router(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await router.start()
        log.info("router up: mode=%s policy=%s workers=%s", cfg.mode, cfg.policy, [w.spec.url for w in router.workers])
        yield
        await router.stop()

    app = FastAPI(title="kvbench-router", lifespan=lifespan)
    app.state.router = router

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await router.handle(await request.json(), "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await router.handle(await request.json(), "/v1/chat/completions")

    @app.get("/v1/models")
    async def models():
        for w in router.workers:
            if w.healthy and router.client is not None:
                try:
                    r = await router.client.get(w.spec.url.rstrip("/") + "/v1/models", timeout=5.0)
                    return JSONResponse(r.json(), status_code=r.status_code)
                except Exception:  # noqa: BLE001
                    continue
        return {"object": "list", "data": [{"id": cfg.model, "object": "model"}]}

    @app.get("/health")
    async def health():
        healthy = [w.name for w in router.workers if w.healthy]
        status = 200 if healthy else 503
        return JSONResponse({"healthy_workers": healthy, "total_workers": len(router.workers)}, status_code=status)

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(router.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/admin/state")
    async def admin_state():
        return router.state()

    @app.get("/admin/policies")
    async def admin_policies():
        return {"available": sorted(POLICIES), "current": router.policy.name}

    @app.post("/admin/policy")
    async def admin_policy(request: Request):
        body = await request.json()
        try:
            router.set_policy(body["policy"], body.get("params"), body.get("decode_policy"))
        except (KeyError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"policy": router.policy.name, "decode_policy": router.decode_policy.name}

    @app.post("/admin/reset")
    async def admin_reset(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        return await router.reset(bool(body.get("flush_workers", True)), bool(body.get("clear_index", True)))

    @app.get("/admin/worker_metrics")
    async def admin_worker_metrics():
        assert router.client is not None
        await asyncio.gather(*(scrape_worker(router.client, w) for w in router.workers))
        return {w.name: {"healthy": w.healthy, "role": w.spec.role, "metrics": w.raw_metrics} for w in router.workers}

    return app


def run_router(cfg: RouterConfig, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host or cfg.host, port=port or cfg.port, log_level="warning", access_log=False)
