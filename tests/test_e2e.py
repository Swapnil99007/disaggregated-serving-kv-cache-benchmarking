"""End-to-end: real HTTP between benchmark client -> router -> simulated engines."""

import asyncio

import httpx

from kvbench.bench.client import run_closed_loop
from kvbench.router.app import create_app as create_router
from kvbench.router.config import RouterConfig, WorkerSpec
from kvbench.sim.config import SimConfig
from kvbench.sim.server import create_app as create_sim
from kvbench.workload.rag import RagWorkloadConfig, generate_rag_trace
from kvbench.workload.trace import group_conversations

from conftest import ServerThread


def _trace(n=24):
    cfg = RagWorkloadConfig(num_prompts=n, num_topics=4, docs_per_topic=3, doc_words=(80, 120),
                            system_prompt_words=(40, 60), output_tokens_median=8, output_tokens_range=(4, 12))
    return generate_rag_trace(cfg)


def _sim(port, role="both"):
    return create_sim(SimConfig(engine_id=f"sim{port}", role=role, speed=50.0, kv_capacity_tokens=65536), "127.0.0.1", port)


def _router_cfg(workers, mode="colocated", policy="cache_aware"):
    return RouterConfig(mode=mode, policy=policy, chars_per_token=5.0, scrape_interval_s=0.2,
                        workers=workers, policy_params={"cache_aware": {"prefill_tps": 20000}})


def test_colocated_cache_aware_end_to_end(port_factory):
    p1, p2, pr = port_factory(), port_factory(), port_factory()
    workers = [WorkerSpec(name="a", url=f"http://127.0.0.1:{p1}", backend="sim"),
               WorkerSpec(name="b", url=f"http://127.0.0.1:{p2}", backend="sim")]
    with ServerThread(_sim(p1), p1), ServerThread(_sim(p2), p2), ServerThread(create_router(_router_cfg(workers)), pr):
        url = f"http://127.0.0.1:{pr}"
        results, wall = asyncio.run(run_closed_loop(group_conversations(_trace()), url, "kvbench-model", concurrency=4))
        assert len(results) == 24
        assert all(r.status == "ok" for r in results), [r.error for r in results if r.status != "ok"]
        assert all(r.ttft_s > 0 and r.e2e_s >= r.ttft_s for r in results)
        assert all(r.worker in ("a", "b") for r in results)
        # follow-up turns should hit the cache on the worker that served turn 0
        later = [r for r in results if r.turn > 0]
        assert later and sum(r.cached_tokens > 0 for r in later) >= 0.8 * len(later)

        st = httpx.get(url + "/admin/state").json()
        assert st["requests_total"] == 24
        assert all(w["inflight"] == 0 for w in st["workers"])
        # admin: switch policy, reset + flush
        assert httpx.post(url + "/admin/policy", json={"policy": "round_robin"}).json()["policy"] == "round_robin"
        flushed = httpx.post(url + "/admin/reset", json={"flush_workers": True}).json()["flushed"]
        assert all(v["ok"] for v in flushed.values())
        wm = httpx.get(url + "/admin/worker_metrics").json()
        assert wm["a"]["metrics"]["vllm:prefix_cache_queries_total"] > 0


def test_non_streaming_and_chat(port_factory):
    p1, pr = port_factory(), port_factory()
    workers = [WorkerSpec(name="a", url=f"http://127.0.0.1:{p1}", backend="sim")]
    with ServerThread(_sim(p1), p1), ServerThread(create_router(_router_cfg(workers, policy="least_loaded")), pr):
        url = f"http://127.0.0.1:{pr}"
        r = httpx.post(url + "/v1/completions", json={"model": "m", "prompt": "hello " * 50, "max_tokens": 5}, timeout=30)
        assert r.status_code == 200 and r.headers["x-kvbench-worker"] == "a"
        assert r.json()["usage"]["completion_tokens"] == 5
        r = httpx.post(url + "/v1/chat/completions",
                       json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}, timeout=30)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"]


def test_disaggregated_end_to_end(port_factory):
    pp, pd, pr = port_factory(), port_factory(), port_factory()
    workers = [WorkerSpec(name="p", url=f"http://127.0.0.1:{pp}", backend="sim", role="prefill"),
               WorkerSpec(name="d", url=f"http://127.0.0.1:{pd}", backend="sim", role="decode")]
    cfg = _router_cfg(workers, mode="disagg")
    seen: dict[str, list] = {"p": [], "d": []}
    sim_p, sim_d = _sim(pp, "prefill"), _sim(pd, "decode")
    for name, app in (("p", sim_p), ("d", sim_d)):
        def record(name=name):
            async def mw(request, call_next):
                if request.url.path.startswith("/v1/"):
                    seen[name].append(request.headers.get("x-request-id"))
                return await call_next(request)
            return mw
        app.middleware("http")(record())
    with ServerThread(sim_p, pp), ServerThread(sim_d, pd), ServerThread(create_router(cfg), pr):
        url = f"http://127.0.0.1:{pr}"
        results, _ = asyncio.run(run_closed_loop(group_conversations(_trace(12)), url, "kvbench-model", concurrency=3))
        assert all(r.status == "ok" for r in results), [r.error for r in results if r.status != "ok"]
        assert all(r.prefill_worker == "p" and r.decode_worker == "d" for r in results)
        assert all(r.prefill_prompt_tokens and r.prefill_cached_tokens is not None for r in results)
        # every output token was generated by the decode worker (max_tokens honoured end to end)
        assert all(r.completion_tokens == r.max_tokens for r in results)
        wm = httpx.get(url + "/admin/worker_metrics").json()
        assert wm["d"]["metrics"]["sim:kv_tokens_transferred_total"] > 0
        assert wm["d"]["metrics"]["sim:prefill_tokens_computed_total"] <= 12   # ~1 token per request
    # vLLM's NixlConnector frees the prefiller's KV by request id, so both legs of a request must
    # carry the same X-Request-Id (and different requests different ids)
    assert len(seen["p"]) == len(seen["d"]) == 12
    # (compare as sorted lists: with 3 concurrent sessions, decode legs can arrive in a different order)
    assert all(seen["p"]) and all(seen["d"]), "router must send X-Request-Id on both legs"
    assert sorted(seen["p"]) == sorted(seen["d"]) and len(set(seen["p"])) == 12


def _track_concurrency(app, live: dict, peak: dict, key: str):
    """ASGI wrapper: count /v1/ requests in flight, including the whole streamed body."""
    async def asgi(scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await app(scope, receive, send)
        live[key] += 1
        peak[key] = max(peak[key], live[key])
        try:
            await app(scope, receive, send)
        finally:
            live[key] -= 1
    return asgi


def test_disagg_admission_control_caps_decode_inflight(port_factory):
    # 6 sessions, decode capped at 2: requests wait in the router BEFORE their prefill leg, so
    # neither engine ever sees more than 2 at once, and nothing fails
    pp, pd, pr = port_factory(), port_factory(), port_factory()
    workers = [WorkerSpec(name="p", url=f"http://127.0.0.1:{pp}", backend="sim", role="prefill"),
               WorkerSpec(name="d", url=f"http://127.0.0.1:{pd}", backend="sim", role="decode", max_inflight=2)]
    cfg = _router_cfg(workers, mode="disagg")
    live, peak = {"p": 0, "d": 0}, {"p": 0, "d": 0}
    sim_p = _track_concurrency(_sim(pp, "prefill"), live, peak, "p")
    sim_d = _track_concurrency(_sim(pd, "decode"), live, peak, "d")
    with ServerThread(sim_p, pp), ServerThread(sim_d, pd), ServerThread(create_router(cfg), pr):
        url = f"http://127.0.0.1:{pr}"
        results, _ = asyncio.run(run_closed_loop(group_conversations(_trace(24)), url, "kvbench-model", concurrency=6))
        state = httpx.get(url + "/admin/state").json()
    assert all(r.status == "ok" for r in results), [r.error for r in results if r.status != "ok"]
    assert peak["d"] <= 2 and peak["p"] <= 2, peak
    assert peak["d"] == 2                                  # the cap was actually reached
    assert state["admission"]["waits"] > 0                 # ... and requests queued behind it
    assert any(r.queue_ms and r.queue_ms > 0 for r in results)
    d = next(w for w in state["workers"] if w["name"] == "d")
    assert d["admitted"] == 0 and d["inflight"] == 0       # every slot was released


def test_max_inflight_rejected_outside_disagg_decode():
    import pytest
    with pytest.raises(ValueError, match="max_inflight"):
        RouterConfig(mode="colocated", workers=[WorkerSpec(name="w", url="http://x", max_inflight=4)]).validate()
    with pytest.raises(ValueError, match="max_inflight"):
        RouterConfig(mode="disagg", workers=[WorkerSpec(name="p", url="http://x", role="prefill", max_inflight=4),
                                             WorkerSpec(name="d", url="http://y", role="decode")]).validate()


def test_admission_slot_released_when_prefill_fails(port_factory):
    # prefill worker unreachable: every request fails, but none may keep its decode slot
    pp, pd, pr = port_factory(), port_factory(), port_factory()
    workers = [WorkerSpec(name="p", url=f"http://127.0.0.1:{pp}", backend="sim", role="prefill"),
               WorkerSpec(name="d", url=f"http://127.0.0.1:{pd}", backend="sim", role="decode", max_inflight=1)]
    cfg = _router_cfg(workers, mode="disagg")
    cfg.unhealthy_after_failures = 10_000          # keep "p" in the pool after it dies
    sim_p = ServerThread(_sim(pp, "prefill"), pp)
    with sim_p, ServerThread(_sim(pd, "decode"), pd), ServerThread(create_router(cfg), pr):
        url = f"http://127.0.0.1:{pr}"
        sim_p.__exit__(None, None, None)         # stop the prefill server
        body = {"model": "kvbench-model", "prompt": "hello there", "max_tokens": 4}
        codes = [httpx.post(url + "/v1/completions", json=body, timeout=30).status_code for _ in range(3)]
        d = next(w for w in httpx.get(url + "/admin/state").json()["workers"] if w["name"] == "d")
    assert codes == [502, 502, 502]                # with a leaked slot, call 2 would hang instead
    assert d["admitted"] == 0 and d["inflight"] == 0
