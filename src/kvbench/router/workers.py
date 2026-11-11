"""Worker state, backend adapters, and telemetry scraping."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from kvbench.common.prom import first_max, first_sum, parse_prometheus_text
from kvbench.router.config import WorkerSpec
from kvbench.router.index import PrefixIndex

log = logging.getLogger("kvbench.router")


@dataclass(frozen=True)
class Backend:
    running: tuple[str, ...]
    waiting: tuple[str, ...]
    kv_usage: tuple[str, ...]
    prefix_queries: tuple[str, ...]
    prefix_hits: tuple[str, ...]
    prompt_tokens: tuple[str, ...]
    gen_tokens: tuple[str, ...]
    cache_hit_rate: tuple[str, ...]
    gpu_mem_used: tuple[str, ...]
    gpu_mem_total: tuple[str, ...]
    flush_method: str
    flush_path: str


_VLLM = dict(
    running=("vllm:num_requests_running",),
    waiting=("vllm:num_requests_waiting",),
    # renamed in vLLM v0.10+: kv_cache_usage_perc; older: gpu_cache_usage_perc
    kv_usage=("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    prefix_queries=("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    prefix_hits=("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    prompt_tokens=("vllm:prompt_tokens_total",),
    gen_tokens=("vllm:generation_tokens_total",),
    cache_hit_rate=(),
    flush_method="POST",
    flush_path="/reset_prefix_cache",   # requires VLLM_SERVER_DEV_MODE=1
)

BACKENDS: dict[str, Backend] = {
    "vllm": Backend(**_VLLM, gpu_mem_used=(), gpu_mem_total=()),
    "sim": Backend(**_VLLM, gpu_mem_used=("sim:gpu_memory_used_bytes",), gpu_mem_total=("sim:gpu_memory_total_bytes",)),
    "sglang": Backend(
        running=("sglang:num_running_reqs",),
        waiting=("sglang:num_queue_reqs",),
        kv_usage=("sglang:token_usage",),
        prefix_queries=(),
        prefix_hits=("sglang:cached_tokens_total",),
        prompt_tokens=("sglang:prompt_tokens_total",),
        gen_tokens=("sglang:generation_tokens_total",),
        cache_hit_rate=("sglang:cache_hit_rate",),
        gpu_mem_used=(),
        gpu_mem_total=(),
        flush_method="POST",
        flush_path="/flush_cache",
    ),
}


@dataclass
class WorkerState:
    spec: WorkerSpec
    index: PrefixIndex
    healthy: bool = True
    # router-side load accounting (exact: the router sees every request it proxies)
    inflight: int = 0
    prefill_inflight: int = 0          # dispatched, first token not yet seen
    pending_prefill_tokens: float = 0.0
    admitted: int = 0                  # disagg decode: requests holding an admission slot
    routed_total: int = 0
    errors_total: int = 0
    # prediction quality (fraction of the prompt predicted cached vs reported cached)
    pred_samples: int = 0
    pred_frac_sum: float = 0.0
    actual_frac_sum: float = 0.0
    overpredictions: int = 0
    reports_cached: bool = False       # has this worker ever reported usage cached_tokens?
    # engine telemetry (scraped)
    engine_running: float | None = None
    engine_waiting: float | None = None
    kv_usage: float | None = None
    prefix_queries: float | None = None
    prefix_hits: float | None = None
    cache_hit_rate: float | None = None
    gpu_mem_used: float | None = None
    gpu_mem_total: float | None = None
    last_scrape: float = 0.0
    consecutive_failures: int = 0
    raw_metrics: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def backend(self) -> Backend:
        return BACKENDS[self.spec.backend]

    @property
    def decode_inflight(self) -> int:
        return max(0, self.inflight - self.prefill_inflight)

    def to_dict(self) -> dict:
        n = max(1, self.pred_samples)
        return {
            "name": self.name,
            "url": self.spec.url,
            "backend": self.spec.backend,
            "role": self.spec.role,
            "healthy": self.healthy,
            "inflight": self.inflight,
            "prefill_inflight": self.prefill_inflight,
            "pending_prefill_tokens": round(self.pending_prefill_tokens, 1),
            "admitted": self.admitted,
            "max_inflight": self.spec.max_inflight,
            "routed_total": self.routed_total,
            "errors_total": self.errors_total,
            "engine_running": self.engine_running,
            "engine_waiting": self.engine_waiting,
            "kv_usage": self.kv_usage,
            "prefix_queries": self.prefix_queries,
            "prefix_hits": self.prefix_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "gpu_mem_used": self.gpu_mem_used,
            "gpu_mem_total": self.gpu_mem_total,
            "index_chunks": len(self.index),
            "index_capacity": self.index.capacity,
            "pred_samples": self.pred_samples,
            "pred_cached_frac_mean": self.pred_frac_sum / n,
            "actual_cached_frac_mean": self.actual_frac_sum / n,
            "overpredictions": self.overpredictions,
            "last_scrape_age_s": (time.monotonic() - self.last_scrape) if self.last_scrape else None,
        }


async def scrape_worker(client: httpx.AsyncClient, w: WorkerState, timeout: float = 2.0) -> dict | None:
    """Fetch and parse /metrics; update telemetry fields. Returns parsed metrics or None."""
    try:
        r = await client.get(w.spec.url.rstrip("/") + "/metrics", timeout=timeout)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        w.consecutive_failures += 1
        log.debug("scrape %s failed: %s", w.name, e)
        return None
    m = parse_prometheus_text(r.text)
    b = w.backend
    prev_queries = w.prefix_queries
    w.engine_running = first_sum(m, list(b.running))
    w.engine_waiting = first_sum(m, list(b.waiting))
    w.kv_usage = first_max(m, list(b.kv_usage))
    w.prefix_queries = first_sum(m, list(b.prefix_queries))
    w.prefix_hits = first_sum(m, list(b.prefix_hits))
    w.cache_hit_rate = first_max(m, list(b.cache_hit_rate))
    w.gpu_mem_used = first_sum(m, list(b.gpu_mem_used))
    w.gpu_mem_total = first_sum(m, list(b.gpu_mem_total))
    w.raw_metrics = {k: sum(v) for k, v in m.items() if not k.endswith("_bucket")}
    w.last_scrape = time.monotonic()
    w.consecutive_failures = 0
    # counter went backwards -> worker restarted, its cache is empty
    if prev_queries is not None and w.prefix_queries is not None and w.prefix_queries < prev_queries:
        log.warning("worker %s counters reset (restart?) -> clearing prefix index", w.name)
        w.index.clear()
    return m


async def check_health(client: httpx.AsyncClient, w: WorkerState, timeout: float = 2.0) -> bool:
    try:
        r = await client.get(w.spec.url.rstrip("/") + "/health", timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def flush_worker(client: httpx.AsyncClient, w: WorkerState, timeout: float = 30.0) -> tuple[bool, str]:
    b = w.backend
    url = w.spec.url.rstrip("/") + b.flush_path
    try:
        r = await client.request(b.flush_method, url, timeout=timeout)
        ok = r.status_code < 400
        return ok, f"{r.status_code} {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


async def telemetry_loop(client: httpx.AsyncClient, workers: list[WorkerState], interval: float, unhealthy_after: int) -> None:
    while True:
        await asyncio.gather(*(scrape_worker(client, w) for w in workers))
        for w in workers:
            if w.consecutive_failures >= unhealthy_after and w.healthy:
                log.warning("worker %s marked unhealthy", w.name)
                w.healthy = False
                w.index.clear()
            elif w.consecutive_failures == 0 and not w.healthy:
                log.warning("worker %s healthy again", w.name)
                w.healthy = True
            w.index.expire()
        await asyncio.sleep(interval)
