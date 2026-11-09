"""Continuous-batching engine simulator.

One asyncio task runs the scheduler loop. Each iteration ("step"):

1. **Admit** waiting requests FCFS while there are sequence slots and KV blocks.
   On admission the prefix cache is probed; hit tokens are not recomputed.
2. **Schedule** a batch under a token budget: every decoding sequence contributes
   1 token, then prefilling sequences get prompt chunks (chunked prefill).
3. **Cost** the step with the cost model and sleep for that long.
4. **Apply**: advance prefill progress, emit one token per decoding sequence.

Because prefill chunks and decodes share a step, a long prefill inflates the
step time for every decoding sequence in the batch. That prefill->decode
interference is exactly what disaggregated serving removes.

Disaggregation (mirrors vLLM's NixlConnector request protocol):
* ``remote_decode`` (sent to a prefill worker): run prefill, emit 1 token, finish,
  and return ``kv_transfer_params`` describing where the KV lives.
* ``remote_prefill`` (sent to a decode worker): allocate blocks, *pull* the prompt
  KV over a modeled link (latency + bytes / bandwidth, serialized per engine to
  model contention), compute only the last prompt token, then decode.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field

from kvbench.sim.blocks import BlockManager
from kvbench.sim.config import SimConfig


class RequestTooLarge(Exception):
    pass


@dataclass
class SimRequest:
    rid: str
    prompt_len: int
    block_hashes: list[int]
    max_tokens: int
    remote_prefill: bool = False
    remote_decode: bool = False
    arrival: float = field(default_factory=time.monotonic)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    state: str = "waiting"   # waiting | transferring | prefill | decode | finished | aborted
    hit_hashes: list[int] = field(default_factory=list)
    committed_hashes: list[int] = field(default_factory=list)
    committed: bool = False
    n_new_blocks: int = 0
    cached_tokens: int = 0       # local prefix-cache hit at admission
    prefilled: int = 0           # prompt tokens whose KV is present (cached + transferred + computed)
    prefill_computed: int = 0    # prompt tokens actually computed by this engine
    transferred_tokens: int = 0
    generated: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    aborted: bool = False


@dataclass
class Counters:
    requests_total: int = 0
    requests_finished: int = 0
    requests_aborted: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prefix_queries: int = 0
    prefix_hits: int = 0
    prefill_tokens_computed: int = 0
    kv_tokens_transferred: int = 0
    steps: int = 0
    busy_s: float = 0.0


class SimEngine:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.bs = cfg.block_size
        self.bm = BlockManager(max(1, cfg.kv_capacity_tokens // cfg.block_size), cfg.enable_prefix_caching)
        self.waiting: deque[SimRequest] = deque()
        self.running: list[SimRequest] = []
        self.c = Counters()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._link_free_at = 0.0
        self.started_at = time.monotonic()

    # ---- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ---- API -----------------------------------------------------------------
    def total_blocks(self, req: SimRequest) -> int:
        return math.ceil((req.prompt_len + req.max_tokens) / self.bs)

    def submit(self, req: SimRequest) -> None:
        if self.total_blocks(req) > self.bm.num_blocks:
            raise RequestTooLarge(
                f"request needs {self.total_blocks(req)} blocks; KV pool has {self.bm.num_blocks}"
            )
        self.c.requests_total += 1
        self.c.prompt_tokens += req.prompt_len
        self.waiting.append(req)
        self._wake.set()

    def abort(self, req: SimRequest) -> None:
        if req.state in ("finished", "aborted"):
            return
        req.aborted = True
        self.c.requests_aborted += 1
        if req.state == "waiting":
            try:
                self.waiting.remove(req)
            except ValueError:
                pass
            req.state = "aborted"
        else:
            self._finish(req, aborted=True)
        self._wake.set()

    def reset_prefix_cache(self) -> int:
        return self.bm.reset_prefix_cache()

    # ---- scheduler -----------------------------------------------------------
    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            req = self.waiting[0]
            n_hit = self.bm.match_prefix(req.block_hashes)
            if n_hit * self.bs >= req.prompt_len:       # always compute >= 1 prompt token
                n_hit = (req.prompt_len - 1) // self.bs
            n_new = self.total_blocks(req) - n_hit
            hit = req.block_hashes[:n_hit]
            if not self.bm.can_allocate(hit, n_new):
                break                                   # FCFS: head-of-line waits for memory
            self.bm.allocate(hit, n_new)
            self.waiting.popleft()
            req.hit_hashes = hit
            req.n_new_blocks = n_new
            req.cached_tokens = n_hit * self.bs
            req.prefilled = req.cached_tokens
            self.c.prefix_queries += req.prompt_len
            self.c.prefix_hits += req.cached_tokens
            self.running.append(req)
            if req.remote_prefill:
                req.state = "transferring"
                asyncio.get_running_loop().create_task(self._transfer(req))
            else:
                req.state = "prefill"

    async def _transfer(self, req: SimRequest) -> None:
        cfg = self.cfg
        pull = max(0, req.prompt_len - 1 - req.prefilled)
        loop = asyncio.get_running_loop()
        now = loop.time()
        start = max(now, self._link_free_at)
        dur = pull * cfg.kv_bytes_per_token / (cfg.transfer_gb_per_s * 1e9) / cfg.speed
        self._link_free_at = start + dur
        await asyncio.sleep((start - now) + dur + cfg.transfer_latency_ms / 1000.0 / cfg.speed)
        if req.aborted:
            return
        req.transferred_tokens = pull
        self.c.kv_tokens_transferred += pull
        req.prefilled = req.prompt_len - 1
        self._commit(req)
        req.state = "prefill"            # one remaining prompt token computed locally
        self._wake.set()

    def _commit(self, req: SimRequest) -> None:
        if req.committed:
            return
        req.committed = True
        req.committed_hashes = self.bm.commit(req.block_hashes[len(req.hit_hashes):])

    def _schedule(self) -> tuple[list[SimRequest], list[tuple[SimRequest, int]]]:
        budget = self.cfg.max_num_batched_tokens
        decodes = [r for r in self.running if r.state == "decode"][:budget]
        budget -= len(decodes)
        prefills: list[tuple[SimRequest, int]] = []
        for r in self.running:
            if budget <= 0:
                break
            if r.state == "prefill":
                chunk = min(r.prompt_len - r.prefilled, budget)
                if chunk > 0:
                    prefills.append((r, chunk))
                    budget -= chunk
        return decodes, prefills

    def step_time(self, decodes: list[SimRequest], prefills: list[tuple[SimRequest, int]]) -> float:
        cfg = self.cfg
        t = cfg.step_overhead_ms / 1000.0
        ptoks = sum(c for _, c in prefills)
        t += ptoks / cfg.prefill_tokens_per_s
        if cfg.prefill_attn_ns_per_token_ctx > 0:
            t += sum(c * (r.prefilled + c / 2) for r, c in prefills) * cfg.prefill_attn_ns_per_token_ctx * 1e-9
        if decodes:
            ctx = sum(r.prompt_len + r.generated for r in decodes)
            t += cfg.decode_weights_ms / 1000.0
            t += ctx * cfg.decode_kv_ns_per_token * 1e-9
            t += len(decodes) * cfg.decode_per_seq_ms / 1000.0
        return t

    def _apply(self, decodes: list[SimRequest], prefills: list[tuple[SimRequest, int]]) -> None:
        for r, c in prefills:
            if r.aborted or r.state != "prefill":
                continue
            r.prefilled += c
            r.prefill_computed += c
            self.c.prefill_tokens_computed += c
            if r.prefilled >= r.prompt_len:
                self._commit(r)
                r.state = "decode"
                self._emit(r)            # first output token comes out of the prefill step
        for r in decodes:
            if r.aborted or r.state != "decode":
                continue
            self._emit(r)

    def _emit(self, r: SimRequest) -> None:
        r.generated += 1
        self.c.generation_tokens += 1
        if r.first_token_time is None:
            r.first_token_time = time.monotonic()
        r.queue.put_nowait(("token", r.generated))
        if r.generated >= r.max_tokens:
            self._finish(r)

    def _finish(self, r: SimRequest, aborted: bool = False) -> None:
        if r in self.running:
            self.running.remove(r)
        n_unhashed = r.n_new_blocks - len(r.committed_hashes)
        self.bm.release(r.hit_hashes + r.committed_hashes, n_unhashed)
        r.hit_hashes, r.committed_hashes, r.n_new_blocks = [], [], 0
        r.finish_time = time.monotonic()
        if aborted:
            r.state = "aborted"
        else:
            r.state = "finished"
            self.c.requests_finished += 1
            r.queue.put_nowait(("done", None))

    async def _loop(self) -> None:
        while True:
            self._admit()
            decodes, prefills = self._schedule()
            if not decodes and not prefills:
                self._wake.clear()
                await self._wake.wait()
                continue
            dt = self.step_time(decodes, prefills)
            self.c.steps += 1
            self.c.busy_s += dt
            await asyncio.sleep(dt / self.cfg.speed)
            self._apply(decodes, prefills)

    # ---- telemetry -----------------------------------------------------------
    def snapshot(self) -> dict:
        cfg = self.cfg
        used_bytes = cfg.weights_gb * 1e9 + self.bm.used * self.bs * cfg.kv_bytes_per_token
        running = [r for r in self.running]
        return {
            "engine_id": cfg.engine_id,
            "role": cfg.role,
            "num_running": len(running),
            "num_waiting": len(self.waiting),
            "num_prefilling": sum(1 for r in running if r.state == "prefill"),
            "num_decoding": sum(1 for r in running if r.state == "decode"),
            "num_transferring": sum(1 for r in running if r.state == "transferring"),
            "kv_usage": self.bm.usage(),
            "kv_blocks_total": self.bm.num_blocks,
            "kv_blocks_used": self.bm.used,
            "kv_blocks_cached": self.bm.cached_blocks(),
            "kv_blocks_evictable": len(self.bm.evictable),
            "evictions": self.bm.evictions,
            "gpu_mem_used_bytes": used_bytes,
            "gpu_mem_total_bytes": cfg.gpu_mem_gb * 1e9,
            "uptime_s": time.monotonic() - self.started_at,
            **self.c.__dict__,
        }
