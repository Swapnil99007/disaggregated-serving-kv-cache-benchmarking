"""Routing policies.

* ``round_robin``      — cache-oblivious, load-oblivious baseline.
* ``least_loaded``     — load-aware, cache-oblivious (fewest in-flight requests).
* ``prefix_affinity``  — pure cache affinity: rendezvous-hash the leading prefix to a
                         worker. Maximizes reuse but ignores load, so hot prefixes
                         create hotspots.
* ``weighted``         — the linear score from the write-up:
                         0.5*hit - 0.3*queue_pressure - 0.2*memory_pressure.
                         Easy to explain, but the terms have no common unit.
* ``cache_aware``      — cache-aware *load balancing*: estimate each worker's time
                         to first token as (queued prefill + this request's
                         uncached prefill) / prefill throughput, plus a decode-batch
                         term and a KV-memory-pressure penalty; pick the minimum.
                         A load-imbalance guard falls back to least-loaded so a
                         popular cache owner cannot be overloaded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from kvbench.router.workers import WorkerState


@dataclass
class RouteRequest:
    hashes: list[int]
    est_prompt_tokens: float
    prompt_chars: int


@dataclass
class Decision:
    worker: WorkerState
    matched_chunks: int
    reason: str
    detail: dict = field(default_factory=dict)


class Policy:
    name = "base"

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})

    def select(self, req: RouteRequest, workers: list[WorkerState]) -> Decision:
        raise NotImplementedError


def _match_frac(w: WorkerState, req: RouteRequest) -> tuple[int, float]:
    m = w.index.match(req.hashes)
    return m, (m / len(req.hashes)) if req.hashes else 0.0


class RoundRobin(Policy):
    name = "round_robin"

    def __init__(self, params=None):
        super().__init__(params)
        self._i = 0

    def select(self, req, workers):
        w = workers[self._i % len(workers)]
        self._i += 1
        return Decision(w, w.index.match(req.hashes), "round_robin")


class LeastLoaded(Policy):
    name = "least_loaded"

    def __init__(self, params=None):
        super().__init__(params)
        self._i = 0

    def select(self, req, workers):
        # rotate the starting point so ties don't always land on worker 0
        n = len(workers)
        order = [workers[(self._i + k) % n] for k in range(n)]
        self._i += 1
        w = min(order, key=lambda x: (x.inflight, x.pending_prefill_tokens))
        return Decision(w, w.index.match(req.hashes), "least_loaded")


class PrefixAffinity(Policy):
    """Rendezvous (highest-random-weight) hashing on the first ``key_chunks`` chunks."""

    name = "prefix_affinity"

    def select(self, req, workers):
        key_chunks = int(self.params.get("key_chunks", 16))
        if req.hashes:
            key = req.hashes[min(len(req.hashes), key_chunks) - 1]
        else:
            key = req.prompt_chars
        kb = key.to_bytes(8, "little", signed=False)

        def weight(w: WorkerState) -> int:
            return int.from_bytes(hashlib.blake2b(kb + w.name.encode(), digest_size=8).digest(), "little")

        w = max(workers, key=weight)
        return Decision(w, w.index.match(req.hashes), "affinity")


class Weighted(Policy):
    name = "weighted"

    def select(self, req, workers):
        w_hit = float(self.params.get("w_hit", 0.5))
        w_queue = float(self.params.get("w_queue", 0.3))
        w_mem = float(self.params.get("w_mem", 0.2))
        max_inflight = max(1, max(w.inflight for w in workers))
        best, best_score, best_m = None, None, 0
        for w in workers:
            m, frac = _match_frac(w, req)
            queue_pressure = w.inflight / max_inflight
            mem_pressure = w.kv_usage or 0.0
            score = w_hit * frac - w_queue * queue_pressure - w_mem * mem_pressure
            if best_score is None or score > best_score:
                best, best_score, best_m = w, score, m
        return Decision(best, best_m, "cache_hit" if best_m else "score", {"score": best_score})


class CacheAware(Policy):
    name = "cache_aware"

    def select(self, req, workers):
        p = self.params
        prefill_tps = float(p.get("prefill_tps", 16000.0))
        decode_cost_s = float(p.get("decode_cost_s", 0.005))
        mem_hw = float(p.get("mem_high_watermark", 0.9))
        mem_pen_s = float(p.get("mem_penalty_s", 1.0))
        imb_abs = int(p.get("imbalance_abs", 8))
        imb_rel = float(p.get("imbalance_rel", 1.5))

        loads = [w.inflight for w in workers]
        lo, hi = min(loads), max(loads)
        if hi - lo > imb_abs and hi > imb_rel * lo:
            w = min(workers, key=lambda x: (x.inflight, x.pending_prefill_tokens))
            return Decision(w, w.index.match(req.hashes), "load_guard")

        best, best_key, best_m, best_detail = None, None, 0, {}
        for w in workers:
            m, frac = _match_frac(w, req)
            new_prefill = req.est_prompt_tokens * (1.0 - frac)
            queue_s = (w.pending_prefill_tokens + new_prefill) / prefill_tps
            decode_s = decode_cost_s * w.decode_inflight
            kv = w.kv_usage or 0.0
            mem_s = mem_pen_s * max(0.0, kv - mem_hw) / max(1e-6, 1.0 - mem_hw)
            cost = queue_s + decode_s + mem_s
            # ties (e.g. all idle, no hits): fewer in-flight, then the emptiest index — new
            # prefixes spread across workers instead of piling onto worker 0
            key = (round(cost, 6), w.inflight, len(w.index))
            if best_key is None or key < best_key:
                best, best_key, best_m = w, key, m
                best_detail = {"cost_s": cost, "queue_s": queue_s, "decode_s": decode_s, "mem_s": mem_s}
        return Decision(best, best_m, "cache_hit" if best_m else "cost_min", best_detail)


POLICIES: dict[str, type[Policy]] = {
    p.name: p for p in (RoundRobin, LeastLoaded, PrefixAffinity, Weighted, CacheAware)
}


def make_policy(name: str, all_params: dict | None = None) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]((all_params or {}).get(name, {}))
