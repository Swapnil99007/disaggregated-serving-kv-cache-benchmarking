"""Approximate per-worker prefix index.

The router never sees a worker's real KV cache, so it records which prefix
chunks it has *sent* to each worker and assumes they are cached there. The
approximation is kept honest by:

* **Capacity bound + LRU**: the index holds at most as many chunks as the worker's
  KV pool could hold, evicting least-recently-used chunks (roughly mirroring
  the worker's own LRU eviction).
* **TTL**: entries not touched within ``ttl_s`` are treated as stale.
* **Explicit invalidation**: cleared when the worker's cache is flushed, it
  restarts (Prometheus counters go backwards), or it is marked unhealthy.
* **Feedback** (optional): when the worker reports fewer cached tokens than
  predicted, the capacity estimate shrinks (``adaptive_capacity``).
"""

from __future__ import annotations

import time
from collections import OrderedDict


class PrefixIndex:
    def __init__(self, capacity_chunks: int, ttl_s: float = 600.0):
        self.capacity = max(1, int(capacity_chunks))
        self.max_capacity = self.capacity
        self.ttl_s = ttl_s
        self._entries: OrderedDict[int, float] = OrderedDict()   # chunk hash -> last access time

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, hashes: list[int], now: float | None = None) -> int:
        """Number of leading chunks of ``hashes`` believed cached (does not touch LRU)."""
        now = time.monotonic() if now is None else now
        n = 0
        for h in hashes:
            ts = self._entries.get(h)
            if ts is None or now - ts > self.ttl_s:
                break
            n += 1
        return n

    def insert(self, hashes: list[int], now: float | None = None) -> None:
        """Record that ``hashes`` are (about to be) cached on this worker."""
        now = time.monotonic() if now is None else now
        # insert tail-first so the shared head of the prefix ends up most recent
        for h in reversed(hashes):
            self._entries[h] = now
            self._entries.move_to_end(h)
        self._evict()

    def _evict(self) -> None:
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def expire(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        n = 0
        while self._entries:
            h, ts = next(iter(self._entries.items()))
            if now - ts <= self.ttl_s:
                break
            self._entries.popitem(last=False)
            n += 1
        return n

    def clear(self) -> None:
        self._entries.clear()

    def shrink(self, factor: float = 0.95, floor_frac: float = 0.25) -> None:
        self.capacity = max(int(self.max_capacity * floor_frac), int(self.capacity * factor))
        self._evict()

    def grow(self, step: int = 1) -> None:
        self.capacity = min(self.max_capacity, self.capacity + step)
