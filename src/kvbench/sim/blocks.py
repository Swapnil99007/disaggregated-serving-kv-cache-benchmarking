"""Paged KV-cache block manager with hash-based prefix caching.

Modeled on vLLM's design:

* The KV pool is ``num_blocks`` fixed-size blocks.
* A *full* prompt block, once computed, is registered under its chained hash.
* Blocks referenced by running requests are pinned (refcount > 0).
* When the last reference is dropped, a hashed block is not discarded: it moves to
  an LRU *evictable* pool and can still be hit by later requests.
* Allocation takes truly free blocks first, then evicts the least-recently-used
  evictable blocks.
* Requests release blocks tail-first, so the shared head of a prefix is the most
  recently used and is evicted last.

"KV usage" (what vLLM reports as kv_cache_usage_perc) counts only pinned blocks;
evictable cached blocks are considered free.
"""

from __future__ import annotations

from collections import OrderedDict


class BlockManager:
    def __init__(self, num_blocks: int, enable_caching: bool = True):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        self.ref: dict[int, int] = {}                       # hashed, pinned blocks -> refcount
        self.evictable: OrderedDict[int, None] = OrderedDict()  # hashed, refcount 0, LRU order
        self.unhashed_used = 0                              # pinned blocks without a hash
        self.evictions = 0

    # ---- accounting ----------------------------------------------------------
    @property
    def used(self) -> int:
        return len(self.ref) + self.unhashed_used

    @property
    def free(self) -> int:
        """Blocks that hold nothing at all (excludes evictable cached blocks)."""
        return self.num_blocks - len(self.ref) - len(self.evictable) - self.unhashed_used

    def usage(self) -> float:
        return self.used / self.num_blocks

    def cached_blocks(self) -> int:
        return len(self.ref) + len(self.evictable)

    # ---- prefix cache --------------------------------------------------------
    def match_prefix(self, hashes: list[int]) -> int:
        if not self.enable_caching:
            return 0
        n = 0
        for h in hashes:
            if h in self.ref or h in self.evictable:
                n += 1
            else:
                break
        return n

    def can_allocate(self, hit_hashes: list[int], n_new: int) -> bool:
        hit_evictable = sum(1 for h in hit_hashes if h in self.evictable)
        return n_new <= self.free + len(self.evictable) - hit_evictable

    def allocate(self, hit_hashes: list[int], n_new: int) -> None:
        """Pin ``hit_hashes`` and reserve ``n_new`` fresh (unhashed) blocks."""
        if not self.can_allocate(hit_hashes, n_new):
            raise RuntimeError("insufficient KV blocks")
        for h in hit_hashes:
            if h in self.evictable:
                del self.evictable[h]
                self.ref[h] = 1
            else:
                self.ref[h] += 1
        deficit = n_new - self.free
        for _ in range(max(0, deficit)):
            self.evictable.popitem(last=False)
            self.evictions += 1
        self.unhashed_used += n_new

    def commit(self, new_hashes: list[int]) -> list[int]:
        """Freshly computed full prompt blocks become shareable under their hashes.

        If another request registered the same hash meanwhile, share that block
        and give ours back (deduplication). Returns the hashes now pinned on the
        caller's behalf (empty when caching is disabled: blocks stay unhashed).
        """
        if not self.enable_caching:
            return []
        for h in new_hashes:
            self.unhashed_used -= 1
            if h in self.ref:
                self.ref[h] += 1
            elif h in self.evictable:
                del self.evictable[h]
                self.ref[h] = 1
            else:
                self.ref[h] = 1
        return list(new_hashes)

    def release(self, hashed: list[int], n_unhashed: int) -> None:
        self.unhashed_used -= n_unhashed
        if not self.enable_caching:
            return
        for h in reversed(hashed):  # tail first -> head of prefix stays most recent
            c = self.ref.get(h)
            if c is None:
                continue
            if c <= 1:
                del self.ref[h]
                self.evictable[h] = None
                self.evictable.move_to_end(h)
            else:
                self.ref[h] = c - 1

    def reset_prefix_cache(self) -> int:
        """Drop all evictable cached blocks (pinned blocks are untouched)."""
        n = len(self.evictable)
        self.evictable.clear()
        return n
