import asyncio

import pytest

from kvbench.sim.blocks import BlockManager
from kvbench.sim.config import SimConfig
from kvbench.sim.engine import RequestTooLarge, SimEngine, SimRequest
from kvbench.sim.tokenizer import block_hashes, tokenize


# ------------------------------------------------------------------ block manager
def test_block_manager_hit_commit_release_cycle():
    bm = BlockManager(10)
    assert bm.match_prefix([1, 2, 3]) == 0
    bm.allocate([], 4)                       # 3 prompt blocks + 1 output block
    assert bm.used == 4 and bm.free == 6
    bm.commit([1, 2, 3])                     # prompt blocks now hashed
    assert bm.match_prefix([1, 2, 3, 4]) == 3
    bm.release([1, 2, 3], n_unhashed=1)
    assert bm.used == 0
    assert len(bm.evictable) == 3            # still cached, just not pinned
    assert bm.match_prefix([1, 2, 9]) == 2


def test_block_manager_evicts_lru_tail_first():
    bm = BlockManager(4)
    bm.allocate([], 3)
    bm.commit([1, 2, 3])
    bm.release([1, 2, 3], 0)                 # released tail-first -> 3 is the LRU
    bm.allocate([], 2)                       # 1 free + must evict 1
    assert bm.evictions == 1
    assert bm.match_prefix([1, 2, 3]) == 2   # head survived, tail evicted


def test_block_manager_capacity_and_dedup():
    bm = BlockManager(4)
    bm.allocate([], 2)
    bm.commit([1, 2])
    assert not bm.can_allocate([], 3)
    # a second request computes the same blocks concurrently -> dedup on commit
    bm.allocate([], 2)
    bm.commit([1, 2])
    assert bm.used == 2                      # duplicates were given back
    assert bm.ref[1] == 2


def test_block_manager_hit_on_evictable_does_not_count_as_free():
    bm = BlockManager(3)
    bm.allocate([], 3)
    bm.commit([1, 2, 3])
    bm.release([1, 2, 3], 0)
    # reusing block 1 from cache leaves only 2 evictable blocks for new allocations
    assert bm.can_allocate([1], 2)
    assert not bm.can_allocate([1], 3)


def test_block_manager_no_caching():
    bm = BlockManager(4, enable_caching=False)
    bm.allocate([], 2)
    assert bm.commit([1, 2]) == []
    bm.release([], 2)
    assert bm.used == 0 and bm.match_prefix([1]) == 0


def test_tokenizer_and_block_hashes_prefix_stable():
    a = tokenize("The cache is warm. The cache is cold!")
    assert a[:4] == ["The", "cache", "is", "warm"]
    long = " ".join(f"w{i}" for i in range(100))
    h1 = block_hashes(tokenize(long), 16)
    h2 = block_hashes(tokenize(long + " extra words here"), 16)
    assert h1 == h2[: len(h1)]


# ------------------------------------------------------------------ engine
def _req(engine: SimEngine, n_tokens: int, max_tokens: int, prefix: str = "p", **kw) -> SimRequest:
    toks = [f"{prefix}{i}" for i in range(n_tokens)]
    return SimRequest(rid=f"r{prefix}{n_tokens}{max_tokens}", prompt_len=n_tokens,
                      block_hashes=block_hashes(toks, engine.bs), max_tokens=max_tokens, **kw)


async def _drain(req: SimRequest) -> int:
    n = 0
    while True:
        kind, _ = await req.queue.get()
        if kind == "done":
            return n
        n += 1


def test_engine_generates_and_reuses_prefix():
    async def go():
        eng = SimEngine(SimConfig(speed=200.0, kv_capacity_tokens=8192))
        eng.start()
        r1 = _req(eng, 640, 5)
        eng.submit(r1)
        assert await _drain(r1) == 5
        assert r1.prefill_computed == 640 and r1.cached_tokens == 0
        r2 = _req(eng, 640, 3)               # identical prompt -> hit (all but last block)
        eng.submit(r2)
        assert await _drain(r2) == 3
        assert r2.cached_tokens == 624       # (640-1)//16 blocks cached
        assert r2.prefill_computed == 16
        assert eng.bm.used == 0              # everything released
        await eng.stop()

    asyncio.run(go())


def test_engine_rejects_oversized_request():
    eng = SimEngine(SimConfig(kv_capacity_tokens=256))
    with pytest.raises(RequestTooLarge):
        eng.submit(_req(eng, 300, 10))


def test_engine_concurrent_batch_and_abort():
    async def go():
        eng = SimEngine(SimConfig(speed=200.0, kv_capacity_tokens=65536, max_num_seqs=4))
        eng.start()
        reqs = [_req(eng, 200, 8, prefix=f"x{i}_") for i in range(6)]
        for r in reqs:
            eng.submit(r)
        eng.abort(reqs[-1])                  # aborted while waiting
        outs = await asyncio.gather(*(_drain(r) for r in reqs[:-1]))
        assert outs == [8] * 5
        assert eng.bm.used == 0
        await eng.stop()

    asyncio.run(go())


def test_engine_step_time_prefill_interference():
    eng = SimEngine(SimConfig())
    d = _req(eng, 100, 10)
    d.state = "decode"
    p = _req(eng, 2000, 10, prefix="q")
    t_decode_only = eng.step_time([d], [])
    t_mixed = eng.step_time([d], [(p, 2000)])
    assert t_mixed > t_decode_only + 0.05    # a 2k-token prefill chunk stalls the decode step


def test_engine_remote_prefill_transfer():
    async def go():
        eng = SimEngine(SimConfig(speed=200.0, role="decode"))
        eng.start()
        r = _req(eng, 1000, 4, remote_prefill=True)
        eng.submit(r)
        assert await _drain(r) == 4
        assert r.transferred_tokens == 999
        assert r.prefill_computed == 1       # only the last prompt token is computed locally
        await eng.stop()

    asyncio.run(go())
