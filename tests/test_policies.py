import pytest

from kvbench.router.config import WorkerSpec
from kvbench.router.index import PrefixIndex
from kvbench.router.policies import RouteRequest, make_policy
from kvbench.router.workers import WorkerState


def make_workers(n=3):
    return [WorkerState(spec=WorkerSpec(name=f"w{i}", url=f"http://w{i}"), index=PrefixIndex(1000)) for i in range(n)]


def rr(hashes, tokens=4000.0):
    return RouteRequest(hashes=hashes, est_prompt_tokens=tokens, prompt_chars=int(tokens * 4))


def test_round_robin_cycles():
    ws = make_workers(3)
    p = make_policy("round_robin")
    picks = [p.select(rr([1]), ws).worker.name for _ in range(6)]
    assert picks == ["w0", "w1", "w2", "w0", "w1", "w2"]


def test_least_loaded_picks_min_inflight():
    ws = make_workers(3)
    ws[0].inflight, ws[1].inflight, ws[2].inflight = 5, 1, 3
    assert make_policy("least_loaded").select(rr([1]), ws).worker.name == "w1"


def test_prefix_affinity_is_sticky_and_load_oblivious():
    ws = make_workers(4)
    p = make_policy("prefix_affinity", {"prefix_affinity": {"key_chunks": 2}})
    first = p.select(rr([10, 20, 30]), ws).worker
    first.inflight = 100                     # overloaded, but affinity ignores load
    assert p.select(rr([10, 20, 99]), ws).worker is first   # same first 2 chunks -> same worker


def test_cache_aware_prefers_cache_hit():
    ws = make_workers(3)
    hashes = list(range(1, 21))
    ws[2].index.insert(hashes)
    d = make_policy("cache_aware").select(rr(hashes), ws)
    assert d.worker.name == "w2"
    assert d.matched_chunks == 20 and d.reason == "cache_hit"


def test_cache_aware_avoids_busy_cache_owner():
    """Worker A has the prefix but a deep prefill queue; worker C has it and is idle."""
    ws = make_workers(3)
    hashes = list(range(1, 21))
    ws[0].index.insert(hashes)
    ws[2].index.insert(hashes)
    ws[0].pending_prefill_tokens = 50_000
    ws[0].inflight = ws[0].prefill_inflight = 6
    assert make_policy("cache_aware").select(rr(hashes), ws).worker.name == "w2"


def test_cache_aware_hit_not_worth_queueing():
    """The only cache owner is so backed up that recomputing elsewhere is cheaper."""
    ws = make_workers(2)
    hashes = list(range(1, 21))
    ws[0].index.insert(hashes)
    ws[0].pending_prefill_tokens = 200_000   # ~10 s of queued prefill at 20k tok/s
    ws[0].inflight = ws[0].prefill_inflight = 4
    d = make_policy("cache_aware", {"cache_aware": {"prefill_tps": 20000}}).select(rr(hashes, 4000), ws)
    assert d.worker.name == "w1"


def test_cache_aware_memory_pressure_penalty():
    ws = make_workers(2)
    hashes = list(range(1, 11))
    ws[0].index.insert(hashes)
    ws[1].index.insert(hashes)
    ws[0].kv_usage = 0.99                     # nearly full KV pool -> risk of eviction/preemption
    ws[1].kv_usage = 0.40
    assert make_policy("cache_aware").select(rr(hashes), ws).worker.name == "w1"


def test_cache_aware_load_guard():
    ws = make_workers(3)
    hashes = list(range(1, 11))
    ws[0].index.insert(hashes)
    ws[0].inflight, ws[1].inflight, ws[2].inflight = 20, 2, 5
    d = make_policy("cache_aware", {"cache_aware": {"imbalance_abs": 8, "imbalance_rel": 1.5}}).select(rr(hashes), ws)
    assert d.reason == "load_guard" and d.worker.name == "w1"


def test_cache_aware_tie_spreads_new_prefixes():
    ws = make_workers(3)
    ws[0].index.insert([100, 101, 102])      # w0 already holds other content
    d = make_policy("cache_aware").select(rr([1, 2, 3]), ws)
    assert d.worker.name in ("w1", "w2")     # idle tie -> emptiest index, not w0


def test_weighted_policy():
    ws = make_workers(2)
    hashes = list(range(1, 11))
    ws[0].index.insert(hashes)
    assert make_policy("weighted").select(rr(hashes), ws).worker.name == "w0"
    # w0: 0.4*hit(1.0) - 0.3*queue(1.0) - 0.2*mem(1.0) = -0.1 ; w1: 0 - 0 - 0 = 0.0 -> w1 wins
    ws[0].inflight, ws[0].kv_usage = 10, 1.0
    ws[1].inflight, ws[1].kv_usage = 0, 0.0
    d = make_policy("weighted", {"weighted": {"w_hit": 0.4}}).select(rr(hashes), ws)
    assert d.worker.name == "w1"


def test_unknown_policy():
    with pytest.raises(ValueError):
        make_policy("nope")
