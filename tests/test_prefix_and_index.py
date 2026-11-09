from kvbench.common.prefix import chunk_hashes, extract_prompt_text
from kvbench.common.prom import first_max, first_sum, parse_prometheus_text
from kvbench.router.index import PrefixIndex


def test_chunk_hashes_share_prefix():
    base = "A" * 1000
    a = chunk_hashes(base + "x" * 600, 256)
    b = chunk_hashes(base + "y" * 600, 256)
    assert len(a) == len(b) == 6
    # first 3 chunks (768 chars) are identical, the 4th straddles the divergence at 1000
    assert a[:3] == b[:3]
    assert a[3] != b[3]


def test_chunk_hashes_chain_and_namespace():
    t = "hello world " * 100
    assert chunk_hashes(t, 64) == chunk_hashes(t, 64)
    assert chunk_hashes(t, 64, "model-a") != chunk_hashes(t, 64, "model-b")
    # same chunk content at a different position hashes differently (chained)
    h = chunk_hashes("ab" * 64, 64)
    assert h[0] != h[1]
    assert chunk_hashes("short", 256) == []


def test_extract_prompt_text():
    assert extract_prompt_text({"prompt": "hi"}) == "hi"
    assert extract_prompt_text({"prompt": ["a", "b"]}) == "ab"
    chat = extract_prompt_text({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]})
    assert "S" in chat and "U" in chat and chat.index("S") < chat.index("U")


def test_prefix_index_match_insert_lru():
    idx = PrefixIndex(capacity_chunks=4, ttl_s=100)
    idx.insert([1, 2, 3], now=0)
    assert idx.match([1, 2, 3, 4], now=1) == 3
    assert idx.match([2, 3], now=1) == 2       # match is by leading hash only
    assert idx.match([9, 1], now=1) == 0
    idx.insert([5, 6], now=2)                  # 5 entries > capacity 4 -> evict LRU
    assert len(idx) == 4


def test_prefix_index_head_survives_eviction():
    # inserted tail-first so the shared head is most recent and evicted last
    idx = PrefixIndex(capacity_chunks=3, ttl_s=100)
    idx.insert([1, 2, 3], now=0)
    idx.insert([7], now=1)                    # evicts the oldest = tail (3)
    assert idx.match([1, 2, 3], now=1) == 2


def test_prefix_index_ttl_and_expire():
    idx = PrefixIndex(capacity_chunks=10, ttl_s=5)
    idx.insert([1, 2], now=0)
    assert idx.match([1, 2], now=4) == 2
    assert idx.match([1, 2], now=6) == 0
    assert idx.expire(now=6) == 2
    assert len(idx) == 0


def test_prefix_index_shrink_grow():
    idx = PrefixIndex(capacity_chunks=100)
    idx.shrink(0.5)
    assert idx.capacity == 50
    idx.grow(1000)
    assert idx.capacity == 100


def test_prometheus_parser():
    text = """
# HELP x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m",engine="0"} 3.0
vllm:num_requests_running{model_name="m",engine="1"} 2.0
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:prefix_cache_hits_total{model_name="m"} 1.5e3
plain_metric 7
"""
    m = parse_prometheus_text(text)
    assert first_sum(m, ["vllm:num_requests_running"]) == 5.0
    assert first_max(m, ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]) == 0.42
    assert first_sum(m, ["vllm:prefix_cache_hits_total"]) == 1500.0
    assert first_sum(m, ["plain_metric"]) == 7.0
    assert first_sum(m, ["missing"]) is None
