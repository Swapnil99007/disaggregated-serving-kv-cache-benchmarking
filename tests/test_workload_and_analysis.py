from kvbench.analysis.prefill import prefill_table
from kvbench.analysis.slo import matched_slo_table, throughput_at_slo
from kvbench.analysis.summarize import aggregate, summarize_run
from kvbench.workload.rag import RagWorkloadConfig, build_corpus, generate_rag_trace, generate_warmup_trace
from kvbench.workload.trace import group_conversations, load_trace, save_trace, trace_stats


def small_cfg(**kw):
    d = dict(num_prompts=60, num_topics=8, docs_per_topic=3, doc_words=(50, 80), system_prompt_words=(30, 40))
    d.update(kw)
    return RagWorkloadConfig(**d)


def test_trace_is_deterministic_and_sized(tmp_path):
    a = generate_rag_trace(small_cfg())
    b = generate_rag_trace(small_cfg())
    assert len(a) == 60
    assert [r.prompt for r in a] == [r.prompt for r in b]
    p = tmp_path / "t.jsonl"
    save_trace(p, a)
    assert [r.prompt for r in load_trace(p)] == [r.prompt for r in a]


def test_multi_turn_prompts_extend_previous_turn():
    trace = generate_rag_trace(small_cfg(mean_turns=4))
    for conv in group_conversations(trace):
        for prev, cur in zip(conv, conv[1:]):
            # turn t+1 = turn t's prompt minus the trailing "Assistant:" + answer + new question
            stem = prev.prompt[: -len("Assistant:")]
            assert cur.prompt.startswith(stem)


def test_trace_stats_reuse_bounds():
    trace = generate_rag_trace(small_cfg())
    s = trace_stats(trace, chunk_chars=64)
    assert 0.0 < s["max_reuse_fraction"] < 1.0
    assert s["unique_prefix_chars"] <= s["prompt_chars_total"]


def test_warmup_trace_shares_corpus_not_requests():
    cfg = small_cfg()
    main = generate_rag_trace(cfg)
    warm = generate_warmup_trace(cfg, 20)
    assert len(warm) == 20 and all(r.request_id.startswith("warm-") for r in warm)
    assert [r.prompt for r in warm] != [r.prompt for r in main[:20]]     # different request stream
    corpus = build_corpus(cfg)                                            # ...over the same corpus
    assert all(any(w.prompt.startswith(sp) for sp in corpus.system_prompts) for w in warm)


def _rec(i, ttft, e2e, out=10, prompt=100, cached=50, worker="w0"):
    return {"status": "ok", "start_ts": float(i), "end_ts": float(i) + e2e, "ttft_s": ttft, "e2e_s": e2e,
            "tpot_s": (e2e - ttft) / (out - 1), "itl_mean_s": 0.01, "completion_tokens": out,
            "prompt_tokens": prompt, "cached_tokens": cached,
            "prefill_computed_tokens": (prompt - cached) if cached is not None else None,
            "prompt_chars": prompt * 4, "worker": worker, "matched_chunks": 1, "prompt_chunks": 2}


def test_summarize_run():
    recs = [_rec(i, 0.1, 1.0, worker=f"w{i % 2}") for i in range(20)]
    recs.append({"status": "error", "start_ts": 0, "end_ts": 1})
    s = summarize_run(recs)
    assert s["n"] == 21 and s["n_ok"] == 20
    assert abs(s["error_rate"] - 1 / 21) < 1e-9
    assert abs(s["ttft_p50"] - 0.1) < 1e-9
    assert abs(s["cache_hit_rate"] - 0.5) < 1e-9
    assert s["prefill_computed_total"] == 20 * 50
    assert s["load_imbalance"] == 1.0
    assert s["steady_throughput_rps"] > 0


def test_summarize_counts_cold_misses_when_backend_omits_cached_field():
    # vLLM drops prompt_tokens_details when cached_tokens == 0 -> cached_tokens arrives as None
    recs = [_rec(i, 0.05, 1.0, prompt=1000, cached=900) if i % 2 == 0 else _rec(i, 0.2, 1.0, prompt=1000, cached=None)
            for i in range(20)]
    s = summarize_run(recs)
    assert s["cache_info_coverage"] == 1.0
    assert s["prefill_computed_total"] == 10 * 100 + 10 * 1000      # misses count as full prefill
    assert abs(s["cache_hit_rate"] - 9000 / 20000) < 1e-9
    # a backend that never reports cached tokens -> unknown, not zero
    s2 = summarize_run([_rec(i, 0.2, 1.0, prompt=1000, cached=None) for i in range(20)])
    assert s2["prefill_computed_total"] is None and s2["cache_hit_rate"] is None


def test_throughput_at_slo_interpolates():
    pts = [(8, 1.0, 10.0), (16, 2.0, 18.0), (32, 4.0, 22.0)]
    x, note = throughput_at_slo(pts, 3.0)
    assert note == "ok" and abs(x - 20.0) < 1e-9
    x, note = throughput_at_slo(pts, 0.5)
    assert x is None
    x, note = throughput_at_slo(pts, 10.0)
    assert x == 22.0 and "lower bound" in note


def test_throughput_at_slo_flags_flat_region():
    # unsaturated closed loop: latency ~constant while throughput doubles per step
    pts = [(8, 1.10, 13.0), (16, 1.12, 26.0), (32, 1.30, 50.0), (64, 2.0, 70.0)]
    x, note = throughput_at_slo(pts, 1.11)
    assert "flat region" in note
    x, note = throughput_at_slo(pts, 1.6)            # crossing on the steep 32->64 segment
    assert note == "ok" and 50.0 < x < 70.0


def test_matched_slo_auto_target_uses_peak_baseline_latency():
    runs = [_run("d/round_robin", c, 0, l, x, 800) for c, l, x in [(8, 1.0, 10.0), (32, 3.0, 20.0)]]
    runs += [_run("d/cache_aware", c, 0, l, x, 600) for c, l, x in [(8, 0.9, 11.0), (32, 2.0, 24.0)]]
    slo = matched_slo_table(aggregate(runs), runs, "d/round_robin")
    assert slo["auto_target"] and slo["target_s"] == 3.0
    by = {r["label"]: r for r in slo["rows"]}
    assert by["d/round_robin"]["throughput_at_target"] == 20.0
    assert "lower bound" in by["d/cache_aware"]["note"]
    assert abs(by["d/cache_aware"]["improvement_pct"] - 20.0) < 1e-9


def _run(label, conc, rep, lat, thr, computed):
    s = {"e2e_p95": lat, "steady_throughput_rps": thr, "prompt_tokens_total": 1000.0,
         "prefill_computed_total": computed, "prompt_chars_total": 4000.0, "cache_hit_rate": 1 - computed / 1000}
    return {"label": label, "deployment": "d", "policy": label.split("/")[1], "concurrency": conc, "repeat": rep,
            "summary": s, "trace_stats": {"unique_prefix_chars": 2000.0}, "run_dir": "/tmp/x"}


def test_matched_slo_and_prefill_tables():
    runs = []
    for rep in range(2):
        runs += [_run("d/round_robin", 8, rep, 1.0, 10.0, 800), _run("d/round_robin", 16, rep, 2.0, 15.0, 800),
                 _run("d/round_robin", 32, rep, 4.0, 18.0, 800),
                 _run("d/cache_aware", 8, rep, 0.8, 11.0, 600), _run("d/cache_aware", 16, rep, 1.5, 17.0, 600),
                 _run("d/cache_aware", 32, rep, 3.0, 22.0, 600)]
    agg = aggregate(runs)
    slo = matched_slo_table(agg, runs, "d/round_robin", "e2e", 95, target_s=2.0)
    by = {r["label"]: r for r in slo["rows"]}
    assert abs(by["d/round_robin"]["throughput_at_target"] - 15.0) < 1e-9
    # cache_aware: between (1.5, 17) and (3.0, 22) at 2.0 -> 17 + 5/3
    assert abs(by["d/cache_aware"]["throughput_at_target"] - (17 + 5 / 3)) < 1e-9
    assert by["d/cache_aware"]["improvement_pct"] > 0
    assert by["d/cache_aware"]["repeats"] == 2

    pre = prefill_table(runs, "d/round_robin")
    all_ca = [r for r in pre if r["label"] == "d/cache_aware" and r["concurrency"] == "all"][0]
    assert abs(all_ca["total_prefill_reduction_pct"] - 25.0) < 1e-9
    # ideal = 2000 chars * (1000 tok / 4000 chars) = 500 tokens per run -> redundant 300 vs 100
    assert abs(all_ca["redundant_prefill_reduction_pct"] - (300 - 100) / 300 * 100) < 1e-9
