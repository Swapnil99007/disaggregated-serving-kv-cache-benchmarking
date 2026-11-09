"""Per-run summaries and cross-repeat aggregation."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from kvbench.common.io import read_json, read_jsonl
from kvbench.common.stats import describe, mean_ci95

LAT_METRICS = ("ttft", "tpot", "e2e", "itl_mean")
LAT_STATS = ("mean", "p50", "p90", "p95", "p99")

# metrics averaged across repeats in the aggregate table
AGG_KEYS = (
    [f"{m}_{s}" for m in LAT_METRICS for s in LAT_STATS]
    + [
        "throughput_rps", "steady_throughput_rps", "output_tok_s", "error_rate",
        "prompt_tokens_total", "cached_tokens_total", "prefill_computed_total", "cache_hit_rate",
        "predicted_match_frac", "load_imbalance", "engine_prefix_hit_rate",
    ]
)


def summarize_run(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    n, n_ok = len(records), len(ok)
    out: dict = {"n": n, "n_ok": n_ok, "error_rate": (n - n_ok) / n if n else 0.0}
    for m in LAT_METRICS:
        d = describe(r.get(f"{m}_s") for r in ok)
        for s in LAT_STATS:
            out[f"{m}_{s}"] = d[s]
    if not ok:
        return out
    starts = [r["start_ts"] for r in ok]
    ends = sorted(r["end_ts"] for r in ok)
    makespan = max(ends) - min(starts)
    out["makespan_s"] = makespan
    out["throughput_rps"] = n_ok / makespan if makespan > 0 else None
    out["output_tok_s"] = sum(r.get("completion_tokens") or 0 for r in ok) / makespan if makespan > 0 else None
    # steady-state throughput: completions between the 10th and 90th percentile completion
    # times, which excludes ramp-up and the drain tail of a closed-loop run
    if n_ok >= 10:
        i10 = int(math.floor(0.1 * (n_ok - 1)))
        i90 = int(math.ceil(0.9 * (n_ok - 1)))
        window = ends[i90] - ends[i10]
        out["steady_throughput_rps"] = (i90 - i10) / window if window > 0 else None
    else:
        out["steady_throughput_rps"] = out["throughput_rps"]

    # vLLM omits usage.prompt_tokens_details entirely when a request had ZERO cached tokens,
    # so "missing" means 0 whenever the backend reports cached tokens at all. Decide that per
    # run: if any request reported a value, treat missing values as 0 (cold misses must count,
    # or the hit rate is inflated); if none did, the backend isn't reporting -> unknown.
    reports_cache = any(r.get("cached_tokens") is not None or r.get("prefill_cached_tokens") is not None for r in ok)

    def _computed(r: dict) -> tuple[int, int] | None:
        """(prompt_tokens, computed_prefill_tokens) for one request, or None if unknown."""
        if r.get("prefill_prompt_tokens") is not None:          # disagg: prefill worker's view
            pt, ct = r["prefill_prompt_tokens"], r.get("prefill_cached_tokens")
        elif r.get("prompt_tokens") is not None:
            pt, ct = r["prompt_tokens"], r.get("cached_tokens")
        else:
            return None
        if ct is None and not reports_cache:
            return None
        return pt, pt - (ct or 0)

    with_cache = [x for x in (_computed(r) for r in ok) if x is not None]
    prompt_total = sum(pt for pt, _ in with_cache)
    computed = sum(c for _, c in with_cache)
    out["cache_info_coverage"] = len(with_cache) / n_ok
    out["prompt_tokens_total"] = float(sum(r.get("prompt_tokens") or 0 for r in ok))
    out["prompt_chars_total"] = float(sum(r.get("prompt_chars") or 0 for r in ok))
    out["prefill_computed_total"] = float(computed) if with_cache else None
    out["cached_tokens_total"] = float(prompt_total - computed) if with_cache else None
    out["cache_hit_rate"] = (prompt_total - computed) / prompt_total if with_cache and prompt_total else None

    pm = [r["matched_chunks"] / r["prompt_chunks"] for r in ok if r.get("matched_chunks") is not None and r.get("prompt_chunks")]
    out["predicted_match_frac"] = sum(pm) / len(pm) if pm else None

    # load balance: requests per (prefill) worker, max / mean
    counts = Counter((r.get("prefill_worker") or r.get("worker")) for r in ok if (r.get("prefill_worker") or r.get("worker")))
    if counts:
        mean_c = sum(counts.values()) / len(counts)
        out["load_imbalance"] = max(counts.values()) / mean_c if mean_c else None
        out["requests_per_worker"] = dict(counts)
    return out


def engine_totals(engine_deltas: dict) -> dict:
    """Sum per-worker Prometheus counter deltas into cluster totals."""
    tot: dict[str, float] = defaultdict(float)
    for metrics in (engine_deltas or {}).values():
        for k, v in metrics.items():
            tot[k] += v
    out = {}
    q = tot.get("vllm:prefix_cache_queries_total") or tot.get("vllm:prefix_cache_queries")
    h = tot.get("vllm:prefix_cache_hits_total") or tot.get("vllm:prefix_cache_hits")
    out["engine_prefix_query_tokens"] = q
    out["engine_prefix_hit_tokens"] = h
    out["engine_prefix_hit_rate"] = (h / q) if q else None
    out["engine_prefill_computed_tokens"] = tot.get("sim:prefill_tokens_computed_total")
    out["engine_kv_tokens_transferred"] = tot.get("sim:kv_tokens_transferred_total")
    return out


def load_runs(exp_dirs: list[str | Path]) -> list[dict]:
    """Load every run under the given experiment directories and (re)compute summaries."""
    runs = []
    for exp in exp_dirs:
        exp = Path(exp)
        trace_stats = read_json(exp / "trace_stats.json") if (exp / "trace_stats.json").exists() else {}
        for run_json in sorted(exp.glob("runs/*/run.json")):
            meta = read_json(run_json)
            records = read_jsonl(run_json.parent / "requests.jsonl")
            s = summarize_run(records)
            s.update(engine_totals(meta.get("engine_deltas", {})))
            runs.append({
                "exp_dir": str(exp),
                "run_dir": str(run_json.parent),
                "label": meta["label"],
                "deployment": meta["deployment"],
                "policy": meta["policy"],
                "concurrency": meta["concurrency"],
                "repeat": meta["repeat"],
                "trace_stats": trace_stats,
                "summary": s,
            })
    return runs


def aggregate(runs: list[dict]) -> list[dict]:
    """Mean / std / 95% CI across repeats for each (label, concurrency)."""
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["label"], r["concurrency"])].append(r)
    rows = []
    for (label, conc), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"label": label, "deployment": rs[0]["deployment"], "policy": rs[0]["policy"],
               "concurrency": conc, "repeats": len(rs)}
        for k in AGG_KEYS:
            m, sd, ci = mean_ci95(r["summary"].get(k) for r in rs)
            row[k] = m
            row[f"{k}_std"] = sd
            row[f"{k}_ci95"] = ci
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], (dict, list)):
                keys.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def flatten_runs(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        row = {k: r[k] for k in ("label", "deployment", "policy", "concurrency", "repeat", "run_dir")}
        row.update({k: v for k, v in r["summary"].items() if not isinstance(v, (dict, list))})
        rows.append(row)
    return rows
