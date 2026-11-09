"""Experiment runner: sweeps policies x concurrency x repeats against a running router.

Per run:
  1. switch the router policy (``POST /admin/policy``)
  2. flush worker prefix caches + clear the router index (cold start)
  3. engine warm-up with short unrelated prompts (CUDA graphs, allocator, connections),
     then flush again so warm-up never pollutes the prefix cache
  4. optional prefix warm-up: replay a *different* request stream over the same corpus
     (records warm-cache behaviour; excluded from metrics)
  5. snapshot engine counters, start telemetry samplers
  6. closed-loop replay of the trace at the given concurrency
  7. stop samplers, snapshot counters again, write artifacts

Policies are run in a shuffled order inside each (repeat, concurrency) block so that
slow drift in the environment (thermal, noisy neighbours) is not confounded with
the policy under test.
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx

from kvbench.analysis.summarize import engine_totals, summarize_run
from kvbench.bench.client import run_closed_loop, send_request
from kvbench.bench.config import ExperimentConfig
from kvbench.bench.telemetry import NvmlSampler, RouterStateSampler
from kvbench.common.io import read_yaml, write_json, write_jsonl
from kvbench.workload.rag import RagWorkloadConfig, generate_rag_trace, generate_warmup_trace
from kvbench.workload.trace import TraceRequest, group_conversations, load_trace, save_trace, trace_stats, truncate_trace
from kvbench.workload.words import VOCAB


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_or_generate_trace(trace_cfg: dict) -> tuple[list[TraceRequest], RagWorkloadConfig | None]:
    path = Path(trace_cfg["path"])
    wl_path = trace_cfg.get("workload")
    wl = RagWorkloadConfig.from_dict(read_yaml(wl_path)) if wl_path else None
    if path.exists():
        trace = load_trace(path)
    else:
        if wl is None:
            raise FileNotFoundError(f"trace {path} not found and no workload config given")
        _log(f"generating trace {path} from {wl_path}")
        trace = generate_rag_trace(wl)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_trace(path, trace)
    return truncate_trace(trace, int(trace_cfg.get("max_prompts") or 0)), wl


async def wait_until_ready(client: httpx.AsyncClient, url: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    state: dict = {}
    while time.monotonic() < deadline:
        try:
            state = (await client.get(url + "/admin/state", timeout=5.0)).json()
            ws = state.get("workers", [])
            healthy = [w["name"] for w in ws if w["healthy"]]
            if ws and len(healthy) == len(ws):
                return state
            if time.monotonic() - last_print > 10:
                _log(f"waiting for workers: {len(healthy)}/{len(ws)} healthy")
                last_print = time.monotonic()
        except Exception as e:  # noqa: BLE001
            if time.monotonic() - last_print > 10:
                _log(f"waiting for router at {url}: {type(e).__name__}")
                last_print = time.monotonic()
        await asyncio.sleep(2.0)
    if state.get("workers") and any(w["healthy"] for w in state["workers"]):
        _log("WARNING: not all workers healthy; continuing with the healthy subset")
        return state
    raise TimeoutError(f"router at {url} not ready after {timeout_s}s")


async def _admin(client: httpx.AsyncClient, url: str, path: str, body: dict | None = None) -> dict:
    r = await client.post(url + path, json=body or {}, timeout=60.0)
    r.raise_for_status()
    return r.json()


async def engine_warmup(url: str, model: str, n: int) -> None:
    rng = random.Random(1234)
    reqs = [
        TraceRequest(request_id=f"engine-warmup-{i}", conversation_id=f"ew{i}", turn=0, tenant=-1, topic=-1,
                     doc_ids=[], prompt=" ".join(rng.choice(VOCAB) for _ in range(64)), max_tokens=4)
        for i in range(n)
    ]
    async with httpx.AsyncClient(timeout=120.0) as client:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(send_request(client, url, model, r, t0) for r in reqs))
    bad = [r for r in res if r.status != "ok"]
    if bad:
        raise RuntimeError(f"engine warm-up failed: {bad[0].error}")


def _metric_deltas(before: dict, after: dict) -> dict:
    out = {}
    for wname, a in after.items():
        b = (before.get(wname) or {}).get("metrics", {})
        am = a.get("metrics", {})
        d = {}
        for k, v in am.items():
            if k.endswith("_total") or k in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits"):
                if k in b:
                    d[k] = v - b[k]
        out[wname] = d
    return out


def _progress_printer(label: str):
    state = {"next": 0.1}

    def cb(done: int, total: int) -> None:
        frac = done / total
        if frac >= state["next"] or done == total:
            print(f"    {label}: {done}/{total} ({frac:.0%})", flush=True)
            while state["next"] <= frac:
                state["next"] += 0.1

    return cb


async def run_one(cfg: ExperimentConfig, client: httpx.AsyncClient, convs: list[list[TraceRequest]],
                  warm_convs: list[list[TraceRequest]], policy: str, conc: int, rep: int, run_dir: Path) -> dict:
    url = cfg.router_url.rstrip("/")
    label = f"{cfg.deployment}/{policy}"
    await _admit_policy(client, url, policy, cfg.decode_policy)
    if cfg.flush_between_runs:
        await _admin(client, url, "/admin/reset", {"flush_workers": True})
    n_ew = int(cfg.warmup.get("engine_requests") or 0)
    if n_ew:
        await engine_warmup(url, cfg.model, n_ew)
        await _admin(client, url, "/admin/reset", {"flush_workers": cfg.flush_between_runs})
    if warm_convs:
        _log(f"  prefix warm-up: {sum(len(c) for c in warm_convs)} prompts")
        await run_closed_loop(warm_convs, url, cfg.model, min(conc, 16), cfg.request.get("ignore_eos", True))
        await _admin(client, url, "/admin/reset", {"flush_workers": False, "clear_index": False})

    before = (await client.get(url + "/admin/worker_metrics", timeout=30.0)).json()
    nvml = NvmlSampler(cfg.telemetry.get("interval_s", 0.5)) if cfg.telemetry.get("nvml", True) else None
    rs = RouterStateSampler(url, cfg.telemetry.get("interval_s", 0.5))
    t0 = time.perf_counter()
    if nvml:
        nvml.start(t0)
    rs.start(t0)
    req_cfg = cfg.request
    results, wall = await run_closed_loop(
        convs, url, cfg.model, conc,
        ignore_eos=req_cfg.get("ignore_eos", True),
        temperature=req_cfg.get("temperature", 0.0),
        think_time_s=req_cfg.get("think_time_s", 0.0),
        timeout_s=req_cfg.get("timeout_s", 900.0),
        extra_body=req_cfg.get("extra_body") or None,
        progress=_progress_printer(f"{label} c={conc} r={rep}"),
    )
    router_rows = await rs.stop()
    gpu_rows = nvml.stop() if nvml else []
    after = (await client.get(url + "/admin/worker_metrics", timeout=30.0)).json()
    router_state = (await client.get(url + "/admin/state", timeout=30.0)).json()

    records = [r.to_dict() for r in results]
    deltas = _metric_deltas(before, after)
    summary = summarize_run(records)
    summary.update(engine_totals(deltas))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "requests.jsonl", records)
    write_jsonl(run_dir / "telemetry_router.jsonl", router_rows)
    if gpu_rows:
        write_jsonl(run_dir / "telemetry_gpu.jsonl", gpu_rows)
    write_json(run_dir / "run.json", {
        "label": label, "deployment": cfg.deployment, "policy": policy, "concurrency": conc, "repeat": rep,
        "wall_s": wall, "finished_at": datetime.now().isoformat(),
        "nvml_available": bool(nvml and nvml.available), "nvml_error": nvml.error if nvml else None,
        "engine_deltas": deltas, "router_state_after": router_state, "summary": summary,
    })
    return summary


async def _admit_policy(client: httpx.AsyncClient, url: str, policy: str, decode_policy: str | None) -> None:
    body = {"policy": policy}
    if decode_policy:
        body["decode_policy"] = decode_policy
    await _admin(client, url, "/admin/policy", body)


def _fmt(v, scale=1.0, nd=3) -> str:
    return "-" if v is None else f"{v * scale:.{nd}f}"


async def run_experiment_async(cfg: ExperimentConfig) -> Path:
    trace, wl = load_or_generate_trace(cfg.trace)
    convs = group_conversations(trace)
    n_pw = int(cfg.warmup.get("prefix_warmup_prompts") or 0)
    warm_convs = group_conversations(generate_warmup_trace(wl, n_pw)) if (n_pw and wl) else []

    exp_dir = Path(cfg.output_dir) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    chunk_chars = int(cfg.analysis.get("chunk_chars", 256))
    ts = trace_stats(trace, chunk_chars=chunk_chars)
    write_json(exp_dir / "trace_stats.json", ts)
    write_json(exp_dir / "experiment.json", {"config": asdict(cfg), "argv": sys.argv})
    _log(f"experiment dir: {exp_dir}")
    _log(f"trace: {ts['num_prompts']} prompts, {ts['num_conversations']} conversations, "
         f"mean prompt {ts['prompt_chars_mean']:.0f} chars, max prefix reuse {ts['max_reuse_fraction']:.1%}")

    url = cfg.router_url.rstrip("/")
    async with httpx.AsyncClient() as client:
        state = await wait_until_ready(client, url, cfg.ready_timeout_s)
        write_json(exp_dir / "router_initial_state.json", state)
        _log(f"router mode={state['mode']} workers={[w['name'] for w in state['workers']]}")

        plan = []
        for rep in range(cfg.repeats):
            for conc in cfg.concurrency:
                order = list(cfg.policies)
                if cfg.shuffle_policy_order:
                    random.Random(hash((rep, conc)) & 0xFFFF).shuffle(order)
                plan += [(rep, conc, p) for p in order]
        _log(f"plan: {len(plan)} runs ({len(cfg.policies)} policies x {len(cfg.concurrency)} concurrency x {cfg.repeats} repeats)")

        for i, (rep, conc, pol) in enumerate(plan, 1):
            _log(f"run {i}/{len(plan)}: policy={pol} concurrency={conc} repeat={rep}")
            run_dir = exp_dir / "runs" / f"{cfg.deployment}__{pol}__c{conc:03d}__r{rep}"
            s = await run_one(cfg, client, convs, warm_convs, pol, conc, rep, run_dir)
            _log(f"  ok={s['n_ok']}/{s['n']} thr={_fmt(s.get('steady_throughput_rps'), nd=2)} req/s "
                 f"ttft_p95={_fmt(s.get('ttft_p95'))}s tpot_p50={_fmt(s.get('tpot_p50'), 1000, 1)}ms "
                 f"e2e_p95={_fmt(s.get('e2e_p95'))}s hit={_fmt(s.get('cache_hit_rate'), 100, 1)}% "
                 f"prefill_tok={_fmt(s.get('prefill_computed_total'), nd=0)}")
    return exp_dir


def run_experiment(cfg: ExperimentConfig, analyze: bool = True) -> Path:
    exp_dir = asyncio.run(run_experiment_async(cfg))
    if analyze:
        from kvbench.analysis.report import analyze_experiments

        a = cfg.analysis
        analyze_experiments(
            [exp_dir], out_dir=exp_dir / "analysis",
            baseline_policy=a.get("baseline_policy", "round_robin"),
            slo_metric=a.get("slo_metric", "e2e"), slo_percentile=int(a.get("slo_percentile", 95)),
            slo_target_s=a.get("slo_target_s"),
        )
        _log(f"report: {exp_dir / 'analysis' / 'REPORT.md'}")
    return exp_dir
