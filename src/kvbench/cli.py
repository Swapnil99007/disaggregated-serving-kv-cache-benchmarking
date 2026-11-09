"""Command-line interface: ``kvbench <command> ...``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import fields

from kvbench.common.io import read_yaml, write_json


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if s.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {s}")


# ---------------------------------------------------------------- gen-trace
def cmd_gen_trace(a) -> int:
    from kvbench.workload.rag import RagWorkloadConfig, generate_rag_trace
    from kvbench.workload.trace import save_trace, trace_stats

    d = read_yaml(a.workload) if a.workload else {}
    if a.num_prompts:
        d["num_prompts"] = a.num_prompts
    if a.seed is not None:
        d["seed"] = a.seed
    cfg = RagWorkloadConfig.from_dict(d)
    trace = generate_rag_trace(cfg)
    save_trace(a.out, trace)
    stats = trace_stats(trace, a.chunk_chars)
    write_json(a.out + ".stats.json", stats)
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(trace)} prompts to {a.out}")
    return 0


def cmd_trace_stats(a) -> int:
    from kvbench.workload.trace import load_trace, trace_stats, truncate_trace

    trace = truncate_trace(load_trace(a.trace), a.max_prompts)
    print(json.dumps(trace_stats(trace, a.chunk_chars), indent=2))
    return 0


# ---------------------------------------------------------------- sim-worker
def _add_sim_args(p: argparse.ArgumentParser) -> None:
    from kvbench.sim.config import SimConfig

    for f in fields(SimConfig):
        t = {"bool": _bool, "int": int, "float": float}.get(str(f.type), str)
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=t, default=None,
                       help=f"(default {f.default})")


def cmd_sim_worker(a) -> int:
    from kvbench.sim.config import SimConfig
    from kvbench.sim.server import run_sim_worker

    d = read_yaml(a.config_file) if a.config_file else {}
    for f in fields(SimConfig):
        v = getattr(a, f.name)
        if v is not None:
            d[f.name] = v
    cfg = SimConfig.from_dict(d)
    print(f"sim worker {cfg.engine_id} role={cfg.role} kv_capacity_tokens={cfg.kv_capacity_tokens} "
          f"speed={cfg.speed} on {a.host}:{a.port}", flush=True)
    run_sim_worker(cfg, a.host, a.port, a.advertised_host)
    return 0


# ---------------------------------------------------------------- router
def cmd_router(a) -> int:
    from kvbench.router.app import run_router
    from kvbench.router.config import RouterConfig

    cfg = RouterConfig.load(a.config)
    if a.policy:
        cfg.policy = a.policy
    print(f"router mode={cfg.mode} policy={cfg.policy} workers={[w.url for w in cfg.workers]} "
          f"on {a.host or cfg.host}:{a.port or cfg.port}", flush=True)
    run_router(cfg, a.host, a.port)
    return 0


def cmd_cluster(a) -> int:
    from kvbench.launch import launch_cluster

    overrides = {}
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()
    return launch_cluster(a.config, overrides, a.port)


# ---------------------------------------------------------------- bench / analyze
def cmd_bench(a) -> int:
    from kvbench.bench.config import ExperimentConfig
    from kvbench.bench.runner import run_experiment

    cfg = ExperimentConfig.load(a.experiment)
    if a.router_url:
        cfg.router_url = a.router_url
    if a.policies:
        cfg.policies = [p.strip() for p in a.policies.split(",")]
    if a.concurrency:
        cfg.concurrency = [int(c) for c in a.concurrency.split(",")]
    if a.repeats is not None:
        cfg.repeats = a.repeats
    if a.max_prompts is not None:
        cfg.trace["max_prompts"] = a.max_prompts
    if a.deployment:
        cfg.deployment = a.deployment
    if a.name:
        cfg.name = a.name
    run_experiment(cfg, analyze=not a.no_analyze)
    return 0


def cmd_analyze(a) -> int:
    from kvbench.analysis.report import analyze_experiments

    out = analyze_experiments(a.exp_dirs, a.out, a.baseline, a.slo_metric, a.slo_percentile, a.slo_target)
    print(f"wrote {out / 'REPORT.md'}")
    return 0


def cmd_smoke(a) -> int:
    """Send a handful of requests through the router and print what happened."""
    import httpx

    from kvbench.bench.client import send_request
    from kvbench.workload.rag import RagWorkloadConfig, generate_rag_trace

    trace = generate_rag_trace(RagWorkloadConfig(num_prompts=a.n, seed=a.seed))

    async def go():
        async with httpx.AsyncClient(timeout=600.0) as client:
            import time

            t0 = time.perf_counter()
            for r in trace:
                res = await send_request(client, a.router_url, a.model, r, t0)
                print(f"{r.request_id:>10}  worker={res.worker} reason={res.route_reason} status={res.status} "
                      f"ttft={res.ttft_s and round(res.ttft_s, 3)}s e2e={res.e2e_s and round(res.e2e_s, 3)}s "
                      f"prompt={res.prompt_tokens} cached={res.cached_tokens} out={res.completion_tokens}"
                      + (f"  err={res.error}" if res.error else ""))
            st = (await client.get(a.router_url.rstrip("/") + "/admin/state")).json()
            print(json.dumps({k: st[k] for k in ("mode", "policy", "requests_total", "reasons")}, indent=2))

    asyncio.run(go())
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="kvbench", description="Disaggregated LLM serving & KV-cache-aware routing testbed")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-trace", help="generate a synthetic long-context RAG trace")
    p.add_argument("--workload", default="configs/workloads/rag_default.yaml")
    p.add_argument("--out", default="traces/rag_1000.jsonl")
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chunk-chars", type=int, default=256)
    p.set_defaults(fn=cmd_gen_trace)

    p = sub.add_parser("trace-stats", help="prefix-sharing statistics for a trace")
    p.add_argument("trace")
    p.add_argument("--chunk-chars", type=int, default=256)
    p.add_argument("--max-prompts", type=int, default=0)
    p.set_defaults(fn=cmd_trace_stats)

    p = sub.add_parser("sim-worker", help="run one simulated LLM engine (OpenAI/vLLM-compatible)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--config-file", default=None, help="YAML of SimConfig fields")
    p.add_argument("--advertised-host", default=None, help="host returned in kv_transfer_params")
    _add_sim_args(p)
    p.set_defaults(fn=cmd_sim_worker)

    p = sub.add_parser("router", help="run the KV-cache-aware router")
    p.add_argument("--config", required=True)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--policy", default=None)
    p.set_defaults(fn=cmd_router)

    p = sub.add_parser("cluster", help="launch sim workers + router locally (no Docker)")
    p.add_argument("--config", default="configs/router/sim_local.yaml")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--set", action="append", help="sim override for all workers, e.g. --set speed=2")
    p.set_defaults(fn=cmd_cluster)

    p = sub.add_parser("bench", help="run an experiment sweep against a running router")
    p.add_argument("--experiment", required=True)
    p.add_argument("--router-url", default=None)
    p.add_argument("--policies", default=None, help="comma-separated override")
    p.add_argument("--concurrency", default=None, help="comma-separated override, e.g. 8,16,32,64")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--deployment", default=None, help="label prefix override")
    p.add_argument("--name", default=None, help="experiment name override")
    p.add_argument("--no-analyze", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("analyze", help="aggregate one or more experiment dirs into a report")
    p.add_argument("exp_dirs", nargs="+")
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", default="round_robin", help="policy name or full label")
    p.add_argument("--slo-metric", default="e2e", choices=["e2e", "ttft", "tpot", "itl_mean"])
    p.add_argument("--slo-percentile", type=int, default=95, choices=[50, 90, 95, 99])
    p.add_argument("--slo-target", type=float, default=None, help="seconds; default = auto from baseline")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("smoke", help="send a few requests through the router")
    p.add_argument("--router-url", default="http://localhost:8000")
    p.add_argument("--model", default="kvbench-model")
    p.add_argument("-n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_smoke)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
