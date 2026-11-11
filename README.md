# kvbench: disaggregated LLM serving & KV-cache-aware routing testbed

A Python testbed for comparing **prefill/decode scheduling** and **KV-cache reuse**
under long-context RAG traffic, on **vLLM**, **SGLang**, or a **GPU-free simulator**,
all orchestrated with **Docker**.

It contains:

| Component | What it does |
|---|---|
| **RAG workload generator** | Deterministic multi-turn conversational RAG trace (system prompt + retrieved docs + history). Zipfian topic popularity creates shared prefixes. Every prompt is fully materialized, so replays are identical across runs. |
| **KV-cache-aware router** | OpenAI-compatible reverse proxy. It keeps an approximate per-worker prefix index (chained chunk hashes), scrapes engine telemetry (KV usage, queue depth, prefix-cache counters), and routes with pluggable policies. It runs **colocated** or **disaggregated prefill/decode** (vLLM NixlConnector protocol). |
| **Routing policies** | `round_robin`, `least_loaded`, `prefix_affinity` (pure cache affinity), `weighted` (linear score), `cache_aware` (cost model: estimated prefill queue + uncached prefill + decode load + memory pressure, with a load-imbalance guard). |
| **Benchmark harness** | Closed-loop sessions at 8–64 concurrency, streaming TTFT / TPOT / ITL / E2E, cold-cache resets between runs, engine warm-up, shuffled policy order, repeats, NVML GPU-memory + KV-usage telemetry. |
| **Analysis** | Mean ± 95% CI across repeats, prefill-token accounting (total and *redundant* vs an infinite-cache ideal), **throughput at a matched p95 latency target**, figures, and a Markdown report. |
| **Simulator** | Continuous-batching engine with paged KV blocks, hash-based prefix caching + LRU eviction, chunked prefill, a prefill/decode cost model, and modeled KV transfer for P/D disaggregation. It speaks vLLM's HTTP API and metric names, so the router and benchmark run unchanged on a laptop. |

```
          trace (1,000 RAG prompts)
                   │  closed loop, N sessions
                   ▼
        ┌──────────────────────┐   /metrics scrape (KV usage, queue, prefix hits)
        │  kvbench router      │◄──────────────────────────────────────────┐
        │  prefix index + policy│                                           │
        └──────┬───────┬───────┘                                           │
   colocated   │       │  disaggregated: 1) prefill (max_tokens=1)          │
               ▼       ▼                 2) decode + kv_transfer_params     │
          ┌────────┐ ┌────────┐      ┌───────────┐  KV   ┌───────────┐     │
          │worker 0│ │worker 1│ ...  │ prefill P │──────►│ decode D  │ ────┘
          └────────┘ └────────┘      └───────────┘       └───────────┘
        vLLM / SGLang / simulator     vLLM NixlConnector / simulator
```

**New to the project?** [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md) walks through it
from plain-English basics to the engineering details, with diagrams and the full results.

## Results on a real GPU

NVIDIA RTX A6000, vLLM 0.11.0, 2 replicas of Qwen2.5-1.5B-Instruct (KV cache capped at 131K
tokens each). 1,000 replayed multi-turn RAG prompts, 8–64 concurrent sessions, 3 repeats
(72 runs, 72,000 requests, 0 errors). Compared with round-robin routing:

| Policy | Prefill tokens computed | Redundant prefill | Throughput at matched p95 E2E latency |
|---|---:|---:|---:|
| `cache_aware` | **−36.2%** | **−76.7%** | **≥ +21.2% ± 4.1%** |
| `prefix_affinity` | −42.1% | −89.3% | ≥ +28.5% ± 6.3% |
| `least_loaded` | −26.2% | −55.6% | ≥ +18.2% ± 3.2% |

With only two replicas, plain prefix affinity edges out the cost-model router. In a 4-worker
simulation, affinity creates hotspots (load imbalance 1.40) and `cache_aware` leads, +37.8% vs
+16.9%. See [the explainer](docs/PROJECT_EXPLAINED.md#45-what-the-results-mean) for why, and the
[full report](results/vllm/20261108-233332/analysis/REPORT.md) for every table and chart.

### Disaggregated vs colocated on two real GPUs

2× NVIDIA A40 (PCIe, no NVLink), vLLM 0.11.0 + NIXL 0.6.0, same model, trace and router.
1 prefill + 1 decode instance (1P+1D) vs one complete replica per GPU, same KV budget per GPU.
The KV cache moved between GPUs through host memory (`UCX_TLS=tcp,cuda_copy`), because direct
GPU-to-GPU reads took the decode GPU offline on these PCIe hosts.

| Sessions | Colocated (req/s) | 1P+1D (req/s) | TPOT p50, colocated | TPOT p50, 1P+1D |
|---:|---:|---:|---:|---:|
| 8 | **5.82** | 4.37 | **8.9 ms** | 9.5 ms |
| 32 | **14.75** | 7.71 | 14.4 ms | **13.1 ms** |
| 64 | **18.09** | 2.85 (with admission cap) | 24.0 ms | **9.8 ms** |

- **Throughput:** colocated serves 59% more requests at matched p95 E2E latency (18.09 vs 7.49
  req/s, ±3.6%).
- **Decode stays fast in 1P+1D:** TPOT is 2.4× lower than colocated at 64 sessions, because the
  decode GPU never processes prompts.
- **The bottleneck is the KV handoff.** The staged path topped out near 0.2 GB/s. That is 1–1.6 s
  per request, against 0.1 s of prefill.
- **Without admission control, 48+ sessions collapse.** The decode KV pool fills (100%), requests
  hang, KV the prefiller held past vLLM's 120 s timeout expires, and up to 64% of requests fail.
- **The router fix:** capping in-flight decode requests at 32 (`max_inflight`) turned that into
  queueing, with 0 errors in 4,000 requests.

Details and caveats: [explainer 4.6](docs/PROJECT_EXPLAINED.md#46-disaggregated-prefilldecode),
[comparison report](results/compare_pd_gpu/REPORT.md).

---

## 1. Setup

Requires Python ≥ 3.10.

```bash
make install            # python3 -m venv .venv && pip install -e ".[dev,gpu]"
source .venv/bin/activate
make test               # 46 tests, including end-to-end router <-> simulator runs; ~10 s
```

## 2. Run everything locally on the simulator (no GPU, no Docker)

```bash
# terminal 1: 4 simulated workers + router on :8000
kvbench cluster --config configs/router/sim_local.yaml
#   (add --set speed=4 to run the simulator 4x faster than real time)

# terminal 2
kvbench smoke                                                    # 6 requests; watch cache hits appear
kvbench bench --experiment configs/experiments/sim_quick.yaml    # 4 policies x 4 concurrency levels
```

The bench command generates the trace if it is missing, runs the sweep, and writes
`results/sim_quick/<timestamp>/analysis/REPORT.md` with tables and figures.

**Run time:** `sim_quick` replays 400 prompts per run (16 runs, ~15 min at real-time
speed, ~4 min with `--set speed=4`). `sim_full` is the full methodology (1,000 prompts ×
6 concurrency levels × 3 repeats × 4 policies = 72 runs). `speed` compresses every
simulated duration by the same factor, so ratios between policies are preserved but
absolute latencies shrink. Use `speed=1` for numbers you intend to quote.

**Disaggregated prefill/decode (simulated, same 4 "GPUs"):**

```bash
kvbench cluster --config configs/router/sim_local_disagg.yaml        # 2 prefill + 2 decode
kvbench bench --experiment configs/experiments/sim_disagg.yaml
kvbench analyze results/sim_quick/<ts> results/sim_disagg/<ts> --baseline sim-4w/round_robin
```

## 3. Run the simulator in Docker

```bash
docker compose -f docker/compose.sim.yml up --build -d
kvbench bench --experiment configs/experiments/sim_quick.yaml        # from the host, or:
docker compose -f docker/compose.sim.yml --profile bench run --rm bench
docker compose -f docker/compose.sim.yml down
```

## 4. Run on real GPUs (vLLM / SGLang)

See **[docs/GPU_SETUP.md](docs/GPU_SETUP.md)** for renting a GPU, drivers, and the
NVIDIA container toolkit. On the GPU host:

```bash
cp .env.example .env                       # model, GPU ids, KV-cache cap
make install && source .venv/bin/activate

# vLLM: 2 replicas of Qwen2.5-1.5B-Instruct on one 24 GB GPU + router
docker compose -f docker/compose.vllm.yml --env-file .env up --build -d
kvbench bench --experiment configs/experiments/vllm_quick.yaml   # sanity pass
kvbench bench --experiment configs/experiments/vllm.yaml         # 1,000 prompts x 8..64 x 3 repeats

# SGLang (same methodology)
docker compose -f docker/compose.vllm.yml down
docker compose -f docker/compose.sglang.yml --env-file .env up --build -d
kvbench bench --experiment configs/experiments/sglang.yaml

# Compare everything in one report
kvbench analyze results/vllm/<ts> results/sglang/<ts> --baseline vllm-2w/round_robin
```

Disaggregated vLLM (2 GPUs): follow **[docs/RUN_DISAGG_2GPU.md](docs/RUN_DISAGG_2GPU.md)**.
It uses `scripts/vllm_disagg_baremetal.sh` with three experiment configs:
- `configs/experiments/vllm_disagg.yaml`, the disaggregated run;
- `configs/experiments/vllm_disagg_ac.yaml`, the same with decode admission control;
- `configs/experiments/vllm_2gpu.yaml`, the colocated baseline.

Alternatively, use `docker/compose.vllm-disagg.yml`.

---

## 5. Methodology

| Control | How |
|---|---|
| Model / hardware | Same weights, `--served-model-name kvbench-model`, same GPU type and count per comparison |
| Prompt distribution | Same deterministic trace for every run (`traces/rag_1000.jsonl`; mean ≈ 18K chars ≈ 3.4K Qwen2.5 tokens) |
| Output lengths | `max_tokens` from the trace + `ignore_eos: true` → identical generation lengths |
| Concurrency | Closed loop, 8 / 16 / 24 / 32 / 48 / 64 sessions; each session replays whole conversations in turn order |
| Cache state | Prefix caches flushed + router index cleared before every run (cold start). Optional prefix warm-up from a different request stream over the same corpus (`warmup.prefix_warmup_prompts`) |
| Warm-up | Short unrelated requests before each run (CUDA graphs, allocator, connections), then flush again |
| Ordering | Policy order shuffled within each (repeat, concurrency) block to decorrelate drift |
| Repeats | 3 per point; tables report mean ± 95% CI (Student t) |
| Throughput | Steady state: completions between the 10th and 90th percentile completion times |

### Metrics

- **TTFT**: request sent → first token (client-side, includes routing and queueing).
- **TPOT**: (E2E − TTFT) / (completion_tokens − 1). **ITL**: gaps between streamed chunks.
- **E2E**: request sent → last chunk.
- **Computed prefill tokens**: Σ(prompt_tokens − cached_tokens) from the engines' own
  `usage.prompt_tokens_details.cached_tokens` (vLLM `--enable-prompt-tokens-details`,
  SGLang `--enable-cache-report`). The engines' Prometheus prefix-cache counters are
  recorded as a cross-check.
- **Redundant prefill**: computed − ideal, where *ideal* is what a single infinitely
  large prefix cache would still compute (distinct prefix chunks in the trace).
- **Reduction %** = (baseline − policy) / baseline, for both total and redundant prefill
  tokens. These are **token** counts, not GPU-time or latency reductions.
- **Throughput at matched p95**: for each policy, the highest throughput whose p95 E2E
  latency is ≤ a target T (interpolated between adjacent concurrency points). By default
  T is the baseline's p95 at its highest concurrency level (the tail latency it needed at
  peak load); override with `--slo-target`. The report flags results that are a
  `lower bound` (the policy never reached T) or fall in a `flat region` (unsaturated part
  of the curve, where latency barely moves and the comparison is unreliable). The sweep
  must reach saturation for this metric to mean anything.

---

## 6. Repository layout

```
src/kvbench/
  workload/   rag.py (generator), trace.py (format, stats, ideal-prefill), words.py
  router/     app.py (proxy, colocated + disagg), policies.py, index.py (prefix index),
              workers.py (telemetry scraping, backend adapters), config.py
  sim/        engine.py (scheduler + cost model), blocks.py (paged KV + prefix cache),
              server.py (OpenAI/vLLM HTTP surface), tokenizer.py, config.py
  bench/      client.py (streaming client, closed loop), runner.py (sweeps), telemetry.py (NVML, KV usage)
  analysis/   summarize.py, prefill.py, slo.py (matched-latency throughput), plots.py, report.py
  common/     prefix.py (chained chunk hashes), prom.py, stats.py, io.py
  cli.py, launch.py
configs/      sim/, router/, workloads/, experiments/
docker/       Dockerfile, compose.{sim,sim-disagg,vllm,sglang,vllm-disagg}.yml
docs/         PROJECT_EXPLAINED.md, GPU_SETUP.md, images/
tests/        unit + end-to-end tests (no GPU needed)
```

## 7. Router admin API

| Endpoint | Purpose |
|---|---|
| `POST /v1/completions`, `POST /v1/chat/completions` | Routed inference (stream or not). Responses carry `x-kvbench-worker`, `-reason`, `-matched-chunks`, `-prompt-chunks`, and in disagg mode `-prefill-worker`, `-decode-worker`, `-prefill-cached-tokens`, `-prefill-ms` |
| `GET /admin/state` | Policy, per-worker in-flight / admitted / pending prefill / KV usage / index size / prediction accuracy, decode admission-queue stats |
| `POST /admin/policy` | `{"policy": "cache_aware", "params": {...}, "decode_policy": "least_loaded"}` |
| `POST /admin/reset` | `{"flush_workers": true, "clear_index": true}`: clears the index and flushes engine prefix caches |
| `GET /admin/worker_metrics` | Fresh scrape of every worker's `/metrics` |
| `GET /metrics` | Router Prometheus metrics |

## 8. Known limitations

- The router's prefix index is **approximate**: it tracks what it *sent*, not what the
  engine still holds. Mitigations: capacity-bounded LRU, TTL, invalidation on flush or
  restart, and logged prediction accuracy (`pred_cached_frac_mean` vs `actual_cached_frac_mean`).
- The router is a single Python process. Its overhead is the same for every policy,
  so comparisons stay fair, but it is not a production data plane.
- The simulator's cost model gives plausible *relative* costs only; validate on real GPUs before quoting absolute numbers.
- vLLM's KV-connector API changes between releases; the disaggregated compose file is pinned and marked experimental.
- The real-GPU disaggregation numbers use a host-staged KV path (~0.2 GB/s) on PCIe GPUs. They describe that
  setup, not disaggregation over NVLink or RDMA.
