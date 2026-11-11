# KV-cache-aware routing benchmark report

Generated 2026-11-11 04:17 from: `results/vllm_disagg/20261111-020116`

- Labels compared: `vllm-disagg-1p1d/cache_aware`
- Baseline: `vllm-disagg-1p1d/cache_aware`
- Concurrency levels: [8, 16, 24, 32, 48, 64]; repeats per point: 3
- Trace: 1000 prompts in 369 conversations; mean prompt 18,047 chars (p95 24,909); max achievable prefix reuse 81.3% of prompt characters

## Headline numbers

**Prefill work** (summed over all concurrency levels, mean across repeats):

| label | prompt tokens | computed prefill tokens | hit rate | ideal (∞ cache) | redundant | total prefill Δ vs baseline | redundant prefill Δ vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-disagg-1p1d/cache_aware` | 17137209 | 4251695 | 75.2% | 3762381 | 489313 | 0.0% | 0.0% |

Positive Δ = less prefill work than the baseline. These are *token* counts; they are not GPU-time or latency reductions.

**Throughput at matched e2e p95 ≤ 97.561 s** (auto: the baseline’s tail latency at its highest concurrency level):

Read the *note* column before quoting a number: `lower bound` = the policy stayed under the target at every tested concurrency (true value is higher); `flat region` = the target falls where latency barely rises with load, so small latency differences become large throughput differences. Do not quote flat-region results; extend the concurrency sweep or pass `--slo-target` nearer saturation.

| label | throughput at target (req/s) | Δ vs baseline (mean curve) | Δ per repeat (mean ± 95% CI) | note |
|---|---:|---:|---:|---|
| `vllm-disagg-1p1d/cache_aware` | 7.71 | 0.0% | 0.0 ± 0.0% | ok |

![throughput vs latency](throughput_vs_latency.png)

![prefill tokens](prefill_tokens.png)

## Latency and throughput by concurrency

Mean across repeats ± 95% CI half-width (when repeats > 1). TTFT/E2E in seconds, TPOT in milliseconds.

| label | conc | TTFT p50 | TTFT p95 | TPOT p50 (ms) | TPOT p95 (ms) | E2E p50 | E2E p95 | throughput (req/s) | output tok/s | hit rate | imbalance | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vllm-disagg-1p1d/cache_aware` | 8 | 0.326 ± 0.015 | 1.449 ± 0.164 | 9.5 ± 0.3 | 10.0 ± 0.6 | 1.665 ± 0.007 | 3.499 ± 0.184 | 4.37 ± 0.08 | 606 | 77.6% | 1.00 | 0.00% |
| `vllm-disagg-1p1d/cache_aware` | 16 | 0.663 ± 0.186 | 2.611 ± 0.513 | 11.0 ± 0.0 | 11.8 ± 0.2 | 2.201 ± 0.178 | 4.807 ± 0.524 | 6.57 ± 0.49 | 874 | 77.4% | 1.00 | 0.07% |
| `vllm-disagg-1p1d/cache_aware` | 24 | 1.193 ± 0.095 | 4.512 ± 0.875 | 12.3 ± 0.3 | 13.8 ± 0.2 | 2.886 ± 0.147 | 6.688 ± 0.616 | 7.43 ± 0.37 | 971 | 77.4% | 1.00 | 0.07% |
| `vllm-disagg-1p1d/cache_aware` | 32 | 1.901 ± 0.280 | 6.864 ± 0.865 | 13.1 ± 1.9 | 15.9 ± 1.1 | 3.729 ± 0.369 | 8.947 ± 0.286 | 7.71 ± 0.40 | 984 | 77.5% | 1.00 | 0.00% |
| `vllm-disagg-1p1d/cache_aware` | 48 | 13.143 ± 4.458 | 23.166 ± 9.709 | 11.6 ± 0.9 | 16.9 ± 1.5 | 14.599 ± 4.197 | 25.334 ± 10.030 | 1.46 ± 4.33 | 223 | 73.6% | 1.00 | 25.03% |
| `vllm-disagg-1p1d/cache_aware` | 64 | 47.434 ± 116.076 | 96.502 ± 139.254 | 11.4 ± 2.8 | 15.0 ± 6.2 | 48.885 ± 115.363 | 97.561 ± 139.820 | 2.05 ± 3.96 | 44 | 60.5% | 1.00 | 64.37% |

![lat](latency_vs_concurrency.png)

![hit](cache_hit_rate.png)

![imb](load_imbalance.png)

![kv](kv_usage_timeseries.png)

![gpu](gpu_memory.png)

## Prefill work by concurrency

| label | conc | computed prefill tokens | hit rate | total Δ vs baseline | redundant Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| `vllm-disagg-1p1d/cache_aware` | 8 | 750542 | 77.6% | 0.0% | 0.0% |
| `vllm-disagg-1p1d/cache_aware` | 16 | 758727 | 77.4% | 0.0% | 0.0% |
| `vllm-disagg-1p1d/cache_aware` | 24 | 756922 | 77.4% | 0.0% | 0.0% |
| `vllm-disagg-1p1d/cache_aware` | 32 | 756377 | 77.5% | 0.0% | 0.0% |
| `vllm-disagg-1p1d/cache_aware` | 48 | 662690 | 73.6% | 0.0% | 0.0% |
| `vllm-disagg-1p1d/cache_aware` | 64 | 566436 | 60.5% | 0.0% | – |

## Metric definitions

- **TTFT**: request sent → first non-empty token received (client-side, includes routing and queueing).
- **TPOT**: (E2E − TTFT) / (completion tokens − 1).
- **E2E**: request sent → final chunk received.
- **Throughput**: completed requests per second between the 10th and 90th percentile completion times (excludes closed-loop ramp-up and drain).
- **Computed prefill tokens**: Σ (prompt_tokens − cached_tokens) using engine-reported `usage.prompt_tokens_details.cached_tokens` (prefill worker's usage in disaggregated mode).
- **Ideal prefill**: distinct prefix chunks in the trace (what one infinite cache would compute), converted to tokens with the measured tokens/char ratio. **Redundant** = computed − ideal.
- **Matched-latency throughput**: highest throughput (measured or linearly interpolated between adjacent concurrency points) whose tail latency is ≤ the target.
- **Load imbalance**: max / mean requests handled per worker (1.0 = perfectly even).
