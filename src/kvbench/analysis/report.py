"""Analysis entry point: CSVs, figures, and a Markdown report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kvbench.analysis import plots
from kvbench.analysis.prefill import prefill_table
from kvbench.analysis.slo import matched_slo_table
from kvbench.analysis.summarize import aggregate, flatten_runs, load_runs, write_csv
from kvbench.common.io import write_json


def _f(v, nd=3, scale=1.0, suffix="") -> str:
    if v is None:
        return "–"
    return f"{v * scale:.{nd}f}{suffix}"


def _pm(m, ci, nd=3, scale=1.0) -> str:
    if m is None:
        return "–"
    if ci is None:
        return f"{m * scale:.{nd}f}"
    return f"{m * scale:.{nd}f} ± {ci * scale:.{nd}f}"


def _resolve_baseline(labels: list[str], baseline: str) -> str:
    if baseline in labels:
        return baseline
    matches = [lab for lab in labels if lab.endswith("/" + baseline)]
    if matches:
        return sorted(matches)[0]
    return sorted(labels)[0]


def analyze_experiments(
    exp_dirs: list[str | Path],
    out_dir: str | Path | None = None,
    baseline_policy: str = "round_robin",
    slo_metric: str = "e2e",
    slo_percentile: int = 95,
    slo_target_s: float | None = None,
) -> Path:
    exp_dirs = [Path(d) for d in exp_dirs]
    out = Path(out_dir) if out_dir else exp_dirs[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(exp_dirs)
    if not runs:
        raise SystemExit(f"no runs found under {[str(d) for d in exp_dirs]}")
    labels = sorted({r["label"] for r in runs})
    baseline = _resolve_baseline(labels, baseline_policy)

    agg = aggregate(runs)
    write_csv(out / "summary_runs.csv", flatten_runs(runs))
    write_csv(out / "summary_agg.csv", agg)
    pre = prefill_table(runs, baseline)
    write_csv(out / "prefill.csv", pre)
    slo = matched_slo_table(agg, runs, baseline, slo_metric, slo_percentile, slo_target_s)
    write_csv(out / "matched_slo.csv", slo["rows"])
    write_json(out / "analysis.json", {"baseline": baseline, "labels": labels, "slo": slo, "prefill": pre})

    colors = plots.color_map(labels)
    figs = {
        "tl": plots.plot_throughput_vs_latency(agg, slo, out, colors),
        "lat": plots.plot_latency_panels(agg, out, colors),
        "pre": plots.plot_prefill(pre, out, colors),
        "hit": plots.plot_simple_metric(agg, "cache_hit_rate", "Prefix-cache hit rate (reported by engines)",
                                        "Prompt tokens served from cache (%)", "cache_hit_rate.png", out, colors, 100.0),
        "imb": plots.plot_simple_metric(agg, "load_imbalance", "Load imbalance (max / mean requests per worker)",
                                        "Max / mean", "load_imbalance.png", out, colors),
        "kv": plots.plot_kv_timeseries(runs, out),
        "gpu": plots.plot_gpu_memory(runs, out),
    }

    _write_report(out, exp_dirs, runs, agg, pre, slo, baseline, labels, figs)
    return out


def _write_report(out, exp_dirs, runs, agg, pre, slo, baseline, labels, figs) -> None:
    L = []
    ts = runs[0].get("trace_stats") or {}
    L.append("# KV-cache-aware routing benchmark report")
    L.append("")
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} from: " + ", ".join(f"`{d}`" for d in exp_dirs))
    L.append("")
    L.append(f"- Labels compared: {', '.join(f'`{x}`' for x in labels)}")
    L.append(f"- Baseline: `{baseline}`")
    reps = sorted({r['repeat'] for r in runs})
    concs = sorted({r['concurrency'] for r in runs})
    L.append(f"- Concurrency levels: {concs}; repeats per point: {len(reps)}")
    if ts:
        L.append(f"- Trace: {ts.get('num_prompts')} prompts in {ts.get('num_conversations')} conversations; "
                 f"mean prompt {ts.get('prompt_chars_mean', 0):,.0f} chars (p95 {ts.get('prompt_chars_p95', 0):,.0f}); "
                 f"max achievable prefix reuse {100 * ts.get('max_reuse_fraction', 0):.1f}% of prompt characters")
    L.append("")

    # --- headline ------------------------------------------------------------
    L.append("## Headline numbers")
    L.append("")
    all_rows = [r for r in pre if r["concurrency"] == "all"]
    if all_rows:
        L.append("**Prefill work** (summed over all concurrency levels, mean across repeats):")
        L.append("")
        L.append("| label | prompt tokens | computed prefill tokens | hit rate | ideal (∞ cache) | redundant | total prefill Δ vs baseline | redundant prefill Δ vs baseline |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(all_rows, key=lambda r: plots.label_order(labels).index(r["label"])):
            L.append(f"| `{r['label']}` | {_f(r['prompt_tokens'], 0)} | {_f(r['computed'], 0)} | {_f(r['hit_rate'], 1, 100, '%')} | "
                     f"{_f(r['ideal'], 0)} | {_f(r['redundant'], 0)} | {_f(r['total_prefill_reduction_pct'], 1, 1, '%')} | "
                     f"{_f(r['redundant_prefill_reduction_pct'], 1, 1, '%')} |")
        L.append("")
        L.append("Positive Δ = less prefill work than the baseline. These are *token* counts; they are not GPU-time or latency reductions.")
        L.append("")
    if slo.get("rows"):
        tgt = slo["target_s"]
        L.append(f"**Throughput at matched {slo['metric'].replace('_', ' ')} ≤ {_f(tgt, 3)} s** "
                 f"({'auto: the baseline’s tail latency at its highest concurrency level' if slo.get('auto_target') else 'user-specified target'}):")
        L.append("")
        L.append("Read the *note* column before quoting a number: `lower bound` = the policy stayed under the target at every "
                 "tested concurrency (true value is higher); `flat region` = the target falls where latency barely rises with "
                 "load, so small latency differences become large throughput differences. Do not quote flat-region results; "
                 "extend the concurrency sweep or pass `--slo-target` nearer saturation.")
        L.append("")
        L.append("| label | throughput at target (req/s) | Δ vs baseline (mean curve) | Δ per repeat (mean ± 95% CI) | note |")
        L.append("|---|---:|---:|---:|---|")
        for r in sorted(slo["rows"], key=lambda r: plots.label_order(labels).index(r["label"])):
            L.append(f"| `{r['label']}` | {_f(r['throughput_at_target'], 2)} | {_f(r['improvement_pct'], 1, 1, '%')} | "
                     f"{_pm(r['improvement_pct_repeat_mean'], r['improvement_pct_repeat_ci95'], 1)}% | {r['note']} |")
        L.append("")
    if figs.get("tl"):
        L.append(f"![throughput vs latency]({figs['tl']})")
        L.append("")
    if figs.get("pre"):
        L.append(f"![prefill tokens]({figs['pre']})")
        L.append("")

    # --- latency table -------------------------------------------------------
    L.append("## Latency and throughput by concurrency")
    L.append("")
    L.append("Mean across repeats ± 95% CI half-width (when repeats > 1). TTFT/E2E in seconds, TPOT in milliseconds.")
    L.append("")
    L.append("| label | conc | TTFT p50 | TTFT p95 | TPOT p50 (ms) | TPOT p95 (ms) | E2E p50 | E2E p95 | throughput (req/s) | output tok/s | hit rate | imbalance | errors |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(agg, key=lambda r: (plots.label_order(labels).index(r["label"]), r["concurrency"])):
        L.append(
            f"| `{r['label']}` | {r['concurrency']} | {_pm(r['ttft_p50'], r['ttft_p50_ci95'])} | {_pm(r['ttft_p95'], r['ttft_p95_ci95'])} | "
            f"{_pm(r['tpot_p50'], r['tpot_p50_ci95'], 1, 1000)} | {_pm(r['tpot_p95'], r['tpot_p95_ci95'], 1, 1000)} | "
            f"{_pm(r['e2e_p50'], r['e2e_p50_ci95'])} | {_pm(r['e2e_p95'], r['e2e_p95_ci95'])} | "
            f"{_pm(r['steady_throughput_rps'], r['steady_throughput_rps_ci95'], 2)} | {_f(r['output_tok_s'], 0)} | "
            f"{_f(r['cache_hit_rate'], 1, 100, '%')} | {_f(r['load_imbalance'], 2)} | {_f(r['error_rate'], 2, 100, '%')} |"
        )
    L.append("")
    for key in ("lat", "hit", "imb", "kv", "gpu"):
        if figs.get(key):
            L.append(f"![{key}]({figs[key]})")
            L.append("")

    # --- per-concurrency prefill ----------------------------------------------
    L.append("## Prefill work by concurrency")
    L.append("")
    L.append("| label | conc | computed prefill tokens | hit rate | total Δ vs baseline | redundant Δ vs baseline |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in pre:
        if r["concurrency"] == "all":
            continue
        L.append(f"| `{r['label']}` | {r['concurrency']} | {_f(r['computed'], 0)} | {_f(r['hit_rate'], 1, 100, '%')} | "
                 f"{_f(r['total_prefill_reduction_pct'], 1, 1, '%')} | {_f(r['redundant_prefill_reduction_pct'], 1, 1, '%')} |")
    L.append("")

    # --- definitions ---------------------------------------------------------
    L.append("## Metric definitions")
    L.append("")
    L.append("- **TTFT**: request sent → first non-empty token received (client-side, includes routing and queueing).")
    L.append("- **TPOT**: (E2E − TTFT) / (completion tokens − 1).")
    L.append("- **E2E**: request sent → final chunk received.")
    L.append("- **Throughput**: completed requests per second between the 10th and 90th percentile completion "
             "times (excludes closed-loop ramp-up and drain).")
    L.append("- **Computed prefill tokens**: Σ (prompt_tokens − cached_tokens) using engine-reported "
             "`usage.prompt_tokens_details.cached_tokens` (prefill worker's usage in disaggregated mode).")
    L.append("- **Ideal prefill**: distinct prefix chunks in the trace (what one infinite cache would compute), "
             "converted to tokens with the measured tokens/char ratio. **Redundant** = computed − ideal.")
    L.append("- **Matched-latency throughput**: highest throughput (measured or linearly interpolated between adjacent "
             "concurrency points) whose tail latency is ≤ the target.")
    L.append("- **Load imbalance**: max / mean requests handled per worker (1.0 = perfectly even).")
    L.append("")
    (out / "REPORT.md").write_text("\n".join(L))
