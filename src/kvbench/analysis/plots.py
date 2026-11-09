"""Static report figures (matplotlib, PNG).

Visual rules: one y-axis per chart (small multiples instead of dual axes), 2px
lines with surface-ringed markers, hairline recessive grid, a legend whenever
there are 2+ series plus sparing end-of-line labels, and colors assigned to each
label in a fixed order so the same policy has the same color in every figure.
The categorical palette was validated for colour-vision-deficiency separation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

from kvbench.common.io import read_jsonl  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERFLOW = MUTED

# fixed policy order -> stable colours across all figures
POLICY_ORDER = ["cache_aware", "round_robin", "least_loaded", "prefix_affinity", "weighted"]


def _policy_rank(p: str) -> int:
    return POLICY_ORDER.index(p) if p in POLICY_ORDER else len(POLICY_ORDER)


def label_order(labels: list[str]) -> list[str]:
    def key(lab: str):
        dep, _, pol = lab.rpartition("/")
        return (dep, _policy_rank(pol), pol)

    return sorted(set(labels), key=key)


def color_map(labels: list[str]) -> dict[str, str]:
    ordered = label_order(labels)
    return {lab: (SERIES[i] if i < len(SERIES) else OVERFLOW) for i, lab in enumerate(ordered)}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
    })


def _line(ax, xs, ys, color, label):
    ax.plot(xs, ys, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round",
            marker="o", markersize=6, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=label, zorder=3)


def _end_labels(ax, series: dict[str, tuple[list, list]], fmt=None) -> None:
    """Direct labels at line ends — only when there are few series (<= 4)."""
    if len(series) > 4:
        return
    ends = {}
    for label, (xs, ys) in series.items():
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ends[label] = pts[-1]
    # labels whose end points nearly coincide would stack on top of each other:
    # drop those and let the legend carry identity
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xr, yr = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
    clash = set()
    items = list(ends.items())
    for i, (la, (xa, ya)) in enumerate(items):
        for lb, (xb, yb) in items[i + 1:]:
            if abs(xa - xb) / xr < 0.12 and abs(ya - yb) / yr < 0.05:
                clash.update((la, lb))
    for label, (x, y) in ends.items():
        if label in clash:
            continue
        ax.annotate(label.rpartition("/")[2], (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_throughput_vs_latency(agg_rows, slo: dict, out: Path, colors: dict) -> str | None:
    lat_key = slo.get("metric") or "e2e_p95"
    tkey = slo.get("throughput_key", "steady_throughput_rps")
    curves = defaultdict(list)
    for r in agg_rows:
        curves[r["label"]].append((r["concurrency"], r.get(lat_key), r.get(tkey) or r.get("throughput_rps")))
    if not curves:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    series = {}
    for label in label_order(list(curves)):
        pts = sorted((c, l, x) for c, l, x in curves[label] if l is not None and x is not None)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        _line(ax, xs, ys, colors[label], label)
        series[label] = (xs, ys)
        for c, l, x in pts:
            ax.annotate(f"{c}", (l, x), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    if slo.get("target_s") is not None:
        ax.axvline(slo["target_s"], color=INK_2, linewidth=1, zorder=2)
        # label at the bottom of the line: the top is where curve end-labels live
        ax.annotate(f"p{lat_key.split('_p')[-1]} target {slo['target_s']:.2f}s", (slo["target_s"], 0.0),
                    xycoords=("data", "axes fraction"), xytext=(4, 4), textcoords="offset points",
                    va="bottom", fontsize=8, color=INK_2)
    _end_labels(ax, series)
    metric_name = lat_key.replace("_p", " p").upper().replace("E2E", "E2E latency").replace("TTFT", "TTFT")
    ax.set_xlabel(f"{metric_name} (s)")
    ax.set_ylabel("Steady-state throughput (req/s)")
    ax.set_title("Throughput vs tail latency (point labels = concurrent sessions)")
    ax.set_ylim(bottom=0)
    # saturating curves bend right, so high-latency/low-throughput stays empty; stop short of the
    # right edge, where the matched-latency target line usually sits
    ax.legend(loc="lower right", bbox_to_anchor=(0.9, 0.02))
    return _save(fig, out / "throughput_vs_latency.png")


def plot_latency_panels(agg_rows, out: Path, colors: dict) -> str | None:
    panels = [("ttft_p95", "TTFT p95 (s)", 1.0), ("tpot_p50", "TPOT p50 (ms)", 1000.0), ("e2e_p95", "E2E p95 (s)", 1.0)]
    by_label = defaultdict(list)
    for r in agg_rows:
        by_label[r["label"]].append(r)
    if not by_label:
        return None
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, scale) in zip(axes, panels):
        for label in label_order(list(by_label)):
            rows = sorted(by_label[label], key=lambda r: r["concurrency"])
            xs = [r["concurrency"] for r in rows]
            ys = [(r.get(key) * scale) if r.get(key) is not None else None for r in rows]
            _line(ax, xs, ys, colors[label], label)
        ax.set_title(title)
        ax.set_xlabel("Concurrent sessions")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["concurrency"] for r in agg_rows}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    # balanced rows: 6 labels -> 2 rows of 3, never 5 + a lonely 1
    ncol = len(labels) if len(labels) <= 4 else (len(labels) + 1) // 2
    rows = (len(labels) + ncol - 1) // ncol
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.02 + 0.06 * rows))
    fig.tight_layout()
    return _save(fig, out / "latency_vs_concurrency.png")


def plot_prefill(prefill_rows, out: Path, colors: dict) -> str | None:
    rows = [r for r in prefill_rows if r["concurrency"] == "all" and r.get("computed") is not None]
    if not rows:
        return None
    _style()
    ordered = label_order([r["label"] for r in rows])
    by = {r["label"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.5 * len(ordered)))
    ys = list(range(len(ordered)))[::-1]
    vals = [by[lab]["computed"] for lab in ordered]
    ax.barh(ys, vals, height=0.5, color=[colors[lab] for lab in ordered], edgecolor=SURFACE, linewidth=2, zorder=3)
    for y, v, lab in zip(ys, vals, ordered):
        red = by[lab].get("total_prefill_reduction_pct")
        txt = f"{v / 1e6:.2f}M" + (f"  ({-red:+.1f}% vs baseline)" if red is not None and abs(red) > 1e-9 else "")
        ax.annotate(txt, (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ideal = [by[lab].get("ideal") for lab in ordered if by[lab].get("ideal")]
    if ideal:
        ax.axvline(ideal[0], color=INK_2, linewidth=1, zorder=4)
        # below the plot area, so it never collides with the title
        ax.annotate("ideal (infinite cache)", (ideal[0], 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -26), textcoords="offset points", ha="center", va="top", fontsize=8, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels(ordered)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
    ax.set_xlabel("Prompt tokens computed in prefill (summed over concurrency levels)", labelpad=18)
    ax.set_title("Prefill work by routing policy (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.35)
    return _save(fig, out / "prefill_tokens.png")


def plot_simple_metric(agg_rows, key: str, title: str, ylabel: str, fname: str, out: Path, colors: dict,
                       scale: float = 1.0) -> str | None:
    by_label = defaultdict(list)
    for r in agg_rows:
        if r.get(key) is not None:
            by_label[r["label"]].append(r)
    if not by_label:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    series = {}
    for label in label_order(list(by_label)):
        rows = sorted(by_label[label], key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in rows]
        ys = [r[key] * scale for r in rows]
        _line(ax, xs, ys, colors[label], label)
        series[label] = (xs, ys)
    _end_labels(ax, series)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["concurrency"] for r in agg_rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Concurrent sessions")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    return _save(fig, out / fname)


def plot_kv_timeseries(runs: list[dict], out: Path) -> str | None:
    """Per-worker KV-cache usage over time for the highest-concurrency run of each label."""
    best: dict[str, dict] = {}
    for r in runs:
        if r["repeat"] != 0:
            continue
        if r["label"] not in best or r["concurrency"] > best[r["label"]]["concurrency"]:
            best[r["label"]] = r
    panels = []
    for label in label_order(list(best)):
        p = Path(best[label]["run_dir"]) / "telemetry_router.jsonl"
        if p.exists():
            rows = read_jsonl(p)
            if rows:
                panels.append((label, best[label]["concurrency"], rows))
    if not panels:
        return None
    _style()
    n = len(panels)
    cols = min(2, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(6.5 * cols, 3.0 * nrows), squeeze=False, sharey=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (label, conc, rows) in zip(axes.flat, panels):
        by_w = defaultdict(list)
        for row in rows:
            if row.get("kv_usage") is not None:
                by_w[row["worker"]].append((row["ts"], row["kv_usage"] * 100))
        for i, w in enumerate(sorted(by_w)):
            pts = sorted(by_w[w])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i % len(SERIES)] if i < len(SERIES) else OVERFLOW,
                    linewidth=1.5, label=w)
        ax.set_title(f"{label}  (c={conc})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("KV cache in use (%)")
        ax.set_ylim(0, 100)
        if len(by_w) >= 2:
            ax.legend(loc="upper right", ncol=min(4, len(by_w)))
    fig.suptitle("KV-cache usage per worker (engine telemetry)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out / "kv_usage_timeseries.png")


def plot_gpu_memory(runs: list[dict], out: Path) -> str | None:
    """NVML GPU memory over time (real-GPU runs only)."""
    panels = []
    seen = set()
    for r in sorted(runs, key=lambda r: -r["concurrency"]):
        if r["label"] in seen or r["repeat"] != 0:
            continue
        p = Path(r["run_dir"]) / "telemetry_gpu.jsonl"
        if p.exists():
            panels.append((r["label"], r["concurrency"], read_jsonl(p)))
            seen.add(r["label"])
    if not panels:
        return None
    _style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.6 * len(panels)), squeeze=False, sharex=False)
    for ax, (label, conc, rows) in zip(axes.flat, panels):
        by_g = defaultdict(list)
        for row in rows:
            by_g[row["gpu"]].append((row["ts"], row["mem_used_bytes"] / 2**30))
        for i, g in enumerate(sorted(by_g)):
            pts = sorted(by_g[g])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i % len(SERIES)], linewidth=1.5, label=f"GPU {g}")
        ax.set_title(f"{label}  (c={conc})")
        ax.set_ylabel("GPU memory used (GiB)")
        ax.set_xlabel("Time (s)")
        if len(by_g) >= 2:
            ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, out / "gpu_memory.png")
