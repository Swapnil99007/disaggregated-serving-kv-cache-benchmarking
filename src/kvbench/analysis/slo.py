"""Throughput at a matched tail-latency target.

For each label we have a curve of (concurrency -> p95 latency, throughput). In a
closed loop both rise with concurrency until the system saturates, after which
throughput flattens and latency keeps climbing. "Throughput at matched p95" asks:
*what is the most throughput this configuration sustains while p95 latency stays
at or under T?* We take the best measured point under T and also linearly
interpolate along each segment that crosses T.

Comparing policies at the same T prevents a policy from "winning" throughput by
simply accepting worse tail latency.

Two pitfalls this module guards against:

* **Unsaturated (flat) region.** Below saturation, latency is nearly constant while
  throughput grows ~linearly with sessions (Little's law). If T falls there, a few
  milliseconds of latency difference become a huge throughput difference. Results
  interpolated on such a segment are flagged ``flat region``. The comparison is
  only meaningful where latency is actually rising.
* **Target choice.** The default T is the baseline's p95 at its *highest* concurrency
  level, i.e. the tail latency the baseline needed at peak load. Policies that stay
  under T everywhere are reported as a (conservative) lower bound.
"""

from __future__ import annotations

from collections import defaultdict

from kvbench.common.stats import mean_ci95

# a crossing segment whose relative latency growth is < this fraction of its relative
# throughput growth is "flat": latency is not the binding constraint there
FLAT_ELASTICITY = 0.25


def throughput_at_slo(points: list[tuple[float, float | None, float | None]], target: float) -> tuple[float | None, str]:
    """points: [(concurrency, latency, throughput)]. Returns (throughput, note)."""
    pts = sorted((c, l, x) for c, l, x in points if l is not None and x is not None)
    if not pts:
        return None, "no data"
    best, best_flat = None, False
    for _, l, x in pts:
        if l <= target and (best is None or x > best):
            best, best_flat = x, False
    for (_, l1, x1), (_, l2, x2) in zip(pts, pts[1:]):
        if (l1 - target) * (l2 - target) < 0:
            x = x1 + (target - l1) / (l2 - l1) * (x2 - x1)
            rel_l = abs(l2 - l1) / max(min(l1, l2), 1e-9)
            rel_x = abs(x2 - x1) / max(min(x1, x2), 1e-9)
            flat = rel_x > 0 and (rel_l / rel_x) < FLAT_ELASTICITY
            if best is None or x > best:
                best, best_flat = x, flat
    if best is None:
        return None, "target below every measured point"
    if all(l < target for _, l, _ in pts):
        return best, "lower bound: never reached target (add higher concurrency)"
    if best_flat:
        return best, "flat region: latency barely rises here, result is unreliable"
    return best, "ok"


def matched_slo_table(
    agg_rows: list[dict],
    runs: list[dict],
    baseline_label: str,
    metric: str = "e2e",
    percentile: int = 95,
    target_s: float | None = None,
    throughput_key: str = "steady_throughput_rps",
) -> dict:
    lat_key = f"{metric}_p{percentile}"
    curves: dict[str, list[tuple]] = defaultdict(list)
    for r in agg_rows:
        curves[r["label"]].append((r["concurrency"], r.get(lat_key), r.get(throughput_key) or r.get("throughput_rps")))

    auto = False
    if target_s is None:
        base = sorted(p for p in curves.get(baseline_label, []) if p[1] is not None)
        if not base:
            return {"rows": [], "target_s": None, "metric": lat_key, "auto_target": True}
        target_s = base[-1][1]     # baseline's tail latency at its highest concurrency level
        auto = True

    # per-repeat curves -> spread of the improvement estimate
    per_rep: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for r in runs:
        s = r["summary"]
        per_rep[(r["label"], r["repeat"])].append((r["concurrency"], s.get(lat_key), s.get(throughput_key) or s.get("throughput_rps")))

    base_x, _ = throughput_at_slo(curves.get(baseline_label, []), target_s)
    rows = []
    for label, pts in sorted(curves.items()):
        x, note = throughput_at_slo(pts, target_s)
        improvements = []
        for (lab, rep), rpts in per_rep.items():
            if lab != label:
                continue
            bx, _ = throughput_at_slo(per_rep.get((baseline_label, rep), []), target_s)
            xx, _ = throughput_at_slo(rpts, target_s)
            if bx and xx is not None:
                improvements.append(100.0 * (xx / bx - 1.0))
        imp_m, imp_sd, imp_ci = mean_ci95(improvements)
        rows.append({
            "label": label,
            "throughput_at_target": x,
            "improvement_pct": (100.0 * (x / base_x - 1.0)) if (x is not None and base_x) else None,
            "improvement_pct_repeat_mean": imp_m,
            "improvement_pct_repeat_std": imp_sd,
            "improvement_pct_repeat_ci95": imp_ci,
            "repeats": len(improvements),
            "note": note,
        })
    return {"rows": rows, "target_s": target_s, "metric": lat_key, "auto_target": auto,
            "throughput_key": throughput_key, "baseline": baseline_label}
