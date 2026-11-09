"""Prefill-work accounting: how much prompt computation did each policy avoid?

Definitions (all per replay of the same trace, averaged across repeats):

* ``prompt_tokens``     total prompt tokens submitted.
* ``computed``          prompt tokens actually computed by the engines
                        = prompt_tokens - cached_tokens (from per-request
                        ``usage.prompt_tokens_details.cached_tokens``; in disagg
                        mode, from the prefill worker's usage).
* ``ideal``             minimum possible prefill work: tokens that would still need
                        computing with one infinitely large prefix cache. Computed
                        from the trace's distinct prefix chunks (characters), then
                        converted to tokens with the run's measured tokens/char.
* ``redundant``         computed - ideal: work that a perfect cache would have skipped.

Two reductions are reported versus the baseline policy:

* total prefill reduction     = (B.computed  - X.computed)  / B.computed
* redundant prefill reduction = (B.redundant - X.redundant) / B.redundant

These are *token* counts, not GPU time or latency. Report them as such.
"""

from __future__ import annotations

from collections import defaultdict

from kvbench.common.stats import mean_ci95


def _ideal_tokens(run: dict) -> float | None:
    ts = run.get("trace_stats") or {}
    s = run["summary"]
    chars, toks = s.get("prompt_chars_total"), s.get("prompt_tokens_total")
    if not ts.get("unique_prefix_chars") or not chars or not toks:
        return None
    return ts["unique_prefix_chars"] * (toks / chars)


def prefill_table(runs: list[dict], baseline_label: str) -> list[dict]:
    """One row per (label, concurrency) plus one 'all' row per label (sum over concurrency)."""
    per: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in runs:
        if r["summary"].get("prefill_computed_total") is None:
            continue
        per[(r["label"], r["concurrency"])].append(r)

    def agg(rs: list[dict]) -> dict:
        prompt = mean_ci95(r["summary"]["prompt_tokens_total"] for r in rs)[0]
        comp_m, comp_sd, _ = mean_ci95(r["summary"]["prefill_computed_total"] for r in rs)
        ideal = mean_ci95(_ideal_tokens(r) for r in rs)[0]
        return {
            "prompt_tokens": prompt,
            "computed": comp_m,
            "computed_std": comp_sd,
            "ideal": ideal,
            "redundant": (comp_m - ideal) if (comp_m is not None and ideal is not None) else None,
            "hit_rate": mean_ci95(r["summary"].get("cache_hit_rate") for r in rs)[0],
            "repeats": len(rs),
        }

    rows = []
    by_label_conc = {k: agg(v) for k, v in per.items()}
    labels = sorted({k[0] for k in per})
    concs = sorted({k[1] for k in per})
    for label in labels:
        for c in concs:
            if (label, c) not in by_label_conc:
                continue
            row = {"label": label, "concurrency": c, **by_label_conc[(label, c)]}
            base = by_label_conc.get((baseline_label, c))
            row.update(_reductions(row, base))
            rows.append(row)
        # 'all' = sum across concurrency levels present for both label and baseline
        common = [c for c in concs if (label, c) in by_label_conc and (baseline_label, c) in by_label_conc]
        if common:
            tot = _sum_rows([by_label_conc[(label, c)] for c in common])
            btot = _sum_rows([by_label_conc[(baseline_label, c)] for c in common])
            row = {"label": label, "concurrency": "all", **tot}
            row.update(_reductions(row, btot))
            rows.append(row)
    return rows


def _sum_rows(rows: list[dict]) -> dict:
    out = {}
    for k in ("prompt_tokens", "computed", "ideal", "redundant"):
        vals = [r[k] for r in rows]
        out[k] = sum(vals) if all(v is not None for v in vals) else None
    out["hit_rate"] = 1 - out["computed"] / out["prompt_tokens"] if out["computed"] is not None and out["prompt_tokens"] else None
    out["repeats"] = min(r["repeats"] for r in rows)
    return out


def _reductions(row: dict, base: dict | None) -> dict:
    out = {"total_prefill_reduction_pct": None, "redundant_prefill_reduction_pct": None}
    if not base:
        return out
    if base.get("computed"):
        out["total_prefill_reduction_pct"] = 100.0 * (base["computed"] - row["computed"]) / base["computed"]
    if base.get("redundant") and row.get("redundant") is not None and base["redundant"] > 0:
        out["redundant_prefill_reduction_pct"] = 100.0 * (base["redundant"] - row["redundant"]) / base["redundant"]
    return out
