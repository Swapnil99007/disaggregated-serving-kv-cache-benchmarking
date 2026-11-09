"""Minimal Prometheus text-format parser (enough for vLLM / SGLang /metrics)."""

from __future__ import annotations


def parse_prometheus_text(text: str) -> dict[str, list[float]]:
    """Parse exposition text into ``{metric_name: [sample values across label sets]}``.

    Labels are discarded; callers decide whether to sum or max across series.
    """
    out: dict[str, list[float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line[: line.index("{")]
            rest = line[line.rindex("}") + 1:].strip()
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            name, rest = parts
        if not rest:
            continue
        try:
            value = float(rest.split()[0])
        except ValueError:
            continue
        out.setdefault(name, []).append(value)
    return out


def first_sum(metrics: dict[str, list[float]], names: list[str]) -> float | None:
    """Sum of the first metric name in ``names`` that is present."""
    for n in names:
        if n in metrics:
            return float(sum(metrics[n]))
    return None


def first_max(metrics: dict[str, list[float]], names: list[str]) -> float | None:
    """Max of the first metric name in ``names`` that is present."""
    for n in names:
        if n in metrics and metrics[n]:
            return float(max(metrics[n]))
    return None
