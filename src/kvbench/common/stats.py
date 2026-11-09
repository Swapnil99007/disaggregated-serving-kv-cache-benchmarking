"""Small statistics helpers (percentiles, confidence intervals)."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# Two-sided 95% Student-t critical values for df = 1..30.
_T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]


def _clean(xs: Iterable[float | None]) -> list[float]:
    return [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def percentile(xs: Iterable[float | None], p: float) -> float | None:
    vals = _clean(xs)
    if not vals:
        return None
    return float(np.percentile(vals, p))


def describe(xs: Iterable[float | None]) -> dict[str, float | None]:
    vals = _clean(xs)
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(vals)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def mean_ci95(xs: Iterable[float | None]) -> tuple[float | None, float | None, float | None]:
    """Return (mean, sample std, 95% CI half-width) using a Student-t interval."""
    vals = _clean(xs)
    if not vals:
        return None, None, None
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return m, 0.0, None
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = _T95[min(n - 1, len(_T95)) - 1]
    return m, sd, t * sd / math.sqrt(n)
