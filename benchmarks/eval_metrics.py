"""Trajectory comparison metrics for sim-vs-empirical evaluation.

Exposes:
  - dtw_distance(a, b)               — dynamic time warping, O(n*m), unit-free
  - ks_statistic(a, b)               — Kolmogorov-Smirnov D against empirical CDF
  - terminal_error(forecast, gt)     — |forecast[-1] - gt|
  - bootstrap_ci(errors, n=2000)     — percentile bootstrap CI on mean error
  - compare_to_ground_truth(series, gt_series) → dict of all above

Design notes:
  - DTW is the right primary metric here because our sim produces trajectories
    on its own schedule and polling dates are discrete/irregular. Euclidean /
    RMSE punishes time-warp even when the shape matches.
  - KS is robust to round-count mismatch (works on marginal distributions of
    positions), and is what readers familiar with forecasting literature expect.
  - Bootstrap CI on error is how we turn "our sim beat baseline by X" into
    "…with 95% CI [X_lo, X_hi]", which is the sentence any reviewer demands.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


# ---------- DTW ----------

def dtw_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Dynamic time warping distance between two 1-D sequences.

    Uses absolute difference as the pointwise cost (not squared) so the
    returned value is on the same scale as the inputs — you can read it as
    "total accumulated misalignment" in units of the input metric.
    """
    if not a or not b:
        raise ValueError("dtw requires non-empty sequences")
    n, m = len(a), len(b)
    INF = float("inf")
    # rolling 2-row DP to keep memory O(min(n,m))
    prev = [INF] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = [INF] * (m + 1)
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m]


def normalized_dtw(a: Sequence[float], b: Sequence[float]) -> float:
    """DTW normalized by the longer path length — so values are comparable
    across scenarios with different round counts."""
    d = dtw_distance(a, b)
    norm = max(len(a), len(b))
    return d / norm if norm else d


# ---------- KS ----------

def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov D statistic (max gap between empirical
    CDFs). No p-value — just the distance."""
    if not a or not b:
        raise ValueError("ks requires non-empty sequences")
    sa = sorted(a)
    sb = sorted(b)
    merged = sorted(set(sa + sb))
    na, nb = len(sa), len(sb)
    d = 0.0
    for v in merged:
        fa = sum(1 for x in sa if x <= v) / na
        fb = sum(1 for x in sb if x <= v) / nb
        gap = abs(fa - fb)
        if gap > d:
            d = gap
    return d


# ---------- Terminal / per-round ----------

def terminal_error(forecast: Sequence[float], gt_value: float) -> float:
    if not forecast:
        raise ValueError("forecast is empty")
    return abs(forecast[-1] - gt_value)


def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        raise ValueError("rmse requires non-empty sequences")
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    s = sum((a[i] - b[i]) ** 2 for i in range(n))
    return math.sqrt(s / n)


# ---------- Bootstrap CI ----------

def bootstrap_ci(
    values: Sequence[float],
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean.

    Returns (mean, ci_low, ci_high)."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_samples)
    hi_idx = int((1 + ci) / 2 * n_samples) - 1
    return (statistics.mean(values), means[lo_idx], means[hi_idx])


# ---------- Composite ----------

@dataclass
class TrajectoryComparison:
    dtw: float
    dtw_normalized: float
    ks: float
    rmse: float
    terminal_error: float | None


def compare_trajectories(
    forecast: Sequence[float],
    ground_truth: Sequence[float],
    terminal_gt: float | None = None,
) -> TrajectoryComparison:
    """Run all the standard trajectory comparisons. `terminal_gt` is the
    verified final value if known (e.g. referendum outcome); falls back to
    `ground_truth[-1]` otherwise."""
    term_gt = terminal_gt if terminal_gt is not None else (
        ground_truth[-1] if ground_truth else None
    )
    term_err = terminal_error(forecast, term_gt) if term_gt is not None else None
    return TrajectoryComparison(
        dtw=dtw_distance(forecast, ground_truth),
        dtw_normalized=normalized_dtw(forecast, ground_truth),
        ks=ks_statistic(forecast, ground_truth),
        rmse=rmse(forecast, ground_truth),
        terminal_error=term_err,
    )
