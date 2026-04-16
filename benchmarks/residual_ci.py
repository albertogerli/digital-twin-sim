"""Residual-bootstrap intervals for turning a point-forecast into a calibrated CI.

Our sim currently emits a single deterministic trajectory per scenario, not a
Monte Carlo ensemble, so `coverage.coverage_from_quantiles` can't be applied
directly. This module fills that gap: given a point-forecast trajectory and its
residuals against the realized series, it resamples residuals to build nominal
intervals, which can then be fed to `compute_coverage`.

The point is not to fake uncertainty — it's to answer: "If we believe the sim
trajectory and use the historical error distribution as our uncertainty proxy,
are the implied intervals well-calibrated?" An over-confident sim shows up as
under-coverage here just as it would with a real ensemble.
"""

from __future__ import annotations

import random
import statistics
from typing import Sequence


def residual_bootstrap_intervals(
    point_forecast: Sequence[float],
    realized: Sequence[float],
    nominal: float = 0.95,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Return (ci_lows, ci_highs) per round using residual resampling.

    Residuals `r_t = realized_t - forecast_t` are pooled across all rounds.
    For each round, we draw `n_samples` residuals with replacement, add them
    to the forecast, and take empirical quantiles for the interval.
    """
    n = len(realized)
    if len(point_forecast) != n:
        raise ValueError(
            f"length mismatch: forecast={len(point_forecast)} realized={n}"
        )
    if n == 0:
        return [], []
    residuals = [r - f for r, f in zip(realized, point_forecast)]
    alpha = (1.0 - nominal) / 2.0
    rng = random.Random(seed)
    ci_lows, ci_highs = [], []
    for f in point_forecast:
        draws = sorted(f + residuals[rng.randrange(n)] for _ in range(n_samples))
        lo_i = max(0, int(alpha * n_samples))
        hi_i = min(n_samples - 1, int((1.0 - alpha) * n_samples))
        ci_lows.append(draws[lo_i])
        ci_highs.append(draws[hi_i])
    return ci_lows, ci_highs


def residual_summary(
    point_forecast: Sequence[float],
    realized: Sequence[float],
) -> dict:
    """Quick stats on residuals: mean (bias), stdev (spread), max |error|."""
    n = len(realized)
    if len(point_forecast) != n or n == 0:
        return {"n": n, "bias": 0.0, "stdev": 0.0, "max_abs": 0.0}
    resid = [r - f for r, f in zip(realized, point_forecast)]
    bias = sum(resid) / n
    stdev = statistics.pstdev(resid) if n > 1 else 0.0
    return {
        "n": n,
        "bias": bias,
        "stdev": stdev,
        "max_abs": max(abs(x) for x in resid),
    }
