"""Null-model forecasters used as baselines for Diebold-Mariano comparison.

A forecaster is any callable that maps a history prefix `h[:t]` plus a
horizon `k` to a point forecast for `h[t+k]`. We only need point forecasts
because DM operates on squared (or absolute) errors against realized values.

`naive_persistence` and `random_walk` are the canonical bars every forecast
evaluation must clear. If our simulation cannot beat `naive_persistence` on a
given scenario, that scenario either has a trivial trajectory or our model is
adding no signal beyond "tomorrow looks like today".
"""

from __future__ import annotations

from typing import Callable, Sequence

import math

Forecaster = Callable[[Sequence[float], int], float]


def naive_persistence(history: Sequence[float], horizon: int) -> float:
    """y_hat(t+k) = y(t). The classic "no-change" baseline."""
    if not history:
        return 0.0
    return history[-1]


def random_walk_mean(history: Sequence[float], horizon: int) -> float:
    """Mean of observed history — random walk without drift, collapsed."""
    if not history:
        return 0.0
    return sum(history) / len(history)


def linear_trend(history: Sequence[float], horizon: int) -> float:
    """OLS linear trend extrapolated `horizon` steps past the last point.

    Uses closed-form slope on indices 0..n-1. Returns `history[-1]` when
    there are fewer than two points (no slope).
    """
    n = len(history)
    if n < 2:
        return history[-1] if history else 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(history) / n
    num = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return y_mean
    slope = num / den
    intercept = y_mean - slope * x_mean
    return intercept + slope * (n - 1 + horizon)


def ar1(history: Sequence[float], horizon: int) -> float:
    """AR(1): y_hat(t+1) = mu + phi * (y(t) - mu), phi from sample autocorr.

    Iterates `horizon` steps. Shrinks toward the mean as horizon grows,
    which is the right behavior — a stationary AR(1) eventually regresses.
    """
    n = len(history)
    if n < 3:
        return naive_persistence(history, horizon)
    mu = sum(history) / n
    num = sum((history[i] - mu) * (history[i - 1] - mu) for i in range(1, n))
    den = sum((history[i] - mu) ** 2 for i in range(n))
    phi = 0.0 if den == 0 else max(-0.99, min(0.99, num / den))
    y = history[-1]
    for _ in range(max(1, horizon)):
        y = mu + phi * (y - mu)
    return y


BASELINES: dict[str, Forecaster] = {
    "persistence": naive_persistence,
    "mean": random_walk_mean,
    "linear_trend": linear_trend,
    "ar1": ar1,
}


def generate_baseline_trajectory(
    forecaster: Forecaster,
    realized: Sequence[float],
    train_frac: float = 0.3,
) -> list[float]:
    """Produce a one-step-ahead forecast trajectory aligned with `realized`.

    The first `split = max(1, int(len(realized) * train_frac))` points are
    held out as history. From there we emit one forecast per realized point,
    rolling the observed value forward after each step (one-step-ahead, not
    recursive multistep — keeps baselines honest).
    """
    n = len(realized)
    if n == 0:
        return []
    split = max(1, int(n * train_frac))
    out: list[float] = list(realized[:split])
    for t in range(split, n):
        out.append(forecaster(list(realized[:t]), 1))
    return out


def forecast_errors(
    forecasts: Sequence[float], realized: Sequence[float]
) -> list[float]:
    """Squared errors aligned one-to-one. Lengths must match."""
    if len(forecasts) != len(realized):
        raise ValueError(
            f"length mismatch: forecasts={len(forecasts)} realized={len(realized)}"
        )
    return [(f - r) ** 2 for f, r in zip(forecasts, realized)]


def rmse(forecasts: Sequence[float], realized: Sequence[float]) -> float:
    errs = forecast_errors(forecasts, realized)
    return math.sqrt(sum(errs) / len(errs)) if errs else float("inf")
