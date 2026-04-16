"""Diebold-Mariano (1995) test of equal predictive accuracy, with the
Harvey-Leybourne-Newbold (1997) small-sample correction.

H0: both forecasts have the same expected loss.
H1: the one with lower mean loss is strictly better.

Usage:
    stat, p_value = dm_test(errors_a, errors_b, horizon=1)

A significantly negative DM statistic (p < 0.05) means forecast A beats B.

We implement HLN explicitly because scipy.stats doesn't ship it and the
uncorrected DM over-rejects for short horizons (our scenarios have 5-9
rounds, which is squarely in HLN territory).

References:
    Diebold, F. X., & Mariano, R. S. (1995). JBES 13(3).
    Harvey, D., Leybourne, S., & Newbold, P. (1997). IJF 13(2).
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

from dataclasses import dataclass


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_loss_a: float
    mean_loss_b: float
    n: int
    horizon: int
    correction: Literal["none", "hln"]
    better: Literal["a", "b", "tie"]

    def summary(self) -> str:
        star = "*" if self.p_value < 0.05 else " "
        return (
            f"DM={self.statistic:+.3f} p={self.p_value:.3f}{star}  "
            f"E[L_a]={self.mean_loss_a:.4f} E[L_b]={self.mean_loss_b:.4f}  "
            f"→ {self.better}"
        )


def _student_t_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of Student's t. Two-sided p = 2·sf(|x|).

    Uses the regularized incomplete beta identity:
        P(T > x) = 0.5 * I_{df/(df+x^2)}(df/2, 1/2)
    """
    if df <= 0:
        return 1.0
    if x == 0:
        return 0.5
    bx = df / (df + x * x)
    prob = 0.5 * _betainc(df / 2.0, 0.5, bx)
    return prob if x > 0 else 1.0 - prob


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta via Lentz's continued fraction (Num. Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3.0e-12) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def dm_test(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    horizon: int = 1,
    correction: Literal["none", "hln"] = "hln",
) -> DMResult:
    """Compute the DM statistic and two-sided p-value.

    Inputs are PER-OBSERVATION losses (squared or absolute errors).
    `correction="hln"` applies Harvey-Leybourne-Newbold's finite-sample
    rescaling and uses Student's t distribution with T-1 df instead of N(0,1),
    which is the right move for short horizons (T < 50).
    """
    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"length mismatch: a={len(errors_a)} b={len(errors_b)}"
        )
    n = len(errors_a)
    if n < 2:
        raise ValueError("need at least 2 observations")

    d = [a - b for a, b in zip(errors_a, errors_b)]
    mean_d = sum(d) / n

    # HAC-style variance estimate with h-1 autocovariance lags (DM-1995).
    def autocov(lag: int) -> float:
        return sum((d[t] - mean_d) * (d[t - lag] - mean_d) for t in range(lag, n)) / n

    var = autocov(0)
    for lag in range(1, horizon):
        var += 2.0 * autocov(lag)

    # Guard against numerically-negative variance when series is nearly constant.
    var = max(var, 1e-12)
    stat = mean_d / math.sqrt(var / n)

    if correction == "hln":
        # HLN scaling: (n + 1 - 2h + h(h-1)/n) / n under the sqrt.
        scale = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        scale = max(scale, 1e-12)
        stat *= math.sqrt(scale)
        # Two-sided t(n-1) tail.
        p_value = 2.0 * _student_t_sf(abs(stat), df=n - 1)
    else:
        # Normal approximation.
        p_value = 2.0 * (1.0 - _phi(abs(stat)))

    mean_a = sum(errors_a) / n
    mean_b = sum(errors_b) / n
    if p_value < 0.05:
        better: Literal["a", "b", "tie"] = "a" if stat < 0 else "b"
    else:
        better = "tie"

    return DMResult(
        statistic=stat,
        p_value=p_value,
        mean_loss_a=mean_a,
        mean_loss_b=mean_b,
        n=n,
        horizon=horizon,
        correction=correction,
        better=better,
    )


def _phi(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
