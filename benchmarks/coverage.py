"""Empirical calibration coverage for Monte Carlo confidence intervals.

For a simulation that produces a nominal X% CI on a per-round quantity
(polarization, avg_position, …), the EMPIRICAL coverage is the fraction of
realized observations that actually fall inside the interval. If the model
is well-calibrated, empirical ≈ nominal. Systematic over-coverage means the
intervals are too wide (low informational value); under-coverage means the
model is overconfident.

This module is scenario-agnostic: it takes any aligned sequence of
(ci_low, ci_high, realized) triples and computes the coverage plus a
bootstrap confidence band on the coverage itself.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CoverageReport:
    nominal: float           # e.g. 0.95
    empirical: float         # fraction in [0, 1]
    n: int
    ci_low: float            # bootstrap lower bound on empirical
    ci_high: float           # bootstrap upper bound on empirical
    mean_width: float        # average |high - low|, informational cost
    miss_below: int          # realizations below ci_low
    miss_above: int          # realizations above ci_high

    @property
    def is_calibrated(self) -> bool:
        """True iff the nominal sits inside the bootstrap CI on empirical."""
        return self.ci_low <= self.nominal <= self.ci_high

    def summary(self) -> str:
        flag = "OK " if self.is_calibrated else "OFF"
        return (
            f"[{flag}] nom={self.nominal:.0%} emp={self.empirical:.0%} "
            f"(bootstrap 95% CI [{self.ci_low:.0%},{self.ci_high:.0%}])  "
            f"n={self.n} width={self.mean_width:.3f}  "
            f"miss↓{self.miss_below} ↑{self.miss_above}"
        )


def compute_coverage(
    ci_lows: Sequence[float],
    ci_highs: Sequence[float],
    realized: Sequence[float],
    nominal: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> CoverageReport:
    n = len(realized)
    if not (len(ci_lows) == len(ci_highs) == n):
        raise ValueError("CI and realized sequences must have equal length")
    if n == 0:
        raise ValueError("empty series")

    hits = [1 if lo <= r <= hi else 0 for lo, hi, r in zip(ci_lows, ci_highs, realized)]
    empirical = sum(hits) / n
    miss_below = sum(1 for lo, r in zip(ci_lows, realized) if r < lo)
    miss_above = sum(1 for hi, r in zip(ci_highs, realized) if r > hi)
    widths = [abs(hi - lo) for lo, hi in zip(ci_lows, ci_highs)]
    mean_width = sum(widths) / n

    # Bootstrap confidence band on the empirical coverage itself.
    rng = random.Random(seed)
    boot_emp = []
    for _ in range(n_bootstrap):
        sample = [hits[rng.randrange(n)] for _ in range(n)]
        boot_emp.append(sum(sample) / n)
    boot_emp.sort()
    lo_idx = max(0, int(0.025 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap))

    return CoverageReport(
        nominal=nominal,
        empirical=empirical,
        n=n,
        ci_low=boot_emp[lo_idx],
        ci_high=boot_emp[hi_idx],
        mean_width=mean_width,
        miss_below=miss_below,
        miss_above=miss_above,
    )


def coverage_from_quantiles(
    simulated_samples_per_round: Sequence[Sequence[float]],
    realized: Sequence[float],
    nominal: float = 0.95,
    **kwargs,
) -> CoverageReport:
    """Convenience wrapper: given a Monte Carlo ensemble per round, derive
    the nominal-level interval from empirical quantiles of the simulated
    samples, then compute coverage against the realized values."""
    if len(simulated_samples_per_round) != len(realized):
        raise ValueError("per-round sample count must match realized length")
    alpha = (1.0 - nominal) / 2.0
    ci_lows, ci_highs = [], []
    for samples in simulated_samples_per_round:
        sorted_s = sorted(samples)
        k = len(sorted_s)
        if k == 0:
            raise ValueError("empty Monte Carlo bucket")
        lo_i = max(0, int(alpha * k))
        hi_i = min(k - 1, int((1.0 - alpha) * k))
        ci_lows.append(sorted_s[lo_i])
        ci_highs.append(sorted_s[hi_i])
    return compute_coverage(ci_lows, ci_highs, realized, nominal=nominal, **kwargs)
