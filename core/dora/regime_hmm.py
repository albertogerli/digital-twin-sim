"""Two-state Gaussian HMM on log(VIX) for posterior regime inference.

Sprint E.2 (Andrew Lo / regime-mixture).

Replaces hand-coded "calm/stressed/crisis" regime labels with a probabilistic
posterior — each historical incident gets P(state=high_vol | observation
sequence up to incident date), letting α be a soft mixture of regime-specific
calibrations rather than a hard category pick.

Model:
  Hidden state Z_t ∈ {0=low, 1=high}
  Emission   X_t = log(VIX_t) | Z_t ~ N(μ_z, σ_z²)
  Transition A[i,j] = P(Z_{t+1}=j | Z_t=i)

EM (Baum-Welch):
  E-step: forward-backward → γ (state posteriors), ξ (transition posteriors)
  M-step: re-estimate (π, A, μ, σ²) by closed-form weighted MLE

Numerical safety: log-space forward-backward to avoid underflow on
~350 observations × hundreds of EM iterations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIX_CACHE_PATH = REPO_ROOT / "shared" / "vix_monthly_cache.json"
HMM_FIT_CACHE_PATH = REPO_ROOT / "shared" / "regime_hmm_fit.json"


def _load_vix_series() -> tuple[list[str], np.ndarray]:
    """Returns (months_list_sorted, log_vix_array)."""
    data = json.loads(VIX_CACHE_PATH.read_text())
    rows = sorted(data["monthly"], key=lambda r: r["month"])
    months = [r["month"] for r in rows]
    log_vix = np.log(np.array([r["vix_close"] for r in rows], dtype=float))
    return months, log_vix


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_normal_pdf(x: float, mu: float, sigma2: float) -> float:
    if sigma2 <= 0:
        return -1e9
    return -0.5 * math.log(2 * math.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2


def fit_hmm_2state(
    obs: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int = 42,
) -> dict:
    """EM (Baum-Welch) for a 2-state Gaussian HMM.

    Returns:
      {
        pi:      [P(z_0=0), P(z_0=1)],
        A:       [[a00, a01], [a10, a11]],
        mu:      [mu_low, mu_high],
        sigma2:  [s2_low, s2_high],
        log_lik: final log-likelihood,
        n_iter:  iterations until convergence,
        gamma:   T×2 posterior state probabilities,
      }
    State 1 is enforced to be the high-mean state (high-vol regime).
    """
    rng = np.random.default_rng(seed)
    T = len(obs)
    K = 2

    # Init: cluster by quantile to break symmetry
    mu = np.array([np.quantile(obs, 0.25), np.quantile(obs, 0.75)])
    sigma2 = np.array([np.var(obs) * 0.5, np.var(obs) * 0.5])
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])  # persistent regimes prior

    log_pi = np.log(pi + 1e-12)
    log_A = np.log(A + 1e-12)
    log_lik_prev = -np.inf

    for it in range(max_iter):
        # E-step: forward-backward in log-space
        log_b = np.array([
            [_log_normal_pdf(obs[t], mu[k], sigma2[k]) for k in range(K)]
            for t in range(T)
        ])  # T×K

        # Forward (alpha)
        log_alpha = np.full((T, K), -1e9)
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_b[t, j]

        log_lik = _logsumexp(log_alpha[T - 1])

        # Backward (beta)
        log_beta = np.full((T, K), -1e9)
        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = _logsumexp(log_A[i, :] + log_b[t + 1] + log_beta[t + 1])

        # gamma (state posteriors)
        log_gamma = log_alpha + log_beta - log_lik
        gamma = np.exp(log_gamma)

        # xi (transition posteriors), summed over t
        log_xi_sum = np.full((K, K), -1e9)
        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t][:, None] + log_A + log_b[t + 1][None, :]
                + log_beta[t + 1][None, :] - log_lik
            )
            # log-sum-accumulate
            for i in range(K):
                for j in range(K):
                    log_xi_sum[i, j] = np.logaddexp(log_xi_sum[i, j], log_xi_t[i, j])

        # M-step
        new_pi = gamma[0] / gamma[0].sum()
        new_log_pi = np.log(new_pi + 1e-12)
        # A: row-normalised xi
        xi_sum = np.exp(log_xi_sum)
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        new_A = xi_sum / np.maximum(row_sums, 1e-12)
        new_log_A = np.log(new_A + 1e-12)
        # mu, sigma² weighted by gamma
        new_mu = np.zeros(K)
        new_sigma2 = np.zeros(K)
        for k in range(K):
            w = gamma[:, k]
            wsum = w.sum()
            new_mu[k] = (w * obs).sum() / max(wsum, 1e-12)
            new_sigma2[k] = (w * (obs - new_mu[k]) ** 2).sum() / max(wsum, 1e-12)
            new_sigma2[k] = max(new_sigma2[k], 1e-6)

        pi, log_pi = new_pi, new_log_pi
        A, log_A = new_A, new_log_A
        mu, sigma2 = new_mu, new_sigma2

        if abs(log_lik - log_lik_prev) < tol:
            break
        log_lik_prev = log_lik

    # Enforce state 1 = high-mean (high-vol regime)
    if mu[0] > mu[1]:
        mu = mu[::-1]
        sigma2 = sigma2[::-1]
        pi = pi[::-1]
        A = A[::-1, ::-1]
        gamma = gamma[:, ::-1]

    return {
        "pi": pi.tolist(),
        "A": A.tolist(),
        "mu": mu.tolist(),
        "sigma2": sigma2.tolist(),
        "log_lik": float(log_lik),
        "n_iter": it + 1,
        "gamma": gamma.tolist(),
    }


def fit_and_cache_regime_hmm() -> dict:
    """Fit the HMM on cached VIX, persist to disk, return summary."""
    months, log_vix = _load_vix_series()
    fit = fit_hmm_2state(log_vix)
    summary = {
        "_source": "core.dora.regime_hmm.fit_hmm_2state on log(^VIX) monthly",
        "_n_months": len(months),
        "_first_month": months[0],
        "_last_month": months[-1],
        "mu_low_logvix": round(fit["mu"][0], 4),
        "mu_high_logvix": round(fit["mu"][1], 4),
        "vix_low_implied": round(math.exp(fit["mu"][0]), 2),
        "vix_high_implied": round(math.exp(fit["mu"][1]), 2),
        "sigma_low": round(math.sqrt(fit["sigma2"][0]), 4),
        "sigma_high": round(math.sqrt(fit["sigma2"][1]), 4),
        "transition_persistence": {
            "P(low→low)": round(fit["A"][0][0], 4),
            "P(high→high)": round(fit["A"][1][1], 4),
        },
        "log_likelihood": round(fit["log_lik"], 2),
        "n_iter": fit["n_iter"],
        "gamma_by_month": {months[i]: round(fit["gamma"][i][1], 4) for i in range(len(months))},
    }
    HMM_FIT_CACHE_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


_GAMMA_CACHE: Optional[dict[str, float]] = None


def _load_gamma_cache() -> dict[str, float]:
    """Load month → P(state=high) from the cached HMM fit."""
    global _GAMMA_CACHE
    if _GAMMA_CACHE is not None:
        return _GAMMA_CACHE
    if not HMM_FIT_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(HMM_FIT_CACHE_PATH.read_text())
        _GAMMA_CACHE = {m: float(p) for m, p in data.get("gamma_by_month", {}).items()}
        return _GAMMA_CACHE
    except Exception:
        return {}


def regime_posterior_for_date(iso_date: str) -> Optional[float]:
    """Return P(state=high_vol | observation sequence) for the month
    containing iso_date (YYYY-MM-DD). None if the month is outside the cache."""
    if not iso_date or len(iso_date) < 7:
        return None
    month = iso_date[:7]  # YYYY-MM
    cache = _load_gamma_cache()
    return cache.get(month)


if __name__ == "__main__":
    summary = fit_and_cache_regime_hmm()
    print(f"VIX low ≈ {summary['vix_low_implied']}, high ≈ {summary['vix_high_implied']}")
    print(f"Persistence: P(low→low)={summary['transition_persistence']['P(low→low)']}, "
          f"P(high→high)={summary['transition_persistence']['P(high→high)']}")
    print(f"Iterations: {summary['n_iter']}, log-lik: {summary['log_likelihood']}")
