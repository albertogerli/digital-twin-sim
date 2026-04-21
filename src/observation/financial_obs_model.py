"""Financial observation model — connects opinion dynamics to market returns.

Links simulator outputs (opinion trajectories, event shocks, polarization)
to observed financial market returns via sector beta coefficients.

Likelihood: actual_return(t) ~ Normal(expected_return(t), σ_market)

where expected_return(t) = sector_beta × (
    w_opinion × Δopinion(t) +
    w_event × shock(t) +
    w_polar × Δpolarization(t)
)

All functions are pure JAX, jit-compilable, differentiable.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import json
import os


# ── Financial Observation Data ───────────────────────────────────────────────

class FinancialObsData(NamedTuple):
    """Financial market observations for a single scenario.

    Fields:
        returns_pct: [n_rounds] actual primary ticker return per round (%).
                     Rounds without data should be 0.0.
        return_mask: [n_rounds] boolean — True for rounds with valid market data.
        benchmark_returns: [n_rounds] benchmark (SPY) return per round (%).
        sector_beta: scalar — political_beta for the primary sector.
        cumulative_return: scalar — total return over scenario window (%).
    """
    returns_pct: jnp.ndarray        # [n_rounds]
    return_mask: jnp.ndarray        # [n_rounds] (bool)
    benchmark_returns: jnp.ndarray  # [n_rounds]
    sector_beta: float              # scalar
    cumulative_return: float        # scalar


# ── Sector Beta Lookup ───────────────────────────────────────────────────────
# Calibration-time political_beta snapshot for the IT regime. The authoritative
# source is `MarketContext(geography="IT").get_beta(sector).political_beta` —
# this dict is kept in-tree because the calibration pipeline runs under JAX
# and must not depend on the full market-context stack.

POLITICAL_BETAS = {
    "banking": 1.85,
    "insurance": 1.45,
    "automotive": 1.30,
    "energy_fossil": 1.10,
    "energy_renewable": 0.75,
    "utilities": 0.60,
    "defense": 0.90,
    "telecom": 1.15,
    "tech": 0.85,
    "healthcare": 0.55,
    "real_estate": 1.20,
    "infrastructure": 0.80,
    "luxury": 0.70,
    "food_consumer": 0.50,
    "sovereign_debt": 2.20,
}


# ── Linkage Function ─────────────────────────────────────────────────────────

def compute_expected_returns(
    pro_fraction: jnp.ndarray,     # [n_rounds] 0-1
    trajectories: jnp.ndarray,     # [n_rounds, n_agents]
    events: jnp.ndarray,           # [n_rounds, 2] (magnitude, direction)
    agent_mask: jnp.ndarray,       # [n_agents] True for real agents
    sector_beta: float,
    w_opinion: float,
    w_event: float,
    w_polar: float,
) -> jnp.ndarray:
    """Compute expected per-round market return from simulator outputs.

    The linkage function translates three opinion dynamics signals into
    expected market returns:

    1. Opinion shift: Δpro_fraction(t) — sentiment change drives markets
    2. Event shock: magnitude × direction — exogenous events move markets
    3. Polarization change: Δstd(positions) — uncertainty premium

    Returns:
        expected_returns: [n_rounds] expected return per round (%)
    """
    n_rounds = pro_fraction.shape[0]

    # 1. Opinion shift signal (first-difference, 0 for round 0)
    delta_opinion = jnp.zeros(n_rounds)
    delta_opinion = delta_opinion.at[1:].set(pro_fraction[1:] - pro_fraction[:-1])

    # 2. Event shock signal (magnitude × direction)
    shock_signal = events[:, 0] * events[:, 1]  # [n_rounds]

    # 3. Polarization signal (std of agent positions, masked)
    mask = agent_mask[None, :]  # [1, n_agents]
    masked_traj = jnp.where(mask, trajectories, jnp.nan)
    # nanstd per round
    round_mean = jnp.nanmean(masked_traj, axis=1)  # [n_rounds]
    round_var = jnp.nanmean(
        jnp.where(mask, (trajectories - round_mean[:, None]) ** 2, 0.0), axis=1
    )
    polarization = jnp.sqrt(round_var + 1e-8)  # [n_rounds]
    delta_polar = jnp.zeros(n_rounds)
    delta_polar = delta_polar.at[1:].set(polarization[1:] - polarization[:-1])

    # Scale factors: opinion shift in [-1,1] → market return in %
    # Events typically 0-1 magnitude → scale to market impact
    OPINION_SCALE = 10.0   # 100% opinion swing ≈ 10% market move
    EVENT_SCALE = 5.0      # max event shock ≈ 5% market move
    POLAR_SCALE = 8.0      # polarization increase ≈ uncertainty premium

    composite = (
        w_opinion * delta_opinion * OPINION_SCALE +
        w_event * shock_signal * EVENT_SCALE +
        w_polar * delta_polar * POLAR_SCALE
    )

    # Apply sector beta
    expected = sector_beta * composite

    return expected


# ── Financial Log-Likelihood ─────────────────────────────────────────────────

def log_likelihood_financial_one(
    pro_fraction: jnp.ndarray,     # [R]
    trajectories: jnp.ndarray,     # [R, A]
    events: jnp.ndarray,           # [R, 2]
    agent_mask: jnp.ndarray,       # [A]
    fin_returns: jnp.ndarray,      # [R] actual returns
    fin_mask: jnp.ndarray,         # [R] validity mask
    fin_sector_beta: float,
    w_opinion: float,
    w_event: float,
    w_polar: float,
    log_sigma_market: float,
) -> jnp.ndarray:
    """Compute financial return log-likelihood for one scenario.

    Returns scalar log-likelihood (sum over valid market rounds).
    """
    expected = compute_expected_returns(
        pro_fraction, trajectories, events, agent_mask,
        fin_sector_beta, w_opinion, w_event, w_polar,
    )

    sigma = jnp.exp(log_sigma_market)

    # Excess return (vs benchmark) — if benchmark data exists, already subtracted
    residual = fin_returns - expected

    # Normal log-likelihood per round
    ll_per_round = (
        -0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(sigma)
        - 0.5 * (residual / sigma) ** 2
    )

    # Mask: only sum over rounds with valid market data
    ll = jnp.sum(jnp.where(fin_mask, ll_per_round, 0.0))
    return ll


# ── Loader ───────────────────────────────────────────────────────────────────

def load_financial_observations(
    scenario_json_path: str,
    n_rounds: int,
) -> tuple[bool, FinancialObsData | None]:
    """Load financial observations from enriched scenario JSON.

    Returns:
        (has_financial_obs, fin_obs_or_None)
    """
    with open(scenario_json_path) as f:
        scenario = json.load(f)

    market = scenario.get("market_observations")
    if market is None:
        return False, None

    per_round = market.get("per_round_returns", [])
    benchmark = market.get("benchmark_per_round", [])
    sector_key = market.get("primary_sector", "")

    # Build per-round arrays
    returns = jnp.zeros(n_rounds)
    masks = jnp.zeros(n_rounds, dtype=jnp.bool_)
    bench = jnp.zeros(n_rounds)

    for entry in per_round:
        r = entry["round"] - 1  # 0-based
        if 0 <= r < n_rounds and entry.get("return_pct") is not None:
            returns = returns.at[r].set(float(entry["return_pct"]))
            masks = masks.at[r].set(True)

    for entry in benchmark:
        r = entry["round"] - 1
        if 0 <= r < n_rounds and entry.get("return_pct") is not None:
            bench = bench.at[r].set(float(entry["return_pct"]))

    # Use excess return (primary - benchmark) for cleaner signal
    excess_returns = returns - bench

    # Look up sector beta
    sector_beta = POLITICAL_BETAS.get(sector_key, 1.0)
    cumulative = market.get("cumulative_return_pct", 0.0)

    # Need at least 2 rounds of data
    n_valid = int(jnp.sum(masks))
    if n_valid < 2:
        return False, None

    fin_obs = FinancialObsData(
        returns_pct=excess_returns,
        return_mask=masks,
        benchmark_returns=bench,
        sector_beta=sector_beta,
        cumulative_return=cumulative,
    )

    return True, fin_obs
