"""Opinion Dynamics v2 — JAX port for differentiable inference.

Pure-JAX reimplementation of DynamicsV2 with:
  - Gauge-fixed softmax: α_direct = 0 (reference level)
  - Smooth thresholds for differentiability (sigmoid + tanh)
  - EMA force standardization (decay=0.7, ~3 round memory, aligned with NumPy rolling buffer)
  - Sparse interaction matrix (K=5 neighbors per agent, aligned with NumPy feed-based)
  - jax.lax.scan for jit-compatible round loop
  - vmap-ready for batched scenario simulation

Parameters (D=8):
  Calibrable: alpha_herd, alpha_anchor, alpha_social, alpha_event
  Frozen: log_lambda_elite, log_lambda_citizen, logit_herd_threshold, logit_anchor_drift
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from .param_utils import CALIBRABLE_PARAMS, FROZEN_PARAMS, ALPHA_KEYS

# ── Constants ────────────────────────────────────────────────────

N_FORCES = 5
EMA_DECAY = 0.3        # calibrated: best alignment with NumPy rolling buffer
K_NEIGHBORS = 5        # sparse interaction: neighbors per agent (matches NumPy feed top_n=5)
BUFFER_TAIL = 8        # mimic NumPy deque(maxlen=8): stats from last 8 agent forces

# ── Scenario Data ────────────────────────────────────────────────

class ScenarioData(NamedTuple):
    """Static scenario inputs (no JAX tracing through these shapes)."""
    initial_positions: jnp.ndarray    # [n_agents]         p_i(0) in [-1, +1]
    agent_types: jnp.ndarray          # [n_agents]         0=elite, 1=citizen
    agent_rigidities: jnp.ndarray     # [n_agents]         rigidity in [0, 1]
    agent_tolerances: jnp.ndarray     # [n_agents]         tolerance for bounded confidence
    events: jnp.ndarray               # [n_rounds, 2]      (magnitude, direction)
    llm_shifts: jnp.ndarray           # [n_rounds, n_agents] precomputed ΔLLM
    interaction_matrix: jnp.ndarray   # [n_agents, n_agents] sparse social graph (K nonzero per row)


# ── Smooth primitives ───────────────────────────────────────────

TAU_HERD = 0.02   # sigmoid steepness for herd threshold
TAU_BC = 0.02     # sigmoid steepness for bounded-confidence tolerance


def smooth_step(x, threshold, tau):
    """Differentiable step: σ((x - threshold) / τ)."""
    return sigmoid((x - threshold) / tau)


def smooth_clamp(x, max_val):
    """Differentiable clamp: max_val * tanh(x / max_val)."""
    return max_val * jnp.tanh(x / (max_val + 1e-8))


# ── Sparse interaction matrix construction ───────────────────────

def build_sparse_interaction(influences: jnp.ndarray, k: int = K_NEIGHBORS,
                              seed: int = 42) -> jnp.ndarray:
    """Build a sparse interaction matrix with K random neighbors per agent.

    For each agent i, selects K random agents (excluding self) as neighbors.
    This mirrors the NumPy SobolPlatform behavior where each agent sees a
    random subset of posts, avoiding the systematic bias of top-K selection.

    Weight = influence_j * 0.1 (same scale as the full matrix).
    Selection is deterministic (seeded) and fixed for the scenario.

    Returns a dense [n, n] matrix with exactly K nonzero entries per row.
    """
    n = influences.shape[0]
    k = min(k, n - 1)  # Can't pick more neighbors than available agents

    # Full weight matrix: w[i,j] = influence_j * 0.1
    full_weights = jnp.broadcast_to(influences[None, :] * 0.1, (n, n))

    # Zero out self-connections
    mask_self = 1.0 - jnp.eye(n)
    full_weights = full_weights * mask_self

    # Random K neighbors per agent (deterministic via seed)
    # Use Gumbel-max trick for differentiable-friendly random selection
    key = jax.random.PRNGKey(seed)
    # Add -inf for self to exclude, random noise for others
    gumbel_noise = jax.random.gumbel(key, shape=(n, n))
    # Mask self with -inf so self is never selected
    gumbel_noise = gumbel_noise + jnp.log(mask_self + 1e-30)

    # Select top-K by noisy score → random K neighbors
    _, top_indices = jax.lax.top_k(gumbel_noise, k)

    # Build sparse mask
    sparse_mask = jnp.zeros((n, n))
    rows = jnp.repeat(jnp.arange(n), k)
    cols = top_indices.ravel()
    sparse_mask = sparse_mask.at[rows, cols].set(1.0)

    return full_weights * sparse_mask


# ── EMA standardization ─────────────────────────────────────────

def ema_standardize(
    forces: jnp.ndarray,     # [n_agents, 5]
    ema_mu: jnp.ndarray,     # [5]
    ema_sigma: jnp.ndarray,  # [5]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """EMA-based force standardization, differentiable and scan-compatible.

    Computes current-round statistics from ALL agents (not a subset),
    then applies EMA temporal smoothing across rounds (decay=0.3).

    This is order-independent: permuting the agent array produces
    identical output because mean/variance are symmetric statistics
    over the full agent set.

    Returns: (standardized_forces, new_ema_mu, new_ema_sigma)
    """
    # Current round statistics from ALL agents (order-independent)
    cur_mu = jnp.mean(forces, axis=0)            # [5]
    # Use sqrt(var + eps) instead of jnp.std for gradient stability:
    # jnp.std has NaN gradient when all values are identical (0/0 in d_std/d_x).
    cur_var = jnp.mean((forces - cur_mu[None, :]) ** 2, axis=0)  # [5]
    cur_sigma = jnp.sqrt(cur_var + 1e-8)       # [5], gradient-safe

    # EMA temporal smoothing across rounds
    new_mu = EMA_DECAY * ema_mu + (1.0 - EMA_DECAY) * cur_mu
    new_sigma = EMA_DECAY * ema_sigma + (1.0 - EMA_DECAY) * cur_sigma

    # Standardize using updated EMA stats
    std_forces = (forces - new_mu[None, :]) / (new_sigma[None, :] + 1e-6)

    return std_forces, new_mu, new_sigma


# ── Force computations ──────────────────────────────────────────

def compute_forces(
    positions: jnp.ndarray,         # [n_agents]
    original_positions: jnp.ndarray,# [n_agents]
    agent_rigidities: jnp.ndarray,  # [n_agents]
    agent_tolerances: jnp.ndarray,  # [n_agents]
    interaction_matrix: jnp.ndarray,# [n_agents, n_agents] (sparse, K nonzero/row)
    llm_shift: jnp.ndarray,        # [n_agents]
    shock_mag: jnp.ndarray,        # scalar
    shock_dir: jnp.ndarray,        # scalar
    herd_threshold: jnp.ndarray,   # scalar
) -> jnp.ndarray:
    """Compute all 5 force terms for all agents.

    Returns: [n_agents, 5] array — columns: direct, herd, anchor, social, event
    """
    # --- 1. Direct: LLM shift × susceptibility ---
    susceptibility = (1.0 - agent_rigidities) * jnp.maximum(0.0, 1.0 - jnp.abs(positions))
    f_direct = llm_shift * susceptibility  # [n_agents]

    # --- 2. Social influence (smooth bounded confidence) ---
    # interaction_matrix[i, j] = weight of j's influence on i (sparse: K nonzero per row)
    pos_diff = positions[None, :] - positions[:, None]  # [n, n] — (j - i)
    distance = jnp.abs(pos_diff)

    # Smooth bounded confidence: influence decays when distance > tolerance
    bc_weight = sigmoid((agent_tolerances[:, None] - distance) / TAU_BC)

    # Combined weight: sparse interaction graph × bounded confidence
    w = interaction_matrix * bc_weight  # [n, n] — sparse × dense = sparse-ish

    # Weighted mean pull: Σ_j w_ij * (p_j - p_i) / Σ_j w_ij
    weighted_pull = jnp.sum(w * pos_diff, axis=1)  # [n]
    weight_sum = jnp.sum(w, axis=1) + 1e-8         # [n]
    f_social = weighted_pull / weight_sum            # [n]

    # --- 3. Event shock ---
    f_event = shock_mag * shock_dir * (1.0 - agent_rigidities)  # [n]

    # --- 4. Herd effect (smooth threshold) ---
    # Feed average = weighted mean of neighbors' positions (same sparse graph)
    feed_weights = interaction_matrix / (jnp.sum(interaction_matrix, axis=1, keepdims=True) + 1e-8)
    feed_avg = jnp.sum(feed_weights * positions[None, :], axis=1)  # [n]
    gap = feed_avg - positions  # [n]

    # Smooth herd activation: σ((|gap| - threshold) / τ)
    herd_activation = smooth_step(jnp.abs(gap), herd_threshold, TAU_HERD)
    f_herd = gap * (1.0 - agent_rigidities) * herd_activation  # [n]

    # --- 5. Anchor pull ---
    f_anchor = agent_rigidities * (original_positions - positions)  # [n]

    # Stack: [n_agents, 5] — order matches FORCE_NAMES
    return jnp.stack([f_direct, f_herd, f_anchor, f_social, f_event], axis=-1)


# ── Single round step ───────────────────────────────────────────

def step_round(
    positions: jnp.ndarray,          # [n_agents]
    original_positions: jnp.ndarray, # [n_agents]
    ema_mu: jnp.ndarray,             # [5] EMA running mean
    ema_sigma: jnp.ndarray,          # [5] EMA running std
    params: dict,
    scenario: ScenarioData,
    round_idx: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one round. Returns (new_positions, new_original, new_ema_mu, new_ema_sigma)."""

    # Extract constrained params
    lambda_elite = jnp.exp(params["log_lambda_elite"])
    lambda_citizen = jnp.exp(params["log_lambda_citizen"])
    herd_threshold = sigmoid(params["logit_herd_threshold"])
    anchor_drift = sigmoid(params["logit_anchor_drift"])

    # Per-agent step size: elite (type=0) or citizen (type=1)
    lambda_per_agent = jnp.where(
        scenario.agent_types == 0, lambda_elite, lambda_citizen
    )

    # Delta caps: elite=0.15, citizen=0.25
    cap_per_agent = jnp.where(scenario.agent_types == 0, 0.15, 0.25)

    # Event data for this round
    shock_mag = scenario.events[round_idx, 0]
    shock_dir = scenario.events[round_idx, 1]
    llm_shift = scenario.llm_shifts[round_idx]

    # Compute raw forces [n_agents, 5]
    raw_forces = compute_forces(
        positions, original_positions,
        scenario.agent_rigidities, scenario.agent_tolerances,
        scenario.interaction_matrix, llm_shift,
        shock_mag, shock_dir, herd_threshold,
    )

    # EMA standardization (differentiable, stateful)
    std_forces, new_ema_mu, new_ema_sigma = ema_standardize(
        raw_forces, ema_mu, ema_sigma,
    )

    # Softmax mixing weights: α_direct=0 (gauge-fixed), then herd, anchor, social, event
    alpha_vec = jnp.array([
        0.0,  # direct — gauge-fixed reference
        params["alpha_herd"],
        params["alpha_anchor"],
        params["alpha_social"],
        params["alpha_event"],
    ])
    pi = jax.nn.softmax(alpha_vec)  # [5]

    # Weighted combination: Σ π_k · f̃_k  →  [n_agents]
    combined = jnp.sum(pi[None, :] * std_forces, axis=-1)

    # Scale by per-agent step size
    delta_p_raw = lambda_per_agent * combined

    # Smooth clamp
    delta_p = smooth_clamp(delta_p_raw, cap_per_agent)

    # Update positions, clamp to [-1, 1]
    new_positions = jnp.clip(positions + delta_p, -1.0, 1.0)

    # Anchor drift
    new_original = original_positions + anchor_drift * (new_positions - original_positions)

    return new_positions, new_original, new_ema_mu, new_ema_sigma


# ── Scan-compatible step ────────────────────────────────────────

def _scan_body(carry, round_idx):
    """Body function for jax.lax.scan over rounds."""
    positions, original_positions, ema_mu, ema_sigma, params, scenario = carry
    new_pos, new_orig, new_mu, new_sig = step_round(
        positions, original_positions, ema_mu, ema_sigma,
        params, scenario, round_idx,
    )
    return (new_pos, new_orig, new_mu, new_sig, params, scenario), new_pos


# ── Main entry point ────────────────────────────────────────────

def simulate_scenario(params: dict, scenario_data: ScenarioData) -> dict:
    """Run full scenario simulation.

    Args:
        params: Dict with all 8 parameters (calibrable + frozen).
        scenario_data: ScenarioData NamedTuple (n_rounds inferred from events shape).

    Returns:
        {
            'trajectories': array[n_rounds, n_agents],
            'pro_fraction': array[n_rounds],
            'mix_weights': array[5],
            'final_pro_pct': float,
        }
    """
    init_pos = scenario_data.initial_positions
    init_orig = init_pos  # original_position starts equal to position

    # EMA initial state: μ=0, σ=1 (uninformative prior)
    init_ema_mu = jnp.zeros(N_FORCES)
    init_ema_sigma = jnp.ones(N_FORCES)

    # n_rounds from events array shape — static at JIT compile time
    n_rounds = scenario_data.events.shape[0]

    carry = (init_pos, init_orig, init_ema_mu, init_ema_sigma, params, scenario_data)
    round_indices = jnp.arange(n_rounds)

    _, trajectories = jax.lax.scan(_scan_body, carry, round_indices)
    # trajectories: [n_rounds, n_agents]

    # Pro fraction per round — smooth sigmoid for differentiability
    tau_pro = 0.02  # steepness
    soft_pro = sigmoid((trajectories - 0.05) / tau_pro)        # [n_rounds, n_agents]
    soft_against = sigmoid((-trajectories - 0.05) / tau_pro)   # [n_rounds, n_agents]
    soft_decided = soft_pro + soft_against                      # [n_rounds, n_agents]
    decided_count = jnp.sum(soft_decided, axis=1) + 1e-8       # [n_rounds]
    pro_fraction = jnp.sum(soft_pro, axis=1) / decided_count   # [n_rounds]

    # Mix weights
    alpha_vec = jnp.array([
        0.0, params["alpha_herd"], params["alpha_anchor"],
        params["alpha_social"], params["alpha_event"],
    ])
    mix_weights = jax.nn.softmax(alpha_vec)

    final_pro_pct = pro_fraction[-1] * 100.0

    return {
        "trajectories": trajectories,
        "pro_fraction": pro_fraction,
        "mix_weights": mix_weights,
        "final_pro_pct": final_pro_pct,
    }


# ── Loss helper (for grad / optimization) ───────────────────────

# Jitted version for inference (not for grad — use simulate_scenario directly)
simulate_scenario_jit = jax.jit(simulate_scenario)


def scenario_loss(params: dict, scenario_data: ScenarioData, target_pro_pct: float) -> jnp.ndarray:
    """MSE loss between simulated and target pro percentage. Scalar output for jax.grad."""
    result = simulate_scenario(params, scenario_data)
    return (result["final_pro_pct"] - target_pro_pct) ** 2
