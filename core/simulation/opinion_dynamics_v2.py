"""Opinion Dynamics v2 — reparametrized model with standardized forces,
softmax mixing weights, and per-tier step sizes.

Key changes from v1:
  - Force terms are standardized (zero mean, unit variance) via rolling buffer
  - Mixing weights π = softmax(α) ensure interpretable relative importance
  - Step size λ is separated per agent tier (elite / institutional / citizen)
  - Single final clamp on Δp instead of per-term clamping
"""

import logging
import math
from collections import deque
from statistics import mean

import numpy as np

logger = logging.getLogger(__name__)

# --- Defaults ---

DEFAULT_LOGITS = {
    "direct": 1.0,
    "social": 0.0,
    "event": -0.2,
    "herd": -0.5,
    "anchor": -0.3,
}

DEFAULT_STEP_SIZES = {
    "elite": 0.12,
    "institutional": 0.10,
    "citizen": 0.20,
}

TIER_MAP = {1: "elite", 2: "institutional", 3: "citizen"}

BUFFER_SIZE = 8  # rolling window for standardization


def softmax(logits: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a named dict of logits."""
    keys = list(logits.keys())
    vals = np.array([logits[k] for k in keys], dtype=np.float64)
    vals -= vals.max()  # numerical stability
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    return {k: float(exp_vals[i] / total) for i, k in enumerate(keys)}


class ForceStandardizer:
    """Maintains rolling mean/std for each force term and standardizes them."""

    def __init__(self, force_names: list[str], buffer_size: int = BUFFER_SIZE):
        self.buffers: dict[str, deque] = {
            name: deque(maxlen=buffer_size) for name in force_names
        }

    def observe(self, raw_forces: dict[str, list[float]]):
        """Record raw force values from a full step (all agents)."""
        for name, values in raw_forces.items():
            if name in self.buffers and values:
                self.buffers[name].extend(values)

    def standardize(self, raw: dict[str, float]) -> dict[str, float]:
        """Standardize a single agent's raw forces using rolling stats.

        Returns z-scored values. Falls back to raw / max(|raw|, 1) if
        the buffer has fewer than 2 observations.
        """
        result = {}
        for name, value in raw.items():
            buf = self.buffers.get(name)
            if buf is None or len(buf) < 2:
                # Not enough data — use simple normalization
                result[name] = value
                continue
            arr = np.array(buf, dtype=np.float64)
            mu = arr.mean()
            sigma = arr.std()
            if sigma < 1e-10:
                result[name] = 0.0
            else:
                result[name] = (value - mu) / sigma
        return result

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return current mean/std for each force (for diagnostics)."""
        stats = {}
        for name, buf in self.buffers.items():
            if len(buf) < 2:
                stats[name] = {"mean": 0.0, "std": 1.0, "n": len(buf)}
            else:
                arr = np.array(buf, dtype=np.float64)
                stats[name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n": len(buf),
                }
        return stats


class DynamicsV2:
    """Reparametrized opinion dynamics with softmax mixing and per-tier step sizes.

    Parameters
    ----------
    alpha : dict
        Raw logits for each force. π = softmax(α) gives the mixing weights.
    step_sizes : dict
        Per-tier step sizes λ. Controls amplitude of position updates.
    herd_threshold : float
        Activation threshold for herd effect.
    anchor_drift_rate : float
        Rate at which original_position drifts toward current position.
    buffer_size : int
        Rolling window for force standardization.
    """

    FORCE_NAMES = ["direct", "social", "event", "herd", "anchor"]

    def __init__(
        self,
        alpha: dict[str, float] | None = None,
        step_sizes: dict[str, float] | None = None,
        herd_threshold: float = 0.21,
        anchor_drift_rate: float = 0.25,
        buffer_size: int = BUFFER_SIZE,
    ):
        self.alpha = dict(alpha) if alpha else dict(DEFAULT_LOGITS)
        self.step_sizes = dict(step_sizes) if step_sizes else dict(DEFAULT_STEP_SIZES)
        self.herd_threshold = herd_threshold
        self.anchor_drift_rate = anchor_drift_rate
        self.standardizer = ForceStandardizer(self.FORCE_NAMES, buffer_size)

        # Delta caps per tier (hard safety clamp after λ scaling)
        self.delta_caps = {"elite": 0.15, "institutional": 0.12, "citizen": 0.25}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_mix_weights(self) -> dict[str, float]:
        """Return the interpretable mixing weights π = softmax(α)."""
        return softmax(self.alpha)

    def step(
        self,
        agents: list,
        platform_engine,
        event: dict,
    ) -> dict[str, float]:
        """Run one round of opinion dynamics on all agents.

        Returns a dict mapping agent_id → new_position.
        """
        from ..platform.feed_algorithm import FeedAlgorithm

        feed_algo = FeedAlgorithm(platform_engine)
        round_num = event.get("round", 0)
        shock_mag = event.get("shock_magnitude", 0.3)
        shock_dir = event.get("shock_direction", 0.0)

        pi = softmax(self.alpha)
        agent_positions = {a.id: a.position for a in agents}
        all_positions = [a.position for a in agents]
        global_mean_pos = mean(all_positions) if all_positions else 0.0

        # Collect raw forces across all agents (for standardizer buffer)
        all_raw_forces: dict[str, list[float]] = {k: [] for k in self.FORCE_NAMES}

        # --- Pass 1: compute raw forces for each agent ---
        agent_raw: list[tuple] = []  # (agent, raw_forces_dict, feed_data)

        for agent in agents:
            # Get feed — use FeedAlgorithm if platform has DB connection,
            # otherwise fall back to top posts (for testing / simple platforms)
            if agent.tier <= 2 and hasattr(platform_engine, 'conn') and platform_engine.conn is not None:
                feed_posts = feed_algo.get_feed(agent.id, round_num, feed_size=10)
            else:
                feed_posts = platform_engine.get_top_posts(round_num, top_n=5)

            feed_data = self._build_feed_data(feed_posts, agents, agent_positions)

            raw = self._compute_raw_forces(
                agent, feed_data, shock_mag, shock_dir, global_mean_pos,
            )

            for k in self.FORCE_NAMES:
                all_raw_forces[k].append(raw[k])

            agent_raw.append((agent, raw, feed_data))

        # Update standardizer with this round's observations
        self.standardizer.observe(all_raw_forces)

        # --- Pass 2: standardize, mix, scale, clamp, apply ---
        updated = {}
        for agent, raw, _feed_data in agent_raw:
            std_forces = self.standardizer.standardize(raw)

            # Weighted combination: Σ π_k · f̃_k
            combined = sum(pi[k] * std_forces[k] for k in self.FORCE_NAMES)

            # Scale by per-tier step size
            tier_name = TIER_MAP.get(getattr(agent, "tier", 1), "citizen")
            lam = self.step_sizes.get(tier_name, 0.15)
            delta_p = lam * combined

            # Single final clamp
            cap = self.delta_caps.get(tier_name, 0.15)
            delta_p = max(-cap, min(cap, delta_p))

            new_pos = max(-1.0, min(1.0, agent.position + delta_p))
            agent.position = new_pos
            updated[agent.id] = new_pos

        # --- Anchor drift (same as v1) ---
        if self.anchor_drift_rate > 0:
            for agent in agents:
                agent.original_position += self.anchor_drift_rate * (
                    agent.position - agent.original_position
                )

        return updated

    # ------------------------------------------------------------------ #
    #  V1 compatibility — drop-in replacement for update_all_agents       #
    # ------------------------------------------------------------------ #

    def update_all_agents(self, agents: list, platform_engine, event: dict):
        """V1-compatible interface. Calls step() internally."""
        self.step(agents, platform_engine, event)

    # ------------------------------------------------------------------ #
    #  Conversion from/to v1 parameters                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_v1_params(
        cls,
        anchor_weight: float = 0.1,
        social_weight: float = 0.15,
        event_weight: float = 0.05,
        herd_weight: float = 0.05,
        direct_shift_weight: float = 0.4,
        herd_threshold: float = 0.2,
        anchor_drift_rate: float = 0.2,
    ) -> "DynamicsV2":
        """Approximate conversion from v1 weights to v2 logits + step sizes.

        Since v1 weights combine both amplitude and composition, we factor them:
          - step_size ≈ sum of all weights (total amplitude)
          - α_k ≈ log(w_k / Σw) (logit of weight share)
        """
        weights = {
            "direct": direct_shift_weight,
            "social": social_weight,
            "event": event_weight,
            "herd": herd_weight,
            "anchor": anchor_weight,
        }
        total = sum(weights.values()) or 1.0

        # Convert normalized weights to logits: α = log(w/Σw)
        # Add small epsilon to avoid log(0)
        alpha = {
            k: float(np.log(max(v / total, 1e-8))) for k, v in weights.items()
        }

        # Step sizes: use total weight scaled per tier
        step_sizes = {
            "elite": total * 0.16,
            "institutional": total * 0.13,
            "citizen": total * 0.27,
        }

        return cls(
            alpha=alpha,
            step_sizes=step_sizes,
            herd_threshold=herd_threshold,
            anchor_drift_rate=anchor_drift_rate,
        )

    def to_v1_params(self) -> dict:
        """Approximate back-conversion to v1 parameter space."""
        pi = self.get_mix_weights()
        avg_lambda = mean(self.step_sizes.values())
        return {
            "direct_shift_weight": pi["direct"] * avg_lambda,
            "social_weight": pi["social"] * avg_lambda,
            "event_weight": pi["event"] * avg_lambda,
            "herd_weight": pi["herd"] * avg_lambda,
            "anchor_weight": pi["anchor"] * avg_lambda,
            "herd_threshold": self.herd_threshold,
            "anchor_drift_rate": self.anchor_drift_rate,
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _build_feed_data(
        self,
        feed_posts: list[dict],
        agents: list,
        agent_positions: dict[str, float],
    ) -> list[tuple[float, float, float]]:
        """Build (author_pos, author_influence, engagement) tuples from feed."""
        agent_map = {a.id: a for a in agents}
        feed_data = []
        for fp in feed_posts:
            author_id = fp.get("author_id", "")
            author_pos = agent_positions.get(author_id, 0.0)
            author = agent_map.get(author_id)
            author_inf = author.influence if author else 0.5
            engagement = (
                fp.get("likes", 0) + fp.get("reposts", 0) * 2
            ) / 100.0
            feed_data.append((author_pos, author_inf, min(1.0, engagement)))
        return feed_data

    def _compute_raw_forces(
        self,
        agent,
        feed_data: list[tuple[float, float, float]],
        shock_mag: float,
        shock_dir: float,
        global_mean_pos: float,
    ) -> dict[str, float]:
        """Compute raw (un-standardized) force terms for one agent.

        Each force is computed to have roughly comparable magnitude before
        standardization — the standardizer will handle residual scale differences.
        """
        rigidity = getattr(agent, "rigidity", 0.5)
        tolerance = getattr(agent, "tolerance", 0.4)

        # 1. Direct: event shock on susceptible agents
        susceptibility = (1 - rigidity) * max(0, 1 - abs(agent.position))
        f_direct = shock_mag * shock_dir * susceptibility

        # 2. Social influence (bounded confidence)
        f_social = 0.0
        if feed_data:
            influence_sum = 0.0
            influence_weight = 0.0
            for author_pos, author_inf, engagement in feed_data:
                distance = abs(author_pos - agent.position)
                if distance < tolerance:
                    w = author_inf * max(0.1, engagement)
                    influence_sum += w * (author_pos - agent.position)
                    influence_weight += w
            if influence_weight > 0:
                f_social = influence_sum / influence_weight

        # 3. Event shock (general push)
        f_event = shock_mag * shock_dir * (1 - rigidity)

        # 4. Herd effect
        f_herd = 0.0
        if feed_data:
            feed_avg = mean(pos for pos, _, _ in feed_data)
            gap = feed_avg - agent.position
            if abs(gap) > self.herd_threshold:
                f_herd = gap * (1 - rigidity)

        # 5. Anchor pull
        f_anchor = rigidity * (agent.original_position - agent.position)

        return {
            "direct": f_direct,
            "social": f_social,
            "event": f_event,
            "herd": f_herd,
            "anchor": f_anchor,
        }
