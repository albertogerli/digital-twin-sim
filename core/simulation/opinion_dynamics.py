"""Mathematical opinion shift model — bounded confidence with social influence.
This module is entirely domain-agnostic."""

import logging
import math
from statistics import mean

logger = logging.getLogger(__name__)


class OpinionDynamics:
    """Implements the bounded confidence model with social influence."""

    def __init__(self, anchor_weight=0.1, social_weight=0.15,
                 event_weight=0.05, herd_weight=0.05, herd_threshold=0.2,
                 direct_shift_weight=0.4, anchor_drift_rate=0.2,
                 calibrated_params_path: str = ""):
        # Load calibrated params if available, otherwise use defaults
        if calibrated_params_path:
            from calibration.parameter_tuner import load_calibrated_params
            params = load_calibrated_params(calibrated_params_path)
            if params:
                logger.info(f"Loaded calibrated params from {calibrated_params_path}")
                anchor_weight = params.get("anchor_weight", anchor_weight)
                social_weight = params.get("social_weight", social_weight)
                event_weight = params.get("event_weight", event_weight)
                herd_weight = params.get("herd_weight", herd_weight)
                herd_threshold = params.get("herd_threshold", herd_threshold)
                direct_shift_weight = params.get("direct_shift_weight", direct_shift_weight)
                anchor_drift_rate = params.get("anchor_drift_rate", anchor_drift_rate)

        self.anchor_weight = anchor_weight
        self.social_weight = social_weight
        self.event_weight = event_weight
        self.herd_weight = herd_weight
        self.herd_threshold = herd_threshold
        self.direct_shift_weight = direct_shift_weight
        self.anchor_drift_rate = anchor_drift_rate

    def update_position(
        self,
        agent_position: float,
        agent_original_position: float,
        agent_rigidity: float,
        agent_tolerance: float,
        feed_authors_positions: list[tuple[float, float, float]],
        event_shock_magnitude: float,
        event_shock_direction: float,
    ) -> float:
        """Update agent position based on multiple forces.

        Args:
            agent_position: Current position (-1 to +1)
            agent_original_position: Starting position
            agent_rigidity: How resistant to change (0-1)
            agent_tolerance: Bounded confidence threshold
            feed_authors_positions: List of (author_position, author_influence, post_engagement)
            event_shock_magnitude: How big the event is (0-1)
            event_shock_direction: Direction of event (-1 to +1)

        Returns:
            New position clamped to [-1, +1]
        """
        # 1. Self-anchoring: rigidity pulls toward original position
        anchor_pull = (
            agent_rigidity
            * (agent_original_position - agent_position)
            * self.anchor_weight
        )

        # 2. Social influence from consumed content (bounded confidence)
        influence_sum = 0.0
        influence_count = 0.0
        for author_pos, author_inf, post_engagement in feed_authors_positions:
            distance = abs(author_pos - agent_position)
            if distance < agent_tolerance:
                weight = author_inf * max(0.1, post_engagement)
                influence_sum += weight * (author_pos - agent_position)
                influence_count += weight

        social_pull = 0.0
        if influence_count > 0:
            social_pull = (influence_sum / influence_count) * self.social_weight

        # 3. Event shock
        event_shock = (
            event_shock_magnitude
            * event_shock_direction
            * (1 - agent_rigidity)
            * self.event_weight
        )

        # 4. Herd effect
        herd_pull = 0.0
        if feed_authors_positions:
            feed_avg = mean(pos for pos, _, _ in feed_authors_positions)
            if abs(feed_avg - agent_position) > self.herd_threshold:
                herd_pull = (
                    (feed_avg - agent_position)
                    * self.herd_weight
                    * (1 - agent_rigidity)
                )

        # Combine and clamp
        delta = anchor_pull + social_pull + event_shock + herd_pull
        new_position = agent_position + delta
        return max(-1.0, min(1.0, new_position))

    def update_all_agents(self, agents: list, platform_engine, event: dict):
        """Update positions for all agents based on their feeds and the round event.

        Three-phase update (aligned with calibration model):
        1. Direct event shift on susceptible agents (near center, low rigidity)
        2. Social influence via bounded confidence model
        3. Anchor drift (original_position moves toward current, enabling reversals)
        """
        from ..platform.feed_algorithm import FeedAlgorithm

        feed_algo = FeedAlgorithm(platform_engine)
        round_num = event.get("round", 0)
        shock_mag = event.get("shock_magnitude", 0.3)
        shock_dir = event.get("shock_direction", 0.0)

        # Phase 1: Direct event shift on susceptible agents
        if shock_mag > 0.15 and abs(shock_dir) > 0.05:
            for agent in agents:
                susceptibility = (1 - agent.rigidity) * max(0, 1 - abs(agent.position))
                shift = shock_mag * shock_dir * susceptibility * self.direct_shift_weight
                agent.position = max(-1.0, min(1.0, agent.position + shift))

        # Phase 2: Social influence via feeds
        agent_positions = {a.id: a.position for a in agents}

        for agent in agents:
            if agent.tier <= 2:
                feed_posts = feed_algo.get_feed(agent.id, round_num, feed_size=10)
            else:
                feed_posts = platform_engine.get_top_posts(round_num, top_n=5)

            feed_data = []
            for fp in feed_posts:
                author_id = fp.get("author_id", "")
                author_pos = agent_positions.get(author_id, 0.0)
                author_inf = 0.5
                for a in agents:
                    if a.id == author_id:
                        author_inf = a.influence
                        break
                engagement = (
                    fp.get("likes", 0) + fp.get("reposts", 0) * 2
                ) / 100.0
                feed_data.append((author_pos, author_inf, min(1.0, engagement)))

            if not feed_data:
                continue

            new_pos = self.update_position(
                agent_position=agent.position,
                agent_original_position=agent.original_position,
                agent_rigidity=agent.rigidity,
                agent_tolerance=getattr(agent, "tolerance", 0.4),
                feed_authors_positions=feed_data,
                event_shock_magnitude=shock_mag,
                event_shock_direction=shock_dir,
            )

            agent.position = new_pos

        # Phase 3: Anchor drift — enables opinion reversals over time
        if self.anchor_drift_rate > 0:
            for agent in agents:
                agent.original_position += self.anchor_drift_rate * (
                    agent.position - agent.original_position
                )
