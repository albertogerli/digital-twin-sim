"""Escalation Engine — dynamic agent activation across rounds.

Monitors simulation engagement metrics and decides when to escalate:
  - Round 1: Only wave_1 agents active (local press, unions, mayors)
  - If engagement stays low → crisis contained, no escalation
  - If engagement spikes → activate wave_2, then wave_3
  - If engagement explodes → spawn reserve agents mid-simulation

The engine computes an `engagement_score` (0-1) each round from:
  - Post volume vs. baseline
  - Reaction amplification (shares/likes ratio)
  - Sentiment volatility
  - Polarization velocity (how fast polarization is changing)
  - Narrative convergence (are agents coalescing around a single frame?)

This score feeds into ActivationPlan.agents_for_round() to determine
which agents participate in each round.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from core.orchestrator.retriever import ActivationPlan, RelevanceScore

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Engagement metrics snapshot for a single round."""
    round_num: int
    post_count: int = 0
    reaction_count: int = 0
    repost_ratio: float = 0.0         # reposts / posts
    avg_engagement_per_post: float = 0.0
    polarization: float = 0.0
    sentiment_negative_pct: float = 0.0
    top_post_engagement: int = 0
    unique_authors: int = 0
    shock_magnitude: float = 0.0

    @property
    def amplification(self) -> float:
        """How much content is being amplified beyond creation."""
        if self.post_count == 0:
            return 0.0
        return self.reaction_count / self.post_count


@dataclass
class EscalationState:
    """Tracks escalation decisions across rounds."""
    current_wave: int = 1              # Highest wave currently active
    engagement_scores: list[float] = field(default_factory=list)
    round_metrics: list[RoundMetrics] = field(default_factory=list)
    spawned_agents: list[str] = field(default_factory=list)  # IDs of dynamically spawned agents
    escalation_events: list[dict] = field(default_factory=list)  # Log of escalation decisions

    @property
    def latest_engagement(self) -> float:
        return self.engagement_scores[-1] if self.engagement_scores else 0.0

    @property
    def engagement_trend(self) -> float:
        """Positive = accelerating, negative = cooling down."""
        if len(self.engagement_scores) < 2:
            return 0.0
        return self.engagement_scores[-1] - self.engagement_scores[-2]

    def to_dict(self) -> dict:
        return {
            "current_wave": self.current_wave,
            "engagement_scores": [round(s, 3) for s in self.engagement_scores],
            "engagement_trend": round(self.engagement_trend, 3),
            "spawned_agents": self.spawned_agents,
            "escalation_events": self.escalation_events,
        }


class EscalationEngine:
    """Decides when and how to escalate agent activation.

    Computes a composite engagement_score each round and uses it to:
    1. Tell the ActivationPlan which waves should be active
    2. Trigger mid-simulation agent spawning from reserve pool
    3. Log escalation events for the contagion scorer

    Thresholds are calibrated so that:
    - A local factory closing with low media echo stays at wave 1
    - A trending hashtag + union reaction triggers wave 2
    - PM involvement threshold is only reached if multiple indicators spike
    """

    # ── Thresholds ──────────────────────────────────────────────────────
    WAVE_2_THRESHOLD = 0.45    # engagement_score to activate national media
    WAVE_3_THRESHOLD = 0.70    # engagement_score to activate heads of state
    RESERVE_THRESHOLD = 0.88   # engagement_score to spawn reserve agents
    COOLDOWN_RATE = 0.15       # How fast engagement decays when nothing happens

    # ── Weights for engagement_score composition ────────────────────────
    W_VOLUME = 0.15            # Post volume vs. baseline
    W_AMPLIFICATION = 0.25     # Reaction amplification ratio
    W_POLARIZATION = 0.20      # Absolute polarization level
    W_POLAR_VELOCITY = 0.15    # Change in polarization from previous round
    W_SENTIMENT = 0.10         # Negative sentiment share
    W_SHOCK = 0.15             # Event shock magnitude

    # Baseline expectations (calibrated from historical simulations)
    BASELINE_POSTS_PER_ROUND = 25
    BASELINE_REACTIONS_PER_POST = 3.0

    def __init__(self, activation_plan: ActivationPlan):
        self.plan = activation_plan
        self.state = EscalationState()
        self._baseline_posts = self.BASELINE_POSTS_PER_ROUND

    def compute_engagement_score(self, metrics: RoundMetrics) -> float:
        """Compute a 0-1 engagement score from round metrics.

        Each component is normalized to [0, 1] then weighted.
        """
        # Volume: how many posts relative to baseline
        volume_ratio = metrics.post_count / max(self._baseline_posts, 1)
        volume_score = min(1.0, volume_ratio / 3.0)  # 3x baseline = 1.0

        # Amplification: reactions per post vs. baseline
        amp_ratio = metrics.amplification / max(self.BASELINE_REACTIONS_PER_POST, 1)
        amp_score = min(1.0, amp_ratio / 4.0)  # 4x baseline = 1.0

        # Polarization: 0-10 scale → 0-1
        polar_score = min(1.0, metrics.polarization / 8.0)  # 8/10 = max

        # Polarization velocity: compare with previous round
        polar_velocity = 0.0
        if self.state.round_metrics:
            prev_polar = self.state.round_metrics[-1].polarization
            delta = metrics.polarization - prev_polar
            polar_velocity = min(1.0, max(0.0, delta / 3.0))  # +3 points = 1.0

        # Negative sentiment share → higher = more crisis
        sentiment_score = min(1.0, metrics.sentiment_negative_pct / 0.6)  # 60% negative = 1.0

        # Shock magnitude (from event injector)
        shock_score = min(1.0, metrics.shock_magnitude)

        # Weighted combination
        raw = (
            self.W_VOLUME * volume_score +
            self.W_AMPLIFICATION * amp_score +
            self.W_POLARIZATION * polar_score +
            self.W_POLAR_VELOCITY * polar_velocity +
            self.W_SENTIMENT * sentiment_score +
            self.W_SHOCK * shock_score
        )

        # Apply momentum: if trend is rising, boost slightly
        if len(self.state.engagement_scores) >= 1:
            prev = self.state.engagement_scores[-1]
            if raw > prev:
                # Rising momentum bonus (max +0.1)
                raw = raw + 0.1 * (raw - prev)
            else:
                # Cooling: apply partial decay toward raw
                raw = raw * (1 - self.COOLDOWN_RATE) + prev * self.COOLDOWN_RATE

        return max(0.0, min(1.0, raw))

    def process_round(
        self,
        round_num: int,
        post_count: int,
        reaction_count: int,
        polarization: float,
        sentiment_pcts: dict[str, float],
        shock_magnitude: float = 0.0,
        top_post_engagement: int = 0,
    ) -> dict:
        """Process a completed round and decide on escalation.

        Call this AFTER each round completes but BEFORE the next round starts.

        Returns:
            dict with:
                - engagement_score: float 0-1
                - active_wave: int (1, 2, or 3)
                - escalated: bool (did we just escalate?)
                - spawn_reserve: bool (should we spawn reserve agents?)
                - agents_for_next_round: list[RelevanceScore]
        """
        # Build metrics
        metrics = RoundMetrics(
            round_num=round_num,
            post_count=post_count,
            reaction_count=reaction_count,
            repost_ratio=reaction_count / max(post_count, 1),
            avg_engagement_per_post=reaction_count / max(post_count, 1),
            polarization=polarization,
            sentiment_negative_pct=sentiment_pcts.get("negative", 0.0),
            top_post_engagement=top_post_engagement,
            shock_magnitude=shock_magnitude,
        )
        self.state.round_metrics.append(metrics)

        # Compute engagement
        score = self.compute_engagement_score(metrics)
        self.state.engagement_scores.append(score)

        # Determine wave
        prev_wave = self.state.current_wave
        if score >= self.WAVE_3_THRESHOLD:
            self.state.current_wave = 3
        elif score >= self.WAVE_2_THRESHOLD:
            self.state.current_wave = max(self.state.current_wave, 2)

        # Never de-escalate (once activated, waves stay active)
        self.state.current_wave = max(self.state.current_wave, prev_wave)

        escalated = self.state.current_wave > prev_wave
        spawn_reserve = score >= self.RESERVE_THRESHOLD and not self.state.spawned_agents

        # Log escalation events
        if escalated:
            event = {
                "round": round_num,
                "from_wave": prev_wave,
                "to_wave": self.state.current_wave,
                "engagement_score": round(score, 3),
                "trigger": self._escalation_trigger(metrics, score),
            }
            self.state.escalation_events.append(event)
            logger.info(
                f"ESCALATION: wave {prev_wave}→{self.state.current_wave} "
                f"(engagement={score:.3f}, round={round_num})"
            )

        if spawn_reserve:
            reserve_ids = [r.stakeholder_id for r in self.plan.reserve]
            self.state.spawned_agents.extend(reserve_ids)
            self.state.escalation_events.append({
                "round": round_num,
                "type": "reserve_spawn",
                "engagement_score": round(score, 3),
                "agents_spawned": reserve_ids,
            })
            logger.info(
                f"RESERVE SPAWN: {len(reserve_ids)} agents activated "
                f"(engagement={score:.3f})"
            )

        # Get agents for next round
        next_round_agents = self.plan.agents_for_round(
            round_num + 1, score
        )

        result = {
            "engagement_score": round(score, 3),
            "active_wave": self.state.current_wave,
            "escalated": escalated,
            "spawn_reserve": spawn_reserve,
            "agents_for_next_round": next_round_agents,
            "trend": round(self.state.engagement_trend, 3),
        }

        logger.info(
            f"Round {round_num}: engagement={score:.3f}, "
            f"wave={self.state.current_wave}, trend={self.state.engagement_trend:+.3f}"
        )
        return result

    def get_active_agents(self, round_num: int) -> list[RelevanceScore]:
        """Get agents that should be active for the given round."""
        score = self.state.latest_engagement
        return self.plan.agents_for_round(round_num, score)

    def should_spawn_agent(self, stakeholder_id: str) -> bool:
        """Check if a specific stakeholder should be dynamically spawned."""
        return stakeholder_id in self.state.spawned_agents

    def _escalation_trigger(self, metrics: RoundMetrics, score: float) -> str:
        """Identify the primary trigger for an escalation event."""
        triggers = []
        if metrics.amplification > self.BASELINE_REACTIONS_PER_POST * 3:
            triggers.append(f"amplification_spike({metrics.amplification:.1f}x)")
        if metrics.polarization > 6.0:
            triggers.append(f"high_polarization({metrics.polarization:.1f})")
        if metrics.sentiment_negative_pct > 0.5:
            triggers.append(f"negative_sentiment({metrics.sentiment_negative_pct:.0%})")
        if metrics.shock_magnitude > 0.6:
            triggers.append(f"shock({metrics.shock_magnitude:.2f})")
        if self.state.engagement_trend > 0.15:
            triggers.append(f"rising_trend(+{self.state.engagement_trend:.2f})")
        return "; ".join(triggers) if triggers else f"composite_score({score:.3f})"

    def predict_next_escalation(self) -> Optional[dict]:
        """Predict when the next escalation might happen.

        Uses engagement trend to estimate rounds until threshold breach.
        Returns None if crisis is cooling down.
        """
        if self.state.current_wave >= 3 and self.state.spawned_agents:
            return None  # Already fully escalated

        trend = self.state.engagement_trend
        current = self.state.latest_engagement

        if trend <= 0:
            return {"prediction": "cooling", "message": "Crisis is de-escalating"}

        # Next threshold
        if self.state.current_wave == 1:
            target = self.WAVE_2_THRESHOLD
            label = "national_media"
        elif self.state.current_wave == 2:
            target = self.WAVE_3_THRESHOLD
            label = "heads_of_state"
        else:
            target = self.RESERVE_THRESHOLD
            label = "reserve_spawn"

        gap = target - current
        if gap <= 0:
            return {"prediction": "imminent", "target": label, "rounds_until": 0}

        # Linear extrapolation (conservative)
        rounds_est = math.ceil(gap / trend)
        return {
            "prediction": "escalating",
            "target": label,
            "rounds_until": rounds_est,
            "current_engagement": round(current, 3),
            "trend": round(trend, 3),
        }

    def get_state_summary(self) -> dict:
        """Return a summary for logging/dashboard."""
        return {
            **self.state.to_dict(),
            "prediction": self.predict_next_escalation(),
        }
