"""Position Updater — conservative EMA-based position updates.

The core algorithm that prevents hallucinated flips from single articles.
Uses exponential moving average with drift clamping and rigidity dampening.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from stakeholder_graph.schema import Position, Stakeholder
from stakeholder_graph.updater.analysis.signal import (
    ArticleAnalysis,
    InfluenceSignal,
    PositionSignal,
    QuoteSignal,
)
from stakeholder_graph.updater.config import UpdaterConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionDelta:
    """A single position change to apply."""
    stakeholder_id: str
    topic_tag: str
    old_value: float
    new_value: float
    n_signals: int
    evidence: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    is_new_topic: bool = False
    new_confidence: Optional[str] = None

    @property
    def delta(self) -> float:
        return self.new_value - self.old_value


@dataclass
class InfluenceDelta:
    """An influence score change to apply."""
    stakeholder_id: str
    old_value: float
    new_value: float
    reason: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass
class QuoteDelta:
    """New quotes to append to a position."""
    stakeholder_id: str
    topic_tag: str
    quotes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class StakeholderUpdate:
    """All updates for a single stakeholder in one run."""
    stakeholder_id: str
    position_deltas: list[PositionDelta] = field(default_factory=list)
    influence_delta: Optional[InfluenceDelta] = None
    quote_deltas: list[QuoteDelta] = field(default_factory=list)
    new_last_updated: str = ""

    @property
    def has_changes(self) -> bool:
        return bool(self.position_deltas) or self.influence_delta is not None or bool(self.quote_deltas)


class PositionUpdater:
    """Computes conservative position updates from extracted signals."""

    def __init__(self, config: UpdaterConfig):
        self.alpha = config.ema_alpha
        self.max_drift = config.max_drift_per_run
        self.strength_weights = config.strength_weights
        self.rigidity_dampening = config.rigidity_dampening
        self.confidence_thresholds = config.confidence_upgrade_signals

    def compute_position_delta(
        self,
        current: Position,
        signals: list[PositionSignal],
        rigidity: float = 0.5,
    ) -> Optional[PositionDelta]:
        """Compute a single position update from signals.

        Algorithm:
        1. Weight signals by strength (strong=0.3, moderate=0.15, weak=0.05)
        2. Compute weighted average signal direction
        3. Apply EMA: new = alpha * signal + (1-alpha) * current
        4. Clamp drift to max_drift_per_run
        5. Apply rigidity dampening for rigid stakeholders
        """
        if not signals:
            return None

        # Weighted average of all signals
        weighted_sum = 0.0
        weight_total = 0.0
        for s in signals:
            w = self.strength_weights.get(s.strength, 0.10)
            weighted_sum += s.direction * w
            weight_total += w

        if weight_total < 0.01:
            return None

        signal_value = weighted_sum / weight_total

        # EMA update
        raw_new = self.alpha * signal_value + (1 - self.alpha) * current.value

        # Clamp drift
        delta = raw_new - current.value
        delta = max(-self.max_drift, min(self.max_drift, delta))

        # Rigidity dampening: high-rigidity stakeholders resist change
        if self.rigidity_dampening and rigidity > 0.6:
            dampening_factor = 1.0 - (rigidity - 0.6) * 1.5  # 0.6→1.0, 0.8→0.7, 1.0→0.4
            dampening_factor = max(0.3, dampening_factor)
            delta *= dampening_factor

        new_value = current.value + delta
        new_value = max(-1.0, min(1.0, round(new_value, 4)))

        # Skip trivial changes
        if abs(new_value - current.value) < 0.005:
            return None

        return PositionDelta(
            stakeholder_id=signals[0].stakeholder_id,
            topic_tag=current.topic_tag,
            old_value=current.value,
            new_value=new_value,
            n_signals=len(signals),
            evidence=[s.evidence for s in signals if s.evidence],
            sources=list(set(s.source_url for s in signals if s.source_url)),
        )

    def compute_new_position(
        self,
        signals: list[PositionSignal],
        min_signals: int = 2,
    ) -> Optional[PositionDelta]:
        """Create a new Position entry from is_new_topic signals.

        Requires at least min_signals concordant signals to create a new position.
        """
        if len(signals) < min_signals:
            return None

        # Check concordance: signals should roughly agree on direction
        directions = [s.direction for s in signals]
        avg_direction = sum(directions) / len(directions)
        variance = sum((d - avg_direction) ** 2 for d in directions) / len(directions)

        if variance > 0.25:  # too much disagreement
            logger.debug(f"New topic signals too divergent (var={variance:.2f}), skipping")
            return None

        return PositionDelta(
            stakeholder_id=signals[0].stakeholder_id,
            topic_tag=signals[0].topic_tag,
            old_value=0.0,
            new_value=round(max(-1.0, min(1.0, avg_direction)), 4),
            n_signals=len(signals),
            evidence=[s.evidence for s in signals if s.evidence],
            sources=list(set(s.source_url for s in signals)),
            is_new_topic=True,
            new_confidence="low",
        )

    def compute_influence_delta(
        self,
        stakeholder: Stakeholder,
        signals: list[InfluenceSignal],
    ) -> Optional[InfluenceDelta]:
        """Compute influence score update from influence signals."""
        if not signals:
            return None

        # Aggregate: majority direction wins
        ups = sum(1 for s in signals if s.direction == "up")
        downs = sum(1 for s in signals if s.direction == "down")

        if ups == downs:
            return None

        direction = 1.0 if ups > downs else -1.0
        avg_magnitude = sum(s.magnitude for s in signals) / len(signals)

        # Very conservative influence updates: max 0.05 per run
        delta = direction * min(avg_magnitude * 0.1, 0.05)
        new_value = round(max(0.0, min(1.0, stakeholder.influence + delta)), 4)

        if abs(new_value - stakeholder.influence) < 0.005:
            return None

        reasons = [s.reason for s in signals if s.reason]
        return InfluenceDelta(
            stakeholder_id=stakeholder.id,
            old_value=stakeholder.influence,
            new_value=new_value,
            reason="; ".join(reasons[:3]),
            sources=list(set(s.source_url for s in signals if s.source_url)),
        )

    def compute_all(
        self,
        analyses: list[ArticleAnalysis],
        stakeholders: dict[str, Stakeholder],
    ) -> list[StakeholderUpdate]:
        """Compute all updates for all stakeholders from analyses.

        Groups signals by stakeholder and topic, then computes deltas.
        """
        from datetime import date

        # Group signals by stakeholder
        position_signals: dict[str, dict[str, list[PositionSignal]]] = defaultdict(lambda: defaultdict(list))
        influence_signals: dict[str, list[InfluenceSignal]] = defaultdict(list)
        quote_signals: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        quote_sources: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        for analysis in analyses:
            for sig in analysis.position_signals:
                position_signals[sig.stakeholder_id][sig.topic_tag].append(sig)
            if analysis.influence_signal:
                influence_signals[analysis.influence_signal.stakeholder_id].append(analysis.influence_signal)
            for q in analysis.quotes:
                if q.topic_tag:
                    quote_signals[q.stakeholder_id][q.topic_tag].append(q.quote)
                    quote_sources[q.stakeholder_id][q.topic_tag].append(q.source_url)

        # Compute updates
        updates = []
        today = date.today().isoformat()

        for sid, topic_signals in position_signals.items():
            stakeholder = stakeholders.get(sid)
            if not stakeholder:
                continue

            update = StakeholderUpdate(
                stakeholder_id=sid,
                new_last_updated=today,
            )

            for topic_tag, signals in topic_signals.items():
                # Find existing position
                existing = next(
                    (p for p in stakeholder.positions if p.topic_tag == topic_tag),
                    None,
                )

                if existing:
                    delta = self.compute_position_delta(existing, signals, stakeholder.rigidity)
                    if delta:
                        update.position_deltas.append(delta)
                else:
                    # New topic — only create if we have concordant signals
                    new_topic_signals = [s for s in signals if s.is_new_topic]
                    if new_topic_signals:
                        delta = self.compute_new_position(new_topic_signals)
                        if delta:
                            update.position_deltas.append(delta)

            # Influence update
            inf_sigs = influence_signals.get(sid, [])
            inf_delta = self.compute_influence_delta(stakeholder, inf_sigs)
            if inf_delta:
                update.influence_delta = inf_delta

            # Quote updates
            for topic_tag, quotes in quote_signals.get(sid, {}).items():
                if quotes:
                    update.quote_deltas.append(QuoteDelta(
                        stakeholder_id=sid,
                        topic_tag=topic_tag,
                        quotes=quotes[:3],  # cap at 3 new quotes per topic per run
                        sources=quote_sources.get(sid, {}).get(topic_tag, []),
                    ))

            if update.has_changes:
                updates.append(update)

        logger.info(
            f"Computed updates for {len(updates)} stakeholders: "
            f"{sum(len(u.position_deltas) for u in updates)} position changes, "
            f"{sum(1 for u in updates if u.influence_delta)} influence changes"
        )
        return updates
