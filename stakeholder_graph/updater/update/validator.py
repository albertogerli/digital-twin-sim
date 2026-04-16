"""Validator — pre-commit checks on computed updates.

Rejects outliers, flags suspicious changes for human review,
ensures all values stay within bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from stakeholder_graph.updater.update.position_updater import StakeholderUpdate

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating all updates."""
    approved: list[StakeholderUpdate] = field(default_factory=list)
    flagged: list[tuple[StakeholderUpdate, str]] = field(default_factory=list)
    rejected: list[tuple[StakeholderUpdate, str]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Approved: {len(self.approved)}, "
            f"Flagged: {len(self.flagged)}, "
            f"Rejected: {len(self.rejected)}"
        )


class Validator:
    """Validates computed updates before persistence."""

    def __init__(
        self,
        max_drift: float = 0.10,
        max_position_deltas_per_stakeholder: int = 5,
        max_influence_drift: float = 0.05,
        suspicious_signal_count: int = 10,
    ):
        self.max_drift = max_drift
        self.max_position_deltas = max_position_deltas_per_stakeholder
        self.max_influence_drift = max_influence_drift
        self.suspicious_signal_count = suspicious_signal_count

    def check(self, updates: list[StakeholderUpdate]) -> ValidationResult:
        """Validate all updates, splitting into approved/flagged/rejected."""
        result = ValidationResult()

        for update in updates:
            issues = self._check_one(update)

            if not issues:
                result.approved.append(update)
            elif any(severity == "reject" for severity, _ in issues):
                reason = "; ".join(msg for _, msg in issues)
                result.rejected.append((update, reason))
                logger.warning(f"Rejected update for {update.stakeholder_id}: {reason}")
            else:
                reason = "; ".join(msg for _, msg in issues)
                result.flagged.append((update, reason))
                # Flagged updates are still applied but logged for review
                result.approved.append(update)
                logger.info(f"Flagged update for {update.stakeholder_id}: {reason}")

        logger.info(f"Validation: {result.summary}")
        return result

    def _check_one(self, update: StakeholderUpdate) -> list[tuple[str, str]]:
        """Check a single stakeholder update. Returns list of (severity, message)."""
        issues = []

        # Check position deltas
        for delta in update.position_deltas:
            # Bounds check
            if not (-1.0 <= delta.new_value <= 1.0):
                issues.append(("reject", f"Position {delta.topic_tag} out of bounds: {delta.new_value}"))

            # Drift check
            if abs(delta.delta) > self.max_drift:
                issues.append(("reject", f"Position {delta.topic_tag} drift too large: {delta.delta:.4f}"))

            # Sign flip check (changing from positive to negative or vice versa)
            if delta.old_value != 0 and (delta.old_value * delta.new_value < 0):
                issues.append(("flag", f"Position {delta.topic_tag} sign flip: {delta.old_value} → {delta.new_value}"))

        # Too many position changes in one run
        if len(update.position_deltas) > self.max_position_deltas:
            issues.append(("flag", f"Too many position changes: {len(update.position_deltas)}"))

        # Influence drift check
        if update.influence_delta:
            drift = abs(update.influence_delta.new_value - update.influence_delta.old_value)
            if drift > self.max_influence_drift:
                issues.append(("reject", f"Influence drift too large: {drift:.4f}"))

        # Suspicious signal count (stakeholder in too many articles)
        total_signals = sum(d.n_signals for d in update.position_deltas)
        if total_signals > self.suspicious_signal_count:
            issues.append(("flag", f"Unusually high signal count: {total_signals}"))

        return issues
