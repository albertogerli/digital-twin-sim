"""Changelog — append-only audit trail for all graph updates.

Every position change, influence shift, and quote addition is logged
as a single JSONL line with full provenance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from stakeholder_graph.updater.update.position_updater import (
    InfluenceDelta,
    PositionDelta,
    QuoteDelta,
    StakeholderUpdate,
)

logger = logging.getLogger(__name__)


class Changelog:
    """Append-only JSONL changelog for graph updates."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict):
        """Append a single entry to the changelog."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def append_update(self, update: StakeholderUpdate, run_id: str):
        """Append all changes from a StakeholderUpdate."""
        timestamp = datetime.now(timezone.utc).isoformat()

        for delta in update.position_deltas:
            self.append({
                "timestamp": timestamp,
                "run_id": run_id,
                "type": "position_change",
                "stakeholder_id": delta.stakeholder_id,
                "topic_tag": delta.topic_tag,
                "old_value": delta.old_value,
                "new_value": delta.new_value,
                "delta": round(delta.delta, 4),
                "n_signals": delta.n_signals,
                "is_new_topic": delta.is_new_topic,
                "evidence": delta.evidence[:3],
                "sources": delta.sources[:5],
            })

        if update.influence_delta:
            d = update.influence_delta
            self.append({
                "timestamp": timestamp,
                "run_id": run_id,
                "type": "influence_change",
                "stakeholder_id": d.stakeholder_id,
                "old_value": d.old_value,
                "new_value": d.new_value,
                "delta": round(d.new_value - d.old_value, 4),
                "reason": d.reason,
                "sources": d.sources[:5],
            })

        for qd in update.quote_deltas:
            self.append({
                "timestamp": timestamp,
                "run_id": run_id,
                "type": "quotes_added",
                "stakeholder_id": qd.stakeholder_id,
                "topic_tag": qd.topic_tag,
                "quotes": qd.quotes,
                "sources": qd.sources[:5],
            })

    def append_all(self, updates: list[StakeholderUpdate], run_id: str):
        """Append all updates from a pipeline run."""
        for update in updates:
            self.append_update(update, run_id)

    def append_run_summary(self, run_id: str, report: dict):
        """Append a summary entry for the entire run."""
        self.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "type": "run_summary",
            **report,
        })

    def query(
        self,
        stakeholder_id: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query changelog entries (reads file, filters in memory)."""
        if not self.path.exists():
            return []

        entries = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if stakeholder_id and entry.get("stakeholder_id") != stakeholder_id:
                    continue
                if since and entry.get("timestamp", "") < since:
                    continue

                entries.append(entry)

        # Return most recent first
        entries.reverse()
        return entries[:limit]
