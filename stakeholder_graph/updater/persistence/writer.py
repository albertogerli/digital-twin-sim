"""Writer — atomic JSON file updates with backup.

Applies StakeholderUpdate deltas to the on-disk JSON files,
ensuring atomic writes and optional backup.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

from stakeholder_graph.updater.update.position_updater import StakeholderUpdate

logger = logging.getLogger(__name__)


class Writer:
    """Applies updates to stakeholder JSON files atomically."""

    def __init__(self, data_dir: Path, backup: bool = True):
        self.data_dir = data_dir
        self.backup = backup
        self._file_index: dict[str, Path] = {}  # stakeholder_id -> json file path
        self._build_index()

    def _build_index(self):
        """Build mapping from stakeholder_id to their source JSON file."""
        for country_dir in sorted(self.data_dir.iterdir()):
            if not country_dir.is_dir() or country_dir.name.startswith("."):
                continue
            for json_file in sorted(country_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    entries = data if isinstance(data, list) else data.get("stakeholders", [])
                    for entry in entries:
                        sid = entry.get("id")
                        if sid:
                            self._file_index[sid] = json_file
                except Exception as e:
                    logger.error(f"Writer index error for {json_file}: {e}")

        logger.info(f"Writer indexed {len(self._file_index)} stakeholders across {len(set(self._file_index.values()))} files")

    def commit(self, updates: list[StakeholderUpdate], run_id: str) -> int:
        """Apply all updates to JSON files.

        Groups updates by file, loads each file once, applies changes,
        writes atomically.

        Returns:
            Number of stakeholders updated.
        """
        # Group updates by file
        file_updates: dict[Path, list[StakeholderUpdate]] = {}
        for update in updates:
            if not update.has_changes:
                continue
            json_file = self._file_index.get(update.stakeholder_id)
            if not json_file:
                logger.warning(f"No file found for stakeholder {update.stakeholder_id}")
                continue
            file_updates.setdefault(json_file, []).append(update)

        n_updated = 0
        for json_file, file_update_list in file_updates.items():
            try:
                n_updated += self._commit_file(json_file, file_update_list, run_id)
            except Exception as e:
                logger.error(f"Failed to commit updates to {json_file}: {e}")

        logger.info(f"Writer committed {n_updated} stakeholder updates across {len(file_updates)} files")
        return n_updated

    def _commit_file(
        self,
        json_file: Path,
        updates: list[StakeholderUpdate],
        run_id: str,
    ) -> int:
        """Apply updates to a single JSON file atomically."""
        # Load current data
        with open(json_file) as f:
            data = json.load(f)

        is_list = isinstance(data, list)
        entries = data if is_list else data.get("stakeholders", [])

        # Build lookup
        entry_index = {e["id"]: e for e in entries if "id" in e}

        n_updated = 0
        for update in updates:
            entry = entry_index.get(update.stakeholder_id)
            if not entry:
                continue

            changed = self._apply_update(entry, update)
            if changed:
                n_updated += 1

        if n_updated == 0:
            return 0

        # Reconstruct data preserving original order
        new_entries = []
        for e in entries:
            sid = e.get("id")
            if sid and sid in entry_index:
                new_entries.append(entry_index[sid])
            else:
                new_entries.append(e)

        if is_list:
            new_data = new_entries
        else:
            new_data = {**data, "stakeholders": new_entries}

        # Backup
        if self.backup:
            backup_dir = self.data_dir / ".backups" / run_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            rel_path = json_file.relative_to(self.data_dir)
            backup_path = backup_dir / str(rel_path).replace("/", "_")
            shutil.copy2(json_file, backup_path)

        # Atomic write
        tmp_path = json_file.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.rename(json_file)

        return n_updated

    def _apply_update(self, entry: dict, update: StakeholderUpdate) -> bool:
        """Apply a StakeholderUpdate to a raw dict entry. Returns True if changed."""
        changed = False

        # Position updates
        positions = entry.get("positions", [])
        positions_by_tag = {p["topic_tag"]: p for p in positions if "topic_tag" in p}

        for delta in update.position_deltas:
            if delta.is_new_topic:
                # Add new position
                new_pos = {
                    "topic_tag": delta.topic_tag,
                    "value": delta.new_value,
                    "source": "auto_updated",
                    "confidence": delta.new_confidence or "low",
                }
                if delta.evidence:
                    new_pos["quotes"] = delta.evidence[:2]
                positions.append(new_pos)
                changed = True
            elif delta.topic_tag in positions_by_tag:
                pos = positions_by_tag[delta.topic_tag]
                pos["value"] = delta.new_value
                # Keep the source as-is but mark it was auto-updated
                if "auto_updated" not in pos.get("source", ""):
                    pos["source"] = pos.get("source", "inferred")
                if delta.new_confidence:
                    pos["confidence"] = delta.new_confidence
                changed = True

        entry["positions"] = positions

        # Influence update
        if update.influence_delta:
            entry["influence"] = update.influence_delta.new_value
            changed = True

        # Quote updates
        for qd in update.quote_deltas:
            if qd.topic_tag in positions_by_tag:
                pos = positions_by_tag[qd.topic_tag]
                existing_quotes = pos.get("quotes", [])
                # Append new quotes, dedup, cap at 5
                for q in qd.quotes:
                    if q not in existing_quotes:
                        existing_quotes.append(q)
                pos["quotes"] = existing_quotes[:5]
                changed = True

        # Update last_updated
        if changed and update.new_last_updated:
            entry["last_updated"] = update.new_last_updated

        return changed
