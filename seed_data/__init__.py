"""Seed data loader — loads verified stakeholders, demographics, and historical context."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .schema import SeedDataBundle, VerifiedStakeholder, VerifiedDemographic

logger = logging.getLogger(__name__)


class SeedDataLoader:
    """Loads and validates seed data for grounding simulations in reality."""

    def __init__(self, seed_data_path: str):
        self.path = Path(seed_data_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Seed data path not found: {seed_data_path}")

    def load(self) -> SeedDataBundle:
        """Load all seed data from the directory into a SeedDataBundle."""
        context_text = self._load_text("context.md")
        historical_text = self._load_text("historical.md")
        stakeholders = self._load_json_list("stakeholders.json", VerifiedStakeholder)
        demographics = self._load_json_list("demographics.json", VerifiedDemographic)
        known_events = self._load_json_file("known_events.json", default=[])

        bundle = SeedDataBundle(
            context_text=context_text,
            stakeholders=stakeholders,
            demographics=demographics,
            historical_text=historical_text,
            known_events=known_events,
        )
        logger.info(
            f"Loaded seed data: {len(bundle.stakeholders)} stakeholders, "
            f"{len(bundle.demographics)} demographics, "
            f"{len(bundle.known_events)} known events"
        )
        return bundle

    def _load_text(self, filename: str) -> str:
        filepath = self.path / filename
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return ""

    def _load_json_file(self, filename: str, default=None):
        filepath = self.path / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return default if default is not None else {}

    def _load_json_list(self, filename: str, model_class, default=None):
        data = self._load_json_file(filename, default=[])
        if not data:
            return []
        return [model_class(**item) for item in data]
