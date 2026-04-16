"""StakeholderDB — in-memory database backed by JSON files.

Loads all stakeholder JSON files from data/<country>/ at init.
Provides query, filter, and export methods.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from stakeholder_graph.schema import Stakeholder

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class StakeholderDB:
    """In-memory stakeholder database with JSON persistence."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self._stakeholders: dict[str, Stakeholder] = {}
        self._load_all()

    def _load_all(self):
        """Load all JSON files from data/<country>/*.json."""
        if not self.data_dir.exists():
            logger.warning(f"Data dir {self.data_dir} does not exist")
            return

        for country_dir in sorted(self.data_dir.iterdir()):
            if not country_dir.is_dir():
                continue
            for json_file in sorted(country_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    entries = data if isinstance(data, list) else data.get("stakeholders", [])
                    for entry in entries:
                        s = Stakeholder(**entry)
                        if s.id in self._stakeholders:
                            logger.warning(f"Duplicate ID {s.id} in {json_file}")
                        self._stakeholders[s.id] = s
                except Exception as e:
                    logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(self._stakeholders)} stakeholders from {self.data_dir}")

    @property
    def size(self) -> int:
        return len(self._stakeholders)

    def get(self, stakeholder_id: str) -> Optional[Stakeholder]:
        return self._stakeholders.get(stakeholder_id)

    def all(self) -> list[Stakeholder]:
        return list(self._stakeholders.values())

    def query(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        categories: Optional[list[str]] = None,
        party_or_org: Optional[str] = None,
        topic_tag: Optional[str] = None,
        min_influence: float = 0.0,
        min_tier: int = 3,
        active_only: bool = True,
        limit: int = 0,
    ) -> list[Stakeholder]:
        """Query stakeholders with filters.

        Args:
            country: ISO country code (e.g. "IT")
            category: Single category filter
            categories: Multiple category filter (OR)
            party_or_org: Party or organization substring match
            topic_tag: Only return stakeholders with a position on this topic
            min_influence: Minimum influence threshold
            min_tier: Maximum tier value (1=elite only, 2=+institutional, 3=all)
            active_only: Only return active stakeholders
            limit: Max results (0=unlimited)

        Returns:
            List of Stakeholder sorted by influence descending.
        """
        results = []
        cats = set()
        if category:
            cats.add(category)
        if categories:
            cats.update(categories)

        for s in self._stakeholders.values():
            if active_only and not s.active:
                continue
            if country and s.country != country:
                continue
            if cats and s.category not in cats:
                continue
            if party_or_org and party_or_org.lower() not in s.party_or_org.lower():
                continue
            if s.influence < min_influence:
                continue
            if s.tier > min_tier:
                continue
            if topic_tag:
                has_topic = any(p.topic_tag == topic_tag for p in s.positions)
                if not has_topic:
                    continue
            results.append(s)

        results.sort(key=lambda s: s.influence, reverse=True)
        if limit > 0:
            results = results[:limit]
        return results

    def query_for_scenario(
        self,
        country: str,
        topic_tags: list[str],
        n_elite: int = 10,
        n_institutional: int = 6,
    ) -> dict:
        """Query stakeholders optimized for scenario agent generation.

        Returns:
            Dict with 'elite_agents' and 'institutional_agents' lists
            in AgentSpec-compatible format.
        """
        primary_topic = topic_tags[0] if topic_tags else ""

        # Elite agents: high-influence individuals
        elite_candidates = self.query(
            country=country,
            categories=["politician", "journalist", "ceo", "union_leader",
                        "magistrate", "academic", "activist"],
            min_influence=0.3,
            min_tier=1,
        )

        # Prefer candidates with positions on the topic
        def score(s: Stakeholder) -> float:
            has_topic = any(p.topic_tag in topic_tags for p in s.positions)
            return s.influence + (0.3 if has_topic else 0.0)

        elite_candidates.sort(key=score, reverse=True)
        elite = elite_candidates[:n_elite]

        # Institutional agents: orgs, parties
        institutional_candidates = self.query(
            country=country,
            categories=["institutional"],
            min_tier=2,
        )
        institutional = institutional_candidates[:n_institutional]

        return {
            "elite_agents": [s.to_agent_spec(primary_topic) for s in elite],
            "institutional_agents": [s.to_agent_spec(primary_topic) for s in institutional],
            "metadata": {
                "source": "stakeholder_graph",
                "country": country,
                "topic_tags": topic_tags,
                "n_elite": len(elite),
                "n_institutional": len(institutional),
            },
        }

    def get_relationships(self, stakeholder_id: str) -> list[dict]:
        """Get all relationships for a stakeholder, resolved to names."""
        s = self.get(stakeholder_id)
        if not s:
            return []
        results = []
        for rel in s.relationships:
            target = self.get(rel.target_id)
            results.append({
                "source": s.name,
                "target": target.name if target else rel.target_id,
                "type": rel.type,
                "strength": rel.strength,
                "context": rel.context,
            })
        return results

    def stats(self) -> dict:
        """Return summary statistics."""
        by_country: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_tier: dict[int, int] = {}
        n_positions = 0
        n_relationships = 0

        for s in self._stakeholders.values():
            by_country[s.country] = by_country.get(s.country, 0) + 1
            by_category[s.category] = by_category.get(s.category, 0) + 1
            by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
            n_positions += len(s.positions)
            n_relationships += len(s.relationships)

        return {
            "total": len(self._stakeholders),
            "by_country": by_country,
            "by_category": by_category,
            "by_tier": by_tier,
            "total_positions": n_positions,
            "total_relationships": n_relationships,
        }
