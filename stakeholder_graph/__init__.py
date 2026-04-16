"""Global Stakeholder Graph — persistent database of real-world actors.

Replaces on-the-fly LLM elite agent generation with a curated, historically
grounded database of politicians, journalists, union leaders, CEOs, and
institutional figures.

Usage:
    from stakeholder_graph import StakeholderDB

    db = StakeholderDB()
    agents = db.query(country="IT", domain="political", topic_tags=["judiciary_reform"])
"""

from stakeholder_graph.schema import Stakeholder, Position, Relationship
from stakeholder_graph.db import StakeholderDB

__all__ = ["StakeholderDB", "Stakeholder", "Position", "Relationship"]
