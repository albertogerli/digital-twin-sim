"""Graph Updater — Continuous Integration for the Stakeholder Graph.

Nightly daemon that crawls news sources, matches articles to stakeholders,
extracts position signals via LLM, and applies conservative EMA updates.
"""

from stakeholder_graph.updater.pipeline import UpdatePipeline, UpdateReport

__all__ = ["UpdatePipeline", "UpdateReport"]
