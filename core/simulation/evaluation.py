"""Realism evaluation service.

Extracted from `engine.py`. Owns the post-simulation realism scoring
pipeline — collects agent positions, dispatches to `evaluation.realism_scorer`,
prints the summary panel, and serialises the result to disk.

Stateless relative to the simulation engine: the engine passes the final
agent collections and the service returns / persists the score.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..llm.base_client import BaseLLMClient
    from ..agents.elite_agent import EliteAgent
    from ..agents.institutional_agent import InstitutionalAgent
    from ..agents.citizen_swarm import CitizenSwarm
    from .event_injector import EventInjector

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


class EvaluationService:
    """Runs realism scoring and persists the verdict."""

    def __init__(
        self,
        llm: "BaseLLMClient",
        scenario_name: str,
        output_dir: str,
        elite_only: bool = False,
    ):
        self.llm = llm
        self.scenario_name = scenario_name
        self.output_dir = output_dir
        self.elite_only = elite_only

    async def evaluate(
        self,
        elite_agents: list["EliteAgent"],
        institutional_agents: list["InstitutionalAgent"],
        citizen_swarm: Optional["CitizenSwarm"],
        event_injector: Optional["EventInjector"],
    ) -> Optional[dict]:
        """Score and print realism. Returns the scorer output or None on failure."""
        try:
            from evaluation.realism_scorer import compute_realism_score
        except Exception as e:
            logger.warning(f"Realism evaluator not available: {e}")
            print(f"\n  ⚠ Realism evaluation skipped: {e}")
            return None

        start_positions = {a.id: a.original_position for a in elite_agents}
        end_positions = {a.id: a.position for a in elite_agents}
        if not self.elite_only:
            for a in institutional_agents:
                start_positions[a.id] = a.original_position
                end_positions[a.id] = a.position

        all_positions = list(end_positions.values())
        if not self.elite_only and citizen_swarm is not None:
            all_positions.extend(c.position for c in citizen_swarm.clusters.values())

        events = event_injector.event_history if event_injector else []

        try:
            realism = await compute_realism_score(
                llm=self.llm,
                scenario_name=self.scenario_name,
                agents_start_positions=start_positions,
                agents_end_positions=end_positions,
                all_final_positions=all_positions,
                alliances=[],
                events=events,
            )
        except Exception as e:
            logger.warning(f"Realism evaluation failed: {e}")
            print(f"\n  ⚠ Realism evaluation skipped: {e}")
            return None

        self._print_summary(realism)
        self._persist(realism)
        return realism

    def _print_summary(self, realism: dict) -> None:
        score = realism["realism_score"]
        print(f"\n  ┌─ REALISM EVALUATION ─────────────────────")
        print(f"  │  Score: {score:.0f}/100")
        print(
            f"  │  Distribution: {realism['distribution_plausibility']:.0f} | "
            f"Drift: {realism['drift_realism']:.0f} | "
            f"Events: {realism['event_coherence']:.0f}"
        )
        for category, issue_list in realism.get("issues", {}).items():
            for issue in issue_list[:2]:
                print(f"  │  ⚠ [{category}] {issue}")
        print(f"  └────────────────────────────────────────────")

    def _persist(self, realism: dict) -> None:
        path = os.path.join(
            self.output_dir, f"{_safe_name(self.scenario_name)}_realism.json"
        )
        with open(path, "w") as f:
            json.dump(realism, f, indent=2)
