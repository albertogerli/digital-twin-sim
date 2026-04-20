"""Simulation setup service — agent/platform/orchestrator bootstrap.

Extracted from `engine.py` to separate one-time bootstrap concerns (spec →
runtime objects, follow-graph wiring, orchestrator init, checkpoint restore)
from the simulation loop itself.

The engine still owns the agent/platform attributes on `self` for backward
compatibility with callers that read them directly; this service simply
builds the objects and returns them as a `SetupResult`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base_client import BaseLLMClient
from ..config.schema import ScenarioConfig
from ..agents.elite_agent import EliteAgent
from ..agents.institutional_agent import InstitutionalAgent
from ..agents.citizen_swarm import CitizenSwarm
from ..agents.citizen_cluster import CitizenCluster
from ..platform.platform_engine import PlatformEngine
from .event_injector import EventInjector
from domains.base_domain import DomainPlugin

logger = logging.getLogger(__name__)


@dataclass
class SetupResult:
    """Runtime objects produced by one setup pass."""

    platform: PlatformEngine
    elite_agents: list[EliteAgent] = field(default_factory=list)
    institutional_agents: list[InstitutionalAgent] = field(default_factory=list)
    citizen_swarm: Optional[CitizenSwarm] = None
    event_injector: Optional[EventInjector] = None
    escalation_engine: object | None = None
    contagion_scorer: object | None = None
    financial_scorer: object | None = None


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


class SimulationSetup:
    """Bootstrap service — builds agents, platform, event injector, orchestrator.

    Stateless with respect to other pipeline stages: the engine instantiates
    one of these per run and discards it after `setup_fresh()` or
    `restore_from_checkpoint()` returns.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        config: ScenarioConfig,
        domain: DomainPlugin,
        output_dir: str,
        elite_only: bool,
        seed_data=None,
        activation_plan=None,
    ):
        self.llm = llm
        self.config = config
        self.domain = domain
        self.output_dir = output_dir
        self.elite_only = elite_only
        self.seed_data = seed_data
        self.activation_plan = activation_plan

    # ── Fresh setup ───────────────────────────────────────────────────────

    def setup_fresh(self) -> SetupResult:
        """Build every runtime object from the scenario config."""
        platform = self._init_platform()

        archetype_channels = self.domain.get_archetype_channel_map()
        elite_system_template = self.domain.get_elite_system_prompt_template()

        elite_agents = self._build_elite_agents(archetype_channels, elite_system_template)
        print(f"  ├─ Loading {len(elite_agents)} Elite agents... ✓")

        institutional_agents: list[InstitutionalAgent] = []
        citizen_swarm: CitizenSwarm
        if not self.elite_only:
            institutional_agents = [
                InstitutionalAgent.from_spec(spec.model_dump())
                for spec in self.config.institutional_agents
            ]
            print(f"  ├─ Loading {len(institutional_agents)} Institutional agents... ✓")

            clusters = [
                CitizenCluster.from_spec(spec.model_dump())
                for spec in self.config.citizen_clusters
            ]
            citizen_swarm = CitizenSwarm(clusters)
            total_pop = sum(c.size for c in citizen_swarm.clusters.values())
            print(f"  ├─ Loading {len(clusters)} Citizen clusters ({total_pop:,} citizens)... ✓")
        else:
            citizen_swarm = CitizenSwarm([])
            print(f"  ├─ Skipping Tier 2+3 (elite-only mode)")

        # Public Opinion aggregate agent (matches legacy calibration v2.3)
        if not self.elite_only and (elite_agents or institutional_agents):
            pub_agent = self._build_public_opinion_agent(elite_agents, institutional_agents)
            institutional_agents.append(pub_agent)
            print(f"  ├─ Public Opinion agent injected (pos={pub_agent.position:+.2f}) ✓")

        self._init_follow_graph(platform, elite_agents, institutional_agents)
        print(f"  ├─ Initializing social platforms... ✓")

        event_injector = self._build_event_injector()
        grounding = "grounded" if self.seed_data else "emergent"
        print(f"  └─ Event injector ready ({grounding} mode) ✓")

        orch = self._init_orchestrator()
        print()

        return SetupResult(
            platform=platform,
            elite_agents=elite_agents,
            institutional_agents=institutional_agents,
            citizen_swarm=citizen_swarm,
            event_injector=event_injector,
            escalation_engine=orch.get("escalation_engine"),
            contagion_scorer=orch.get("contagion_scorer"),
            financial_scorer=orch.get("financial_scorer"),
        )

    # ── Checkpoint restore (What-If branching) ────────────────────────────

    def restore_from_checkpoint(
        self,
        checkpoint: dict,
        resume_round: int,
        agent_overrides: dict | None = None,
    ) -> SetupResult:
        """Rebuild the simulation state from a prior checkpoint."""
        from .checkpoint import restore_agents

        elite_agents, inst_agents, clusters = restore_agents(checkpoint)

        # Apply agent overrides (position / trait tweaks for what-if branches)
        overrides = agent_overrides or {}
        if overrides:
            agent_map = {a.id: a for a in elite_agents}
            agent_map.update({a.id: a for a in inst_agents})
            for agent_id, patch in overrides.items():
                agent = agent_map.get(agent_id)
                if agent:
                    for key, value in patch.items():
                        if hasattr(agent, key):
                            setattr(agent, key, value)
                    logger.info(f"Applied overrides to {agent_id}: {patch}")

        citizen_swarm = CitizenSwarm(clusters)
        platform = self._init_platform()
        self._init_follow_graph(platform, elite_agents, inst_agents)
        event_injector = self._build_event_injector()

        # Give the event injector context of prior rounds
        for prev_round in range(1, resume_round + 1):
            timeline_label = (
                self.config.timeline_labels[prev_round - 1]
                if prev_round <= len(self.config.timeline_labels)
                else f"Round {prev_round}"
            )
            event_injector.event_history.append({
                "round": prev_round,
                "timeline_label": timeline_label,
                "event": f"[inherited from parent scenario round {prev_round}]",
                "shock_magnitude": 0.3,
                "shock_direction": 0.0,
            })

        orch = self._init_orchestrator()

        print(f"  ├─ Restored {len(elite_agents)} Elite agents from checkpoint ✓")
        print(f"  ├─ Restored {len(inst_agents)} Institutional agents ✓")
        print(f"  ├─ Restored {len(citizen_swarm.clusters)} Citizen clusters ✓")
        if overrides:
            print(f"  ├─ Applied {len(overrides)} agent overrides ✓")
        print(f"  └─ Branching from round {resume_round} ✓\n")

        return SetupResult(
            platform=platform,
            elite_agents=elite_agents,
            institutional_agents=inst_agents,
            citizen_swarm=citizen_swarm,
            event_injector=event_injector,
            escalation_engine=orch.get("escalation_engine"),
            contagion_scorer=orch.get("contagion_scorer"),
            financial_scorer=orch.get("financial_scorer"),
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _init_platform(self) -> PlatformEngine:
        db_path = os.path.join(self.output_dir, f"social_{_safe_name(self.config.name)}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        return PlatformEngine(db_path)

    def _build_elite_agents(
        self, archetype_channels: dict, elite_system_template: str,
    ) -> list[EliteAgent]:
        agents: list[EliteAgent] = []
        for spec in self.config.elite_agents:
            primary, secondary = archetype_channels.get(
                spec.archetype,
                (spec.platform_primary or "social", spec.platform_secondary or "forum"),
            )
            spec_dict = spec.model_dump()
            spec_dict["platform_primary"] = primary
            spec_dict["platform_secondary"] = secondary

            pos_desc = self.domain.describe_position(spec.position)
            system_prompt = elite_system_template.format(
                name=spec.name, role=spec.role, bio=spec.bio,
                pos_desc=pos_desc, pos=spec.position,
                traits=", ".join(spec.key_traits) if spec.key_traits else "",
                style=spec.communication_style,
            )
            agents.append(EliteAgent.from_spec(spec_dict, system_prompt))
        return agents

    def _build_public_opinion_agent(
        self,
        elite_agents: list[EliteAgent],
        institutional_agents: list[InstitutionalAgent],
    ) -> InstitutionalAgent:
        """Aggregate citizen sentiment as one high-tolerance institutional agent.

        Initialized at the influence-weighted mean of all named agents so the
        public-opinion baseline isn't arbitrary. Low rigidity / high tolerance
        makes it responsive to events and broadly influenceable.
        """
        all_named = elite_agents + institutional_agents
        total_infl = sum(a.influence for a in all_named) or 1.0
        weighted_pos = sum(a.position * a.influence for a in all_named) / total_infl
        agent = InstitutionalAgent(
            id="public_opinion",
            name="Public Opinion",
            role="aggregate citizen sentiment",
            archetype="public_opinion",
            position=weighted_pos,
            original_position=weighted_pos,
            influence=0.7,
            rigidity=0.1,
            key_trait="responsive to events and social pressure",
            category="public_opinion",
        )
        agent.tolerance = 0.9
        return agent

    def _init_follow_graph(
        self,
        platform: PlatformEngine,
        elite_agents: list[EliteAgent],
        institutional_agents: list[InstitutionalAgent],
    ) -> None:
        """Seed the follower relationships at round 0."""
        channels = self.domain.get_channels()
        default_channel = channels[0].id if channels else "social"

        for elite in elite_agents:
            if not self.elite_only:
                for inst in institutional_agents:
                    if abs(inst.position - elite.position) < 0.6:
                        platform.add_follow(inst.id, elite.id, default_channel, 0)

            for other in elite_agents:
                if other.id != elite.id:
                    platform.add_follow(elite.id, other.id, default_channel, 0)

    def _build_event_injector(self) -> EventInjector:
        historical_context = ""
        if self.seed_data:
            historical_context = self.seed_data.format_historical_context()

        return EventInjector(
            llm=self.llm,
            scenario_context=self.config.scenario_context,
            initial_event=self.config.initial_event,
            event_prompt_template=self.domain.get_event_generation_prompt_template(),
            timeline_labels=self.config.timeline_labels,
            fallback_strings=self.domain.get_fallback_strings(),
            language=self.config.language,
            historical_context=historical_context,
            few_shot_example=self.domain.get_event_few_shot(),
        )

    def _init_orchestrator(self) -> dict:
        """Bootstrap the wave-escalation + contagion + financial-alpha chain.

        Returns a dict with optional keys — empty when no activation plan is
        attached to the config (e.g. lightweight non-financial scenarios).
        """
        if not self.activation_plan:
            return {}
        try:
            from core.orchestrator.escalation import EscalationEngine
            from core.orchestrator.contagion import ContagionScorer
            from core.orchestrator.ticker_relevance import TickerRelevanceScorer
            from core.orchestrator.financial_impact import FinancialImpactScorer

            escalation_engine = EscalationEngine(self.activation_plan)
            contagion_scorer = ContagionScorer(escalation_engine)

            rel_scorer = TickerRelevanceScorer()
            geography = getattr(self.activation_plan, "country", "IT")
            entities = self.activation_plan.detected_sectors
            topics = self.activation_plan.detected_topics
            relevant_universe = rel_scorer.select(
                self.config.domain, entities, geography, topics,
            )

            financial_scorer = FinancialImpactScorer(
                detected_topics=topics,
                detected_sectors=entities,
                llm=self.llm,
                relevant_universe=relevant_universe,
            )

            n_w1 = len(self.activation_plan.wave_1)
            n_w2 = len(self.activation_plan.wave_2)
            n_w3 = len(self.activation_plan.wave_3)
            sectors = financial_scorer.get_sector_summary()
            n_betas = len(sectors.get("sector_betas", {}))
            n_topics = len(sectors.get("detected_topics", []))
            print(f"  ┌─ Orchestrator active: {n_w1}→{n_w2}→{n_w3} wave escalation ✓")
            if n_topics:
                topics_str = ", ".join(sectors["detected_topics"][:4])
                print(
                    f"  ├─ Financial Alpha: {n_topics} topics → {n_betas} sector betas, "
                    f"LLM Flash Notes ✓ ({topics_str})"
                )
            return {
                "escalation_engine": escalation_engine,
                "contagion_scorer": contagion_scorer,
                "financial_scorer": financial_scorer,
            }
        except Exception as e:
            logger.warning(f"Orchestrator init failed: {e}")
            return {}
