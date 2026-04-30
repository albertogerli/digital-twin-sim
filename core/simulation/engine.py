"""Simulation engine — thin coordinator over setup / run / evaluate / report.

The engine was previously a 600-line God Object that did agent bootstrap,
the simulation loop, realism scoring, and report generation inline. Those
four concerns now live in focused services:

  - `SimulationSetup`    — build agents/platform/orchestrator
  - `RoundManager`       — execute one round at a time (unchanged)
  - `EvaluationService`  — post-run realism scoring
  - `ReportingService`   — Markdown report + financial-impact JSON

This module wires them together and preserves the engine's legacy public
attribute surface (`self.elite_agents`, `self._round_manager`, …) so
existing callers (`run.py`, `api/simulation_manager.py`) continue to work.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..llm.base_client import BaseLLMClient
from ..config.schema import ScenarioConfig
from ..agents.elite_agent import EliteAgent
from ..agents.institutional_agent import InstitutionalAgent
from ..agents.citizen_swarm import CitizenSwarm
from ..platform.platform_engine import PlatformEngine
from .event_injector import EventInjector
from .round_manager import RoundManager
from .setup import SimulationSetup, SetupResult
from .evaluation import EvaluationService
from .reporting import ReportingService
from domains.base_domain import DomainPlugin

logger = logging.getLogger(__name__)


def _load_seed_data(seed_data_path: str):
    """Load seed data bundle if path is provided and valid."""
    if not seed_data_path:
        return None
    try:
        from seed_data import SeedDataLoader
        loader = SeedDataLoader(seed_data_path)
        return loader.load()
    except Exception as e:
        logger.warning(f"Failed to load seed data from {seed_data_path}: {e}")
        return None


class SimulationEngine:
    """Coordinates the full simulation pipeline."""

    def __init__(
        self,
        llm: BaseLLMClient,
        config: ScenarioConfig,
        domain: DomainPlugin,
        output_dir: str = "outputs",
        elite_only: bool = False,
        verbose: bool = False,
        progress_callback=None,
        # What-If branching
        resume_checkpoint: dict = None,
        resume_round: int = 0,
        event_override: str = None,
        agent_overrides: dict = None,
    ):
        self.llm = llm
        self.config = config
        self.domain = domain
        self.output_dir = output_dir
        self.elite_only = elite_only
        self.verbose = verbose
        self.progress_callback = progress_callback

        # Branching state
        self.resume_checkpoint = resume_checkpoint
        self.resume_round = resume_round
        self.event_override = event_override
        self.agent_overrides = agent_overrides or {}

        os.makedirs(self.output_dir, exist_ok=True)

        # Load optional seed data used by the event injector for grounding
        self.seed_data = _load_seed_data(config.seed_data_path)
        if self.seed_data:
            print(
                f"  ├─ Seed data loaded: {len(self.seed_data.stakeholders)} stakeholders, "
                f"{len(self.seed_data.demographics)} demographics ✓"
            )

        # Orchestrator plan (attached by scenario loader when relevant)
        self.activation_plan = getattr(config, "_activation_plan", None)

        # Agent + platform attributes are populated by SimulationSetup during run()
        self.platform: Optional[PlatformEngine] = None
        self.elite_agents: list[EliteAgent] = []
        self.institutional_agents: list[InstitutionalAgent] = []
        self.citizen_swarm: Optional[CitizenSwarm] = None
        self.event_injector: Optional[EventInjector] = None
        self.escalation_engine = None
        self.contagion_scorer = None
        self.financial_scorer = None
        self.round_results: list[dict] = []
        self._round_manager: Optional[RoundManager] = None

    # ── Pipeline ──────────────────────────────────────────────────────────

    async def run(self):
        """Execute the full simulation pipeline."""
        self._print_header()

        # Step 1 — bootstrap (or restore from checkpoint for What-If branching)
        setup_service = SimulationSetup(
            llm=self.llm,
            config=self.config,
            domain=self.domain,
            output_dir=self.output_dir,
            elite_only=self.elite_only,
            seed_data=self.seed_data,
            activation_plan=self.activation_plan,
        )
        if self.resume_checkpoint:
            result = setup_service.restore_from_checkpoint(
                checkpoint=self.resume_checkpoint,
                resume_round=self.resume_round,
                agent_overrides=self.agent_overrides,
            )
        else:
            result = setup_service.setup_fresh()
        self._apply_setup(result)

        # Step 2 — rounds
        await self._simulate()

        # Step 2b — serialise financial-impact timeline (frontend bridge)
        ReportingService(
            llm=self.llm, config=self.config, domain=self.domain,
            output_dir=self.output_dir, elite_only=self.elite_only,
        ).save_financial_impact(
            round_results=self.round_results,
            financial_scorer_active=self.financial_scorer is not None,
        )

        # Step 3 — realism evaluation
        await EvaluationService(
            llm=self.llm, scenario_name=self.config.name,
            output_dir=self.output_dir, elite_only=self.elite_only,
        ).evaluate(
            elite_agents=self.elite_agents,
            institutional_agents=self.institutional_agents,
            citizen_swarm=self.citizen_swarm,
            event_injector=self.event_injector,
        )

        # Step 4 — final Markdown report
        report_svc = ReportingService(
            llm=self.llm, config=self.config, domain=self.domain,
            output_dir=self.output_dir, elite_only=self.elite_only,
        )
        md_path = await report_svc.generate_report(
            round_results=self.round_results,
            elite_agents=self.elite_agents,
            citizen_swarm=self.citizen_swarm,
        )

        # Step 4b — printable HTML report (deliverable for the client)
        try:
            report_svc.generate_html_report(
                round_results=self.round_results,
                elite_agents=self.elite_agents,
                citizen_swarm=self.citizen_swarm,
                markdown_report_path=md_path,
                cost=self.llm.stats.total_cost if hasattr(self.llm, "stats") else None,
            )
        except Exception as e:
            logger.warning(f"HTML report generation failed: {e}")

        self._print_footer()

    def _apply_setup(self, result: SetupResult) -> None:
        """Copy SetupResult fields onto self to preserve legacy attribute surface."""
        self.platform = result.platform
        self.elite_agents = result.elite_agents
        self.institutional_agents = result.institutional_agents
        self.citizen_swarm = result.citizen_swarm
        self.event_injector = result.event_injector
        self.escalation_engine = result.escalation_engine
        self.contagion_scorer = result.contagion_scorer
        self.financial_scorer = result.financial_scorer

    async def _simulate(self):
        """Drive the RoundManager across every round."""
        start_round = self.resume_round + 1 if self.resume_round else 1

        round_manager = RoundManager(
            llm=self.llm,
            platform=self.platform,
            event_injector=self.event_injector,
            elite_agents=self.elite_agents,
            institutional_agents=self.institutional_agents,
            citizen_swarm=self.citizen_swarm,
            domain_plugin=self.domain,
            scenario_name=self.config.name,
            checkpoint_dir=self.output_dir,
            elite_only=self.elite_only,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            language=self.config.language,
            scenario_context=self.config.scenario_context,
            metrics_to_track=self.config.metrics_to_track,
            escalation_engine=self.escalation_engine,
            contagion_scorer=self.contagion_scorer,
            financial_scorer=self.financial_scorer,
        )

        # First round of a branched run uses the injected event override
        if self.event_override and start_round > 1:
            round_manager._branch_event_override = self.event_override

        # Wargame mode reaches in via progress_callback; keep the handle exposed
        self._round_manager = round_manager

        for round_num in range(start_round, self.config.num_rounds + 1):
            result = await round_manager.execute_round(round_num)
            self.round_results.append(result)

    # ── I/O ───────────────────────────────────────────────────────────────

    def _print_header(self):
        n_elite = len(self.config.elite_agents)
        n_inst = len(self.config.institutional_agents)
        n_clusters = len(self.config.citizen_clusters)
        n_rounds = self.config.num_rounds

        from .param_loader import CalibratedParamLoader
        loader = CalibratedParamLoader()
        domain_id = getattr(self.domain, "domain_id", "") if self.domain else ""
        cal_info = loader.get_calibration_info()
        params = loader.get_params(domain=domain_id)
        source_label = f"{cal_info['model_version']}/{params.get('_source', 'unknown')}"

        print(f"""
╔══════════════════════════════════════════════════════════╗
║  DigitalTwinSim — {self.domain.domain_label:<38} ║
║  {n_elite} elite | {n_inst} institutional | {n_clusters} clusters | {n_rounds} rounds{' ' * max(0, 10 - len(str(n_rounds)))}║
║  Params: {source_label:<47} ║
╚══════════════════════════════════════════════════════════╝

[{self.config.name}] Initializing...""")

    def _print_footer(self):
        total_stats = self.platform.get_total_stats() if self.platform else {}
        print(f"""
[{self.config.name}] ━━ COMPLETE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total: {total_stats.get('total_posts', 0)} posts | {total_stats.get('total_reactions', 0)} reactions | {self.config.num_rounds} rounds
  Cost: ${self.llm.stats.total_cost:.2f}

{self.llm.stats.summary()}
""")

    def get_simulation_state(self) -> dict:
        return {
            "scenario": self.config.name,
            "domain": self.config.domain,
            "rounds": self.config.num_rounds,
            "round_results": self.round_results,
            "elite_agents": {a.id: a for a in self.elite_agents},
            "institutional_agents": {a.id: a for a in self.institutional_agents},
            "citizen_swarm": self.citizen_swarm,
            "platform": self.platform,
        }
