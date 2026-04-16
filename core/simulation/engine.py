"""Main simulation orchestrator — domain-agnostic."""

import asyncio
import logging
import os
from typing import Optional

from ..llm.base_client import BaseLLMClient
from ..config.schema import ScenarioConfig
from ..agents.elite_agent import EliteAgent
from ..agents.institutional_agent import InstitutionalAgent
from ..agents.citizen_swarm import CitizenSwarm
from ..agents.citizen_cluster import CitizenCluster
from ..platform.platform_engine import PlatformEngine
from ..platform.metrics import EngagementMetrics
from .event_injector import EventInjector
from .round_manager import RoundManager
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
    """Orchestrates the entire simulation pipeline."""

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

        # Load seed data if configured
        self.seed_data = _load_seed_data(config.seed_data_path)
        if self.seed_data:
            print(f"  ├─ Seed data loaded: {len(self.seed_data.stakeholders)} stakeholders, "
                  f"{len(self.seed_data.demographics)} demographics ✓")

        # Will be initialized in setup
        self.platform: Optional[PlatformEngine] = None
        self.elite_agents: list[EliteAgent] = []
        self.institutional_agents: list[InstitutionalAgent] = []
        self.citizen_swarm: Optional[CitizenSwarm] = None
        self.event_injector: Optional[EventInjector] = None
        self.round_results: list[dict] = []

        # Orchestrator (Dynamic Contextual Activation)
        self.activation_plan = getattr(config, "_activation_plan", None)
        self.escalation_engine = None
        self.contagion_scorer = None
        self.financial_scorer = None

    async def run(self):
        """Execute the full simulation pipeline."""
        self._print_header()

        # Step 1: Setup agents and platform
        if self.resume_checkpoint:
            self._restore_from_checkpoint()
        else:
            self._setup()

        # Step 2: Run simulation rounds
        await self._simulate()

        # Step 2b: Serialise financial-impact timeline (frontend bridge)
        self._save_financial_impact()

        # Step 3: Evaluate realism
        await self._evaluate_realism()

        # Step 4: Generate report
        await self._report()

        self._print_footer()

    def _print_header(self):
        n_elite = len(self.config.elite_agents)
        n_inst = len(self.config.institutional_agents)
        n_clusters = len(self.config.citizen_clusters)
        n_rounds = self.config.num_rounds

        # Log calibration parameters source
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

    def _setup(self):
        """Load agents and initialize platforms from config."""
        # Initialize platform
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in self.config.name
        )
        db_path = os.path.join(self.output_dir, f"social_{safe_name}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        self.platform = PlatformEngine(db_path)

        # Get channel mappings from domain
        archetype_channels = self.domain.get_archetype_channel_map()
        elite_system_template = self.domain.get_elite_system_prompt_template()

        # Create Elite agents from config specs
        for spec in self.config.elite_agents:
            # Assign platforms from domain archetype map
            primary, secondary = archetype_channels.get(
                spec.archetype,
                (spec.platform_primary or "social", spec.platform_secondary or "forum")
            )
            spec_dict = spec.model_dump()
            spec_dict["platform_primary"] = primary
            spec_dict["platform_secondary"] = secondary

            # Build system prompt using domain template
            pos_desc = self.domain.describe_position(spec.position)
            system_prompt = elite_system_template.format(
                name=spec.name, role=spec.role, bio=spec.bio,
                pos_desc=pos_desc, pos=spec.position,
                traits=", ".join(spec.key_traits) if spec.key_traits else "",
                style=spec.communication_style,
            )

            agent = EliteAgent.from_spec(spec_dict, system_prompt)
            self.elite_agents.append(agent)
        print(f"  ├─ Loading {len(self.elite_agents)} Elite agents... ✓")

        # Create Institutional agents
        if not self.elite_only:
            for spec in self.config.institutional_agents:
                agent = InstitutionalAgent.from_spec(spec.model_dump())
                self.institutional_agents.append(agent)
            print(f"  ├─ Loading {len(self.institutional_agents)} Institutional agents... ✓")

            # Create Citizen clusters
            clusters = [
                CitizenCluster.from_spec(spec.model_dump())
                for spec in self.config.citizen_clusters
            ]
            self.citizen_swarm = CitizenSwarm(clusters)
            total_pop = sum(c.size for c in self.citizen_swarm.clusters.values())
            print(f"  ├─ Loading {len(clusters)} Citizen clusters ({total_pop:,} citizens)... ✓")
        else:
            self.citizen_swarm = CitizenSwarm([])
            print(f"  ├─ Skipping Tier 2+3 (elite-only mode)")

        # Inject implicit "Public Opinion" agent — represents mass citizen
        # sentiment as a single institutional-tier agent. Initialized at the
        # influence-weighted mean position of all named agents. Low rigidity
        # makes it highly responsive to events; high tolerance means it's
        # influenced by all agents. Aligned with calibration v2.3 model.
        if not self.elite_only and (self.elite_agents or self.institutional_agents):
            all_named = self.elite_agents + self.institutional_agents
            total_infl = sum(a.influence for a in all_named) or 1.0
            weighted_pos = sum(a.position * a.influence for a in all_named) / total_infl
            pub_agent = InstitutionalAgent(
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
            pub_agent.tolerance = 0.9
            self.institutional_agents.append(pub_agent)
            print(f"  ├─ Public Opinion agent injected (pos={weighted_pos:+.2f}) ✓")

        # Initialize follow graph
        self._init_follow_graph()
        print(f"  ├─ Initializing social platforms... ✓")

        # Event injector with optional seed data grounding
        historical_context = ""
        if self.seed_data:
            historical_context = self.seed_data.format_historical_context()

        event_few_shot = self.domain.get_event_few_shot()

        self.event_injector = EventInjector(
            llm=self.llm,
            scenario_context=self.config.scenario_context,
            initial_event=self.config.initial_event,
            event_prompt_template=self.domain.get_event_generation_prompt_template(),
            timeline_labels=self.config.timeline_labels,
            fallback_strings=self.domain.get_fallback_strings(),
            language=self.config.language,
            historical_context=historical_context,
            few_shot_example=event_few_shot,
        )
        grounding = "grounded" if self.seed_data else "emergent"
        print(f"  └─ Event injector ready ({grounding} mode) ✓")

        # Initialize orchestrator if activation plan is available
        if self.activation_plan:
            try:
                from core.orchestrator.escalation import EscalationEngine
                from core.orchestrator.contagion import ContagionScorer
                from core.orchestrator.ticker_relevance import TickerRelevanceScorer
                self.escalation_engine = EscalationEngine(self.activation_plan)
                self.contagion_scorer = ContagionScorer(self.escalation_engine)

                # Global Relevance Scoring — select 10-30 tickers from 190+ universe
                rel_scorer = TickerRelevanceScorer()
                domain = self.config.domain
                geography = getattr(self.activation_plan, "country", "IT")
                entities = self.activation_plan.detected_sectors
                topics = self.activation_plan.detected_topics
                relevant_universe = rel_scorer.select(domain, entities, geography, topics)

                # Financial Impact Scorer
                from core.orchestrator.financial_impact import FinancialImpactScorer
                self.financial_scorer = FinancialImpactScorer(
                    detected_topics=topics,
                    detected_sectors=entities,
                    llm=self.llm,
                    relevant_universe=relevant_universe,
                )
                n_w1 = len(self.activation_plan.wave_1)
                n_w2 = len(self.activation_plan.wave_2)
                n_w3 = len(self.activation_plan.wave_3)
                sectors = self.financial_scorer.get_sector_summary()
                n_betas = len(sectors.get("sector_betas", {}))
                n_topics = len(sectors.get("detected_topics", []))
                print(f"  ┌─ Orchestrator active: {n_w1}→{n_w2}→{n_w3} wave escalation ✓")
                if n_topics:
                    topics_str = ", ".join(sectors["detected_topics"][:4])
                    print(f"  ├─ Financial Alpha: {n_topics} topics → {n_betas} sector betas, LLM Flash Notes ✓ ({topics_str})")
            except Exception as e:
                logger.warning(f"Orchestrator init failed: {e}")
                self.escalation_engine = None
                self.contagion_scorer = None
        print()

    def _init_follow_graph(self):
        """Initialize follower relationships."""
        channels = self.domain.get_channels()
        default_channel = channels[0].id if channels else "social"

        for elite in self.elite_agents:
            if not self.elite_only:
                for inst in self.institutional_agents:
                    if abs(inst.position - elite.position) < 0.6:
                        self.platform.add_follow(
                            inst.id, elite.id, default_channel, 0
                        )

            for other in self.elite_agents:
                if other.id != elite.id:
                    self.platform.add_follow(
                        elite.id, other.id, default_channel, 0
                    )

    def _restore_from_checkpoint(self):
        """Restore agents from a checkpoint dict (for What-If branching)."""
        from .checkpoint import restore_agents

        cp = self.resume_checkpoint
        elite_agents, inst_agents, clusters = restore_agents(cp)

        # Apply agent overrides (position changes, etc.)
        if self.agent_overrides:
            agent_map = {a.id: a for a in elite_agents}
            agent_map.update({a.id: a for a in inst_agents})
            for agent_id, overrides in self.agent_overrides.items():
                agent = agent_map.get(agent_id)
                if agent:
                    for key, value in overrides.items():
                        if hasattr(agent, key):
                            setattr(agent, key, value)
                    logger.info(f"Applied overrides to {agent_id}: {overrides}")

        self.elite_agents = elite_agents
        self.institutional_agents = inst_agents
        self.citizen_swarm = CitizenSwarm(clusters)

        # Initialize platform (fresh DB for the branch)
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in self.config.name
        )
        db_path = os.path.join(self.output_dir, f"social_{safe_name}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        self.platform = PlatformEngine(db_path)
        self._init_follow_graph()

        # Event injector
        historical_context = ""
        if self.seed_data:
            historical_context = self.seed_data.format_historical_context()

        self.event_injector = EventInjector(
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

        # Reconstruct event history from parent checkpoint's coalition_history
        # to give the event injector context of prior rounds
        for prev_round in range(1, self.resume_round + 1):
            self.event_injector.event_history.append({
                "round": prev_round,
                "timeline_label": self.config.timeline_labels[prev_round - 1]
                    if prev_round <= len(self.config.timeline_labels)
                    else f"Round {prev_round}",
                "event": f"[inherited from parent scenario round {prev_round}]",
                "shock_magnitude": 0.3,
                "shock_direction": 0.0,
            })

        n_elite = len(self.elite_agents)
        n_inst = len(self.institutional_agents)
        n_clusters = len(self.citizen_swarm.clusters)
        print(f"  ├─ Restored {n_elite} Elite agents from checkpoint ✓")
        print(f"  ├─ Restored {n_inst} Institutional agents ✓")
        print(f"  ├─ Restored {n_clusters} Citizen clusters ✓")
        if self.agent_overrides:
            print(f"  ├─ Applied {len(self.agent_overrides)} agent overrides ✓")
        print(f"  └─ Branching from round {self.resume_round} ✓\n")

    async def _simulate(self):
        """Run the simulation for all rounds."""
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

        # Inject event override for the first branched round
        if self.event_override and start_round > 1:
            round_manager._branch_event_override = self.event_override

        # Expose round_manager for wargame mode (progress_callback can inject overrides)
        self._round_manager = round_manager

        for round_num in range(start_round, self.config.num_rounds + 1):
            result = await round_manager.execute_round(round_num)
            self.round_results.append(result)

    async def _evaluate_realism(self):
        """Run post-simulation realism evaluation."""
        try:
            from evaluation.realism_scorer import compute_realism_score

            # Collect start/end positions
            start_positions = {a.id: a.original_position for a in self.elite_agents}
            end_positions = {a.id: a.position for a in self.elite_agents}
            if not self.elite_only:
                for a in self.institutional_agents:
                    start_positions[a.id] = a.original_position
                    end_positions[a.id] = a.position

            all_positions = list(end_positions.values())
            if not self.elite_only:
                all_positions.extend(c.position for c in self.citizen_swarm.clusters.values())

            # Collect alliance pairs from round results (simplified)
            alliances = []

            # Collect events
            events = self.event_injector.event_history if self.event_injector else []

            realism = await compute_realism_score(
                llm=self.llm,
                scenario_name=self.config.name,
                agents_start_positions=start_positions,
                agents_end_positions=end_positions,
                all_final_positions=all_positions,
                alliances=alliances,
                events=events,
            )

            score = realism["realism_score"]
            print(f"\n  ┌─ REALISM EVALUATION ─────────────────────")
            print(f"  │  Score: {score:.0f}/100")
            print(f"  │  Distribution: {realism['distribution_plausibility']:.0f} | "
                  f"Drift: {realism['drift_realism']:.0f} | "
                  f"Events: {realism['event_coherence']:.0f}")
            issues = realism.get("issues", {})
            for category, issue_list in issues.items():
                for issue in issue_list[:2]:  # Show max 2 per category
                    print(f"  │  ⚠ [{category}] {issue}")
            print(f"  └────────────────────────────────────────────")

            # Save realism report
            import json
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.config.name)
            realism_path = os.path.join(self.output_dir, f"{safe_name}_realism.json")
            with open(realism_path, "w") as f:
                json.dump(realism, f, indent=2)

        except Exception as e:
            logger.warning(f"Realism evaluation failed: {e}")
            print(f"\n  ⚠ Realism evaluation skipped: {e}")

    async def _report(self):
        """Generate the final report."""
        print(f"\n  Generating report...")
        report_system = self.domain.get_report_system_prompt()
        report_template = self.domain.get_report_prompt_template()

        # Build round summaries
        round_summaries = []
        for r in self.round_results:
            round_summaries.append(
                f"Round {r['round']} ({r.get('timeline_label', '?')}): "
                f"{r['posts']} posts, {r['reactions']} reactions, "
                f"polarization {r['polarization']:.1f}/10"
            )

        # Elite summary
        elite_summary = "\n".join(
            f"- {a.name} ({a.role}): pos {a.position:+.2f}, state {a.emotional_state}"
            for a in self.elite_agents
        )

        # Cluster summary
        cluster_summary = ""
        if not self.elite_only and self.citizen_swarm.clusters:
            cluster_lines = []
            for c in self.citizen_swarm.clusters.values():
                cluster_lines.append(
                    f"- {c.name}: pos {c.position:+.2f}, "
                    f"sentiment {c.dominant_sentiment}, "
                    f"engagement {c.engagement_level:.1f}"
                )
            cluster_summary = "CITIZEN CLUSTERS:\n" + "\n".join(cluster_lines)

        prompt = report_template.format(
            scenario_title=self.config.name,
            num_rounds=self.config.num_rounds,
            round_summaries="\n".join(round_summaries),
            num_elite=len(self.elite_agents),
            elite_summary=elite_summary,
            cluster_summary=cluster_summary,
        )

        # Inject language instruction into BOTH system prompt and user prompt
        lang = getattr(self.config, "language", "en")
        if lang and lang != "en":
            lang_map = {"it": "Italian", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese"}
            lang_name = lang_map.get(lang, lang)
            lang_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: Write the ENTIRE report in {lang_name}. Every heading, paragraph, analysis, conclusion, and narrative MUST be in {lang_name}. Do NOT use English for any part of the report content."
            report_system += lang_instruction
            prompt = lang_instruction + "\n\n" + prompt

        try:
            report_text = await self.llm.generate_text(
                prompt=prompt,
                system_prompt=report_system,
                temperature=0.7,
                max_output_tokens=8000,
                component="report",
            )
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report_text = f"# {self.config.name}\n\nReport generation failed: {e}"

        # Save report
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in self.config.name
        )
        report_path = os.path.join(self.output_dir, f"{safe_name}_report.md")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"  └─ Report: {report_path} ✓")

    def _save_financial_impact(self):
        """Serialise the per-round financial-impact output from the scorer.

        Writes a single JSON array keyed by round, using the Python schema as
        the source of truth for the frontend bridge. Skipped if the scorer
        was not active (e.g., non-financial scenarios without plugin wiring).
        """
        if not self.financial_scorer:
            return

        import json
        from core.orchestrator.financial_impact import FIN_SCHEMA_VERSION

        rounds_payload = []
        for r in self.round_results:
            orch = r.get("orchestrator") or {}
            fin = orch.get("financial_impact")
            if not fin:
                continue
            rounds_payload.append({
                "round": r["round"],
                "timeline_label": r.get("timeline_label", ""),
                **fin,
            })

        if not rounds_payload:
            return

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in self.config.name
        )
        out_path = os.path.join(self.output_dir, f"{safe_name}_financial_impact.json")
        payload = {
            "schema_version": FIN_SCHEMA_VERSION,
            "scenario": self.config.name,
            "domain": self.config.domain,
            "num_rounds": self.config.num_rounds,
            "provenance": "backend-simulated",
            "rounds": rounds_payload,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  └─ Financial impact: {out_path} ✓ ({len(rounds_payload)} rounds)")

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
