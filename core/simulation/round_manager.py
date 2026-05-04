"""Phase-by-phase round execution manager — domain-agnostic."""

import asyncio
import logging
import os
import sys

from ..llm.base_client import BaseLLMClient
from ..agents.elite_agent import EliteAgent
from ..agents.institutional_agent import InstitutionalAgent, process_institutional_batch
from ..agents.citizen_swarm import CitizenSwarm
from ..platform.platform_engine import PlatformEngine
from ..platform.metrics import EngagementMetrics
from .event_injector import EventInjector
from .interaction_resolver import InteractionResolver
from .opinion_dynamics import OpinionDynamics
from .param_loader import CalibratedParamLoader
from .checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class RoundManager:
    """Executes all 7 phases of a simulation round."""

    def __init__(
        self,
        llm: BaseLLMClient,
        platform: PlatformEngine,
        event_injector: EventInjector,
        elite_agents: list[EliteAgent],
        institutional_agents: list[InstitutionalAgent],
        citizen_swarm: CitizenSwarm,
        domain_plugin,
        scenario_name: str,
        checkpoint_dir: str = "outputs",
        elite_only: bool = False,
        verbose: bool = False,
        progress_callback=None,
        language: str = "en",
        scenario_context: str = "",
        metrics_to_track: list[str] = None,
        escalation_engine=None,
        contagion_scorer=None,
        financial_scorer=None,
        rag_store=None,
    ):
        self.llm = llm
        self.platform = platform
        self.event_injector = event_injector
        self.elite_agents = elite_agents
        self.institutional_agents = institutional_agents
        self.citizen_swarm = citizen_swarm
        self.domain = domain_plugin
        self.scenario_name = scenario_name
        self.checkpoint_dir = checkpoint_dir
        self.elite_only = elite_only
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.language = language
        self.scenario_context = scenario_context
        self.metrics_to_track = metrics_to_track or []
        self._phase_labels = self._build_phase_labels(language)
        self.interaction_resolver = InteractionResolver(platform, domain_plugin)
        # Optional RAG retrieval for grounded reasoning
        self.rag_store = rag_store
        # Cache: round_num -> retrieved chunks (so all posts in a round share citations)
        self._round_retrieved: dict = {}

        # Load calibrated params from v2 posterior (with v1 fallback)
        self.param_loader = CalibratedParamLoader()
        domain_id = getattr(domain_plugin, "domain_id", "") if domain_plugin else ""
        params = self.param_loader.get_params(domain=domain_id, include_uncertainty=True)
        # Log param source
        source = params.get("_source", "unknown")
        model_ver = params.get("_model_version", "?")
        logger.info(f"OpinionDynamics params: source={source}, model={model_ver}")
        if params.get("_ci95"):
            for k, (lo, hi) in params["_ci95"].items():
                logger.info(f"  {k}: {params.get(k, '?'):.4f} [{lo:.3f}, {hi:.3f}]")
        # Print comparison for debugging
        self.param_loader.print_comparison(domain=domain_id)
        # Create OpinionDynamics with v1-compatible params (strip metadata keys)
        od_params = {k: v for k, v in params.items() if not k.startswith("_")}
        self.opinion_dynamics = OpinionDynamics(**od_params)
        self._params_info = params  # Keep for CI propagation
        self.metrics = EngagementMetrics(platform)

        # Regime switching: activate for financial/corporate domains
        self._use_regime_switching = domain_id in ("financial", "corporate")
        self._regime_prob = 0.0  # Track current regime probability
        self.coalition_history = []
        self.custom_metrics_history = []

        # Orchestrator: dynamic agent activation
        self.escalation_engine = escalation_engine
        self.contagion_scorer = contagion_scorer
        self.financial_scorer = financial_scorer
        self._active_elite_ids: set[str] = set()  # IDs of currently active elite agents
        self._active_inst_ids: set[str] = set()    # IDs of currently active institutional agents

        # ── ALM Financial Twin (Sprint 1, weak coupling) ───────────────────
        # Only enabled for the banking-style financial domain. Steps in
        # lockstep with the opinion sim, enforces ALM bounds (deposit beta,
        # loan elasticity, NIM compression, regulatory floors). Defaults are
        # Italian commercial-bank reference (EBA / ECB / BdI 2025); override
        # via scenario_context or future scenario.financial_twin_overrides.
        # ── Real-price ticker tracking ─────────────────────────────────────
        # Domain-agnostic: any brief or KPI label that names a known ticker
        # (TIT.MI, ENEL.MI, AAPL, etc.) gets fetched at anchor + repriced
        # per round via the empirical financial twin (sector beta + impulse
        # response). Replaces the old "TLIT.MI Stock Price = 32" 0-100
        # LLM-score anti-pattern with deterministic real prices in the
        # ticker's native currency.
        self.ticker_price_state = None
        try:
            from core.orchestrator.ticker_prices import TickerPriceState
            self.ticker_price_state = TickerPriceState(
                brief_text=scenario_context or "",
                extra_metric_names=self.metrics_to_track,
            )
            if self.ticker_price_state.tickers:
                logger.info(
                    f"TickerPriceState armed: {len(self.ticker_price_state.tickers)} "
                    f"ticker(s) {self.ticker_price_state.tickers}"
                )
        except Exception as exc:
            logger.warning(f"TickerPriceState init failed (continuing without): {exc}")
            self.ticker_price_state = None

        # Strip ticker-shaped names from LLM 0-100 metrics to avoid
        # double-counting + label confusion. Real prices are emitted
        # under round_result["ticker_prices"], which the frontend renders
        # separately with currency formatting.
        if self.ticker_price_state and self.ticker_price_state.tickers:
            ticker_set = set(self.ticker_price_state.tickers)
            from core.orchestrator.ticker_prices import is_ticker
            self.metrics_to_track = [
                m for m in self.metrics_to_track
                if not (m in ticker_set or is_ticker(m))
            ]

        self.financial_twin = None
        if domain_id == "financial":
            try:
                from core.financial.twin import FinancialTwin
                # Sprint 5: per-country params dispatch from scope.geography
                # (set by Layer-0 brief_scope when available). Falls back
                # to IT if no geography or non-supported country.
                country_params = None
                country_used = "IT"
                try:
                    geo = []
                    if scenario_context:
                        # Best-effort extraction; a structured scope object
                        # would be cleaner but isn't always available here.
                        for code in ("IT", "DE", "FR", "ES", "NL", "US", "GB", "UK"):
                            if f" {code}" in scenario_context or scenario_context.endswith(code):
                                geo.append(code)
                    from core.financial.country_params import select_country_params
                    country_used, country_params = select_country_params(geo or ["IT"])
                except Exception as exc:
                    logger.debug(f"country dispatch fallback to IT: {exc}")

                self.financial_twin = FinancialTwin(params=country_params)
                logger.info(f"FinancialTwin country selected: {country_used}")
                # Sprint 4+6: country-aware live anchor refresh (ECB / FRED /
                # BoE depending on jurisdiction). Cache 24h, fallback graceful.
                try:
                    self.financial_twin.refresh_market_anchors(
                        use_cache=True, country=country_used,
                    )
                except Exception as exc:
                    logger.warning(f"FinancialTwin live anchor refresh skipped: {exc}")
                logger.info(
                    f"FinancialTwin initialised: baseline "
                    f"{self.financial_twin.current_state().to_compact_str()}"
                )
            except Exception as exc:
                logger.warning(f"FinancialTwin init failed (continuing without): {exc}")

    def _get_active_elite_agents(self, round_num: int) -> list[EliteAgent]:
        """Get elite agents active in this round (filtered by orchestrator)."""
        if not self.escalation_engine:
            return self.elite_agents  # No orchestrator → all active

        active_scores = self.escalation_engine.get_active_agents(round_num)
        active_ids = {s.stakeholder_id for s in active_scores}
        self._active_elite_ids = active_ids

        active = [a for a in self.elite_agents if a.id in active_ids]
        skipped = len(self.elite_agents) - len(active)
        if skipped > 0:
            logger.info(f"Orchestrator: {len(active)}/{len(self.elite_agents)} elite agents active (round {round_num})")
        return active

    def _get_active_institutional_agents(self, round_num: int) -> list[InstitutionalAgent]:
        """Get institutional agents active in this round."""
        if not self.escalation_engine:
            return self.institutional_agents

        active_scores = self.escalation_engine.get_active_agents(round_num)
        active_ids = {s.stakeholder_id for s in active_scores}
        self._active_inst_ids = active_ids

        # Always keep public_opinion agent active
        active = [a for a in self.institutional_agents
                  if a.id in active_ids or a.category == "public_opinion"]
        return active

    async def _feed_orchestrator(self, round_num: int, round_stats: dict,
                           polarization: float, sentiment_pcts: dict,
                           event: dict, coalitions) -> dict:
        """Feed round results to escalation engine and contagion scorer.

        Returns orchestrator data dict for inclusion in round_complete event.
        """
        if not self.escalation_engine:
            return {}

        # Feed escalation engine
        escalation_result = self.escalation_engine.process_round(
            round_num=round_num,
            post_count=round_stats.get("posts", 0),
            reaction_count=round_stats.get("reactions", 0),
            polarization=polarization,
            sentiment_pcts=sentiment_pcts,
            shock_magnitude=event.get("shock_magnitude", 0),
            top_post_engagement=round_stats.get("max_engagement", 0),
        )

        # Feed contagion scorer
        contagion_cri = 0.0
        contagion_report = None
        if self.contagion_scorer:
            # Detect institutional activation
            active_categories = set()
            for a in self.institutional_agents:
                if a.id in self._active_inst_ids or a.category == "public_opinion":
                    active_categories.add(a.category)

            contagion_cri = self.contagion_scorer.score_round(
                round_num=round_num,
                post_count=round_stats.get("posts", 0),
                reaction_count=round_stats.get("reactions", 0),
                repost_count=int(round_stats.get("reactions", 0) * 0.3),
                top_post_engagement=round_stats.get("max_engagement", 0),
                institutional_actors_active=len(self._active_inst_ids),
                union_activated="union" in active_categories or "union_leader" in active_categories,
                party_activated="politician" in active_categories or "government" in active_categories,
            )
            contagion_report = self.contagion_scorer.generate_report()

        # Feed financial impact scorer
        financial_data = {}
        if self.financial_scorer:
            # Count negative institutional agents
            neg_inst = sum(1 for a in self.institutional_agents
                          if a.position < -0.3 and a.category != "public_opinion")
            total_inst = max(1, sum(1 for a in self.institutional_agents
                                    if a.category != "public_opinion"))
            neg_ceo = sum(1 for a in self.elite_agents
                          if a.position < -0.3 and getattr(a, "archetype", "") in ("ceo", "business_leader"))

            # Polarization velocity
            polar_vel = 0.0
            if self.escalation_engine and len(self.escalation_engine.state.round_metrics) >= 2:
                prev_pol = self.escalation_engine.state.round_metrics[-2].polarization
                polar_vel = polarization - prev_pol

            # Collect active agents for dynamic ticker extraction
            active_agents = list(self.elite_agents) + [
                a for a in self.institutional_agents if a.category != "public_opinion"
            ]

            financial_report = self.financial_scorer.score_round(
                round_num=round_num,
                engagement_score=escalation_result.get("engagement_score", 0),
                contagion_risk=contagion_cri,
                active_wave=escalation_result.get("active_wave", 1),
                polarization=polarization,
                polarization_velocity=polar_vel,
                negative_institutional_pct=neg_inst / total_inst,
                negative_ceo_count=neg_ceo,
                active_agents=active_agents,
            )

            # Generate LLM Flash Note (async)
            try:
                flash_note = await self.financial_scorer.generate_flash_note(
                    report=financial_report,
                    crisis_brief=self.scenario_context,
                    round_num=round_num,
                )
                financial_report.headline = flash_note
            except Exception as e:
                logger.debug(f"Flash Note generation skipped: {e}")

            financial_data = financial_report.to_dict()

        orchestrator_data = {
            "engagement_score": escalation_result.get("engagement_score", 0),
            "active_wave": escalation_result.get("active_wave", 1),
            "escalated": escalation_result.get("escalated", False),
            "contagion_risk_index": contagion_cri,
            "contagion_risk_label": contagion_report.risk_label if contagion_report else "unknown",
            "containment_window": contagion_report.containment_window if contagion_report else None,
            "financial_impact": financial_data,
        }

        if escalation_result.get("escalated"):
            wave = escalation_result["active_wave"]
            print(f"  ├─ ⚡ ESCALATION → Wave {wave} activated (engagement={escalation_result['engagement_score']:.3f})")
        if contagion_report and contagion_cri > 0.5:
            print(f"  ├─ ⚠ Contagion Risk: {contagion_cri:.3f} ({contagion_report.risk_label.upper()})")
        if financial_data.get("market_volatility_warning") in ("HIGH", "CRITICAL"):
            print(f"  ├─ 💰 {financial_data['market_volatility_warning']}: {financial_data.get('headline', '')[:80]}")

        return orchestrator_data

    def _all_agents_flat(self) -> list:
        agents = list(self.elite_agents)
        if not self.elite_only:
            agents.extend(self.institutional_agents)
        return agents

    def _all_positions(self) -> list[float]:
        positions = [a.position for a in self.elite_agents]
        if not self.elite_only:
            positions.extend(a.position for a in self.institutional_agents)
            positions.extend(c.position for c in self.citizen_swarm.clusters.values())
        return positions

    @staticmethod
    def _build_phase_labels(language: str) -> dict:
        """Return localised phase labels keyed by phase id."""
        if language == "it":
            return {
                "event": "Evento",
                "elite": "Reazioni Elite",
                "institutional": "Agenti Istituzionali",
                "citizens": "Cittadini",
                "platform": "Dinamiche piattaforma",
                "opinion": "Aggiornamento posizioni",
                "checkpoint": "Salvataggio checkpoint",
                # simulation_manager phases
                "round_of": "Round {r} di {t}",
                "round_done": "Round {r} completato",
                "web_research": "Ricerca contesto online...",
                "entity_research": "Analisi entità...",
                "agent_generation": "Generazione agenti...",
                "validation": "Validazione agenti...",
            }
        if language == "es":
            return {
                "event": "Evento",
                "elite": "Reacciones Elite",
                "institutional": "Agentes Institucionales",
                "citizens": "Ciudadanos",
                "platform": "Dinámica de plataforma",
                "opinion": "Actualización de posiciones",
                "checkpoint": "Guardando checkpoint",
                "round_of": "Ronda {r} de {t}",
                "round_done": "Ronda {r} completada",
                "web_research": "Investigación de contexto...",
                "entity_research": "Análisis de entidades...",
                "agent_generation": "Generación de agentes...",
                "validation": "Validación de agentes...",
            }
        # Default: English
        return {
            "event": "Event",
            "elite": "Elite Reactions",
            "institutional": "Institutional Agents",
            "citizens": "Citizens",
            "platform": "Platform Dynamics",
            "opinion": "Updating Positions",
            "checkpoint": "Saving Checkpoint",
            "round_of": "Round {r} of {t}",
            "round_done": "Round {r} completed",
            "web_research": "Researching context...",
            "entity_research": "Analyzing entities...",
            "agent_generation": "Generating agents...",
            "validation": "Validating agents...",
        }

    async def _notify(self, event_type: str, data: dict):
        if self.progress_callback:
            await self.progress_callback(event_type, data)

    async def execute_round(self, round_num: int) -> dict:
        """Execute all phases of a round and return summary."""
        print(f"\n  [{self.scenario_name}] ━━ Round {round_num} ━━━━━━━━━━━━━")
        sys.stdout.flush()
        await self._notify("round_start", {"round": round_num, "cost": self.llm.stats.total_cost})

        # Get viral posts from previous round
        viral_posts_text = ""
        if round_num > 1:
            viral_posts_text = self.platform.format_viral_posts(round_num - 1, top_n=5)

        # Sprint 8: prepend the SYSTEM CLOCK date so agents don't reason
        # in the past ("nel 2022 i tassi erano..."). The LLM otherwise
        # anchors on its training-time priors and produces hallucinations.
        from datetime import datetime as _dt
        today_str = _dt.now().strftime("%Y-%m-%d")
        date_line = (
            f"[DATA CORRENTE: {today_str}] Ragiona come se fosse oggi. "
            f"Riferimenti macro (tassi BCE, spread BTP, mercato del lavoro, "
            f"contesto politico) devono essere coerenti con questa data."
        )
        viral_posts_text = date_line + "\n" + (viral_posts_text or "")

        # ALM context: prepend a compact financial-twin snapshot + previous
        # round's feedback signals so agents see both the balance-sheet state
        # AND the stress signals when reasoning. Domain-only; no-op if
        # FinancialTwin is not active. Uses the previous round's state
        # (the new step happens later in this round).
        if self.financial_twin is not None:
            try:
                ts = self.financial_twin.current_state()
                fb = self.financial_twin.latest_feedback()
                alm_lines = [
                    f"[ALM corrente — banca scenario] {ts.to_compact_str()} "
                    f"(rate {ts.policy_rate_pct:.2f}%, BTP-Bund {ts.btp_bund_spread_bps:.0f}bp)."
                ]
                fb_summary = fb.to_compact_str()
                if fb_summary:
                    alm_lines.append(
                        f"[Segnali di stress finanziario percepiti dal mercato] {fb_summary}."
                    )
                viral_posts_text = "\n".join(alm_lines) + "\n" + (viral_posts_text or "")
            except Exception:
                pass

        # Current stats
        all_positions = self._all_positions()
        polarization = self.metrics.polarization_index(all_positions)
        avg_sentiment = (
            self.citizen_swarm.get_avg_sentiment()
            if not self.elite_only else "neutral"
        )
        top_narratives = ", ".join(
            self.metrics.extract_narratives(
                round_num - 1 if round_num > 1 else None, top_n=3
            )
        )

        # Coalition info from previous round
        coalition_info = ""
        if self.coalition_history:
            last_coalitions = self.coalition_history[-1].get("coalitions", [])
            parts = []
            if isinstance(last_coalitions, list):
                for c in last_coalitions:
                    label = c.get("label", "?")
                    size = c.get("size", 0)
                    avg_pos = c.get("avg_position", 0)
                    parts.append(f"  {label}: {size} members, pos {avg_pos:+.2f}")
            coalition_info = "\n".join(parts) if parts else ""

        # === Phase 1: Event Generation ===
        print(f"  ├─ Phase 1: Event generation   ", end="", flush=True)

        # What-If: use event override for the first branched round
        branch_override = getattr(self, "_branch_event_override", None)
        if branch_override:
            timeline_label = self.event_injector._get_timeline_label(round_num)
            shock_mag = getattr(self, "_wargame_shock_override", 0.5) or 0.5
            event = {
                "round": round_num,
                "timeline_label": timeline_label,
                "event": branch_override,
                "shock_magnitude": shock_mag,
                "shock_direction": 0.0,
                "key_actors": [],
                "institutional_impact": "",
                "public_perception": "",
            }
            self.event_injector.event_history.append(event)
            self._branch_event_override = None  # only override once
            self._wargame_shock_override = None
            is_wargame = "PLAYER ACTION" if "rilascia un" in branch_override else "WHAT-IF OVERRIDE"
            print(f"✓ ({timeline_label}) [{is_wargame}]")
        else:
            event = await self.event_injector.generate_event(
                round_num=round_num,
                elite_agents=self.elite_agents,
                polarization=polarization,
                dominant_sentiment=avg_sentiment,
                top_narratives=top_narratives or "No emerging narrative.",
                coalition_info=coalition_info,
                viral_posts=viral_posts_text,
            )
            timeline_label = event["timeline_label"]
            print(f"✓ ({timeline_label})")
        round_event = event["event"]
        await self._notify("round_phase", {"round": round_num, "phase": "event_generation", "message": f"{self._phase_labels['event']}: {timeline_label}", "phase_index": 1, "total_phases": 7})
        if self.verbose:
            print(f"     Event: {round_event[:120]}...")

        # Get domain-specific prompt templates and channel info
        channel_descs = {
            ch.id: ch.description for ch in self.domain.get_channels()
        }
        channel_max_lens = self.domain.get_channel_max_lengths()
        elite_prompt = self.domain.get_elite_prompt_template()
        inst_prompt = self.domain.get_institutional_batch_prompt_template()
        cluster_prompt = self.domain.get_cluster_prompt_template()
        channel_map = self.domain.get_archetype_channel_map()
        profile_template = self.domain.get_mini_profile_template()

        # Inject few-shot examples into prompts.
        # The few-shot content is JSON and contains unescaped {}; these must be
        # doubled so the subsequent prompt_template.format() in
        # elite_agent.generate_round / citizen_swarm.simulate_round treats them
        # as literal braces, not placeholders. Blinded mode drops the example
        # to avoid culturally-specific contamination (e.g. Italian roster).
        blinded_mode = EventInjector._detect_blinded(self.scenario_context or "")
        elite_few_shot = self.domain.get_elite_few_shot() if not blinded_mode else ""
        cluster_few_shot = self.domain.get_cluster_few_shot() if not blinded_mode else ""

        def _esc(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        if elite_few_shot:
            elite_prompt = elite_prompt + f"\n\nEXAMPLE OF A HIGH-QUALITY RESPONSE (use as format/quality reference, do NOT copy content):\n{_esc(elite_few_shot)}\n"
        if cluster_few_shot:
            cluster_prompt = cluster_prompt + f"\n\nEXAMPLE OF A HIGH-QUALITY RESPONSE (use as format/quality reference, do NOT copy content):\n{_esc(cluster_few_shot)}\n"

        # Inject language instruction into prompts
        if self.language and self.language != "en":
            lang_map = {"it": "Italian", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ar": "Arabic", "ru": "Russian"}
            lang_name = lang_map.get(self.language, self.language)
            lang_instruction = f"\n\nIMPORTANT: Write ALL text content (posts, reactions, descriptions, reasoning, narratives, events) in {lang_name}. Only JSON keys must remain in English.\n"
            elite_prompt = lang_instruction + elite_prompt
            inst_prompt = lang_instruction + inst_prompt
            cluster_prompt = lang_instruction + cluster_prompt

        # === RAG retrieval (once per round) ─────────────────────
        # Query the rag_store with the round event so all subsequent agent
        # prompts can be grounded in retrieved snippets, and so we can
        # attribute citations to each post emitted this round.
        rag_context_block = self._rag_setup_round(round_num, round_event)
        if rag_context_block:
            elite_prompt   = elite_prompt   + rag_context_block
            inst_prompt    = inst_prompt    + rag_context_block
            cluster_prompt = cluster_prompt + rag_context_block

        # === Phase 2: Elite Agent Generation ===
        # Filter active elite agents via orchestrator
        active_elite = self._get_active_elite_agents(round_num)
        wave_label = f" [wave {self.escalation_engine.state.current_wave}]" if self.escalation_engine else ""
        print(f"  ├─ Phase 2: Elite reactions{wave_label}  ", end="", flush=True)
        elite_results = await self._run_elite_agents_filtered(
            active_elite,
            round_num, timeline_label, round_event, viral_posts_text,
            polarization, avg_sentiment, top_narratives,
            elite_prompt, channel_descs, channel_max_lens,
        )
        for r in elite_results:
            if r:
                for p in r.get("posts", []):
                    self._attach_citations(p, round_num)
                    self.platform.add_post(p, round_num)
        cost_str = f"${self.llm.stats.total_cost:.2f}"
        print(f"✓ {len([r for r in elite_results if r])}/{len(active_elite)}  ({cost_str})")
        await self._notify("round_phase", {"round": round_num, "phase": "elite_reactions", "message": f"{self._phase_labels['elite']}: {len([r for r in elite_results if r])}/{len(active_elite)}", "phase_index": 2, "total_phases": 7})

        # === Phase 3: Institutional Agent Batch ===
        if not self.elite_only:
            active_inst = self._get_active_institutional_agents(round_num)
            print(f"  ├─ Phase 3: Institutional batch ", end="", flush=True)
            inst_results = await self._run_institutional_agents_filtered(
                active_inst,
                round_num, timeline_label, round_event, viral_posts_text,
                inst_prompt, channel_max_lens, profile_template,
            )
            for batch in inst_results:
                if isinstance(batch, list):
                    for r in batch:
                        for p in r.get("posts", []):
                            self._attach_citations(p, round_num)
                            self.platform.add_post(p, round_num)
            print(f"✓ ({cost_str})")
            await self._notify("round_phase", {"round": round_num, "phase": "institutional_batch", "message": f"{self._phase_labels['institutional']}: {len(active_inst)}", "phase_index": 3, "total_phases": 7})

            # === Phase 4: Citizen Swarm ===
            print(f"  ├─ Phase 4: Citizen swarm       ", end="", flush=True)
            cluster_results = await self.citizen_swarm.simulate_round(
                self.llm, round_num, timeline_label, round_event,
                viral_posts_text, cluster_prompt, channel_map,
                channel_descs, channel_max_lens,
            )
            for cr in cluster_results:
                for p in cr.get("posts", []):
                    self._attach_citations(p, round_num)
                    self.platform.add_post(p, round_num)
            print(f"✓ {len(cluster_results)} clusters ({cost_str})")
            await self._notify("round_phase", {"round": round_num, "phase": "citizen_swarm", "message": f"{self._phase_labels['citizens']}: {len(cluster_results)} clusters", "phase_index": 4, "total_phases": 7})
        else:
            print(f"  ├─ Phase 3-4: Skipped (elite-only mode)")

        # === Phase 5: Platform Dynamics ===
        await self._notify("round_phase", {"round": round_num, "phase": "platform_dynamics", "message": self._phase_labels["platform"], "phase_index": 5, "total_phases": 7})
        print(f"  ├─ Phase 5: Platform dynamics   [resolving engagement, feeds, follows]")
        all_agents = self._all_agents_flat()
        all_round_posts = self.platform.get_posts_by_round(round_num)
        coalitions = self.interaction_resolver.resolve_round(
            round_num, all_agents, all_round_posts
        )
        self.coalition_history.append({"round": round_num, "coalitions": coalitions})

        # === Phase 6: Opinion Dynamics ===
        await self._notify("round_phase", {"round": round_num, "phase": "opinion_dynamics", "message": f"{self._phase_labels['opinion']}: {len(all_agents)}", "phase_index": 6, "total_phases": 7})
        print(f"  ├─ Phase 6: Opinion dynamics    [updating {len(all_agents)} positions]")
        self.opinion_dynamics.update_all_agents(all_agents, self.platform, event)

        # Save agent states
        for agent in all_agents:
            self.platform.save_agent_state(
                agent.id, round_num, agent.position,
                agent.emotional_state, agent.engagement_level, ""
            )
        if not self.elite_only:
            for cluster in self.citizen_swarm.clusters.values():
                self.platform.save_agent_state(
                    cluster.id, round_num, cluster.position,
                    cluster.dominant_sentiment, cluster.engagement_level, ""
                )

        # === Phase 7: Checkpoint ===
        await self._notify("round_phase", {"round": round_num, "phase": "checkpoint", "message": self._phase_labels["checkpoint"], "phase_index": 7, "total_phases": 7})
        # Collect orchestrator state for wargame rollback
        orch_state = None
        if self.escalation_engine:
            orch_state = {
                "escalation": {
                    "current_wave": self.escalation_engine.state.current_wave,
                    "round_metrics": [
                        {"round": m.round_num, "engagement": m.avg_engagement_per_post,
                         "polarization": m.polarization, "volume": m.post_count}
                        for m in self.escalation_engine.state.round_metrics
                    ],
                },
            }
            if self.contagion_scorer:
                orch_state["contagion"] = {
                    "cri_history": list(self.contagion_scorer.cri_history),
                }
            if self.financial_scorer:
                orch_state["financial"] = {
                    "impact_history": list(self.financial_scorer.impact_history),
                }

        checkpoint = save_checkpoint(
            self.checkpoint_dir, self.scenario_name, round_num,
            self.elite_agents, self.institutional_agents,
            self.citizen_swarm, self.coalition_history,
            self.llm.stats.total_cost, self.elite_only,
            domain=getattr(self.domain, "domain_id", ""),
            confidence_interval=getattr(self, "_last_ci", None),
            regime_info=getattr(self, "_last_regime", None),
            params_used={
                k: v for k, v in self._params_info.items()
                if not k.startswith("_") or k in ("_source", "_model_version")
            } if hasattr(self, "_params_info") else None,
            orchestrator_state=orch_state,
        )
        print(f"  ├─ Phase 7: Checkpoint [saved: {checkpoint}]")

        # Domain-specific metrics
        domain_metrics = {}
        if self.domain:
            domain_metrics = self.domain.compute_domain_metrics(
                all_agents,
                list(self.citizen_swarm.clusters.values()),
                self.platform, round_num,
            )

        # LLM-evaluated custom metrics
        custom_metrics = {}
        if self.metrics_to_track:
            custom_metrics = await self._evaluate_custom_metrics(
                round_num, timeline_label, round_event,
                all_agents, coalitions, domain_metrics,
            )
            self.custom_metrics_history.append({"round": round_num, **custom_metrics})

        # Round stats
        round_stats = self.platform.get_round_stats(round_num)
        all_positions = self._all_positions()
        polarization = self.metrics.polarization_index(all_positions)
        print(f"  ├─ Stats: {round_stats['posts']} posts | "
              f"{round_stats['reactions']} reactions | "
              f"polarization: {polarization:.1f}/10")
        if custom_metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in custom_metrics.items())
            print(f"  ├─ Metrics: {metrics_str}")

        print(f"  └─ Running cost: ${self.llm.stats.total_cost:.2f}")

        # Build enriched round data for live dashboard
        top_posts_raw = self.platform.get_top_posts(round_num, top_n=10)
        top_posts_data = []
        for p in top_posts_raw:
            top_posts_data.append({
                "id": p.get("id"),
                "author_id": p.get("author_id", ""),
                "author_name": p.get("author_name", p.get("author_id", "?")),
                "platform": p.get("platform", "social"),
                "text": p.get("content", ""),
                "likes": p.get("likes", 0),
                "reposts": p.get("reposts", 0),
                "replies": p.get("reply_count", 0),
                "total_engagement": p.get("likes", 0) + p.get("reposts", 0) * 2 + p.get("reply_count", 0) * 3,
            })

        # Agent snapshot
        agents_snapshot = []
        for a in all_agents:
            snap = {
                "id": a.id,
                "name": a.name,
                "role": getattr(a, "role", ""),
                "position": round(a.position, 3),
                "emotional_state": getattr(a, "emotional_state", "neutral"),
                "tier": getattr(a, "tier", 1),
            }
            if getattr(a, "category", "") == "public_opinion":
                snap["is_synthetic"] = True
            agents_snapshot.append(snap)
        # Add citizen clusters
        if not self.elite_only:
            for c in self.citizen_swarm.clusters.values():
                agents_snapshot.append({
                    "id": c.id,
                    "name": c.name,
                    "role": "citizen_cluster",
                    "position": round(c.position, 3),
                    "emotional_state": getattr(c, "emotional_state", "neutral"),
                    "tier": 3,
                    "cluster_size": getattr(c, "population", 0),
                })

        # Sentiment distribution
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for a in agents_snapshot:
            es = a.get("emotional_state", "neutral").lower()
            if es in ("satisfied", "triumphant", "optimistic", "hopeful"):
                sentiments["positive"] += 1
            elif es in ("combative", "furious", "worried", "frustrated"):
                sentiments["negative"] += 1
            else:
                sentiments["neutral"] += 1
        total_s = max(sum(sentiments.values()), 1)
        sentiment_pcts = {k: round(v / total_s, 2) for k, v in sentiments.items()}

        # Feed orchestrator (escalation engine + contagion scorer)
        orchestrator_data = await self._feed_orchestrator(
            round_num, round_stats, polarization, sentiment_pcts, event, coalitions
        )
        if orchestrator_data:
            orch_str = (
                f"engagement={orchestrator_data.get('engagement_score', 0):.3f} "
                f"wave={orchestrator_data.get('active_wave', 1)} "
                f"CRI={orchestrator_data.get('contagion_risk_index', 0):.3f}"
            )
            print(f"  ├─ Orchestrator: {orch_str}")

        # Coalition data for frontend
        coalitions_data = []
        if isinstance(coalitions, list):
            coalitions_data = coalitions
        elif isinstance(coalitions, dict):
            coalitions_data = coalitions.get("coalitions", [])

        # Build confidence interval from posterior uncertainty
        ci95 = self._params_info.get("_ci95", {}) if hasattr(self, "_params_info") else {}
        confidence_interval = None
        if ci95:
            # Approximate pro_pct CI from parameter uncertainty
            # Use sigma_delta (model discrepancy) as primary uncertainty source
            disc = self.param_loader.get_discrepancy(
                domain=getattr(self.domain, "domain_id", None)
            )
            sigma_pp = disc["sigma_delta"] * 25  # logit → pp rough conversion
            avg_pos = sum(p for p in all_positions) / max(len(all_positions), 1) if all_positions else 0
            pro_est = (1 + avg_pos) / 2 * 100  # rough position → pro% mapping
            confidence_interval = {
                "pro_pct_mean": round(pro_est, 1),
                "pro_pct_ci95_lo": round(max(0, pro_est - 1.96 * sigma_pp), 1),
                "pro_pct_ci95_hi": round(min(100, pro_est + 1.96 * sigma_pp), 1),
                "sigma_pp": round(sigma_pp, 1),
            }

        # Regime info (defaults to normal)
        regime_info = {
            "regime_prob": round(self._regime_prob, 3),
            "regime_label": "crisis" if self._regime_prob > 0.5 else "normal",
        }

        # Estimate regime probability from shock magnitude
        shock_mag = event.get("shock_magnitude", 0)
        if self._use_regime_switching and shock_mag > 0.4:
            # Simple heuristic: large shocks push toward crisis
            self._regime_prob = min(1.0, self._regime_prob * 0.9 + 0.6 * (shock_mag - 0.3))
        else:
            self._regime_prob = max(0.0, self._regime_prob * 0.7)
        regime_info["regime_prob"] = round(self._regime_prob, 3)
        regime_info["regime_label"] = "crisis" if self._regime_prob > 0.5 else "normal"

        # Store for checkpoint (will be saved in next round's checkpoint)
        self._last_ci = confidence_interval
        self._last_regime = regime_info

        # ── Step the FinancialTwin (Sprint 1 + Sprint 2 closed loop) ─────
        # Sprint 2: opinion is now weighted by per-agent financial exposure
        # (depositors / borrowers / competitors) so that retail negativity
        # drives runoff harder than institutional moderation.
        financial_twin_state = None
        financial_feedback = None
        if self.financial_twin is not None:
            try:
                from core.financial import (
                    aggregate_opinion_by_exposure,
                    infer_financial_exposure,
                )
                opinion_aggregate = sum(all_positions) / max(len(all_positions), 1)
                # Build per-agent exposure-tagged positions
                exposure_rows = []
                for a in (self.elite_agents or []):
                    ex = infer_financial_exposure(
                        archetype=getattr(a, "archetype", ""),
                        role=getattr(a, "role", ""),
                        party_or_org=getattr(a, "domain_attributes", {}).get("party_or_org", ""),
                    )
                    exposure_rows.append({
                        "position": getattr(a, "position", 0.0),
                        "exposure": ex,
                        "weight": getattr(a, "influence", 0.5),
                    })
                # Citizens count as full retail
                if not self.elite_only and self.citizen_swarm is not None:
                    for c in self.citizen_swarm.clusters.values():
                        ex = infer_financial_exposure(archetype="citizen", role="citizen")
                        exposure_rows.append({
                            "position": getattr(c, "position", 0.0),
                            "exposure": ex,
                            "weight": float(getattr(c, "population", 100)) / 1000.0,
                        })
                exposure_weighted = aggregate_opinion_by_exposure(exposure_rows)

                shock_mag = float(event.get("shock_magnitude", 0) or 0)
                shock_dir = float(event.get("shock_direction", 0) or 0)
                rate_change_bps = shock_mag * shock_dir * 200.0
                ts = self.financial_twin.step(
                    round_num=round_num,
                    rate_change_bps=rate_change_bps,
                    opinion_aggregate=opinion_aggregate,
                    polarization=polarization,
                    narrative=str(round_event)[:160],
                    opinion_by_exposure=exposure_weighted,
                )
                financial_twin_state = ts.to_dict()
                fb = self.financial_twin.latest_feedback()
                financial_feedback = fb.to_dict()
                # Patch the checkpoint that was already saved earlier in this
                # round, so the export pipeline (which reads checkpoints) sees
                # the financial twin state. We can't move save_checkpoint
                # earlier because the twin needs post-agent-update positions.
                try:
                    import json as _json
                    safe_name = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in self.scenario_name
                    )
                    cp_path = f"{self.checkpoint_dir}/state_{safe_name}_r{round_num}.json"
                    with open(cp_path, "r") as _f:
                        _cp_data = _json.load(_f)
                    _cp_data["financial_twin"] = financial_twin_state
                    _cp_data["financial_feedback"] = financial_feedback
                    with open(cp_path, "w") as _f:
                        _json.dump(_cp_data, _f, indent=2, ensure_ascii=False)
                except Exception as _exc:
                    logger.warning(f"Could not patch checkpoint with financial twin: {_exc}")
            except Exception as exc:
                logger.warning(f"FinancialTwin step failed at round {round_num}: {exc}")

        # ── Step the TickerPriceState (real-price tracking) ────────────────
        # Uses CRI from contagion_scorer + intensity proxy from
        # escalation_engine.active_wave (1-5). Skipped silently if no
        # tickers were extracted from the brief.
        ticker_prices = {}
        if self.ticker_price_state is not None and self.ticker_price_state.tickers:
            try:
                cri_val = float(orchestrator_data.get("contagion_risk_index", 0.0) or 0.0)
                intensity_val = float(orchestrator_data.get("active_wave", 1) or 1)
                ticker_prices = self.ticker_price_state.step(
                    cri=cri_val, intensity=intensity_val,
                )
            except Exception as exc:
                logger.warning(f"TickerPriceState step failed at round {round_num}: {exc}")

        result = {
            "round": round_num,
            "timeline_label": timeline_label,
            "event": round_event,
            "posts": round_stats["posts"],
            "reactions": round_stats["reactions"],
            "polarization": polarization,
            "coalitions": coalitions,
            "domain_metrics": domain_metrics,
            "custom_metrics": custom_metrics,
            "ticker_prices": ticker_prices,
            "cost": self.llm.stats.total_cost,
            "confidence_interval": confidence_interval,
            "regime_info": regime_info,
            "orchestrator": orchestrator_data,
            # Persisted for downstream reporting (HTML report, exports)
            "top_posts": top_posts_data,
            "sentiment": sentiment_pcts,
            "shock_magnitude": event.get("shock_magnitude", 0),
            "shock_direction": event.get("shock_direction", 0),
            "financial_twin": financial_twin_state,
            "financial_feedback": financial_feedback,
        }

        # Flush any pending SQLite writes before reporting round complete so
        # subsequent readers (e.g. export) see a consistent snapshot.
        if self.platform is not None:
            try:
                await self.platform.aflush()
            except Exception:
                pass

        await self._notify("round_complete", {
            "round": round_num,
            "cost": self.llm.stats.total_cost,
            "timeline_label": timeline_label,
            "event": round_event,
            "posts_count": round_stats["posts"],
            "reactions_count": round_stats["reactions"],
            "polarization": polarization,
            "top_posts": top_posts_data,
            "agents": agents_snapshot,
            "sentiment": sentiment_pcts,
            "coalitions": coalitions_data,
            "domain_metrics": domain_metrics,
            "custom_metrics": custom_metrics,
            "ticker_prices": ticker_prices,
            "shock_magnitude": event.get("shock_magnitude", 0),
            "shock_direction": event.get("shock_direction", 0),
            "confidence_interval": confidence_interval,
            "regime_info": regime_info,
            "orchestrator": orchestrator_data,
            "calibration_source": self._params_info.get("_model_version", "v1") + "_" + self._params_info.get("_source", "unknown") if hasattr(self, "_params_info") else "v1_default",
        })
        return result

    async def _evaluate_custom_metrics(self, round_num, timeline_label, round_event,
                                          all_agents, coalitions, domain_metrics):
        """Use LLM to evaluate scenario-specific metrics for this round."""
        if not self.metrics_to_track:
            return {}

        # Build state summary for LLM
        agent_summary = []
        for a in all_agents[:15]:  # Top 15 agents
            agent_summary.append(f"- {a.name} ({getattr(a, 'role', '?')}): pos={a.position:+.2f}, mood={getattr(a, 'emotional_state', '?')}")

        viral_text = self.platform.format_viral_posts(round_num, top_n=5)

        coalition_text = ""
        coal_list = coalitions if isinstance(coalitions, list) else coalitions.get("coalitions", []) if isinstance(coalitions, dict) else []
        for c in coal_list:
            coalition_text += f"- {c.get('label', '?')}: {c.get('size', 0)} members, avg_pos={c.get('avg_position', 0):+.2f}\n"

        metrics_list = "\n".join(f'- "{m}"' for m in self.metrics_to_track)
        prev_metrics_text = ""
        if self.custom_metrics_history:
            prev = self.custom_metrics_history[-1]
            prev_metrics_text = "Previous round metrics: " + ", ".join(
                f"{k}={v}" for k, v in prev.items() if k != "round"
            )

        prompt = f"""You are evaluating a simulation scenario: "{self.scenario_name}".
Context: {self.scenario_context[:500]}

Round {round_num} ({timeline_label}):
Event: {round_event}

Key agents:
{chr(10).join(agent_summary)}

Coalitions:
{coalition_text or 'None detected'}

Viral posts this round:
{viral_text or 'None'}

{prev_metrics_text}

Evaluate these scenario-specific metrics for the current round state.
Each metric should be a number from 0 to 100 (where 0=minimum, 100=maximum).

Metrics to evaluate:
{metrics_list}

Respond with JSON only:
{{
{', '.join(f'  "{m}": <0-100>' for m in self.metrics_to_track)}
}}"""

        lang_instruction = ""
        if self.language and self.language != "en":
            lang_instruction = f"\nNote: The scenario is in {self.language}. Understand the content but respond with the JSON metrics only."
            prompt += lang_instruction

        try:
            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.3,
                max_output_tokens=500,
                component="metrics_evaluation",
            )
            # Ensure all values are numbers 0-100
            clean = {}
            for m in self.metrics_to_track:
                val = result.get(m, 50)
                if isinstance(val, (int, float)):
                    clean[m] = max(0, min(100, round(val)))
                else:
                    clean[m] = 50
            return clean
        except Exception as e:
            logger.warning(f"Custom metrics evaluation failed: {e}")
            return {m: 50 for m in self.metrics_to_track}

    # ── RAG: per-round retrieval + citation attachment ───────────────────

    def _rag_setup_round(self, round_num: int, round_event) -> str:
        """Retrieve top-K KB chunks for this round and cache them.

        Returns a context block to append to agent prompts. If rag_store is
        absent or empty, returns "" and skips citation attachment downstream.
        """
        if not getattr(self, "rag_store", None):
            return ""
        try:
            if self.rag_store.chunk_count == 0:
                return ""
        except Exception:
            return ""

        # Build the query from the round's narrative event
        if isinstance(round_event, dict):
            query = round_event.get("event") or round_event.get("title") or str(round_event)
        else:
            query = str(round_event)
        if self.scenario_context:
            query = f"{self.scenario_context}\n{query}"

        try:
            chunks = self.rag_store.retrieve(query, k=4)
        except Exception as exc:
            logger.warning(f"RAG retrieve failed for round {round_num}: {exc}")
            return ""

        if not chunks:
            self._round_retrieved[round_num] = []
            return ""

        # Cache for citation attachment on every post in this round
        self._round_retrieved[round_num] = [
            {
                "doc_id":   c.doc_id,
                "chunk_id": c.chunk_id,
                "title":    c.title,
                "snippet":  c.snippet,
                "score":    c.score,
            }
            for c in chunks
        ]

        # Build the prompt context block (snippet excerpts to ground reasoning)
        lines = ["", "RETRIEVED REFERENCE DOCUMENTS (use to ground your reasoning):"]
        for c in chunks:
            lines.append(f"  [{c.chunk_id}] {c.title} (score {c.score:.2f})")
            lines.append(f"    \"{c.snippet}\"")
        lines.append("")
        return "\n".join(lines)

    def _attach_citations(self, post_dict: dict, round_num: int) -> None:
        """Annotate a post with the round's retrieved chunks (lightweight)."""
        chunks = self._round_retrieved.get(round_num)
        if not chunks:
            return
        # Carry only the lean fields the frontend renders
        post_dict["citations"] = [
            {
                "doc_id":   c["doc_id"],
                "chunk_id": c["chunk_id"],
                "title":    c["title"],
                "score":    c["score"],
                "snippet":  c["snippet"],
            }
            for c in chunks
        ]

    async def _run_elite_agents_filtered(self, agents: list, round_num,
                                          timeline_label, round_event,
                                          viral_posts, polarization, avg_sentiment,
                                          top_narratives, prompt_template,
                                          channel_descs, channel_max_lens):
        """Run elite agents — accepts filtered agent list from orchestrator."""
        tasks = [
            agent.generate_round(
                self.llm, round_num, timeline_label, round_event,
                viral_posts, polarization, avg_sentiment, top_narratives,
                prompt_template, channel_descs, channel_max_lens,
            )
            for agent in agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed = []
        errors = 0
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                errors += 1
                print(f"    ⚠ Elite agent {agents[i].name} failed: {r}")
                logger.error(f"Elite agent {agents[i].id} error: {r}")
                processed.append(None)
            else:
                posts_count = len(r.get("posts", [])) if r else 0
                if r and posts_count == 0:
                    print(f"    ⚠ Elite agent {agents[i].name} returned 0 posts")
                processed.append(r)
        if errors:
            print(f"    ⚠ {errors}/{len(agents)} elite agents failed!")
            if errors == len(agents):
                await self._notify("round_phase", {
                    "round": round_num,
                    "phase": "warning",
                    "message": f"Tutti gli agenti elite hanno fallito ({errors}/{len(agents)}). Possibile problema LLM.",
                    "phase_index": 2,
                    "total_phases": 7,
                })
                # Generate minimal fallback results to prevent downstream corruption
                # (0 posts → abnormal engagement → math overflow in financial scoring)
                for idx, agent in enumerate(agents):
                    if processed[idx] is None:
                        processed[idx] = {
                            "agent_id": agent.id,
                            "posts": [],
                            "position": agent.position,
                            "emotional_state": agent.emotional_state,
                            "strategic_move": "",
                            "alliances": [],
                            "targets": [],
                            "position_reasoning": "[LLM failure — position unchanged]",
                        }
        return processed

    # Keep old method for backward compatibility
    async def _run_elite_agents(self, round_num, timeline_label, round_event,
                                 viral_posts, polarization, avg_sentiment,
                                 top_narratives, prompt_template,
                                 channel_descs, channel_max_lens):
        return await self._run_elite_agents_filtered(
            self.elite_agents, round_num, timeline_label, round_event,
            viral_posts, polarization, avg_sentiment, top_narratives,
            prompt_template, channel_descs, channel_max_lens,
        )

    async def _run_institutional_agents_filtered(self, agents: list,
                                                  round_num, timeline_label,
                                                  round_event, viral_posts,
                                                  prompt_template, channel_max_lens,
                                                  profile_template):
        """Run institutional agents — accepts filtered agent list from orchestrator."""
        # Exclude passive agents (e.g., Public Opinion) from LLM batch
        active_agents = [a for a in agents if a.category != "public_opinion"]
        batches = []
        batch_size = 10
        for i in range(0, len(active_agents), batch_size):
            batch = active_agents[i:i + batch_size]
            batches.append(batch)

        tasks = [
            process_institutional_batch(
                batch, self.llm, round_num, timeline_label, round_event,
                viral_posts, prompt_template, channel_max_lens, profile_template,
            )
            for batch in batches
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # Keep old method for backward compatibility
    async def _run_institutional_agents(self, round_num, timeline_label,
                                         round_event, viral_posts,
                                         prompt_template, channel_max_lens,
                                         profile_template):
        return await self._run_institutional_agents_filtered(
            self.institutional_agents, round_num, timeline_label,
            round_event, viral_posts, prompt_template, channel_max_lens,
            profile_template,
        )
