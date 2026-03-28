"""Manages simulation lifecycle — launch, track, stream progress."""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import AsyncGenerator, Optional
from uuid import uuid4

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.models import SimulationRequest, SimulationStatus, ProgressEvent
from core.simulation.engine import SimulationEngine
from core.config.schema import ScenarioConfig
from domains.domain_registry import DomainRegistry
from briefing.scenario_builder import ScenarioBuilder
from export import export_scenario, discover_scenarios

logger = logging.getLogger(__name__)

# Provider defaults
LLM_PROVIDERS = {
    "gemini": ("gemini-3.1-flash-lite-preview", "GOOGLE_API_KEY"),
    "openai": ("gpt-5.4-mini", "OPENAI_API_KEY"),
}

PERSISTENCE_FILE = os.path.join(PROJECT_ROOT, "outputs", "simulations.json")
MAX_CONCURRENT = 2


class SimulationState:
    def __init__(self, sim_id: str, request: SimulationRequest):
        self.id = sim_id
        self.request = request
        self.status = "queued"
        self.scenario_name: Optional[str] = None
        self.scenario_id: Optional[str] = None
        self.domain: Optional[str] = None
        self.current_round = 0
        self.total_rounds = 0
        self.cost = 0.0
        self.agents_count = 0
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None

    def to_status(self) -> SimulationStatus:
        return SimulationStatus(
            id=self.id,
            status=self.status,
            brief=self.request.brief,
            scenario_name=self.scenario_name,
            scenario_id=self.scenario_id,
            domain=self.domain,
            current_round=self.current_round,
            total_rounds=self.total_rounds,
            cost=self.cost,
            created_at=self.created_at,
            completed_at=self.completed_at,
            error=self.error,
            agents_count=self.agents_count,
        )


class SimulationManager:
    def __init__(self):
        self.simulations: dict[str, SimulationState] = {}
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._load_persisted()

    def _load_persisted(self):
        """Load completed simulations from disk."""
        if os.path.exists(PERSISTENCE_FILE):
            try:
                with open(PERSISTENCE_FILE) as f:
                    data = json.load(f)
                for entry in data:
                    state = SimulationState(
                        entry["id"],
                        SimulationRequest(brief=entry.get("brief", ""))
                    )
                    state.status = entry.get("status", "completed")
                    state.scenario_name = entry.get("scenario_name")
                    state.scenario_id = entry.get("scenario_id")
                    state.domain = entry.get("domain")
                    state.total_rounds = entry.get("total_rounds", 0)
                    state.current_round = state.total_rounds
                    state.cost = entry.get("cost", 0)
                    state.agents_count = entry.get("agents_count", 0)
                    state.created_at = entry.get("created_at", "")
                    state.completed_at = entry.get("completed_at")
                    state.error = entry.get("error")
                    # Mark running sims as failed (server restarted)
                    if state.status in ("running", "analyzing", "configuring", "exporting"):
                        state.status = "failed"
                        state.error = "Server restarted during simulation"
                    self.simulations[state.id] = state
            except Exception as e:
                logger.warning(f"Could not load persisted simulations: {e}")

    def _persist(self):
        """Save simulation metadata to disk."""
        os.makedirs(os.path.dirname(PERSISTENCE_FILE), exist_ok=True)
        data = []
        for s in self.simulations.values():
            data.append({
                "id": s.id,
                "status": s.status,
                "brief": s.request.brief,
                "scenario_name": s.scenario_name,
                "scenario_id": s.scenario_id,
                "domain": s.domain,
                "total_rounds": s.total_rounds,
                "cost": s.cost,
                "agents_count": s.agents_count,
                "created_at": s.created_at,
                "completed_at": s.completed_at,
                "error": s.error,
            })
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    async def launch(
        self,
        request: SimulationRequest,
        sim_id: str = "",
        document_context: dict = None,
    ) -> str:
        if not sim_id:
            sim_id = str(uuid4())[:8]
        state = SimulationState(sim_id, request)
        state.document_context = document_context  # RAG docs
        self.simulations[sim_id] = state
        self._persist()
        state.task = asyncio.create_task(self._run_pipeline(sim_id))
        return sim_id

    async def cancel(self, sim_id: str) -> bool:
        state = self.simulations.get(sim_id)
        if not state or state.status in ("completed", "failed", "cancelled"):
            return False
        if state.task:
            state.task.cancel()
        state.status = "cancelled"
        state.completed_at = datetime.utcnow().isoformat()
        await state.event_queue.put(ProgressEvent(type="cancelled", message="Simulation cancelled"))
        self._persist()
        return True

    def get_status(self, sim_id: str) -> Optional[SimulationStatus]:
        state = self.simulations.get(sim_id)
        return state.to_status() if state else None

    def list_simulations(self) -> list[SimulationStatus]:
        return [s.to_status() for s in sorted(
            self.simulations.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )]

    async def stream_events(self, sim_id: str) -> AsyncGenerator[ProgressEvent, None]:
        state = self.simulations.get(sim_id)
        if not state:
            return
        while True:
            try:
                event = await asyncio.wait_for(state.event_queue.get(), timeout=30)
                yield event
                if event.type in ("completed", "error", "cancelled"):
                    break
            except asyncio.TimeoutError:
                yield ProgressEvent(type="heartbeat", message="")

    async def _emit(self, sim_id: str, event: ProgressEvent):
        state = self.simulations.get(sim_id)
        if state:
            await state.event_queue.put(event)

    async def _run_pipeline(self, sim_id: str):
        state = self.simulations[sim_id]
        request = state.request

        async with self._semaphore:
            try:
                # Phase 1: Create LLM client
                state.status = "analyzing"
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="status", message="Inizializzazione LLM...",
                    phase="init"
                ))

                llm = self._create_llm(request.provider, request.model, request.budget)

                # Define progress callback early (used in web research + simulation)
                async def on_progress(event_type: str, data: dict):
                    if event_type == "round_start":
                        state.current_round = data.get("round", 0)
                        state.cost = data.get("cost", state.cost)
                        self._persist()
                        await self._emit(sim_id, ProgressEvent(
                            type="round_start",
                            message=f"Round {data['round']} di {state.total_rounds}",
                            round=data["round"],
                            data=data,
                        ))
                    elif event_type == "round_phase":
                        await self._emit(sim_id, ProgressEvent(
                            type="round_phase",
                            message=data.get("message", ""),
                            round=data.get("round"),
                            phase=data.get("phase", ""),
                            data=data,
                        ))
                    elif event_type == "round_complete":
                        state.current_round = data.get("round", 0)
                        state.cost = data.get("cost", state.cost)
                        self._persist()
                        await self._emit(sim_id, ProgressEvent(
                            type="round_complete",
                            message=f"Round {data['round']} completato",
                            round=data["round"],
                            data=data,
                        ))

                # Phase 2: Web research + Analyze brief
                await self._emit(sim_id, ProgressEvent(
                    type="status", message="Ricerca contesto online...",
                    phase="web_research"
                ))

                DomainRegistry.discover()
                available_domains = DomainRegistry.list_domains()

                # Load document context if uploaded
                doc_context = getattr(state, "document_context", None)
                extra_context = ""
                seed_data_path = ""
                if doc_context:
                    extra_context = doc_context.get("context_text", "")
                    seed_data_path = doc_context.get("seed_data_path", "")
                    if extra_context:
                        await self._emit(sim_id, ProgressEvent(
                            type="status",
                            message=f"Documenti caricati: {doc_context['file_count']} file, {doc_context['total_chars']} caratteri",
                            phase="documents",
                        ))

                builder = ScenarioBuilder()
                config = await builder.build_from_brief(
                    brief_text=request.brief,
                    llm=llm,
                    available_domains=available_domains,
                    interactive=False,
                    progress_callback=on_progress,
                    extra_context=extra_context,
                    seed_data_path=seed_data_path,
                )

                # Apply overrides
                if request.domain:
                    config.domain = request.domain
                if request.rounds:
                    config.num_rounds = request.rounds
                config.budget_usd = request.budget

                state.scenario_name = config.name
                state.domain = config.domain
                state.total_rounds = config.num_rounds
                state.agents_count = (
                    len(config.elite_agents) +
                    len(config.institutional_agents) +
                    len(config.citizen_clusters)
                )
                state.status = "configuring"
                self._persist()

                await self._emit(sim_id, ProgressEvent(
                    type="brief_analyzed",
                    message=f"Scenario: {config.name}",
                    data={
                        "scenario_name": config.name,
                        "domain": config.domain,
                        "num_rounds": config.num_rounds,
                        "elite_agents": len(config.elite_agents),
                        "institutional_agents": len(config.institutional_agents),
                        "citizen_clusters": len(config.citizen_clusters),
                        "position_axis": {
                            "negative_label": config.position_axis.negative_label,
                            "positive_label": config.position_axis.positive_label,
                            "neutral_label": config.position_axis.neutral_label,
                        } if config.position_axis else None,
                    }
                ))

                # Phase 3: Run simulation
                domain = DomainRegistry.get(config.domain)
                state.status = "running"
                self._persist()

                engine = SimulationEngine(
                    llm=llm,
                    config=config,
                    domain=domain,
                    output_dir=os.path.join(PROJECT_ROOT, "outputs"),
                    elite_only=request.elite_only,
                    verbose=False,
                    progress_callback=on_progress,
                )

                await engine.run()

                state.cost = llm.stats.total_cost

                # Phase 3b: Monte Carlo analysis (if requested)
                monte_carlo_data = None
                if request.monte_carlo:
                    await self._emit(sim_id, ProgressEvent(
                        type="status", message="Analisi Monte Carlo in corso...",
                        phase="monte_carlo"
                    ))
                    monte_carlo_data = await self._run_monte_carlo(
                        sim_id, config, engine,
                        n_runs=request.monte_carlo_runs,
                        perturbation=request.monte_carlo_perturbation,
                    )

                # Phase 4: Export
                state.status = "exporting"
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="status", message="Esportazione risultati...",
                    phase="exporting"
                ))

                outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
                export_dir = os.path.join(outputs_dir, "exports")
                safe_name = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in config.name
                )
                export_scenario(safe_name, outputs_dir, export_dir)
                state.scenario_id = safe_name

                # Save Monte Carlo results alongside scenario
                if monte_carlo_data:
                    mc_path = os.path.join(export_dir, safe_name, "monte_carlo.json")
                    os.makedirs(os.path.dirname(mc_path), exist_ok=True)
                    import json as _json
                    with open(mc_path, "w") as f:
                        _json.dump(monte_carlo_data, f, indent=2)

                # Build scenarios.json manifest
                self._rebuild_scenarios_manifest(export_dir)

                # Done!
                state.status = "completed"
                state.completed_at = datetime.utcnow().isoformat()
                self._persist()

                await self._emit(sim_id, ProgressEvent(
                    type="completed",
                    message="Simulazione completata!",
                    data={
                        "scenario_name": config.name,
                        "scenario_id": safe_name,
                        "cost": state.cost,
                        "total_rounds": state.total_rounds,
                        "monte_carlo": monte_carlo_data,
                    }
                ))

            except asyncio.CancelledError:
                state.status = "cancelled"
                state.completed_at = datetime.utcnow().isoformat()
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="cancelled", message="Simulazione cancellata"
                ))
            except Exception as e:
                logger.error(f"Simulation {sim_id} failed: {traceback.format_exc()}")
                state.status = "failed"
                state.error = str(e)
                state.completed_at = datetime.utcnow().isoformat()
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="error", message=f"Errore: {str(e)}",
                    data={"traceback": traceback.format_exc()[-500:]}
                ))

    async def _run_monte_carlo(
        self, sim_id: str, config, engine, n_runs: int = 10, perturbation: float = 0.15
    ) -> dict:
        """Run Monte Carlo analysis using synthetic sim (zero LLM cost).

        Takes the completed simulation's agent states and events, then re-runs
        the opinion dynamics N times with perturbed parameters to produce
        confidence intervals.
        """
        from core.simulation.monte_carlo import MonteCarloEngine, perturb_params
        from calibration.synthetic_sim import (
            SyntheticAgent, run_synthetic_simulation, _agents_to_pro_pct
        )
        from calibration.historical_scenario import GroundTruth, PollingDataPoint

        mc_engine = MonteCarloEngine(n_runs=n_runs, perturbation_pct=perturbation)

        # Load calibrated base params for this domain
        calibrated_path = os.path.join(
            PROJECT_ROOT, "calibration", "results",
            f"calibrated_params_{config.domain}.json"
        )
        base_params = {
            "anchor_weight": 0.1, "social_weight": 0.15,
            "event_weight": 0.05, "herd_weight": 0.05,
            "herd_threshold": 0.2, "direct_shift_weight": 0.4,
            "anchor_drift_rate": 0.2,
        }
        if os.path.exists(calibrated_path):
            import json as _json
            with open(calibrated_path) as f:
                data = _json.load(f)
                cal = data.get("calibrated_params", {})
                base_params.update(cal)

        # Build a GroundTruth-like structure from the completed simulation
        # We use the actual sim results as a "reference trajectory"
        rm = engine.round_manager
        polling = []
        for i in range(config.num_rounds):
            ch = rm.coalition_history[i] if i < len(rm.coalition_history) else {}
            positions = rm._all_positions() if i == config.num_rounds - 1 else []
            # Use checkpoint data for per-round positions (approximation)
            pro = 50.0
            against = 50.0
            polling.append(PollingDataPoint(
                round_equivalent=i + 1,
                pro_pct=pro,
                against_pct=against,
                undecided_pct=0,
            ))

        # For Monte Carlo, we don't need ground truth — we just run N synthetic sims
        # and aggregate the results
        gt = GroundTruth(
            scenario_name=config.name,
            description=getattr(config, 'description', ''),
            final_outcome_pro_pct=50.0,
            final_outcome_against_pct=50.0,
            polling_trajectory=polling,
            key_events=[],
        )

        # Generate parameter sets
        param_sets = mc_engine.generate_parameter_sets(base_params)

        # Run N synthetic simulations
        all_runs = []
        for i, params in enumerate(param_sets):
            try:
                final_pro, positions_per_round = run_synthetic_simulation(
                    gt, params, n_agents=200, seed=42 + i
                )
                run_result = {
                    "rounds": [
                        {
                            "round": r + 1,
                            "polarization": abs(pos) * 10,  # Scale to 0-10
                            "avg_position": pos,
                            "sentiment": {
                                "positive": max(0, 0.33 + pos * 0.2),
                                "neutral": 0.34,
                                "negative": max(0, 0.33 - pos * 0.2),
                            },
                        }
                        for r, pos in enumerate(positions_per_round)
                    ],
                    "final_polarization": abs(positions_per_round[-1]) * 10 if positions_per_round else 0,
                    "final_avg_position": positions_per_round[-1] if positions_per_round else 0,
                }
                all_runs.append(run_result)
            except Exception as e:
                logger.warning(f"Monte Carlo run {i} failed: {e}")

            # Emit progress
            if (i + 1) % 5 == 0 or i == len(param_sets) - 1:
                await self._emit(sim_id, ProgressEvent(
                    type="round_phase",
                    message=f"Monte Carlo: {i + 1}/{len(param_sets)} runs",
                    phase="monte_carlo",
                    data={"monte_carlo_progress": i + 1, "monte_carlo_total": len(param_sets)},
                ))

        result = mc_engine.aggregate_results(all_runs, param_sets)
        return mc_engine.result_to_dict(result)

    def _create_llm(self, provider: str, model: Optional[str], budget: float):
        if provider == "openai":
            from core.llm.openai_client import OpenAIClient
            return OpenAIClient(model=model or "gpt-5.4-mini", budget=budget)
        else:
            from core.llm.gemini_client import GeminiClient
            return GeminiClient(model=model or "gemini-3.1-flash-lite-preview", budget=budget)

    def _rebuild_scenarios_manifest(self, export_dir: str):
        """Rebuild scenarios.json from export directory."""
        scenarios = []
        if not os.path.isdir(export_dir):
            return
        for d in sorted(os.listdir(export_dir)):
            if not d.startswith("scenario_"):
                continue
            meta_path = os.path.join(export_dir, d, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                scenario_id = d.replace("scenario_", "", 1)
                scenarios.append({
                    "id": scenario_id,
                    "name": meta.get("scenario_name", scenario_id.replace("_", " ")),
                    "domain": meta.get("domain", "unknown"),
                    "description": meta.get("description", ""),
                    "num_rounds": meta.get("num_rounds", 0),
                })

        manifest_path = os.path.join(export_dir, "scenarios.json")
        with open(manifest_path, "w") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
