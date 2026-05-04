"""Manages simulation lifecycle — launch, track, stream progress."""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import AsyncGenerator, Optional
from uuid import uuid4

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.models import SimulationRequest, SimulationStatus, ProgressEvent, BranchRequest, WargameIntervention
from api import job_queue
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
MAX_CONCURRENT = int(os.getenv("DTS_MAX_CONCURRENT", "4"))

# Bounded SSE queue per simulation. If a client disconnects or lags, events
# accumulate here — without a cap, a single stuck consumer can eat GB of RAM
# in a long simulation. On overflow we drop the OLDEST event (so the latest
# state still reaches the client once it recovers) and emit a drop marker.
SSE_QUEUE_MAX = int(os.getenv("DTS_SSE_QUEUE_MAX", "512"))

# Throttle disk persistence: _persist() rewrites the full simulations.json
# every call, which under N concurrent sims with 10+ events per round turns
# into O(N * events) rewrites. Debounce to this interval.
PERSIST_MIN_INTERVAL_S = float(os.getenv("DTS_PERSIST_INTERVAL", "0.5"))


class SimulationState:
    def __init__(self, sim_id: str, request: SimulationRequest, tenant_id: str = "default"):
        self.id = sim_id
        self.request = request
        self.tenant_id = tenant_id
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
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAX)
        self.events_dropped: int = 0
        self.task: Optional[asyncio.Task] = None

        # Wargame state
        self.wargame_mode: bool = getattr(request, "wargame_mode", False)
        self.player_role: str = getattr(request, "player_role", "")
        self._wargame_resume: asyncio.Event = asyncio.Event()
        self._wargame_intervention: Optional[WargameIntervention] = None
        self._wargame_sitrep: Optional[dict] = None

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
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)  # local fallback
        self._persist_lock = asyncio.Lock()
        self._manifest_lock = asyncio.Lock()
        self._last_persist_ts: float = 0.0
        self._persist_dirty: bool = False
        self._safe_name_cache: dict[str, str] = {}
        # Sync fallback at boot — Postgres rehydration runs in initialize()
        # called from FastAPI startup hook (db calls require an event loop).
        self._load_persisted()

    async def initialize(self):
        """Async startup: rehydrate from Postgres if DATABASE_URL is set.
        Must be called from a FastAPI startup hook, not __init__."""
        try:
            from api import db
            if not db.is_available():
                logger.info("DATABASE_URL not set — using JSON file persistence")
                return
            # Rebuild in-memory state from DB. DB takes precedence over the
            # JSON snapshot: if both exist, DB wins (it's append-only true state).
            await db.mark_running_as_failed()
            rows = await db.list_simulations()
            # First-run migration: DB exists but empty, in-memory state has
            # data from the old JSON file → push it into DB instead of wiping.
            if not rows and self.simulations:
                logger.info(f"Postgres empty, migrating {len(self.simulations)} sims from JSON")
                for s in self.simulations.values():
                    try:
                        await db.upsert_simulation({
                            "id": s.id, "tenant_id": s.tenant_id, "status": s.status,
                            "brief": s.request.brief, "scenario_name": s.scenario_name,
                            "scenario_id": s.scenario_id, "domain": s.domain,
                            "current_round": s.current_round, "total_rounds": s.total_rounds,
                            "cost": s.cost, "agents_count": s.agents_count,
                            "created_at": s.created_at, "completed_at": s.completed_at,
                            "error": s.error,
                            "wargame_mode": getattr(s, "wargame_mode", False),
                            "player_role": getattr(s, "player_role", ""),
                            "wargame_sitrep": getattr(s, "_wargame_sitrep", None),
                        })
                    except Exception as ex:
                        logger.warning(f"Migrate {s.id} failed: {ex}")
                logger.info("JSON→DB migration complete")
                return  # in-memory state preserved
            self.simulations.clear()
            for row in rows:
                req = SimulationRequest(brief=row.get("brief") or "")
                if row.get("wargame_mode"):
                    try:
                        req.wargame_mode = True
                        req.player_role = row.get("player_role") or ""
                    except Exception:
                        pass
                state = SimulationState(
                    row["id"], req,
                    tenant_id=row.get("tenant_id", "default"),
                )
                state.status = row.get("status") or "completed"
                state.scenario_name = row.get("scenario_name")
                state.scenario_id = row.get("scenario_id")
                state.domain = row.get("domain")
                state.total_rounds = row.get("total_rounds") or 0
                state.current_round = row.get("current_round") or 0
                state.cost = float(row.get("cost") or 0)
                state.agents_count = row.get("agents_count") or 0
                ca = row.get("created_at")
                state.created_at = ca.isoformat() if hasattr(ca, "isoformat") else (ca or "")
                cm = row.get("completed_at")
                state.completed_at = cm.isoformat() if hasattr(cm, "isoformat") else cm
                state.error = row.get("error")
                sitrep = row.get("wargame_sitrep")
                if sitrep:
                    state._wargame_sitrep = sitrep
                if state.status == "awaiting_player" and row.get("wargame_mode"):
                    state._restored_after_restart = True
                self.simulations[state.id] = state
            logger.info(f"Postgres rehydrate: {len(self.simulations)} simulations loaded")
        except Exception as e:
            logger.warning(f"Postgres rehydrate failed (continuing with JSON state): {e}")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    def _make_safe_name(self, config_name: str, sim_id: str) -> str:
        """Compute a unique directory-safe scenario name.

        Previously multiple scenarios with colliding sanitized names could
        stomp on each other's export directories. We append a short hash of
        the sim_id to guarantee uniqueness while keeping the name readable.
        """
        cached = self._safe_name_cache.get(sim_id)
        if cached is not None:
            return cached
        base = self._sanitize_name(config_name).strip("_") or "scenario"
        suffix = hashlib.sha1(sim_id.encode("utf-8")).hexdigest()[:6]
        safe = f"{base}__{suffix}"
        self._safe_name_cache[sim_id] = safe
        return safe

    def _load_persisted(self):
        """Load completed simulations from disk."""
        if os.path.exists(PERSISTENCE_FILE):
            try:
                with open(PERSISTENCE_FILE) as f:
                    data = json.load(f)
                for entry in data:
                    # Restore wargame fields by faking a request that carries
                    # them, so SimulationState.__init__ picks them up.
                    req = SimulationRequest(brief=entry.get("brief", ""))
                    if entry.get("wargame_mode"):
                        try:
                            req.wargame_mode = True
                            req.player_role = entry.get("player_role", "")
                        except Exception:
                            pass
                    state = SimulationState(
                        entry["id"], req,
                        tenant_id=entry.get("tenant_id", "default"),
                    )
                    state.status = entry.get("status", "completed")
                    state.scenario_name = entry.get("scenario_name")
                    state.scenario_id = entry.get("scenario_id")
                    state.domain = entry.get("domain")
                    state.total_rounds = entry.get("total_rounds", 0)
                    state.current_round = entry.get("current_round", state.total_rounds)
                    state.cost = entry.get("cost", 0)
                    state.agents_count = entry.get("agents_count", 0)
                    state.created_at = entry.get("created_at", "")
                    state.completed_at = entry.get("completed_at")
                    state.error = entry.get("error")
                    # Wargame: restore sitrep so the UI can show the last
                    # awaiting_player snapshot even after a container restart.
                    sitrep = entry.get("wargame_sitrep")
                    if sitrep:
                        state._wargame_sitrep = sitrep
                    # Wargame paused at awaiting_player: keep the status so the
                    # UI shows the SITREP instead of a blank "failed" screen.
                    # The submit_intervention endpoint will return a clear 410
                    # because state._restored_after_restart is set below.
                    if (state.status == "awaiting_player"
                            and entry.get("wargame_mode")):
                        state._restored_after_restart = True
                    elif state.status in ("running", "analyzing",
                                          "configuring", "exporting"):
                        state.status = "failed"
                        state.error = "Server restarted during simulation"
                    self.simulations[state.id] = state
            except Exception as e:
                logger.warning(f"Could not load persisted simulations: {e}")

    def _persist(self):
        """Save simulation metadata to disk (throttled).

        Events fire per-round-start / per-round-complete / per-phase, and each
        call used to serialize the FULL simulations.json and fsync. Under
        concurrent load this dominated wall-time. We now coalesce writes:
        if the last flush was <PERSIST_MIN_INTERVAL_S ago, set a dirty flag
        and let a background flush (or the next eligible call) catch up.
        Terminal transitions (completed/failed/cancelled) always force-flush.
        """
        self._persist_dirty = True
        now = time.time()
        force = any(
            (s.status in ("completed", "failed", "cancelled") and s.completed_at)
            # Wargame paused → flush immediately so a container restart during
            # the human-thinking window doesn't lose the SITREP.
            or s.status == "awaiting_player"
            for s in self.simulations.values()
        )
        if not force and (now - self._last_persist_ts) < PERSIST_MIN_INTERVAL_S:
            return
        self._flush_persist()

    def _persist_to_db_fire_and_forget(self, entries: list[dict]):
        """Async upsert to Postgres. Called from sync code via create_task."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No event loop (e.g. unit tests) → skip DB, file is enough
        try:
            from api import db
            if not db.is_available():
                return
            async def _upsert_all():
                ok, err = 0, 0
                for e in entries:
                    try:
                        await db.upsert_simulation(e)
                        ok += 1
                    except Exception as ex:
                        err += 1
                        logger.warning(f"DB upsert failed for {e.get('id')}: {ex}")
                if err:
                    logger.warning(f"DB persist batch: {ok} ok, {err} failed")
                else:
                    logger.debug(f"DB persist batch: {ok} sims upserted")
            loop.create_task(_upsert_all())
        except Exception as e:
            logger.warning(f"DB persist scheduling failed: {e}")

    def _flush_persist(self):
        """Write simulations.json + Postgres now. Safe to call from any thread."""
        os.makedirs(os.path.dirname(PERSISTENCE_FILE), exist_ok=True)
        data = []
        for s in self.simulations.values():
            entry = {
                "id": s.id,
                "tenant_id": s.tenant_id,
                "status": s.status,
                "brief": s.request.brief,
                "scenario_name": s.scenario_name,
                "scenario_id": s.scenario_id,
                "domain": s.domain,
                "total_rounds": s.total_rounds,
                "current_round": s.current_round,
                "cost": s.cost,
                "agents_count": s.agents_count,
                "created_at": s.created_at,
                "completed_at": s.completed_at,
                "error": s.error,
            }
            # Wargame fields — let the UI rehydrate after a container restart.
            if getattr(s, "wargame_mode", False):
                entry["wargame_mode"] = True
                entry["player_role"] = getattr(s, "player_role", "")
                sitrep = getattr(s, "_wargame_sitrep", None)
                if sitrep:
                    entry["wargame_sitrep"] = sitrep
            data.append(entry)
        tmp_path = PERSISTENCE_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, PERSISTENCE_FILE)
        self._last_persist_ts = time.time()
        self._persist_dirty = False
        # Mirror to Postgres (no-op if DATABASE_URL unset)
        self._persist_to_db_fire_and_forget(data)

    async def launch(
        self,
        request: SimulationRequest,
        sim_id: str = "",
        document_context: dict = None,
        rag_store=None,
        tenant_id: str = "default",
    ) -> str:
        if not sim_id:
            sim_id = str(uuid4())[:8]
        state = SimulationState(sim_id, request, tenant_id=tenant_id)
        state.document_context = document_context  # RAG raw context (legacy)
        state.rag_store = rag_store                # RAGStore — agent retrieval at round time
        self.simulations[sim_id] = state
        self._persist()
        state.task = asyncio.create_task(self._run_pipeline(sim_id))
        return sim_id

    async def launch_branch(self, request: BranchRequest, tenant_id: str = "default") -> str:
        """Launch a What-If branch from an existing scenario."""
        sim_id = str(uuid4())[:8]

        # Create a dummy SimulationRequest for state tracking
        sim_request = SimulationRequest(
            brief=f"What-If branch from {request.parent_scenario_id} @ round {request.branch_round}: {request.what_if}",
            provider=request.provider,
            model=request.model,
            budget=request.budget,
        )
        state = SimulationState(sim_id, sim_request, tenant_id=tenant_id)
        state.document_context = None
        self.simulations[sim_id] = state
        self._persist()

        # Store branch info on state
        state._branch_request = request
        state.task = asyncio.create_task(self._run_branch_pipeline(sim_id))
        return sim_id

    async def _run_branch_pipeline(self, sim_id: str):
        """Run a What-If branch simulation."""
        state = self.simulations[sim_id]
        branch: BranchRequest = state._branch_request

        async with job_queue.acquire(sim_id):
            try:
                state.status = "analyzing"
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="status",
                    message=f"Caricamento checkpoint round {branch.branch_round}...",
                    phase="init",
                ))

                # 1. Load parent checkpoint
                from core.simulation.checkpoint import load_checkpoint, find_checkpoint
                outputs_dir = os.path.join(PROJECT_ROOT, "outputs")

                # Find parent scenario name from exports
                export_dir = os.path.join(outputs_dir, "exports")
                parent_meta_path = os.path.join(
                    export_dir, f"scenario_{branch.parent_scenario_id}", "metadata.json"
                )
                if not os.path.exists(parent_meta_path):
                    raise FileNotFoundError(f"Parent scenario not found: {branch.parent_scenario_id}")

                with open(parent_meta_path) as f:
                    parent_meta = json.load(f)

                scenario_name = parent_meta["scenario_name"]
                cp_path = find_checkpoint(outputs_dir, scenario_name, branch.branch_round)
                checkpoint = load_checkpoint(cp_path)

                await self._emit(sim_id, ProgressEvent(
                    type="status",
                    message=f"Checkpoint caricato: {scenario_name} round {branch.branch_round}",
                    phase="checkpoint_loaded",
                ))

                # 2. Create LLM
                llm = self._create_llm(branch.provider, branch.model, branch.budget)

                # 3. Load parent config
                parent_config_path = os.path.join(
                    export_dir, f"scenario_{branch.parent_scenario_id}", "metadata.json"
                )
                # Rebuild config from metadata
                from briefing.scenario_builder import ScenarioBuilder
                builder = ScenarioBuilder()
                # Load the YAML config if exists, otherwise build minimal config
                config_yaml_path = os.path.join(outputs_dir, f"{branch.parent_scenario_id}_config.yaml")
                if os.path.exists(config_yaml_path):
                    config = builder.build_from_file(config_yaml_path)
                else:
                    # Build minimal config from metadata
                    config = self._config_from_metadata(parent_meta)

                # Apply branch overrides
                total_rounds = config.num_rounds
                if branch.rounds_to_run:
                    total_rounds = branch.branch_round + branch.rounds_to_run
                config.num_rounds = total_rounds

                # Update scenario name for branch
                branch_label = branch.what_if[:50] if branch.what_if else f"branch_r{branch.branch_round}"
                safe_label = "".join(c if c.isalnum() or c in "-_ " else "_" for c in branch_label).strip()
                config.name = f"{scenario_name} [What-If: {safe_label}]"

                # Inject what-if context into scenario context
                if branch.what_if:
                    config.scenario_context = (
                        f"{config.scenario_context}\n\n"
                        f"WHAT-IF SCENARIO: Starting from round {branch.branch_round}, "
                        f"the following change occurs: {branch.what_if}"
                    )

                state.scenario_name = config.name
                state.domain = config.domain
                state.total_rounds = config.num_rounds - branch.branch_round
                state.agents_count = (
                    len(checkpoint.get("elite_agents", [])) +
                    len(checkpoint.get("institutional_agents", [])) +
                    len(checkpoint.get("citizen_clusters", []))
                )
                state.status = "configuring"
                self._persist()

                await self._emit(sim_id, ProgressEvent(
                    type="brief_analyzed",
                    message=f"What-If: {config.name}",
                    data={
                        "scenario_name": config.name,
                        "domain": config.domain,
                        "num_rounds": state.total_rounds,
                        "elite_agents": len(checkpoint.get("elite_agents", [])),
                        "institutional_agents": len(checkpoint.get("institutional_agents", [])),
                        "citizen_clusters": len(checkpoint.get("citizen_clusters", [])),
                        "branch_from": branch.parent_scenario_id,
                        "branch_round": branch.branch_round,
                        "position_axis": {
                            "negative_label": config.position_axis.negative_label,
                            "positive_label": config.position_axis.positive_label,
                            "neutral_label": config.position_axis.neutral_label,
                        } if config.position_axis else None,
                    }
                ))

                # 4. Define progress callback
                async def on_progress(event_type: str, data: dict):
                    if event_type == "round_start":
                        state.current_round = data.get("round", 0) - branch.branch_round
                        state.cost = data.get("cost", state.cost)
                        self._persist()
                        await self._emit(sim_id, ProgressEvent(
                            type="round_start",
                            message=f"Round {data['round']} (branch +{state.current_round})",
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
                        state.current_round = data.get("round", 0) - branch.branch_round
                        state.cost = data.get("cost", state.cost)
                        self._persist()
                        await self._emit(sim_id, ProgressEvent(
                            type="round_complete",
                            message=f"Round {data['round']} completato (branch +{state.current_round})",
                            round=data["round"],
                            data=data,
                        ))

                # 5. Run simulation from checkpoint
                DomainRegistry.discover()
                domain = DomainRegistry.get(config.domain)

                state.status = "running"
                self._persist()

                engine = SimulationEngine(
                    llm=llm,
                    config=config,
                    domain=domain,
                    output_dir=outputs_dir,
                    elite_only=False,
                    verbose=False,
                    progress_callback=on_progress,
                    resume_checkpoint=checkpoint,
                    resume_round=branch.branch_round,
                    event_override=branch.event_override,
                    agent_overrides=branch.agent_overrides,
                )

                await engine.run()
                state.cost = llm.stats.total_cost

                # 6. Export
                state.status = "exporting"
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="status", message="Esportazione branch...",
                    phase="exporting"
                ))

                safe_name = self._make_safe_name(config.name, sim_id)
                checkpoint_name = self._sanitize_name(config.name)
                export_scenario(safe_name, outputs_dir, export_dir, source_name=checkpoint_name)
                state.scenario_id = safe_name

                await self._rebuild_scenarios_manifest(export_dir)

                state.status = "completed"
                state.completed_at = datetime.utcnow().isoformat()
                self._persist()

                await self._emit(sim_id, ProgressEvent(
                    type="completed",
                    message="Branch What-If completato!",
                    data={
                        "scenario_name": config.name,
                        "scenario_id": safe_name,
                        "cost": state.cost,
                        "total_rounds": state.total_rounds,
                        "branch_from": branch.parent_scenario_id,
                        "branch_round": branch.branch_round,
                    }
                ))

            except asyncio.CancelledError:
                state.status = "cancelled"
                state.completed_at = datetime.utcnow().isoformat()
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="cancelled", message="Branch cancellato"
                ))
            except Exception as e:
                logger.error(f"Branch {sim_id} failed: {traceback.format_exc()}")
                state.status = "failed"
                state.error = str(e)
                state.completed_at = datetime.utcnow().isoformat()
                self._persist()
                await self._emit(sim_id, ProgressEvent(
                    type="error", message=f"Errore: {str(e)}",
                    data={"traceback": traceback.format_exc()[-500:]}
                ))

    def _config_from_metadata(self, meta: dict) -> ScenarioConfig:
        """Build a minimal ScenarioConfig from exported metadata."""
        from core.config.schema import AxisConfig
        axis = meta.get("position_axis", {})
        return ScenarioConfig(
            name=meta.get("scenario_name", "Unknown"),
            description=meta.get("description", ""),
            domain=meta.get("domain", "political"),
            language=meta.get("language", "en"),
            num_rounds=meta.get("num_rounds", 9),
            timeline_unit=meta.get("timeline_unit", "month"),
            timeline_labels=meta.get("timeline_labels", []),
            position_axis=AxisConfig(
                negative_label=axis.get("negative_label", "Against"),
                positive_label=axis.get("positive_label", "In favor"),
                neutral_label=axis.get("neutral_label", "Neutral"),
            ),
            channels=meta.get("channels", []),
            elite_agents=[],
            institutional_agents=[],
            citizen_clusters=[],
            initial_event=meta.get("initial_event", ""),
            scenario_context=meta.get("scenario_context", ""),
            metrics_to_track=meta.get("metrics_to_track", []),
        )

    async def submit_intervention(self, sim_id: str, intervention: WargameIntervention) -> dict:
        """Submit a human player's intervention to a paused wargame simulation."""
        state = self.simulations.get(sim_id)
        if not state:
            return {"error": "Simulation not found"}
        if state.status != "awaiting_player":
            return {"error": f"Simulation is '{state.status}', not awaiting input"}

        # ── KB inject: ingest the doc into the live RAG store BEFORE resuming ──
        # Subsequent rounds will retrieve from this newly-injected content.
        kb_result = None
        if intervention.action_type == "inject_kb" and intervention.kb_doc:
            store = getattr(state, "rag_store", None)
            if store is None:
                # Lazy-create per-sim store if none was attached at launch (no docs uploaded)
                try:
                    from api.rag_store import RAGStore
                    store = RAGStore()
                    state.rag_store = store
                    # Hand the live store to the running engine so RoundManager picks it up
                    engine_ref = getattr(state, "_engine_ref", None)
                    if engine_ref is not None:
                        engine_ref.rag_store = store
                        rm = getattr(engine_ref, "_round_manager", None)
                        if rm is not None:
                            rm.rag_store = store
                except Exception as exc:
                    logger.warning(f"Could not init RAGStore for KB inject: {exc}")
                    store = None

            if store is not None:
                try:
                    # Stable doc_id includes the round so injects are inspectable later
                    inject_round = state.current_round + 1
                    doc_id = f"inject_r{inject_round}_{intervention.kb_doc.source[:16]}"
                    added = store.add_document(
                        doc_id=doc_id,
                        title=intervention.kb_doc.title,
                        text=intervention.kb_doc.text,
                    )
                    kb_result = {"doc_id": doc_id, "chunks_added": added, "total_chunks": store.chunk_count}
                    logger.info(f"Wargame KB inject: {doc_id} → {added} chunks (sim={sim_id})")
                except Exception as exc:
                    logger.warning(f"Wargame KB inject failed: {exc}")
                    kb_result = {"error": str(exc)}

        # Store the intervention
        state._wargame_intervention = intervention

        # Resume the simulation
        state.status = "running"
        self._persist()
        state._wargame_resume.set()

        action_desc = intervention.action_text[:100]
        await self._emit(sim_id, ProgressEvent(
            type="player_action",
            message=f"Player: {action_desc}...",
            round=state.current_round + 1,
            data={
                "action_text": intervention.action_text,
                "action_type": intervention.action_type,
                "target_audience": intervention.target_audience,
                "kb_inject": kb_result,
            }
        ))

        return {
            "status": "accepted",
            "message": f"Intervention accepted. Round {state.current_round + 1} starting with your action.",
            "action_type": intervention.action_type,
            "kb_inject": kb_result,
        }

    async def rollback_to_round(self, sim_id: str, target_round: int) -> dict:
        """Rollback a wargame simulation to a previous round (save scumming).

        Kills the current simulation, loads the checkpoint for target_round,
        and restarts a new simulation from that point, pausing immediately
        for a new player intervention.
        """
        state = self.simulations.get(sim_id)
        if not state:
            return {"error": "Simulation not found"}
        if not state.wargame_mode:
            return {"error": "Rollback only available in wargame mode"}
        if target_round < 1 or target_round >= state.current_round:
            return {"error": f"Invalid target round {target_round} (current: {state.current_round})"}

        # Find the checkpoint file (checkpoints are stored under the legacy
        # un-hashed sanitized name, since they pre-date per-sim uniqueness).
        from core.simulation.checkpoint import find_checkpoint, load_checkpoint
        scenario_name = state.scenario_name or "unknown"
        safe_name = self._sanitize_name(scenario_name)
        checkpoint_dir = os.path.join("outputs", safe_name) if os.path.isdir(os.path.join("outputs", safe_name)) else "outputs"

        try:
            cp_path = find_checkpoint(checkpoint_dir, scenario_name, target_round)
        except FileNotFoundError:
            # Try just "outputs/" directly
            try:
                cp_path = find_checkpoint("outputs", scenario_name, target_round)
            except FileNotFoundError:
                return {"error": f"No checkpoint found for round {target_round}"}

        # Kill current simulation
        if state.task and not state.task.done():
            state.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(state.task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Load checkpoint
        checkpoint = load_checkpoint(cp_path)

        # Create a new simulation that branches from this checkpoint
        # Reuse the original request but as a branch
        from api.models import BranchRequest
        branch_req = BranchRequest(
            parent_scenario_id=state.scenario_id or sim_id,
            branch_round=target_round,
            what_if=f"Wargame rollback to round {target_round}",
            event_override=None,  # Player will provide new intervention
            rounds_to_run=state.total_rounds - target_round,
            provider=state.request.provider,
            model=getattr(state.request, "model", None),
            budget=state.request.budget,
        )

        # Update state to reflect rollback
        state.current_round = target_round
        state.status = "awaiting_player"
        state.completed_at = None
        state.error = None
        self._persist()

        # Build SITREP from checkpoint data
        sitrep = {
            "round": target_round,
            "rollback": True,
            "message": f"Time-travel: rolled back to end of Round {target_round}. "
                       f"Your previous actions from Round {target_round + 1} onwards have been erased. "
                       f"You can now make a different decision.",
            "agents_snapshot": {
                "elite_count": len(checkpoint.get("elite_agents", [])),
                "institutional_count": len(checkpoint.get("institutional_agents", [])),
                "citizen_clusters": len(checkpoint.get("citizen_clusters", [])),
            },
            "coalition_history": checkpoint.get("coalition_history", []),
        }

        # Store checkpoint on state for when the player submits new intervention
        state._rollback_checkpoint = checkpoint
        state._rollback_checkpoint_path = cp_path
        state._rollback_branch_req = branch_req
        state._wargame_sitrep = sitrep

        # Emit SSE events
        await self._emit(sim_id, ProgressEvent(
            type="rollback",
            message=f"Rolled back to Round {target_round}",
            round=target_round,
            data={"target_round": target_round},
        ))
        await self._emit(sim_id, ProgressEvent(
            type="awaiting_intervention",
            message=f"Round {target_round} restored. Submit your new action.",
            round=target_round,
            data=sitrep,
        ))

        return {
            "status": "rolled_back",
            "target_round": target_round,
            "message": f"Simulation rolled back to Round {target_round}. Submit intervention to continue.",
            "checkpoint_available": True,
        }

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

    def get_status(self, sim_id: str, tenant_id: str = "default") -> Optional[SimulationStatus]:
        state = self.simulations.get(sim_id)
        if not state:
            return None
        if tenant_id != "default" and state.tenant_id != tenant_id:
            return None  # Tenant isolation: not your simulation
        return state.to_status()

    def get_state(self, sim_id: str, tenant_id: str = "default") -> Optional[SimulationState]:
        """Get raw SimulationState with tenant isolation check."""
        state = self.simulations.get(sim_id)
        if not state:
            return None
        if tenant_id != "default" and state.tenant_id != tenant_id:
            return None
        return state

    def list_simulations(self, tenant_id: str = "default") -> list[SimulationStatus]:
        sims = self.simulations.values()
        if tenant_id != "default":
            sims = [s for s in sims if s.tenant_id == tenant_id]
        return [s.to_status() for s in sorted(
            sims,
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
        if not state:
            return
        # Strategy: drop-oldest-on-full. If a client is slow/disconnected and
        # the queue saturates, we evict the oldest queued event to make room
        # for the new one — preserving the FRESHEST state. Terminal events
        # (completed/error/cancelled) are guaranteed delivery: they will
        # evict as many older events as needed rather than being dropped.
        is_terminal = event.type in ("completed", "error", "cancelled")
        try:
            state.event_queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        # Queue is full: drop one oldest, try again. For terminal events we
        # keep evicting until we succeed (bounded by queue size anyway).
        max_attempts = state.event_queue.maxsize + 1 if is_terminal else 1
        for _ in range(max_attempts):
            try:
                state.event_queue.get_nowait()
                state.events_dropped += 1
            except asyncio.QueueEmpty:
                pass
            try:
                state.event_queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                continue
        # Only reachable if queue.maxsize == 0, which we don't configure.
        state.events_dropped += 1
        logger.warning(f"[{sim_id}] event {event.type} dropped: queue saturated")

    async def _run_pipeline(self, sim_id: str):
        state = self.simulations[sim_id]
        request = state.request

        # Heartbeat immediato: senza questo l'utente vede "Log eventi (0)" finché
        # job_queue.acquire() rilascia il lock (può richiedere secondi se ci sono
        # sim in coda davanti).
        state.status = "analyzing"
        self._persist()
        await self._emit(sim_id, ProgressEvent(
            type="status",
            message="Simulazione in coda, preparazione...",
            phase="queued",
        ))

        async with job_queue.acquire(sim_id):
            try:
                # Phase 1: Create LLM client
                await self._emit(sim_id, ProgressEvent(
                    type="status", message="Inizializzazione LLM...",
                    phase="init"
                ))

                llm = self._create_llm(request.provider, request.model, request.budget)

                # Phase labels — default Italian, updated after brief analysis
                from core.simulation.round_manager import RoundManager
                phase_labels = RoundManager._build_phase_labels("it")

                # Define progress callback early (used in web research + simulation)
                async def on_progress(event_type: str, data: dict):
                    if event_type == "round_start":
                        state.current_round = data.get("round", 0)
                        state.cost = data.get("cost", state.cost)
                        self._persist()
                        tpl = phase_labels.get("round_of", "Round {r} of {t}")
                        await self._emit(sim_id, ProgressEvent(
                            type="round_start",
                            message=tpl.format(r=data["round"], t=state.total_rounds),
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
                        tpl = phase_labels.get("round_done", "Round {r} completed")
                        await self._emit(sim_id, ProgressEvent(
                            type="round_complete",
                            message=tpl.format(r=data["round"]),
                            round=data["round"],
                            data=data,
                        ))

                        # === WARGAME PAUSE ===
                        # After each round_complete, pause and wait for human input
                        if state.wargame_mode and data.get("round", 0) < state.total_rounds:
                            await self._wargame_pause(sim_id, state, data)

                            # If player submitted an intervention, inject it
                            pending = getattr(state, "_pending_event_override", None)
                            engine_ref = getattr(state, "_engine_ref", None)
                            rm = getattr(engine_ref, "_round_manager", None) if engine_ref else None
                            if pending and rm:
                                rm._branch_event_override = pending
                                # Override shock magnitude if provided
                                shock = getattr(state, "_pending_shock", None)
                                if shock is not None:
                                    rm._wargame_shock_override = shock
                                state._pending_event_override = None
                                state._pending_shock = None

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

                # Update phase labels to match detected language
                phase_labels.update(RoundManager._build_phase_labels(config.language))

                # Apply user-selected KPIs (merge with LLM-extracted ones)
                if request.metrics_to_track:
                    existing = set(config.metrics_to_track)
                    for m in request.metrics_to_track:
                        if m not in existing:
                            config.metrics_to_track.append(m)

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

                # Build activation plan summary for frontend
                activation_plan_data = None
                activation_plan = getattr(config, "_activation_plan", None)
                if activation_plan:
                    activation_plan_data = {
                        "wave_1": len(activation_plan.wave_1),
                        "wave_2": len(activation_plan.wave_2),
                        "wave_3": len(activation_plan.wave_3),
                        "reserve": len(activation_plan.reserve),
                        "detected_sectors": activation_plan.detected_sectors,
                        "detected_regions": activation_plan.detected_regions,
                    }

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
                        "activation_plan": activation_plan_data,
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
                    rag_store=getattr(state, "rag_store", None),
                )

                # Store engine ref for wargame mode (so callback can inject overrides)
                state._engine_ref = engine

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
                safe_name = self._make_safe_name(config.name, sim_id)
                checkpoint_name = self._sanitize_name(config.name)
                export_scenario(safe_name, outputs_dir, export_dir, source_name=checkpoint_name)
                state.scenario_id = safe_name

                # Save Monte Carlo results alongside scenario
                if monte_carlo_data:
                    mc_path = os.path.join(export_dir, safe_name, "monte_carlo.json")
                    os.makedirs(os.path.dirname(mc_path), exist_ok=True)
                    import json as _json
                    with open(mc_path, "w") as f:
                        _json.dump(monte_carlo_data, f, indent=2)

                # Build scenarios.json manifest (lock-guarded)
                await self._rebuild_scenarios_manifest(export_dir)

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
        from core.simulation.monte_carlo import MonteCarloEngine
        from calibration.synthetic_sim import run_synthetic_simulation
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

        # Build a GroundTruth-like structure for Monte Carlo
        polling = [
            PollingDataPoint(
                round_equivalent=i + 1,
                pro_pct=50.0,
                against_pct=40.0,
                undecided_pct=10.0,
            )
            for i in range(config.num_rounds)
        ]

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

    async def _wargame_pause(self, sim_id: str, state: SimulationState, round_data: dict):
        """Pause simulation for human player intervention.

        Builds a situation report from the round data, sends it via SSE,
        then waits for the player to POST to /api/simulations/{id}/intervene.
        """
        round_num = round_data.get("round", 0)

        # Build situation report (SITREP) for the player
        orchestrator = round_data.get("orchestrator", {})
        financial = round_data.get("financial_impact", {})

        # Identify threats
        threats = []
        pol = round_data.get("polarization", 0)
        if pol > 5:
            threats.append(f"Polarizzazione alta ({pol:.1f}/10)")
        neg_pct = round_data.get("sentiment", {}).get("negative", 0)
        if neg_pct > 0.4:
            threats.append(f"Sentiment negativo dominante ({neg_pct:.0%})")
        if orchestrator.get("escalated"):
            threats.append(f"Escalation a Wave {orchestrator.get('active_wave', '?')}")
        cri = orchestrator.get("contagion_risk_index", 0)
        if cri > 0.4:
            threats.append(f"Rischio contagio {orchestrator.get('contagion_risk_label', '?').upper()} ({cri:.2f})")
        if financial.get("market_volatility_warning") in ("HIGH", "CRITICAL"):
            threats.append(f"Impatto mercato: {financial.get('headline', '')}")

        # Top viral posts (what's dominating the narrative)
        top_posts = round_data.get("top_posts", [])[:5]
        narrative_snapshot = []
        for p in top_posts:
            narrative_snapshot.append({
                "author": p.get("author_name", "?"),
                "text": p.get("text", "")[:200],
                "engagement": p.get("total_engagement", 0),
            })

        # Coalitions
        coalitions = round_data.get("coalitions", [])

        # Build SITREP
        sitrep = {
            "round_completed": round_num,
            "next_round": round_num + 1,
            "player_role": state.player_role,
            "status": "AWAITING YOUR MOVE",
            "threats": threats,
            "polarization": round_data.get("polarization", 0),
            "sentiment": round_data.get("sentiment", {}),
            "engagement_score": orchestrator.get("engagement_score", 0),
            "active_wave": orchestrator.get("active_wave", 1),
            "contagion_risk": cri,
            "top_narratives": narrative_snapshot,
            "coalitions": coalitions[:4],
            "financial_impact": financial,
            "prompt": (
                f"Sei {state.player_role or 'il decisore'}. "
                f"Il Round {round_num} è appena terminato. "
                f"{'⚠️ ' + '; '.join(threats[:3]) if threats else 'Situazione sotto controllo.'} "
                f"Che contromossa fai? Scrivi il tuo comunicato stampa, annuncio interno, "
                f"post social, o azione politica. I 744 agenti reagiranno alla tua mossa."
            ),
            "suggested_actions": [
                "Comunicato stampa conciliante",
                "Annuncio di investimenti compensativi",
                "Incontro diretto con i sindacati",
                "Dichiarazione sui social media",
                "Silenzio strategico (skip)",
            ],
        }

        state._wargame_sitrep = sitrep

        # Update status and emit SITREP via SSE
        state.status = "awaiting_player"
        self._persist()

        await self._emit(sim_id, ProgressEvent(
            type="awaiting_intervention",
            message=sitrep["prompt"],
            round=round_num,
            data=sitrep,
        ))

        # === WAIT for human input ===
        state._wargame_resume.clear()
        logger.info(f"Wargame {sim_id}: paused after round {round_num}, awaiting player input")

        # Wait up to 30 minutes for player input (then auto-skip)
        try:
            await asyncio.wait_for(state._wargame_resume.wait(), timeout=1800)
        except asyncio.TimeoutError:
            logger.info(f"Wargame {sim_id}: player timeout after round {round_num}, auto-continuing")
            state._wargame_intervention = WargameIntervention(
                action_text="", skip=True, action_type="timeout"
            )

        # Retrieve the intervention
        intervention = state._wargame_intervention
        state._wargame_intervention = None
        state._wargame_sitrep = None

        if intervention and not intervention.skip and intervention.action_text:
            # Build event text that wraps the player's action
            role_label = state.player_role or "Corporate leadership"
            action_label = {
                "press_release": "comunicato stampa ufficiale",
                "internal_memo": "comunicazione interna",
                "social_post": "dichiarazione pubblica sui social",
                "policy_announcement": "annuncio di politica aziendale/istituzionale",
            }.get(intervention.action_type, "azione")

            event_text = (
                f"BREAKING — {role_label} rilascia un {action_label}: "
                f"\"{intervention.action_text}\" "
            )
            if intervention.target_audience:
                event_text += f" [Indirizzato a: {intervention.target_audience}]"

            # Inject as event override for next round via the engine
            # We store it on the state; the on_progress callback reads it
            state._pending_event_override = event_text
            state._pending_shock = intervention.shock_magnitude or 0.5

            logger.info(f"Wargame {sim_id}: player action injected for round {round_num + 1}")
        else:
            state._pending_event_override = None
            state._pending_shock = None
            logger.info(f"Wargame {sim_id}: player skipped, auto-generating next round")

        state.status = "running"
        self._persist()

    def _create_llm(self, provider: str, model: Optional[str], budget: float):
        if provider == "openai":
            from core.llm.openai_client import OpenAIClient
            return OpenAIClient(model=model or "gpt-5.4-mini", budget=budget)
        else:
            from core.llm.gemini_client import GeminiClient
            return GeminiClient(model=model or "gemini-3.1-flash-lite-preview", budget=budget)

    async def _rebuild_scenarios_manifest(self, export_dir: str):
        """Rebuild scenarios.json. Lock prevents two concurrent exports from
        racing the final write (would otherwise produce truncated JSON)."""
        async with self._manifest_lock:
            await asyncio.to_thread(self._rebuild_scenarios_manifest_sync, export_dir)

    @staticmethod
    def _rebuild_scenarios_manifest_sync(export_dir: str):
        scenarios = []
        if not os.path.isdir(export_dir):
            return
        for d in sorted(os.listdir(export_dir)):
            if not d.startswith("scenario_"):
                continue
            meta_path = os.path.join(export_dir, d, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                scenario_id = d.replace("scenario_", "", 1)
                scenarios.append({
                    "id": scenario_id,
                    "name": meta.get("scenario_name", scenario_id.replace("_", " ")),
                    "domain": meta.get("domain", "unknown"),
                    "description": meta.get("description", ""),
                    "num_rounds": meta.get("num_rounds", 0),
                })

        manifest_path = os.path.join(export_dir, "scenarios.json")
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, manifest_path)
