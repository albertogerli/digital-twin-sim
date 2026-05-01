"""FastAPI application — DigitalTwinSim API."""

import json
import os
import sys
import time
from functools import wraps
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pydantic import BaseModel
from api.models import (
    SimulationRequest, BranchRequest, ObservationInput, WargameIntervention,
    MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_FILES,
)
from api.simulation_manager import SimulationManager
from api.document_processor import save_uploaded_file, process_uploads
from api.auth import Tenant, verify_api_key
from api.rate_limiter import (
    limiter, rate_limit_exceeded_handler,
    LIMIT_CREATE_SIM, LIMIT_SUGGEST_KPIS, LIMIT_READS,
)

# ── Structured logging ──────────────────────────────────────
from api.logging_config import setup_logging
setup_logging()

# ── Sentry (optional) ───────────────────────────────────────
_sentry_dsn = os.getenv("DTS_SENTRY_DSN_BACKEND", "")
if _sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk.init(
            dsn=_sentry_dsn,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            traces_sample_rate=0.2,
            environment=os.getenv("DTS_ENV", "development"),
        )
    except ImportError:
        pass

app = FastAPI(
    title="DigitalTwinSim",
    description="Universal Digital Twin Simulation Platform — agent-based opinion dynamics with LLM-powered scenarios, wargame mode, what-if branching, and Monte Carlo analysis.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Simulations", "description": "Launch, monitor, and control simulations"},
        {"name": "Scenarios", "description": "Access completed scenario exports"},
        {"name": "Wargame", "description": "Interactive wargame mode endpoints"},
        {"name": "Domains", "description": "Available domain modules"},
        {"name": "Observability", "description": "Health, metrics, and usage tracking"},
    ],
)

# ── Request logging middleware ───────────────────────────────
from api.middleware import RequestLoggingMiddleware
app.add_middleware(RequestLoggingMiddleware)

# ── Rate limiting ───────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ── CORS ────────────────────────────────────────────────────
_cors_origins = os.getenv("DTS_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics ───────────────────────────────────────
try:
    from api.metrics import setup_metrics
    setup_metrics(app)
except ImportError:
    pass  # prometheus deps optional

manager = SimulationManager()


@app.on_event("startup")
async def _on_startup():
    """Rehydrate SimulationManager from Postgres if DATABASE_URL is set."""
    await manager.initialize()

# ── In-memory TTL cache for GET endpoints ─────────────────
_cache: dict[str, tuple[float, any]] = {}


def cached(ttl_seconds: float = 5.0):
    """Simple in-memory TTL cache for GET endpoints."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            now = time.monotonic()
            if key in _cache:
                expires, value = _cache[key]
                if now < expires:
                    return value
            result = await func(*args, **kwargs)
            _cache[key] = (now + ttl_seconds, result)
            return result
        return wrapper
    return decorator

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPORTS_DIR = os.path.join(OUTPUTS_DIR, "exports")


def _tenant_id(tenant: Optional[Tenant]) -> str:
    """Extract tenant_id, defaulting to 'default' if auth is disabled."""
    return tenant.tenant_id if tenant else "default"


# ── Health (used by Railway / load balancer) ─────────────────

API_VERSION = "0.6.0"


@app.get("/api/health")
async def health():
    """Lightweight readiness probe — no DB, no LLM. Returns 200 if the
    process is alive and the FastAPI app has started."""
    return {
        "status": "ok",
        "service": "digital-twin-sim-api",
        "version": API_VERSION,
    }


@app.get("/api/disk/stats")
async def disk_stats():
    """Diagnostic: check writability + free space on /app/outputs (volume mount).
    Helps debug 'disk I/O error' from SQLite on Railway volume."""
    import shutil
    out_dir = OUTPUTS_DIR
    info = {"path": out_dir, "exists": os.path.isdir(out_dir)}
    try:
        usage = shutil.disk_usage(out_dir)
        info["total_gb"] = round(usage.total / 1e9, 2)
        info["used_gb"] = round(usage.used / 1e9, 2)
        info["free_gb"] = round(usage.free / 1e9, 2)
        info["used_pct"] = round(usage.used / usage.total * 100, 1) if usage.total else None
    except Exception as e:
        info["disk_usage_error"] = str(e)

    # Test writability with a small probe file
    probe = os.path.join(out_dir, ".write_probe")
    try:
        with open(probe, "w") as f:
            f.write(str(time.time()))
        os.remove(probe)
        info["writable"] = True
    except Exception as e:
        info["writable"] = False
        info["write_error"] = str(e)

    # Test SQLite + WAL specifically (the failure mode the user hit)
    sqlite_probe = os.path.join(out_dir, ".sqlite_probe.db")
    try:
        import sqlite3
        if os.path.exists(sqlite_probe):
            os.remove(sqlite_probe)
        conn = sqlite3.connect(sqlite_probe, timeout=5.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()
        for ext in ("", "-wal", "-shm"):
            f = sqlite_probe + ext
            if os.path.exists(f):
                os.remove(f)
        info["sqlite_wal"] = "ok"
    except Exception as e:
        info["sqlite_wal"] = "FAIL"
        info["sqlite_error"] = str(e)

    # List existing .db files to spot stuck simulations
    try:
        dbs = [f for f in os.listdir(out_dir) if f.endswith(".db")]
        info["existing_dbs"] = dbs[:20]
        info["db_count"] = len(dbs)
    except Exception:
        pass

    return info


@app.get("/api/db/stats")
async def db_stats():
    """Diagnostic: row counts directly from Postgres (bypasses in-memory)."""
    from api import db as _db
    if not _db.is_available():
        return {"db_configured": False, "message": "DATABASE_URL not set"}
    try:
        pool = await _db.get_pool()
        if not pool:
            return {"db_configured": True, "connected": False}
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM simulations")
            by_status = await conn.fetch(
                "SELECT status, COUNT(*) AS n FROM simulations GROUP BY status"
            )
            wargame = await conn.fetchval(
                "SELECT COUNT(*) FROM simulations WHERE wargame_mode = TRUE"
            )
            recent = await conn.fetch(
                "SELECT id, status, scenario_name, wargame_mode, "
                "current_round, total_rounds, created_at "
                "FROM simulations ORDER BY created_at DESC LIMIT 5"
            )
        return {
            "db_configured": True,
            "connected": True,
            "total_simulations": total,
            "by_status": {r["status"]: r["n"] for r in by_status},
            "wargame_count": wargame,
            "recent": [
                {
                    "id": r["id"],
                    "status": r["status"],
                    "scenario_name": r["scenario_name"],
                    "wargame": r["wargame_mode"],
                    "round": f"{r['current_round']}/{r['total_rounds']}",
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in recent
            ],
        }
    except Exception as e:
        return {"db_configured": True, "connected": False, "error": str(e)}


# ── Domains ──────────────────────────────────────────────────

@app.get("/api/domains")
@limiter.limit(LIMIT_READS)
async def list_domains(request: Request, tenant: Optional[Tenant] = Depends(verify_api_key)):
    from domains.domain_registry import DomainRegistry
    DomainRegistry.discover()
    return {"domains": DomainRegistry.list_domains()}


# ── KPI Suggestions ─────────────────────────────────────────

class KpiSuggestRequest(BaseModel):
    brief: str
    domain: Optional[str] = None
    provider: str = "gemini"

@app.post("/api/suggest-kpis")
@limiter.limit(LIMIT_SUGGEST_KPIS)
async def suggest_kpis(
    request: Request,
    body: KpiSuggestRequest,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Use LLM to suggest quantitative KPIs for a scenario brief."""
    llm = manager._create_llm(body.provider, None, 0.5)
    domain_hint = f"\nDomain: {body.domain}" if body.domain else ""
    prompt = f"""Given this scenario brief, suggest 6-8 quantitative KPIs (Key Performance Indicators) that would be meaningful to track during a multi-round simulation.

Brief: {body.brief}{domain_hint}

Requirements:
- Each KPI must be measurable on a 0-100 scale
- KPIs should cover different dimensions (sentiment, risk, trust, behavior, media, etc.)
- Use short, clear names (2-4 words each)
- Write KPI names in the same language as the brief
- Return ONLY a JSON array of strings, no explanations

Example: ["Consenso pubblico", "Rischio reputazionale", "Fiducia investitori", "Viralità mediatica", "Coesione interna", "Pressione regolamentare"]"""

    try:
        result = await llm.generate_json(
            prompt=prompt,
            temperature=0.4,
            max_output_tokens=300,
            component="kpi_suggestion",
        )
        if isinstance(result, list):
            return {"kpis": [str(k) for k in result[:10]]}
        return {"kpis": []}
    except Exception as e:
        return {"kpis": [], "error": str(e)}


# ── Simulations ──────────────────────────────────────────────

@app.post("/api/simulations")
@limiter.limit(LIMIT_CREATE_SIM)
async def create_simulation(
    request: Request,
    body: SimulationRequest,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    sim_id = await manager.launch(body, tenant_id=_tenant_id(tenant))
    return {"id": sim_id, "status": "queued"}


@app.post("/api/simulations/with-documents")
@limiter.limit(LIMIT_CREATE_SIM)
async def create_simulation_with_documents(
    request: Request,
    brief: str = Form(...),
    provider: str = Form("gemini"),
    domain: Optional[str] = Form(None),
    rounds: Optional[int] = Form(None),
    budget: float = Form(5.0),
    elite_only: bool = Form(False),
    metrics_to_track: Optional[str] = Form(None),
    documents: list[UploadFile] = File(default=[]),
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Launch simulation with uploaded documents for RAG grounding."""
    from uuid import uuid4
    sim_id = str(uuid4())[:8]
    tid = _tenant_id(tenant)

    # Validate upload limits
    if len(documents) > MAX_UPLOAD_FILES:
        raise HTTPException(
            400,
            f"Too many files. Maximum {MAX_UPLOAD_FILES} allowed.",
        )

    # Save uploaded documents
    doc_count = 0
    for doc in documents:
        if doc.filename and doc.size and doc.size > 0:
            if doc.size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    400,
                    f"File '{doc.filename}' exceeds {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB limit.",
                )
            content = await doc.read()
            save_uploaded_file(sim_id, doc.filename, content)
            doc_count += 1

    # Process uploads
    doc_result = process_uploads(sim_id) if doc_count > 0 else None

    # Parse metrics_to_track from JSON string
    parsed_metrics = []
    if metrics_to_track:
        try:
            parsed_metrics = json.loads(metrics_to_track)
        except (json.JSONDecodeError, TypeError):
            parsed_metrics = []

    sim_request = SimulationRequest(
        brief=brief,
        provider=provider,
        domain=domain,
        rounds=rounds,
        budget=budget,
        elite_only=elite_only,
        metrics_to_track=parsed_metrics,
    )

    actual_id = await manager.launch(
        sim_request, sim_id=sim_id, document_context=doc_result, tenant_id=tid,
    )
    return {
        "id": actual_id,
        "status": "queued",
        "documents_processed": doc_count,
        "context_chars": doc_result["total_chars"] if doc_result else 0,
    }


@app.get("/api/simulations")
@limiter.limit(LIMIT_READS)
@cached(ttl_seconds=5.0)
async def list_simulations(
    request: Request,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    return manager.list_simulations(tenant_id=_tenant_id(tenant))


@app.get("/api/simulations/{sim_id}")
@limiter.limit(LIMIT_READS)
async def get_simulation(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    status = manager.get_status(sim_id, tenant_id=_tenant_id(tenant))
    if not status:
        raise HTTPException(404, "Simulation not found")
    return status


@app.get("/api/simulations/{sim_id}/stream")
async def stream_simulation(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant))
    if not state:
        raise HTTPException(404, "Simulation not found")

    async def event_generator():
        async for event in manager.stream_events(sim_id):
            yield {
                "event": event.type,
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_generator())


@app.post("/api/simulations/{sim_id}/intervene")
@limiter.limit(LIMIT_CREATE_SIM)
async def wargame_intervene(
    request: Request,
    sim_id: str,
    intervention: WargameIntervention,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Submit a human player's counter-move during a wargame simulation."""
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant))
    if not state:
        raise HTTPException(404, "Simulation not found")
    # Container restarted between rounds: the SITREP survived (in-memory state
    # was rehydrated from disk) but the live asyncio task and LLM clients did
    # not. We can't resume reliably yet — return a clear 410 so the UI prompts
    # the player to start a new run instead of a cryptic 404.
    if getattr(state, "_restored_after_restart", False):
        raise HTTPException(
            410,
            "Simulazione interrotta da un riavvio del server. La sessione "
            "wargame non può essere ripresa. Avvia una nuova simulazione."
        )
    if state.status != "awaiting_player":
        raise HTTPException(
            400,
            f"Simulation is '{state.status}', not awaiting player input. "
            f"Only wargame simulations in paused state accept interventions."
        )

    result = await manager.submit_intervention(sim_id, intervention)
    return result


@app.get("/api/simulations/{sim_id}/wargame-state")
@limiter.limit(LIMIT_READS)
async def get_wargame_state(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Get the current wargame situation report for the player."""
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant))
    if not state:
        raise HTTPException(404, "Simulation not found")

    sitrep = getattr(state, "_wargame_sitrep", None)
    if not sitrep:
        return {"status": state.status, "message": "No situation report available yet"}
    return sitrep


@app.post("/api/simulations/{sim_id}/rollback")
@limiter.limit(LIMIT_CREATE_SIM)
async def wargame_rollback(
    request: Request,
    sim_id: str,
    target_round: int,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Rollback a wargame simulation to a previous round."""
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant))
    if not state:
        raise HTTPException(404, "Simulation not found")
    result = await manager.rollback_to_round(sim_id, target_round)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/scenarios/{scenario_id}/branch")
@limiter.limit(LIMIT_CREATE_SIM)
async def branch_scenario(
    request: Request,
    scenario_id: str,
    body: BranchRequest,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Create a What-If branch from a completed scenario."""
    export_dir = os.path.join(EXPORTS_DIR, f"scenario_{scenario_id}")
    if not os.path.exists(export_dir):
        raise HTTPException(404, f"Scenario not found: {scenario_id}")
    body.parent_scenario_id = scenario_id
    sim_id = await manager.launch_branch(body, tenant_id=_tenant_id(tenant))
    return {"id": sim_id, "status": "queued", "branch_from": scenario_id}


@app.delete("/api/simulations/{sim_id}")
async def cancel_simulation(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant))
    if not state:
        raise HTTPException(404, "Simulation not found")
    ok = await manager.cancel(sim_id)
    if not ok:
        raise HTTPException(400, "Cannot cancel this simulation")
    return {"status": "cancelled"}


# ── Scenarios (completed exports) ────────────────────────────

@app.get("/api/scenarios")
@limiter.limit(LIMIT_READS)
@cached(ttl_seconds=5.0)
async def list_scenarios(
    request: Request,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    tid = _tenant_id(tenant)
    # Try tenant-specific manifest first, fall back to shared
    tenant_manifest = os.path.join(EXPORTS_DIR, tid, "scenarios.json")
    shared_manifest = os.path.join(EXPORTS_DIR, "scenarios.json")

    manifest = tenant_manifest if os.path.exists(tenant_manifest) else shared_manifest
    if not os.path.exists(manifest):
        return []
    with open(manifest) as f:
        return json.load(f)


@app.get("/api/scenarios/{scenario_id}/{filename}")
@limiter.limit(LIMIT_READS)
async def get_scenario_file(
    request: Request,
    scenario_id: str,
    filename: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    # Sanitize
    if ".." in scenario_id or ".." in filename:
        raise HTTPException(400, "Invalid path")
    if not (filename.endswith(".json") or filename.endswith(".md") or filename.endswith(".html")):
        raise HTTPException(400, "Only .json, .md and .html files allowed")

    path = os.path.join(EXPORTS_DIR, f"scenario_{scenario_id}", filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {filename}")

    if filename.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if filename.endswith(".html"):
        return FileResponse(path, media_type="text/html")
    return FileResponse(path, media_type="text/markdown")


# ── Online observations (EnKF) ───────────────────────────────

@app.post("/api/simulations/{sim_id}/observe")
@limiter.limit(LIMIT_CREATE_SIM)
async def submit_observation(
    request: Request,
    sim_id: str,
    observation: ObservationInput,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Submit a real-world observation for EnKF data assimilation."""
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant))
    if not state:
        raise HTTPException(404, "Simulation not found")
    if state.status not in ("running",):
        raise HTTPException(400, f"Cannot observe simulation in status '{state.status}'")

    # Check if simulation has EnKF enabled
    enkf = getattr(state, "_enkf_filter", None)
    if enkf is None:
        raise HTTPException(
            400,
            "This simulation was not launched with online_mode=True. "
            "EnKF data assimilation is not available.",
        )

    try:
        obs_value = observation.pro_pct / 100.0
        obs_noise = 1.0 / max(observation.sample_size or 1000, 100) * 10

        enkf.assimilate(obs_value, obs_noise)

        ensemble_mean = float(enkf.ensemble_mean()) * 100
        ensemble_std = float(enkf.ensemble_std()) * 100
        effective_size = getattr(enkf, "effective_sample_size", lambda: len(enkf.ensemble))()

        prior_ci_width = getattr(state, "_prior_ci_width", ensemble_std * 3.92)
        post_ci_width = ensemble_std * 3.92
        ci_reduction = max(0, (1 - post_ci_width / max(prior_ci_width, 0.01)) * 100)
        state._prior_ci_width = post_ci_width

        return {
            "updated_prediction": {
                "pro_pct_mean": round(ensemble_mean, 1),
                "pro_pct_ci95": [
                    round(max(0, ensemble_mean - 1.96 * ensemble_std), 1),
                    round(min(100, ensemble_mean + 1.96 * ensemble_std), 1),
                ],
                "ci_reduction_pct": round(ci_reduction, 1),
            },
            "ensemble_health": {
                "spread": round(ensemble_std / 100, 4),
                "effective_size": int(effective_size),
            },
        }
    except Exception as e:
        raise HTTPException(500, f"Assimilation failed: {str(e)}")


# ── Usage Tracking ────────────────────────────────────────────

@app.get("/api/usage")
@limiter.limit(LIMIT_READS)
async def get_usage(
    request: Request,
    since: Optional[str] = None,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Get LLM usage records for cost tracking."""
    from api import db
    records = await db.get_usage(tenant_id=_tenant_id(tenant), since=since)
    return {"records": records, "count": len(records)}


# ── Health ───────────────────────────────────────────────────

@app.get("/api/health")
@cached(ttl_seconds=5.0)
async def health():
    from api import db, job_queue

    pg_ok = await db.check_health()
    redis_ok = await job_queue.check_health()
    running = await job_queue.running_count()

    return {
        "status": "ok",
        "postgres": pg_ok,
        "redis": redis_ok,
        "simulations": len(manager.simulations),
        "running": running,
        "max_concurrent": int(os.getenv("DTS_MAX_CONCURRENT", "4")),
        "version": "1.0.0",
    }
