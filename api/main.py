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

from pydantic import BaseModel, Field
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
import logging
logger = logging.getLogger(__name__)

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


def _tenant_id(tenant: Optional[Tenant], request: Optional[Request] = None) -> str:
    """Extract tenant_id, with this priority:

      1. Forwarded `X-Tenant-Id` header (set by the Next.js Edge middleware
         from the cookie's `sub` claim — this is how invite-link sessions
         get isolated workspaces).
      2. The Tenant.tenant_id from the API-key auth dependency.
      3. "default" fallback (auth disabled).

    The header is trusted because the Next.js layer only sets it after
    verifying the HMAC-signed session cookie — clients can't spoof it
    when going through Next, and a direct backend hit needs a valid
    API key anyway.
    """
    if request is not None:
        forwarded = request.headers.get("x-tenant-id")
        if forwarded:
            return forwarded.strip()[:64]
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
    sim_id = await manager.launch(body, tenant_id=_tenant_id(tenant, request))
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
    tid = _tenant_id(tenant, request)

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

    # Process uploads + build per-simulation RAG store (in-memory, lives for run)
    rag_store = None
    if doc_count > 0:
        try:
            from api.rag_store import RAGStore
            rag_store = RAGStore()
        except Exception as exc:
            logger.warning(f"RAGStore init failed: {exc}")
    doc_result = process_uploads(sim_id, rag_store=rag_store) if doc_count > 0 else None

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
        sim_request, sim_id=sim_id, document_context=doc_result, rag_store=rag_store, tenant_id=tid,
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
    return manager.list_simulations(tenant_id=_tenant_id(tenant, request))


@app.get("/api/simulations/{sim_id}")
@limiter.limit(LIMIT_READS)
async def get_simulation(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    status = manager.get_status(sim_id, tenant_id=_tenant_id(tenant, request))
    if not status:
        raise HTTPException(404, "Simulation not found")
    return status


@app.get("/api/simulations/{sim_id}/stream")
async def stream_simulation(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
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
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
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
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
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
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
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
    sim_id = await manager.launch_branch(body, tenant_id=_tenant_id(tenant, request))
    return {"id": sim_id, "status": "queued", "branch_from": scenario_id}


@app.delete("/api/simulations/{sim_id}")
async def cancel_simulation(
    request: Request,
    sim_id: str,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
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
    tid = _tenant_id(tenant, request)
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
    state = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
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
    records = await db.get_usage(tenant_id=_tenant_id(tenant, request), since=since)
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


# ── Compliance: BYOD enclave + DORA export ───────────────────────

@app.get("/api/compliance/byod/status")
@cached(ttl_seconds=5.0)
async def byod_status():
    """Current BYOD mode + aggregate audit summary. Public diagnostic
    so the frontend can show a status badge without credentials."""
    from core.byod.sanitizer import audit_summary, get_mode
    summary = audit_summary()
    return {
        "mode": get_mode().value,
        "n_audit_rows": summary["n_rows"],
        "by_site": summary["by_site"],
        "by_category": summary["by_category"],
    }


@app.get("/api/compliance/byod/audit")
@limiter.limit(LIMIT_READS)
async def byod_audit_recent(
    request: Request,
    limit: int = 50,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Tail the BYOD audit log (last N rows). Used by the /compliance
    page to render a recent-leakage table."""
    import json as _json
    from core.byod.sanitizer import DEFAULT_AUDIT_PATH
    if not DEFAULT_AUDIT_PATH.exists():
        return {"rows": [], "count": 0, "path": str(DEFAULT_AUDIT_PATH)}
    rows = []
    with DEFAULT_AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
    rows = rows[-max(1, min(limit, 1000)):]
    return {"rows": rows, "count": len(rows)}


class _ByodTestBody(BaseModel):
    prompt: str = Field(..., max_length=20_000)
    mode: Optional[str] = None  # OFF / LOG / STRICT / BLOCK


@app.post("/api/compliance/byod/test")
@limiter.limit(LIMIT_READS)
async def byod_test(
    body: _ByodTestBody,
    request: Request,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Interactive sanitizer playground — paste a prompt, see what would
    be redacted in STRICT mode without actually emitting an audit row."""
    from core.byod.sanitizer import (
        BYODMode, BYODLeakError, sanitize_prompt,
    )
    requested = (body.mode or "STRICT").upper()
    try:
        mode = BYODMode(requested)
    except ValueError:
        raise HTTPException(400, f"Invalid mode '{requested}' — use OFF/LOG/STRICT/BLOCK")
    try:
        # Use a tmp audit path so the playground never pollutes real audit log
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            tmp_path = Path(tf.name)
        try:
            res = sanitize_prompt(
                body.prompt,
                call_site="ui:byod_test",
                mode=mode,
                audit_path=tmp_path,
                tenant="ui-playground",
            )
            return {
                "mode": res.mode,
                "input_chars": len(body.prompt),
                "output_chars": len(res.text),
                "sanitized_text": res.text,
                "modified": res.modified,
                "detections": res.detections,
            }
        finally:
            tmp_path.unlink(missing_ok=True)
    except BYODLeakError as e:
        return {
            "mode": "BLOCK",
            "blocked": True,
            "reason": str(e),
            "detections": e.detections,
        }


def _derive_dora_metrics_from_export(sim) -> dict:
    """Compute the 8 DORA-classification inputs from a completed sim's export.

    The export pipeline writes per-scenario JSONs under
    outputs/exports/scenario_<scenario_id>/ (polarization.json,
    replay_round_N.json, agents.json, top_posts.json). We read them and
    derive DORA's 7-criterion inputs as best-effort proxies, since the
    sim doesn't natively produce DORA fields:

      polarization_peak       max polarization across rounds  (DIRECT)
      viral_posts_count       count of posts with engagement >= P95
                              threshold across the run        (DIRECT)
      countries_affected      distinct countries in agents.json with at
                              least one negatively-positioned member
                                                              (PROXY)
      affected_core_functions count institutional agents ending in pos
                              < -0.3                          (PROXY)
      customers_affected      sum citizen-cluster populations × negative
                              sentiment share                 (PROXY)
      economic_impact_eur     |sum shock_magnitude × shock_direction| ×
                              CALIBRATION_EUR_PER_SHOCK_UNIT  (PROXY)
      data_records_lost       0 (sim has no data-loss concept) (N/A)
      downtime_hours          0 unless the brief mentions outage —
                              future enhancement              (N/A)

    All derivations are LOGGED so a CRO can audit how each number was
    obtained. Falls back to all-zeros if the export is missing.
    """
    # Map sim → export dir. SimulationState.scenario_id is the safe_name
    # without the "scenario_" prefix; export pipeline adds the prefix.
    scenario_id = getattr(sim, "scenario_id", None)
    if not scenario_id:
        logger.info(f"DORA metrics: sim {sim.id} has no scenario_id; returning zeros")
        return {}
    export_dir = os.path.join(EXPORTS_DIR, f"scenario_{scenario_id}")
    if not os.path.isdir(export_dir):
        logger.info(f"DORA metrics: export dir not found at {export_dir}; returning zeros")
        return {}

    metrics: dict = {}

    # 1. Peak polarization (max across rounds)
    pol_path = os.path.join(export_dir, "polarization.json")
    if os.path.isfile(pol_path):
        try:
            series = json.load(open(pol_path))
            if isinstance(series, list) and series:
                peak = max(float(r.get("polarization", 0) or 0) for r in series)
                metrics["polarization_peak"] = round(peak, 3)
        except Exception as e:
            logger.warning(f"DORA: polarization parse failed: {e}")

    # 2-3. Walk all replay_round_*.json — viral posts + shock accumulation
    total_shock = 0.0
    viral_count = 0
    high_engagement_threshold = 5  # posts with engagement >= 5 considered viral
    rounds_data = []
    try:
        for fn in sorted(os.listdir(export_dir)):
            if not (fn.startswith("replay_round_") and fn.endswith(".json")):
                continue
            r = json.load(open(os.path.join(export_dir, fn)))
            rounds_data.append(r)
            ev = r.get("event") or {}
            if isinstance(ev, dict):
                sm = float(ev.get("shock_magnitude", 0) or 0)
                sd = float(ev.get("shock_direction", 0) or 0)
                total_shock += abs(sm * sd)
            for p in (r.get("posts") or []):
                eng = (p.get("likes", 0) or 0) + (p.get("reposts", 0) or 0) * 2 + (p.get("reply_count", 0) or 0) * 3
                if eng >= high_engagement_threshold:
                    viral_count += 1
    except Exception as e:
        logger.warning(f"DORA: replay walk failed: {e}")
    metrics["viral_posts_count"] = viral_count

    # 6-pre. Collect ticker_price history for Method B (economic impact
    # via direct market-cap loss). Tries the per-round replay first
    # (newer scenarios export ticker_prices) and falls back to scanning
    # checkpoint state_*.json files (where round_manager patches them).
    ticker_history: list[dict] = []
    for r in rounds_data:
        tp = r.get("ticker_prices")
        if isinstance(tp, dict) and tp:
            ticker_history.append(tp)
    if not ticker_history:
        try:
            cp_glob = os.path.join(OUTPUTS_DIR, f"state_{scenario_id}_r*.json")
            import glob as _glob
            for cp_path in sorted(_glob.glob(cp_glob)):
                try:
                    cp = json.load(open(cp_path))
                    tp = cp.get("ticker_prices")
                    if isinstance(tp, dict) and tp:
                        ticker_history.append(tp)
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"DORA: checkpoint ticker_prices scan skipped: {e}")

    # 4-5. Country & institutional impact from agents.json
    ag_path = os.path.join(export_dir, "agents.json")
    if os.path.isfile(ag_path):
        try:
            agents_doc = json.load(open(ag_path))
            agents_list = agents_doc if isinstance(agents_doc, list) else (agents_doc.get("agents") or [])
            negative_countries: set[str] = set()
            affected_functions = 0
            customers = 0
            negative_share = 0
            total_pop = 0
            for a in agents_list:
                pos = float(a.get("position", 0) or 0)
                country = (a.get("country") or "").strip()
                category = (a.get("category") or a.get("archetype") or "").lower()
                # Country count: any country with at least one negative agent
                if country and pos < -0.2:
                    negative_countries.add(country.upper())
                # Affected core functions: institutional / regulator / central
                # agents ending negatively-shifted
                if pos < -0.3 and category in (
                    "institutional", "regulator", "central_banker", "ministry",
                    "eu_commissioner", "policymaker"
                ):
                    affected_functions += 1
                # Customers: citizen clusters carry a population field
                if category in ("citizen_cluster", "citizen", "cluster"):
                    pop = int(a.get("population", a.get("size", 100)) or 100)
                    total_pop += pop
                    if pos < 0:
                        negative_share += pop
            metrics["countries_affected"] = max(1, len(negative_countries))
            metrics["affected_core_functions"] = affected_functions
            # Customers-affected proxy: portion of citizen population with
            # negative final position. Multiplier 1000 makes the magnitude
            # comparable to a real-bank "thousands of clients" scale.
            customers_proxy = int(negative_share * 1000) if total_pop > 0 else 0
            metrics["customers_affected"] = customers_proxy
        except Exception as e:
            logger.warning(f"DORA: agents parse failed: {e}")

    # 6. Economic impact — combined Method A (calibrated shock anchor)
    # + Method B (direct ticker market-cap loss × contagion γ).
    try:
        from core.dora.economic_impact import (
            estimate_anchor, estimate_ticker, combine,
        )
        anchor_est = estimate_anchor(total_shock_units=total_shock)
        ticker_est = estimate_ticker(ticker_price_history=ticker_history)
        combined = combine(anchor_est, ticker_est)
        metrics["economic_impact_eur"] = float(combined.get("point_eur", 0.0))
        # Expose full breakdown so the UI can render the methodology
        # transparently (selected method, low/high band, per-ticker losses,
        # calibration-α reference incidents).
        metrics["economic_impact_breakdown"] = combined
    except Exception as e:
        logger.warning(f"DORA: economic_impact computation failed ({e}); falling back to legacy 50M anchor")
        metrics["economic_impact_eur"] = round(total_shock * 50_000_000.0, 2)
        metrics["economic_impact_breakdown"] = None

    # 7-8. Sim has no native data-loss / downtime — leave as 0
    metrics.setdefault("data_records_lost", 0)
    metrics.setdefault("downtime_hours", 0.0)

    logger.info(
        f"DORA metrics for {sim.id} (scenario_id={scenario_id}): "
        f"polarization_peak={metrics.get('polarization_peak')}, "
        f"viral={metrics.get('viral_posts_count')}, "
        f"countries={metrics.get('countries_affected')}, "
        f"affected_fn={metrics.get('affected_core_functions')}, "
        f"customers={metrics.get('customers_affected')}, "
        f"eur={metrics.get('economic_impact_eur')}"
    )
    return metrics


@app.get("/api/compliance/dora/preview/{sim_id}")
@limiter.limit(LIMIT_READS)
async def dora_preview(
    sim_id: str,
    request: Request,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Preview the DORA classification for a completed simulation
    *without* emitting the full XML. Used by the /compliance page to
    render the 7-criterion grid before letting the user download."""
    from core.dora.classification import classify_from_simulation
    sim = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
    if not sim:
        raise HTTPException(404, "Simulation not found")
    if sim.status != "completed":
        raise HTTPException(400, "Simulation must be completed to generate DORA report")

    metrics = _derive_dora_metrics_from_export(sim)
    cri = classify_from_simulation(
        customers_affected=int(metrics.get("customers_affected", 0)),
        economic_impact_eur=float(metrics.get("economic_impact_eur", 0.0)),
        countries_affected=int(metrics.get("countries_affected", 1)),
        polarization_peak=float(metrics.get("polarization_peak", 0.0)),
        viral_posts_count=int(metrics.get("viral_posts_count", 0)),
        data_records_lost=int(metrics.get("data_records_lost", 0)),
        affected_core_functions=int(metrics.get("affected_core_functions", 0)),
        downtime_hours=float(metrics.get("downtime_hours", 0.0)),
    )
    return {
        "sim_id": sim_id,
        "scenario_name": (getattr(sim, "scenario_name", None) or sim_id),
        "classification": {
            "clients_affected": cri.clients_affected,
            "data_losses": cri.data_losses,
            "reputational_impact": cri.reputational_impact,
            "duration_downtime_hours": cri.duration_downtime_hours,
            "geographical_spread": cri.geographical_spread,
            "economic_impact_eur_band": cri.economic_impact_eur_band,
            "criticality_of_services_affected": cri.criticality_of_services_affected,
        },
        "is_major": cri.is_major(),
        "metrics_used": metrics,
    }


@app.get("/api/compliance/dora/export/{sim_id}")
@limiter.limit(LIMIT_READS)
async def dora_export(
    sim_id: str,
    request: Request,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Download the DORA Major Incident Report XML for a completed sim."""
    from datetime import datetime, timezone
    from fastapi.responses import Response
    from core.dora.classification import classify_from_simulation
    from core.dora.exporter import build_incident_report
    from core.dora.schema import (
        FinancialEntity, IncidentType, ReportType, RootCauseCategory,
    )

    sim = manager.get_state(sim_id, tenant_id=_tenant_id(tenant, request))
    if not sim:
        raise HTTPException(404, "Simulation not found")
    if sim.status != "completed":
        raise HTTPException(400, "Simulation must be completed to generate DORA report")

    metrics = _derive_dora_metrics_from_export(sim)
    cri = classify_from_simulation(
        customers_affected=int(metrics.get("customers_affected", 0)),
        economic_impact_eur=float(metrics.get("economic_impact_eur", 0.0)),
        countries_affected=int(metrics.get("countries_affected", 1)),
        polarization_peak=float(metrics.get("polarization_peak", 0.0)),
        viral_posts_count=int(metrics.get("viral_posts_count", 0)),
        data_records_lost=int(metrics.get("data_records_lost", 0)),
        affected_core_functions=int(metrics.get("affected_core_functions", 0)),
        downtime_hours=float(metrics.get("downtime_hours", 0.0)),
    )

    # Tenant-derived entity placeholder — real deployment overrides this
    # via env vars or a tenant config table.
    entity = FinancialEntity(
        legal_name=os.getenv("DORA_ENTITY_NAME", "Tenant Entity (placeholder)"),
        lei_code=os.getenv("DORA_ENTITY_LEI", "00000000000000000000"),
        competent_authority=os.getenv("DORA_COMPETENT_AUTHORITY", "Banca d'Italia"),
        country=os.getenv("DORA_ENTITY_COUNTRY", "IT"),
    )

    now = datetime.now(timezone.utc)
    xml = build_incident_report(
        reference_number=f"{(getattr(sim, 'name', sim_id) or sim_id)[:30]}-{sim_id[:8]}",
        report_type=ReportType.FINAL,
        entity=entity,
        classification=cri,
        incident_type=IncidentType.AVAILABILITY,
        root_cause_category=RootCauseCategory.SYSTEM_FAILURE,
        root_cause_description=getattr(sim, "brief", "Simulator-derived crisis scenario.")[:1800],
        detected_at=now,
        classified_at=now,
        customers_affected=int(metrics.get("customers_affected", 0)) or None,
        economic_impact_eur=float(metrics.get("economic_impact_eur", 0.0)) or None,
        notified_clients=bool(metrics.get("notified_clients", False)),
        public_communication_issued=bool(metrics.get("public_communication_issued", False)),
        permanent_remediation_summary=metrics.get("permanent_remediation_summary"),
        lessons_learned=metrics.get("lessons_learned"),
    )
    filename = f"dora_incident_{sim_id[:8]}.xml"
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Self-calibration (continuous) ─────────────────────────────

@app.get("/api/compliance/calibration/summary")
@cached(ttl_seconds=30.0)
async def calibration_summary():
    """Running aggregate from the continuous self-calibration loop —
    n forecasts/evaluations, MAE per horizon, direction acc, per-ticker."""
    from core.calibration.continuous import running_summary
    s = running_summary()
    return {
        "n_forecasts": s.n_forecasts,
        "n_evaluations": s.n_evaluations,
        "last_forecast_date": s.last_forecast_date,
        "last_evaluation_date": s.last_evaluation_date,
        "mae_t1_running": s.mae_t1_running,
        "mae_t3_running": s.mae_t3_running,
        "mae_t7_running": s.mae_t7_running,
        "direction_acc_t1": s.direction_acc_t1,
        "by_ticker": s.by_ticker,
    }


@app.get("/api/compliance/calibration/recent")
@cached(ttl_seconds=10.0)
async def calibration_recent(limit: int = 30):
    """Tail recent evaluations for the UI table."""
    from core.calibration.continuous import recent_evaluations
    rows = recent_evaluations(limit=max(1, min(limit, 200)))
    return {"rows": rows, "count": len(rows)}


# ── Admin jobs (background tasks) ────────────────────────────────

import threading as _threading
import subprocess as _subprocess

# In-memory job status registry (process-local).
# Key: job_name, value: {state, started_at, finished_at, exit_code, output_tail, pid}
_admin_jobs_state: dict[str, dict] = {}
_admin_jobs_lock = _threading.Lock()

_PY = sys.executable or "python"

_ADMIN_JOBS = {
    "calibration-forecast": {
        "label": "Self-calibration forecast",
        "description": "Fetch headlines for the watchlist, run shadow forecast, persist to SQLite.",
        "cmds": [[_PY, "-m", "scripts.continuous_calibration", "forecast"]],
        "icon": "auto_graph",
    },
    "calibration-evaluate": {
        "label": "Self-calibration T+1/T+3/T+7 evaluate",
        "description": "Score pending forecasts against realised yfinance returns.",
        "cmds": [
            [_PY, "scripts/continuous_calibration.py", "evaluate", "--horizon", str(h)]
            for h in (1, 3, 7)
        ],
        "icon": "verified",
    },
    "stakeholder-update": {
        "label": "Stakeholder graph nightly update",
        "description": "Crawl RSS + Google News, EMA-update agent positions on detected mentions.",
        "cmds": [[_PY, "-m", "stakeholder_graph.updater"]],
        "icon": "groups",
    },
}


def _run_admin_job_in_thread(job_name: str):
    job = _ADMIN_JOBS.get(job_name)
    if not job:
        return
    with _admin_jobs_lock:
        _admin_jobs_state[job_name] = {
            "state": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_at": None,
            "exit_code": None,
            "output_tail": [],
        }
    try:
        all_output: list[str] = []
        last_code = 0
        for cmd in job["cmds"]:
            try:
                proc = _subprocess.run(
                    cmd,
                    cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                    timeout=600,
                )
                all_output.extend(proc.stdout.splitlines()[-50:])
                all_output.extend([f"[stderr] {l}" for l in proc.stderr.splitlines()[-20:]])
                last_code = proc.returncode
                if proc.returncode != 0:
                    break
            except _subprocess.TimeoutExpired:
                all_output.append(f"[TIMEOUT] command exceeded 600s: {' '.join(cmd)}")
                last_code = 124
                break
            except Exception as e:
                all_output.append(f"[EXCEPTION] {e}")
                last_code = 1
                break
        with _admin_jobs_lock:
            _admin_jobs_state[job_name].update({
                "state": "completed" if last_code == 0 else "failed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "exit_code": last_code,
                "output_tail": all_output[-100:],
            })
    except Exception as e:
        with _admin_jobs_lock:
            _admin_jobs_state[job_name].update({
                "state": "failed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "exit_code": -1,
                "output_tail": [f"[FATAL] {e}"],
            })


@app.get("/api/admin/jobs")
@cached(ttl_seconds=3.0)
async def admin_jobs_status():
    """Return registry of available admin jobs + their last-run status."""
    with _admin_jobs_lock:
        states = dict(_admin_jobs_state)
    return {
        "jobs": [
            {
                "name": name,
                "label": meta["label"],
                "description": meta["description"],
                "icon": meta["icon"],
                "last_run": states.get(name),
            }
            for name, meta in _ADMIN_JOBS.items()
        ]
    }


@app.post("/api/admin/jobs/{job_name}/run")
@limiter.limit(LIMIT_READS)
async def admin_jobs_run(
    job_name: str,
    request: Request,
    tenant: Optional[Tenant] = Depends(verify_api_key),
):
    """Trigger an admin job in a background thread. Returns immediately."""
    if job_name not in _ADMIN_JOBS:
        raise HTTPException(404, f"Unknown job '{job_name}'. Known: {list(_ADMIN_JOBS)}")
    with _admin_jobs_lock:
        current = _admin_jobs_state.get(job_name, {})
        if current.get("state") == "running":
            raise HTTPException(409, f"Job '{job_name}' is already running")
    t = _threading.Thread(target=_run_admin_job_in_thread, args=(job_name,), daemon=True)
    t.start()
    return {"job": job_name, "state": "queued", "message": "Job started in background"}


# ── Invite usage analytics ─────────────────────────────────────────
# Two storage layers, both append-only on disk so they survive Railway
# redeploys when /app/outputs is mounted as a persistent volume:
#   1. _INVITE_REDEMPTIONS_PATH: one JSONL line per /api/auth/invite/redeem
#      success — captures (sub, label, redeemed_at, ua_hint) so we can show
#      "X invitees clicked the link in the last 7 days" even if they
#      never ran a sim afterwards.
#   2. simulations table (already persisted) — we aggregate sim count
#      and last activity per tenant_id (= sub from the invite token).

_INVITE_REDEMPTIONS_PATH = os.path.join(PROJECT_ROOT, "outputs", "invite_redemptions.jsonl")


@app.post("/api/admin/invites/log-redemption")
async def log_invite_redemption(request: Request):
    """Append-only log of one invite redemption.

    Called by the Edge runtime /api/auth/invite/redeem after it
    validates the HMAC signature. We trust the frontend here because
    (a) the same Vercel project signs both the invite token and the
    log call, and (b) the worst case of a forged log is a misleading
    metric, not a security boundary leak.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid json body")
    sub = (body.get("sub") or "").strip()
    label = (body.get("label") or "").strip()
    if not sub:
        raise HTTPException(400, "sub required")
    entry = {
        "sub": sub,
        "label": label[:120],
        "redeemed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ua_hint": (request.headers.get("user-agent") or "")[:200],
        "ip_hint": (request.client.host if request.client else "")[:64],
    }
    try:
        os.makedirs(os.path.dirname(_INVITE_REDEMPTIONS_PATH), exist_ok=True)
        with open(_INVITE_REDEMPTIONS_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"invite redemption log write failed: {exc}")
        raise HTTPException(500, "log write failed")
    return {"ok": True}


@app.get("/api/admin/invites/stats")
@cached(ttl_seconds=15.0)
async def admin_invites_stats():
    """Per-invitee aggregate: redemptions + sim activity.

    Returns:
      {
        "total_redemptions": N,
        "unique_invitees": M,
        "redemptions_last_7d": K,
        "redemptions_last_30d": K,
        "users": [
          {sub, label, first_redeemed, last_redeemed, redemption_count,
           sim_count, last_sim_at, total_cost, sim_status_breakdown}
        ]
      }
    """
    # 1. Read redemption log (best-effort)
    redemptions_by_sub: dict[str, list[dict]] = {}
    try:
        if os.path.exists(_INVITE_REDEMPTIONS_PATH):
            with open(_INVITE_REDEMPTIONS_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        sub = e.get("sub", "")
                        if sub:
                            redemptions_by_sub.setdefault(sub, []).append(e)
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning(f"invite redemption log read failed: {exc}")

    # 2. Aggregate sim activity per tenant_id
    sims_by_tenant: dict[str, list] = {}
    for sim in manager.simulations.values():
        tid = sim.tenant_id
        if not tid or tid == "default":
            continue
        sims_by_tenant.setdefault(tid, []).append(sim)

    # 3. Merge — every tenant_id we've seen (either in log or in sims)
    all_subs = set(redemptions_by_sub) | set(sims_by_tenant)
    now = time.time()
    cutoff_7d = now - 7 * 24 * 3600
    cutoff_30d = now - 30 * 24 * 3600

    def _ts(s: str) -> float:
        try:
            return time.mktime(time.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError):
            return 0.0

    users = []
    rd_7d = 0
    rd_30d = 0
    for sub in sorted(all_subs):
        rds = redemptions_by_sub.get(sub, [])
        sims = sims_by_tenant.get(sub, [])
        # Pick most recent label across redemptions (or sim brief if no log)
        label = ""
        if rds:
            label = rds[-1].get("label") or ""
        first_rd = rds[0].get("redeemed_at") if rds else None
        last_rd = rds[-1].get("redeemed_at") if rds else None
        # Status breakdown
        status_counts: dict[str, int] = {}
        for s in sims:
            status_counts[s.status] = status_counts.get(s.status, 0) + 1
        # Last sim activity
        last_sim_at = None
        if sims:
            timestamps = [s.completed_at or s.created_at for s in sims if s.completed_at or s.created_at]
            if timestamps:
                last_sim_at = max(timestamps)
        # Total cost
        total_cost = sum(getattr(s, "cost", 0.0) or 0.0 for s in sims)
        users.append({
            "sub": sub,
            "label": label,
            "first_redeemed": first_rd,
            "last_redeemed": last_rd,
            "redemption_count": len(rds),
            "sim_count": len(sims),
            "last_sim_at": last_sim_at,
            "total_cost": round(total_cost, 4),
            "sim_status_breakdown": status_counts,
        })
        for r in rds:
            ts = _ts(r.get("redeemed_at", ""))
            if ts >= cutoff_7d:
                rd_7d += 1
            if ts >= cutoff_30d:
                rd_30d += 1

    # Sort users: most recently active first
    def _activity_key(u):
        return u.get("last_sim_at") or u.get("last_redeemed") or ""
    users.sort(key=_activity_key, reverse=True)

    total_redemptions = sum(len(v) for v in redemptions_by_sub.values())
    return {
        "total_redemptions": total_redemptions,
        "unique_invitees": len(redemptions_by_sub),
        "redemptions_last_7d": rd_7d,
        "redemptions_last_30d": rd_30d,
        "users": users,
        "users_total": len(users),
    }
