"""FastAPI application — DigitalTwinSim API."""

import json
import os
import sys
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.models import SimulationRequest, BranchRequest, ObservationInput
from api.simulation_manager import SimulationManager
from api.document_processor import save_uploaded_file, process_uploads

app = FastAPI(title="DigitalTwinSim API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SimulationManager()

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPORTS_DIR = os.path.join(OUTPUTS_DIR, "exports")


# ── Domains ──────────────────────────────────────────────────

@app.get("/api/domains")
async def list_domains():
    from domains.domain_registry import DomainRegistry
    DomainRegistry.discover()
    return {"domains": DomainRegistry.list_domains()}


# ── Simulations ──────────────────────────────────────────────

@app.post("/api/simulations")
async def create_simulation(request: SimulationRequest):
    sim_id = await manager.launch(request)
    return {"id": sim_id, "status": "queued"}


@app.post("/api/simulations/with-documents")
async def create_simulation_with_documents(
    brief: str = Form(...),
    provider: str = Form("gemini"),
    domain: Optional[str] = Form(None),
    rounds: Optional[int] = Form(None),
    budget: float = Form(5.0),
    elite_only: bool = Form(False),
    documents: list[UploadFile] = File(default=[]),
):
    """Launch simulation with uploaded documents for RAG grounding.

    Accepts multipart/form-data with:
    - brief: scenario description
    - documents: PDF, DOCX, TXT, MD, JSON files
    - All other SimulationRequest fields as form fields

    Documents are extracted and injected as context into the briefing pipeline.
    JSON files with stakeholder data are parsed as structured seed data.
    """
    from uuid import uuid4
    sim_id = str(uuid4())[:8]

    # Save uploaded documents
    doc_count = 0
    for doc in documents:
        if doc.filename and doc.size and doc.size > 0:
            content = await doc.read()
            save_uploaded_file(sim_id, doc.filename, content)
            doc_count += 1

    # Process uploads
    doc_result = process_uploads(sim_id) if doc_count > 0 else None

    request = SimulationRequest(
        brief=brief,
        provider=provider,
        domain=domain,
        rounds=rounds,
        budget=budget,
        elite_only=elite_only,
    )

    actual_id = await manager.launch(request, sim_id=sim_id, document_context=doc_result)
    return {
        "id": actual_id,
        "status": "queued",
        "documents_processed": doc_count,
        "context_chars": doc_result["total_chars"] if doc_result else 0,
    }


@app.get("/api/simulations")
async def list_simulations():
    return manager.list_simulations()


@app.get("/api/simulations/{sim_id}")
async def get_simulation(sim_id: str):
    status = manager.get_status(sim_id)
    if not status:
        raise HTTPException(404, "Simulation not found")
    return status


@app.get("/api/simulations/{sim_id}/stream")
async def stream_simulation(sim_id: str):
    state = manager.simulations.get(sim_id)
    if not state:
        raise HTTPException(404, "Simulation not found")

    async def event_generator():
        async for event in manager.stream_events(sim_id):
            yield {
                "event": event.type,
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_generator())


@app.post("/api/scenarios/{scenario_id}/branch")
async def branch_scenario(scenario_id: str, request: BranchRequest):
    """Create a What-If branch from a completed scenario."""
    # Validate parent exists
    export_dir = os.path.join(EXPORTS_DIR, f"scenario_{scenario_id}")
    if not os.path.exists(export_dir):
        raise HTTPException(404, f"Scenario not found: {scenario_id}")
    # Force parent_scenario_id to match URL
    request.parent_scenario_id = scenario_id
    sim_id = await manager.launch_branch(request)
    return {"id": sim_id, "status": "queued", "branch_from": scenario_id}


@app.delete("/api/simulations/{sim_id}")
async def cancel_simulation(sim_id: str):
    ok = await manager.cancel(sim_id)
    if not ok:
        raise HTTPException(400, "Cannot cancel this simulation")
    return {"status": "cancelled"}


# ── Scenarios (completed exports) ────────────────────────────

@app.get("/api/scenarios")
async def list_scenarios():
    manifest = os.path.join(EXPORTS_DIR, "scenarios.json")
    if not os.path.exists(manifest):
        return []
    with open(manifest) as f:
        return json.load(f)


@app.get("/api/scenarios/{scenario_id}/{filename}")
async def get_scenario_file(scenario_id: str, filename: str):
    # Sanitize
    if ".." in scenario_id or ".." in filename:
        raise HTTPException(400, "Invalid path")
    if not filename.endswith(".json") and not filename.endswith(".md"):
        raise HTTPException(400, "Only .json and .md files allowed")

    path = os.path.join(EXPORTS_DIR, f"scenario_{scenario_id}", filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {filename}")

    if filename.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    else:
        return FileResponse(path, media_type="text/markdown")


# ── Online observations (EnKF) ───────────────────────────────

@app.post("/api/simulations/{sim_id}/observe")
async def submit_observation(sim_id: str, observation: ObservationInput):
    """Submit a real-world observation for EnKF data assimilation.

    Only works for simulations launched with online_mode=True.
    Returns updated prediction with confidence interval and ensemble health.
    """
    state = manager.simulations.get(sim_id)
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
        # Run assimilation step
        obs_value = observation.pro_pct / 100.0  # Convert pct to [0,1]
        obs_noise = 1.0 / max(observation.sample_size or 1000, 100) * 10  # Rough noise from sample size

        enkf.assimilate(obs_value, obs_noise)

        # Get updated ensemble stats
        ensemble_mean = float(enkf.ensemble_mean()) * 100
        ensemble_std = float(enkf.ensemble_std()) * 100
        effective_size = getattr(enkf, "effective_sample_size", lambda: len(enkf.ensemble))()

        # Compute CI reduction vs prior
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


# ── Health ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "simulations": len(manager.simulations)}
