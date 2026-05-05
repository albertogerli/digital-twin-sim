"""DORA Sprint D.4 — replay an historical incident through the sim
and write back the ACTUAL Σ |shock_magnitude × shock_direction| as
the incident's `shock_units` calibration value.

Replaces the hand-waved tier estimates (1.4, 2.4, etc.) with a
self-consistent measure: every incident's α-calibration x-value is
"what the simulator says when fed the incident's brief".

Workflow:
  python -m scripts.replay_historical_incident <incident_id>
  python -m scripts.replay_historical_incident --all
  python -m scripts.replay_historical_incident --dry-run

For each incident:
  1. Look up the incident's brief in shared/dora_replay_briefs.json
     (one entry per incident_id; create new entries by adding to the
     JSON. The shape is { incident_id: { brief, domain, num_rounds,
     budget_usd, _notes } }).
  2. Spin up an in-process simulation engine (no API call) using the
     same code path /api/simulations uses.
  3. Run N rounds (typically 5-9), collect each round's
     shock_magnitude and shock_direction.
  4. Compute Σ |s_mag × s_dir|.
  5. Update shared/dora_reference_incidents.json in-place setting
     `shock_units` to the measured value (and append a `_replay_meta`
     field with sim_id + measured_at timestamp + brief_hash).

Cost per incident ≈ same as a normal sim ($0.40-$0.90). Full-corpus
backfill (40 incidents) ≈ $20-30 + ~3h wall time.

This script is the empirical anchor under everything else: once it's
been run end-to-end, the entire α-calibration pipeline becomes self-
consistent — there's no longer a "shock_units estimated by hand"
caveat in the calibration notes.

Status: INFRASTRUCTURE landed in Sprint D.4. Sample briefs for 6
representative incidents shipped (1 per category). Full backfill is
the operator's call (--all flag). Until then, _calibrated_alpha keeps
using the heuristic shock_units already in the JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REFERENCE_PATH = REPO_ROOT / "shared" / "dora_reference_incidents.json"
BRIEFS_PATH = REPO_ROOT / "shared" / "dora_replay_briefs.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_briefs() -> dict:
    if not BRIEFS_PATH.exists():
        logger.warning(f"briefs file missing: {BRIEFS_PATH} — returning empty")
        return {}
    return json.loads(BRIEFS_PATH.read_text()).get("briefs", {})


def _load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text())


async def _run_sim_and_measure(brief: str, domain: str = "financial",
                                num_rounds: int = 5, budget: float = 1.0) -> tuple[float, dict]:
    """Run the brief through an in-process sim, return (Σ|shock|, meta).

    Uses the same orchestrator path /api/simulations uses. No HTTP
    layer; results never leave the local process.
    """
    # Lazy imports — keeps the script importable for --dry-run testing
    # even on machines without the full sim deps.
    from api.simulation_manager import SimulationManager
    from api.models import SimulationRequest

    mgr = SimulationManager()
    req = SimulationRequest(
        brief=brief, domain=domain, provider="gemini",
        num_rounds=num_rounds, budget_usd=budget,
    )
    sim_id = await mgr.launch(req, tenant_id="dora_replay")
    state = mgr.simulations[sim_id]
    # Wait for completion (poll the in-memory state — replay is local).
    while state.status not in ("completed", "failed", "cancelled"):
        await asyncio.sleep(2.0)
    if state.status != "completed":
        raise RuntimeError(f"sim {sim_id} ended {state.status}: {state.error}")

    # Read replay_round_*.json from the export dir to extract shocks.
    safe_name = state.scenario_id
    if not safe_name:
        raise RuntimeError(f"sim {sim_id} has no scenario_id; export likely failed")
    export_dir = REPO_ROOT / "outputs" / "exports" / f"scenario_{safe_name}"
    total_shock = 0.0
    rounds_seen = 0
    for fn in sorted(export_dir.iterdir()):
        if not (fn.name.startswith("replay_round_") and fn.name.endswith(".json")):
            continue
        r = json.loads(fn.read_text())
        ev = r.get("event") or {}
        if isinstance(ev, dict):
            sm = float(ev.get("shock_magnitude", 0) or 0)
            sd = float(ev.get("shock_direction", 0) or 0)
            total_shock += abs(sm * sd)
            rounds_seen += 1
    return total_shock, {
        "sim_id": sim_id,
        "scenario_id": safe_name,
        "rounds_seen": rounds_seen,
        "cost_usd": state.cost,
    }


def _update_reference(incident_id: str, shock_units: float, meta: dict, brief: str) -> None:
    data = _load_reference()
    bh = hashlib.sha256(brief.encode("utf-8")).hexdigest()[:12]
    for inc in data.get("incidents", []):
        if inc.get("id") == incident_id:
            old = inc.get("shock_units")
            inc["shock_units"] = round(shock_units, 4)
            inc["_replay_meta"] = {
                "measured_shock_units": round(shock_units, 4),
                "previous_shock_units": old,
                "sim_id": meta.get("sim_id"),
                "scenario_id": meta.get("scenario_id"),
                "brief_sha256_12": bh,
                "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rounds_used": meta.get("rounds_seen"),
                "cost_usd": meta.get("cost_usd"),
            }
            REFERENCE_PATH.write_text(json.dumps(data, indent=2))
            logger.info(
                f"updated {incident_id}: shock_units {old} → {round(shock_units, 4)} "
                f"(sim_id={meta.get('sim_id')}, cost ${meta.get('cost_usd', 0):.3f})"
            )
            return
    raise KeyError(f"incident_id {incident_id} not found in {REFERENCE_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("incident_id", nargs="?", help="Single incident to replay")
    parser.add_argument("--all", action="store_true", help="Replay every incident in the briefs file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run; no sim calls")
    args = parser.parse_args()

    briefs = _load_briefs()
    if not briefs:
        logger.error(f"No briefs in {BRIEFS_PATH}. Add briefs and re-run.")
        return 1

    if args.all:
        targets = list(briefs.keys())
    elif args.incident_id:
        if args.incident_id not in briefs:
            logger.error(f"incident_id {args.incident_id!r} not in briefs ({len(briefs)} available)")
            logger.info("Available: " + ", ".join(sorted(briefs)[:10]) + (" …" if len(briefs) > 10 else ""))
            return 1
        targets = [args.incident_id]
    else:
        parser.print_help()
        return 1

    if args.dry_run:
        logger.info(f"DRY RUN — would replay {len(targets)} incident(s):")
        for t in targets:
            b = briefs[t]
            logger.info(f"  {t}: domain={b.get('domain', 'financial')}  rounds={b.get('num_rounds', 5)}  budget=${b.get('budget_usd', 1.0)}")
        return 0

    failures: list[str] = []
    for t in targets:
        brief_entry = briefs[t]
        try:
            logger.info(f"replaying {t}…")
            total_shock, meta = asyncio.run(_run_sim_and_measure(
                brief=brief_entry["brief"],
                domain=brief_entry.get("domain", "financial"),
                num_rounds=int(brief_entry.get("num_rounds", 5)),
                budget=float(brief_entry.get("budget_usd", 1.0)),
            ))
            _update_reference(t, total_shock, meta, brief_entry["brief"])
        except Exception as e:
            logger.exception(f"FAIL {t}: {e}")
            failures.append(t)

    if failures:
        logger.error(f"failures: {failures}")
        return 1
    logger.info("all incidents replayed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
