"""E2E batch runner — esegue gli scenari paper sul Railway production endpoint
e raccoglie risultati di validazione in un CSV.

Uso:
    python scripts/e2e_batch_paper_scenarios.py --select 10 --parallel 3
    python scripts/e2e_batch_paper_scenarios.py --all   # 44 scenari, ~7-10h

Output:
    outputs/e2e_batch_results.csv con campi: scenario_id, domain, country,
    sim_id, status, agents_count, has_financial_twin, has_reasoning_trace,
    breach_count, errors, duration_s.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("DTS_API", "https://digital-twin-sim-production.up.railway.app")
SCENARIOS_DIR = Path("calibration/empirical/scenarios")
OUT_CSV = Path("outputs/e2e_batch_results.csv")

# Selection mix per coverage (10 scenari, 1-3 per dominio principale)
SELECTION_10 = [
    "FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI",
    "FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202",
    "FIN-2021-GAMESTOP",
    "POL-2016-BREXIT",
    "POL-2016-REFERENDUM_COSTITUZIONALE_ITAL",
    "POL-2020-ELEZIONI_PRESIDENZIALI_USA_202",
    "CORP-2019-BOEING_MAX",
    "CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW",
    "PH-2021-COVID_VAX_IT",
    "TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E",
]


def http_get(path: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(f"{API_BASE}{path}",
                                  headers={"User-Agent": "DTS-e2e-batch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "DTS-e2e-batch/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def build_brief(scenario: dict) -> str:
    """Compose a brief from a paper scenario JSON."""
    title = scenario.get("title", "")
    notes = scenario.get("notes", "")
    domain = scenario.get("domain", "")
    country = scenario.get("country", "")
    events = scenario.get("events", []) or []
    event_lines = []
    for e in events[:6]:
        d = e.get("date", "")
        desc = e.get("description", "")
        event_lines.append(f"- ({d}) {desc}")
    parts = [f"E2E TEST — {title}"]
    if domain:
        parts.append(f"Domain: {domain}")
    if country:
        parts.append(f"Country: {country}")
    if notes:
        parts.append(f"Context: {notes}")
    if event_lines:
        parts.append("Key events:")
        parts.extend(event_lines)
    return "\n\n".join(parts)


def load_scenarios(only: list[str] | None = None) -> list[dict]:
    out = []
    for f in sorted(SCENARIOS_DIR.glob("*.json")):
        sid = f.stem
        if "manifest" in sid or "meta" in sid:
            continue
        if only is not None and sid not in only:
            continue
        try:
            with open(f) as fh:
                out.append(json.load(fh))
        except Exception as exc:
            print(f"  load fail {sid}: {exc}", file=sys.stderr)
    return out


def launch_one(scenario: dict) -> tuple[str, str]:
    sid_paper = scenario.get("id") or scenario.get("scenario_id", "?")
    brief = build_brief(scenario)
    payload = {
        "brief": brief,
        "provider": "gemini",
        "budget": 1.5,
    }
    res = http_post("/api/simulations", payload)
    return sid_paper, res.get("id", "")


def poll_status(sim_id: str) -> dict:
    return http_get(f"/api/simulations/{sim_id}")


def fetch_agents(sim_id: str, scenario_id: str) -> dict:
    """Fetch agents.json for the completed scenario. Returns {} on fail."""
    sid = scenario_id or sim_id
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)
    # Best-effort: try a few known patterns
    for path in [
        f"/api/scenarios/{safe}__{sim_id[:6]}/agents.json",
        f"/api/scenarios/{safe}/agents.json",
    ]:
        try:
            return {"path": path, "data": http_get(path)}
        except Exception:
            continue
    return {}


def fetch_round_1(sim_id: str, scenario_id: str) -> dict:
    sid = scenario_id or sim_id
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)
    for path in [
        f"/api/scenarios/{safe}__{sim_id[:6]}/replay_round_1.json",
        f"/api/scenarios/{safe}/replay_round_1.json",
    ]:
        try:
            return {"path": path, "data": http_get(path)}
        except Exception:
            continue
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--max-wait-min", type=int, default=20)
    args = ap.parse_args()

    if args.all:
        scenarios = load_scenarios(only=None)
    else:
        scenarios = load_scenarios(only=SELECTION_10[:args.select])
    print(f"Loaded {len(scenarios)} scenarios")
    if not scenarios:
        print("No scenarios to run.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id", "domain", "country", "sim_id", "status",
        "agents_count", "has_financial_twin", "has_reasoning_trace",
        "duration_s", "errors", "title",
    ]
    csv_file = open(OUT_CSV, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # Launch all (or in batches)
    in_flight = []  # list of (paper_id, sim_id, t_launch, scenario)
    pending = list(scenarios)
    completed = 0

    def launch_next():
        if not pending:
            return False
        scn = pending.pop(0)
        try:
            paper_id, sim_id = launch_one(scn)
            in_flight.append({
                "paper_id": paper_id, "sim_id": sim_id,
                "t_launch": time.time(), "scenario": scn,
            })
            print(f"  ▶ launched {paper_id} → sim_id={sim_id} ({len(in_flight)} in flight)")
            return True
        except Exception as exc:
            print(f"  ✗ launch FAILED for {scn.get('id', '?')}: {exc}")
            writer.writerow({
                "scenario_id": scn.get("id", "?"),
                "domain": scn.get("domain", ""),
                "country": scn.get("country", ""),
                "sim_id": "",
                "status": "launch_error",
                "errors": str(exc),
                "title": scn.get("title", ""),
            })
            csv_file.flush()
            return True

    # Initial parallel launch
    for _ in range(min(args.parallel, len(pending))):
        launch_next()

    # Poll loop
    while in_flight or pending:
        time.sleep(args.poll_interval)
        still = []
        for entry in in_flight:
            sim_id = entry["sim_id"]
            scn = entry["scenario"]
            paper_id = entry["paper_id"]
            t_launch = entry["t_launch"]
            elapsed = time.time() - t_launch
            try:
                status_data = poll_status(sim_id)
            except Exception as exc:
                print(f"  ⚠ poll failed for {sim_id}: {exc}")
                still.append(entry)
                continue
            status = status_data.get("status", "unknown")
            if status in ("completed", "failed", "cancelled") or elapsed > args.max_wait_min * 60:
                # Finalize
                agents_count = status_data.get("agents_count", 0)
                scenario_id = status_data.get("scenario_id") or ""
                has_ft = False
                has_rt = False
                # Probe round 1 for financial_twin presence
                r1 = fetch_round_1(sim_id, scenario_id)
                if r1.get("data"):
                    has_ft = "financial_twin" in r1["data"]
                # agents.json check
                agents_data = fetch_agents(sim_id, scenario_id)
                if agents_data.get("data"):
                    agents_count = len(agents_data["data"])
                writer.writerow({
                    "scenario_id": paper_id,
                    "domain": scn.get("domain", ""),
                    "country": scn.get("country", ""),
                    "sim_id": sim_id,
                    "status": status if elapsed <= args.max_wait_min * 60 else "timeout",
                    "agents_count": agents_count,
                    "has_financial_twin": has_ft,
                    "has_reasoning_trace": has_rt,
                    "duration_s": int(elapsed),
                    "errors": status_data.get("error", "") or "",
                    "title": scn.get("title", ""),
                })
                csv_file.flush()
                completed += 1
                print(f"  ✓ {paper_id} → {status} (agents={agents_count}, ALM={has_ft}, t={int(elapsed)}s)")
                # Slot freed → launch next
                launch_next()
            else:
                still.append(entry)
        in_flight = still
        if in_flight or pending:
            print(f"  ... {len(in_flight)} in flight, {len(pending)} queued, {completed} done")

    csv_file.close()
    print(f"\nDone. Results in {OUT_CSV}")


if __name__ == "__main__":
    main()
