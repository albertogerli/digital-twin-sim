"""Empirical ground-truth loader for real-world polling trajectories.

Wraps the `calibration/empirical/scenarios*` directories, which hold hand-curated
scenarios (Scottish independence, Boeing MAX, SVB collapse, etc.) with polling
trajectories sourced from actual pollsters (Morning Consult, Reuters/Ipsos,
YouGov, …). Each scenario carries a `ground_truth_outcome` verified against a
referendum/election/survey result.

The v2.2 directory is a refined subset of v1 that passed review. Where a file
appears in both, v2.2 wins.

Normalized trajectory convention:
    normalized_support = pro_pct / 100   ∈ [0, 1]
    signed_position    = (pro_pct - against_pct) / 100   ∈ [-1, 1]
These two series mirror the two metrics our sim already exposes
(avg_position ≈ signed_position, polarization ≈ dispersion).

Tagging convention for the scenario matrix:
  • domain   — from scenario.domain, mapped onto our 7-bucket axis
  • region   — from ISO country code via COUNTRY_TO_REGION
  • tension  — derived from trajectory volatility (stdev of round-to-round Δ)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .scenario_matrix import DOMAINS, REGIONS, TENSION_LEVELS, ScenarioCell


DEFAULT_V1 = (
    Path(__file__).resolve().parents[1] / "calibration" / "empirical" / "scenarios"
)
DEFAULT_V22 = (
    Path(__file__).resolve().parents[1] / "calibration" / "empirical" / "scenarios_v2.2"
)


# Map empirical `domain` strings onto the canonical 7-bucket axis.
DOMAIN_ALIAS: dict[str, str] = {
    "political": "political",
    "corporate": "corporate",
    "commercial": "commercial",
    "financial": "financial",
    "public_health": "public_health",
    "environmental": "environmental",
    "labor": "labor",
    # known extras collapsed onto nearest bucket
    "technology": "commercial",
    "energy": "environmental",
    "social": "political",
    "marketing": "commercial",
    "legal": "corporate",
    "economic": "financial",
    "sport": "commercial",
}


COUNTRY_TO_REGION: dict[str, str] = {
    "US": "US",
    "CA": "US",
    "IT": "EU", "GB": "EU", "FR": "EU", "DE": "EU", "ES": "EU",
    "IE": "EU", "MT": "EU", "SE": "EU", "GR": "EU", "EU": "EU",
    "NL": "EU", "BE": "EU", "PT": "EU", "AT": "EU", "PL": "EU",
    "JP": "APAC", "AU": "APAC", "CN": "APAC", "IN": "APAC",
    "KR": "APAC", "SG": "APAC", "NZ": "APAC", "HK": "APAC",
    "BR": "LATAM", "CL": "LATAM", "AR": "LATAM", "MX": "LATAM",
    "CO": "LATAM", "PE": "LATAM", "VE": "LATAM",
    "TR": "GLOBAL",
}


@dataclass
class EmpiricalScenario:
    id: str
    title: str
    domain: str            # canonical 7-bucket
    country: str
    region: str
    tension: str
    n_rounds: int
    pro_pct: list[float]           # per-round, 0-100 scale
    against_pct: list[float]
    undecided_pct: list[float]
    support: list[float]           # pro_pct / 100  ∈ [0, 1]
    signed_position: list[float]   # (pro - against) / 100  ∈ [-1, 1]
    ground_truth_support: float | None  # terminal pro_pct/100 from outcome, or None
    source_path: Path
    version: str                   # "v1" or "v2.2"

    @property
    def cell(self) -> ScenarioCell:
        return ScenarioCell(domain=self.domain, region=self.region, tension=self.tension)


def _infer_tension(signed_position: list[float]) -> str:
    """Map per-round volatility onto our 4 tension buckets.

    Uses the stdev of round-to-round ΔÏ. Thresholds chosen so a referendum that
    swings ~20pp round-on-round counts as 'critical', a fluke pollster +/-5pp
    counts as 'low', with interior buckets roughly log-spaced.
    """
    if len(signed_position) < 2:
        return "low"
    deltas = [
        signed_position[i] - signed_position[i - 1]
        for i in range(1, len(signed_position))
    ]
    vol = statistics.pstdev(deltas) if len(deltas) >= 2 else abs(deltas[0])
    if vol >= 0.15:
        return "critical"
    if vol >= 0.07:
        return "high"
    if vol >= 0.03:
        return "moderate"
    return "low"


def _load_one(path: Path, version: str) -> EmpiricalScenario | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    traj = raw.get("polling_trajectory") or []
    if len(traj) < 2:
        return None

    def _num(v, default=0.0) -> float:
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    pro = [_num(r.get("pro_pct")) for r in traj]
    against = [_num(r.get("against_pct")) for r in traj]
    undecided = [_num(r.get("undecided_pct")) for r in traj]
    support = [p / 100.0 for p in pro]
    signed = [(p - a) / 100.0 for p, a in zip(pro, against)]

    raw_domain = (raw.get("domain") or "").strip().lower()
    domain = DOMAIN_ALIAS.get(raw_domain)
    if domain not in DOMAINS:
        domain = DOMAINS[0]

    country = (raw.get("country") or "").strip().upper() or "??"
    region = COUNTRY_TO_REGION.get(country, "GLOBAL")

    tension = _infer_tension(signed)
    if tension not in TENSION_LEVELS:
        tension = "moderate"

    gt = raw.get("ground_truth_outcome") or {}
    gt_support: float | None = None
    if isinstance(gt, dict) and isinstance(gt.get("pro_pct"), (int, float)):
        gt_support = float(gt["pro_pct"]) / 100.0

    return EmpiricalScenario(
        id=str(raw.get("id") or path.stem),
        title=str(raw.get("title") or path.stem),
        domain=domain,
        country=country,
        region=region,
        tension=tension,
        n_rounds=int(raw.get("n_rounds") or len(traj)),
        pro_pct=pro,
        against_pct=against,
        undecided_pct=undecided,
        support=support,
        signed_position=signed,
        ground_truth_support=gt_support,
        source_path=path,
        version=version,
    )


def load_empirical_scenarios(
    v1_dir: Path = DEFAULT_V1,
    v22_dir: Path = DEFAULT_V22,
) -> list[EmpiricalScenario]:
    """Load scenarios from v1 + v2.2, preferring v2.2 on duplicate filenames."""
    scenarios: dict[str, EmpiricalScenario] = {}
    for directory, version in ((v1_dir, "v1"), (v22_dir, "v2.2")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if ".meta." in path.name:
                continue
            scenario = _load_one(path, version)
            if scenario is None:
                continue
            # v2.2 (second iteration) overwrites v1
            scenarios[path.name] = scenario
    return list(scenarios.values())


def summarize(scenarios: Iterable[EmpiricalScenario]) -> dict:
    """Quick aggregate stats over a corpus."""
    scenarios = list(scenarios)
    by_domain: dict[str, int] = {}
    by_region: dict[str, int] = {}
    by_tension: dict[str, int] = {}
    for s in scenarios:
        by_domain[s.domain] = by_domain.get(s.domain, 0) + 1
        by_region[s.region] = by_region.get(s.region, 0) + 1
        by_tension[s.tension] = by_tension.get(s.tension, 0) + 1
    traj_lens = [s.n_rounds for s in scenarios]
    return {
        "n": len(scenarios),
        "avg_rounds": sum(traj_lens) / len(traj_lens) if traj_lens else 0.0,
        "with_ground_truth": sum(1 for s in scenarios if s.ground_truth_support is not None),
        "by_domain": by_domain,
        "by_region": by_region,
        "by_tension": by_tension,
    }
