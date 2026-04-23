"""Blinding protocol: anonymize an empirical scenario so the LLM cannot
recognize it from surface cues.

Design goals, in priority order:
  1. **No leak**: strip every piece of identifying metadata a motivated LLM
     would use to recognize the real-world event (country, dates, proper
     nouns, institutional names, even distinctive numeric signatures like
     "48.1/51.89" that uniquely fingerprint a specific vote).
  2. **Preserve dynamics**: the simulator consumes initial positions,
     influence levels, shock magnitudes/directions, polling trajectory and
     covariates. Keep all of those. A blinded run must remain a faithful
     simulation substrate.
  3. **Deterministic**: same input scenario → same blinded output. Makes
     A/B comparison against the contaminated run reproducible.
  4. **Reversible for grading only**: the mapping is persisted in a sidecar
     file so downstream evaluation can rejoin ground truth without leaking
     identity into the sim itself.

Rename strategy for agents (by initial_position + influence):
  position ≥ +0.5        →  PRO_LEADER_{k}
  +0.2 ≤ position < 0.5  →  PRO_SUPPORTER_{k}
  -0.2 < position < 0.2  →  NEUTRAL_{k}
  -0.5 < position ≤ -0.2 →  AGAINST_SUPPORTER_{k}
  position ≤ -0.5        →  AGAINST_LEADER_{k}
  (k is 1-based, ordered by influence desc within each bucket.)

Dates → relative T-{k} where T=0 is date_end.

Country → "Country_{H}" where H is the first 2 hex chars of sha1(country).
This is stable across runs and doesn't leak the ISO code.

Events / descriptions / title → scrubbed via (a) the agent rename map
applied as word-boundary regex, (b) a fixed list of geography/people tokens
(see STOPLIST), (c) a template rewrite for the title.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Stoplist: any of these tokens found in scrubbed text is replaced with a
# generic placeholder. Covers political figures/parties/places that are
# most at risk of triggering LLM recognition. This is deliberately narrow —
# the renaming map already handles scenario-specific proper nouns.
STOPLIST: dict[str, str] = {
    # Italian
    "italy": "COUNTRY",
    "italia": "COUNTRY",
    "italian": "",
    "italiana": "",
    "italiano": "",
    "rome": "CITY",
    "roma": "CITY",
    "milan": "CITY",
    "milano": "CITY",
    # UK / EU
    "britain": "COUNTRY",
    "british": "",
    "uk": "COUNTRY",
    "england": "COUNTRY",
    "english": "",
    "london": "CITY",
    "scotland": "COUNTRY",
    "scottish": "",
    "ireland": "COUNTRY",
    "irish": "",
    "european": "",
    "europe": "REGION",
    "brexit": "REFERENDUM",
    # US
    "united states": "COUNTRY",
    "america": "COUNTRY",
    "american": "",
    "usa": "COUNTRY",
    "us": "COUNTRY",
    "washington": "CITY",
    "new york": "CITY",
    "california": "REGION",
    # Other
    "france": "COUNTRY",
    "french": "",
    "paris": "CITY",
    "germany": "COUNTRY",
    "german": "",
    "berlin": "CITY",
    "brazil": "COUNTRY",
    "brazilian": "",
    "turkey": "COUNTRY",
    "turkish": "",
    "chile": "COUNTRY",
    "greece": "COUNTRY",
    "greek": "",
    "catalonia": "REGION",
    "catalan": "",
    "australia": "COUNTRY",
    "australian": "",
    "japan": "COUNTRY",
    "japanese": "",
    "malta": "COUNTRY",
    "maltese": "",
    # Common party tokens
    "democratic party": "RULING_PARTY",
    "republican": "OPPOSITION_BLOC",
    "democrat": "",
    "conservative": "",
    "labour": "",
    "labor party": "OPPOSITION_BLOC",
    "tories": "",
    "tory": "",
    "eu": "INSTITUTIONAL_BLOC",
    "european union": "INSTITUTIONAL_BLOC",
}


@dataclass
class AgentRename:
    original: str
    alias: str
    role: str  # PRO_LEADER | PRO_SUPPORTER | NEUTRAL | AGAINST_SUPPORTER | AGAINST_LEADER
    rank: int
    initial_position: float
    influence: float


@dataclass
class BlindingMap:
    scenario_id: str
    original_title: str
    original_country: str
    country_alias: str
    date_end_original: str
    agents: list[AgentRename] = field(default_factory=list)
    title_template: str = ""

    def name_to_alias(self) -> dict[str, str]:
        return {a.original: a.alias for a in self.agents}


def _country_alias(country: str) -> str:
    if not country:
        return "Country_X"
    h = hashlib.sha1(country.upper().encode("utf-8")).hexdigest()[:2].upper()
    return f"Country_{h}"


def _bucket(pos: float) -> str:
    if pos >= 0.5:
        return "PRO_LEADER"
    if pos >= 0.2:
        return "PRO_SUPPORTER"
    if pos <= -0.5:
        return "AGAINST_LEADER"
    if pos <= -0.2:
        return "AGAINST_SUPPORTER"
    return "NEUTRAL"


def _rename_agents(agents: list[dict]) -> list[AgentRename]:
    # group by bucket, sort by influence desc, assign ranks
    by_bucket: dict[str, list[dict]] = {}
    for a in agents:
        pos = float(a.get("initial_position") or 0.0)
        by_bucket.setdefault(_bucket(pos), []).append(a)

    out: list[AgentRename] = []
    for bucket, members in by_bucket.items():
        members.sort(key=lambda x: -float(x.get("influence") or 0.0))
        for i, m in enumerate(members, 1):
            out.append(AgentRename(
                original=str(m.get("name") or f"UNNAMED_{bucket}_{i}"),
                alias=f"{bucket}_{i}",
                role=bucket,
                rank=i,
                initial_position=float(m.get("initial_position") or 0.0),
                influence=float(m.get("influence") or 0.0),
            ))
    return out


def _expand_rename_map(rename_map: dict[str, str]) -> dict[str, str]:
    """For multi-token names, add each token (len≥4) as a separate key
    mapping to the same alias. Catches 'Renzi' when the agent is 'Matteo Renzi'."""
    expanded = dict(rename_map)
    for orig, alias in list(rename_map.items()):
        tokens = [t for t in re.split(r"[\s\-\.]+", orig) if len(t) >= 4]
        for tok in tokens:
            # don't override a more-specific existing mapping
            if tok not in expanded:
                expanded[tok] = alias
    return expanded


def _scrub(text: str, rename_map: dict[str, str]) -> str:
    """Apply rename + stoplist scrubbing. Case-insensitive, word-boundary."""
    if not text:
        return text
    out = text

    expanded = _expand_rename_map(rename_map)

    # 1. named-entity renames (highest priority — scenario-specific)
    for orig, alias in sorted(expanded.items(), key=lambda kv: -len(kv[0])):
        if not orig:
            continue
        pattern = re.compile(r"\b" + re.escape(orig) + r"\b", re.IGNORECASE)
        out = pattern.sub(alias, out)

    # 2. stoplist — country/city/bloc replacements
    for orig, alias in sorted(STOPLIST.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(r"\b" + re.escape(orig) + r"\b", re.IGNORECASE)
        out = pattern.sub(alias if alias else "", out)

    # 3. collapse whitespace after removals
    out = re.sub(r"\s+", " ", out).strip(" .,;:-")

    # 4. strip 4-digit years
    out = re.sub(r"\b(19|20)\d{2}\b", "YEAR", out)

    return out


def _relative_date(iso_date: str, anchor: datetime | None) -> str:
    if not iso_date or not anchor:
        return "T-?"
    try:
        dt = datetime.strptime(iso_date[:10], "%Y-%m-%d")
    except ValueError:
        return "T-?"
    delta = (anchor - dt).days
    if delta > 0:
        return f"T-{delta}d"
    if delta < 0:
        return f"T+{-delta}d"
    return "T-0"


def _title_template(n_rounds: int, domain: str, tension: str | None = None) -> str:
    # schematic description with no identifying details
    dom = (domain or "political").replace("_", " ")
    base = f"Binary {dom} referendum-style decision over {n_rounds} polling rounds"
    if tension:
        base += f" ({tension} volatility)"
    return base


def blind_scenario(raw: dict) -> tuple[dict, BlindingMap]:
    """Return (blinded_scenario_dict, mapping). Input `raw` is the parsed
    empirical scenario JSON. Output preserves all numeric fields that feed
    the simulator."""

    sid = str(raw.get("id") or "UNKNOWN")
    title = str(raw.get("title") or "")
    country = str(raw.get("country") or "??")
    domain = str(raw.get("domain") or "political")
    date_end = str(raw.get("date_end") or "")
    anchor = None
    try:
        anchor = datetime.strptime(date_end[:10], "%Y-%m-%d") if date_end else None
    except ValueError:
        anchor = None

    renames = _rename_agents(raw.get("agents") or [])
    rename_map = {r.original: r.alias for r in renames}

    country_alias = _country_alias(country)
    blinded_title = _title_template(
        n_rounds=int(raw.get("n_rounds") or len(raw.get("polling_trajectory") or [])),
        domain=domain,
    )

    # --- events ---
    blinded_events: list[dict] = []
    for ev in raw.get("events") or []:
        blinded_events.append({
            "round": ev.get("round"),
            "date_rel": _relative_date(str(ev.get("date") or ""), anchor),
            "description": _scrub(str(ev.get("description") or ""), rename_map),
            "shock_magnitude": ev.get("shock_magnitude"),
            "shock_direction": ev.get("shock_direction"),
        })

    # --- polling trajectory (keep numbers, rewrite labels) ---
    blinded_traj: list[dict] = []
    for r in raw.get("polling_trajectory") or []:
        blinded_traj.append({
            "round": r.get("round"),
            "date_rel": _relative_date(str(r.get("date") or ""), anchor),
            "pro_pct": r.get("pro_pct"),
            "against_pct": r.get("against_pct"),
            "undecided_pct": r.get("undecided_pct"),
        })

    # --- agents ---
    blinded_agents: list[dict] = []
    for ag in raw.get("agents") or []:
        orig_name = str(ag.get("name") or "")
        alias = rename_map.get(orig_name, "UNKNOWN_AGENT")
        blinded_agents.append({
            "name": alias,
            "type": ag.get("type"),
            "initial_position": ag.get("initial_position"),
            "influence": ag.get("influence"),
            "description": _scrub(str(ag.get("description") or ""), rename_map),
        })

    blinded = {
        "id": f"BLINDED_{hashlib.sha1(sid.encode()).hexdigest()[:10]}",
        "original_id": sid,   # kept for our bookkeeping, NOT for the sim prompt
        "domain": domain,
        "title": blinded_title,
        "country": country_alias,
        "date_start_rel": "T-"
                          f"{int(raw.get('n_rounds') or 7) * int(raw.get('round_duration_days') or 13)}d",
        "date_end_rel": "T-0",
        "n_rounds": raw.get("n_rounds"),
        "round_duration_days": raw.get("round_duration_days"),
        "ground_truth_outcome": raw.get("ground_truth_outcome"),  # used by grader only
        "polling_trajectory": blinded_traj,
        "events": blinded_events,
        "agents": blinded_agents,
        "covariates": raw.get("covariates"),
        "notes": "Blinded: entity names, geography, and absolute dates "
                 "redacted. Numeric fields preserved.",
    }

    bmap = BlindingMap(
        scenario_id=sid,
        original_title=title,
        original_country=country,
        country_alias=country_alias,
        date_end_original=date_end,
        agents=renames,
        title_template=blinded_title,
    )

    return blinded, bmap


# ---------- CLI ----------


def _write_blinded(src: Path, out_dir: Path, map_dir: Path) -> tuple[Path, Path]:
    raw = json.loads(src.read_text(encoding="utf-8"))
    blinded, bmap = blind_scenario(raw)

    # keep the original filename prefix, mark blinded
    out_path = out_dir / src.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(blinded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    map_path = map_dir / (src.stem + ".map.json")
    map_dir.mkdir(parents=True, exist_ok=True)
    map_payload = {
        "scenario_id": bmap.scenario_id,
        "original_title": bmap.original_title,
        "original_country": bmap.original_country,
        "country_alias": bmap.country_alias,
        "date_end_original": bmap.date_end_original,
        "title_template": bmap.title_template,
        "agents": [
            {
                "original": r.original,
                "alias": r.alias,
                "role": r.role,
                "rank": r.rank,
                "initial_position": r.initial_position,
                "influence": r.influence,
            }
            for r in bmap.agents
        ],
    }
    map_path.write_text(
        json.dumps(map_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path, map_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs", nargs="+",
        default=[
            "calibration/empirical/scenarios_v2.2",
            "calibration/empirical/scenarios_v2.1",
            "calibration/empirical/scenarios",
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("calibration/empirical/scenarios_blinded"),
    )
    parser.add_argument(
        "--map-dir", type=Path,
        default=Path("calibration/empirical/blinding_maps"),
    )
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Only blind scenarios whose id contains one of these substrings")
    args = parser.parse_args()

    seen: set[str] = set()
    written: list[tuple[Path, Path]] = []

    for dirstr in args.input_dirs:
        d = Path(dirstr)
        if not d.is_dir():
            continue
        for src in sorted(d.glob("*.json")):
            if ".meta." in src.name or src.name == "manifest.json":
                continue
            if src.name in seen:
                continue
            seen.add(src.name)
            if args.ids:
                try:
                    sid = json.loads(src.read_text())["id"]
                except Exception:
                    continue
                if not any(sub in sid for sub in args.ids):
                    continue
            try:
                written.append(_write_blinded(src, args.output_dir, args.map_dir))
            except Exception as e:
                print(f"ERR {src.name}: {e}")

    print(f"wrote {len(written)} blinded scenarios → {args.output_dir}")
    print(f"wrote {len(written)} map files → {args.map_dir}")


if __name__ == "__main__":
    main()
