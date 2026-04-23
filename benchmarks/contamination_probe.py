"""Contamination probe: how much does the LLM already *know* about each
empirical scenario, before the simulation even starts?

For a retrospective benchmark to prove anything, we must quantify how much of
the signal comes from simulation vs. pre-existing memorized ground truth. We
probe the model across four independent axes and score each scenario 0..1 on
a contamination index.

Probe axes (each returns a 0..1 "leaked" score):
  1. outcome      — asked outright who won / what the %s were
  2. trajectory   — asked about polling movement shape (tightened / blew out)
  3. events       — asked which named events drove the campaign
  4. actors       — asked which people/orgs led each side

Each axis uses JSON-structured output so we can grade automatically. The
contamination index is a weighted mean (outcome is double-weighted because
it's the metric most at risk of trivial leakage).

Scoring conventions:
  score = 1.0  full leak: model confidently produces correct answer
  score = 0.5  partial leak: model produces correct direction but wrong level
  score = 0.0  no leak: model declines / says unknown / produces wrong answer

Usage:
    python -m benchmarks.contamination_probe
    python -m benchmarks.contamination_probe --scenarios POL-2016-REFERENDUM_COSTITUZIONALE_ITAL
    python -m benchmarks.contamination_probe --limit 5 --budget 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.llm.gemini_client import GeminiClient
from core.llm.json_parser import parse_json_response

from .historical import EmpiricalScenario, load_empirical_scenarios

logger = logging.getLogger(__name__)


# ---------- Probe prompts ----------

SYSTEM_PROBE = (
    "You are a research assistant answering factual questions about public "
    "events. Answer only in valid JSON matching the requested schema. If you "
    "genuinely don't remember a fact, set the field to null rather than "
    "guessing — we are measuring recall, not creativity."
)


def _p_outcome(title: str, country: str, date_end: str) -> str:
    return (
        f"Event: \"{title}\" ({country}, concluded around {date_end}). "
        "Do you remember the final outcome? Return JSON:\n"
        "{\n"
        '  "known": true|false,               // do you recall this specific event?\n'
        '  "winner": "pro" | "against" | null,  // which side prevailed (null if unknown)\n'
        '  "pro_pct": number|null,             // final % for the pro/yes/supporting side\n'
        '  "confidence": number                 // 0..1, how sure you are\n'
        "}\n"
        "Only fill numeric fields if you are recalling, not estimating."
    )


def _p_trajectory(title: str, country: str, date_start: str, date_end: str) -> str:
    return (
        f"Event: \"{title}\" ({country}, active {date_start} → {date_end}). "
        "Describe the shape of public-opinion/polling movement over the campaign. "
        "Return JSON:\n"
        "{\n"
        '  "known": true|false,\n'
        '  "shape": "tightened" | "blew_out" | "flat" | "reversed" | null,\n'
        '  "leading_side_early": "pro" | "against" | "tied" | null,\n'
        '  "leading_side_late":  "pro" | "against" | "tied" | null,\n'
        '  "confidence": number\n'
        "}"
    )


def _p_events(title: str, country: str, date_end: str) -> str:
    return (
        f"Event: \"{title}\" ({country}, concluded {date_end}). "
        "Name up to 3 specific news / campaign events during the campaign that "
        "materially shifted public opinion. Return JSON:\n"
        "{\n"
        '  "known": true|false,\n'
        '  "events": [{"description": string, "approx_date": "YYYY-MM-DD"|null}],\n'
        '  "confidence": number\n'
        "}\n"
        "List nothing if you don't recall specific events. Don't invent."
    )


def _p_actors(title: str, country: str, date_end: str) -> str:
    return (
        f"Event: \"{title}\" ({country}, concluded {date_end}). "
        "Name the top 3 individuals or organizations who publicly led the "
        "pro side, and the top 3 who led the against side. Return JSON:\n"
        "{\n"
        '  "known": true|false,\n'
        '  "pro_leaders": [string],\n'
        '  "against_leaders": [string],\n'
        '  "confidence": number\n'
        "}\n"
        "Leave arrays empty if you don't recall. Don't invent."
    )


# ---------- Graders ----------


def _grade_outcome(resp: dict, scen: EmpiricalScenario) -> tuple[float, dict]:
    if not resp.get("known"):
        return 0.0, {"reason": "declared unknown"}
    gt_pro = (scen.ground_truth_support or 0.0) * 100.0
    gt_winner = "pro" if gt_pro >= 50.0 else "against"

    claimed_winner = resp.get("winner")
    pro_pct = resp.get("pro_pct")
    confidence = float(resp.get("confidence") or 0.0)

    score = 0.0
    parts = {"winner_match": False, "pct_within_5pp": False, "pct_within_2pp": False}

    if claimed_winner == gt_winner:
        score += 0.4
        parts["winner_match"] = True

    if isinstance(pro_pct, (int, float)) and scen.ground_truth_support is not None:
        err = abs(pro_pct - gt_pro)
        if err <= 2.0:
            score += 0.6
            parts["pct_within_2pp"] = True
            parts["pct_within_5pp"] = True
        elif err <= 5.0:
            score += 0.3
            parts["pct_within_5pp"] = True

    # scale by model's stated confidence so honest "I don't know" is rewarded
    return min(1.0, score) * max(0.3, confidence), parts


def _grade_trajectory(resp: dict, scen: EmpiricalScenario) -> tuple[float, dict]:
    if not resp.get("known"):
        return 0.0, {"reason": "declared unknown"}
    if not scen.signed_position or len(scen.signed_position) < 2:
        return 0.0, {"reason": "no gt trajectory"}

    early, late = scen.signed_position[0], scen.signed_position[-1]
    actual_shape = (
        "flat" if abs(late - early) < 0.05
        else ("blew_out" if (late - early) * (1 if late > 0 else -1) > 0.05
              else "tightened" if abs(late) < abs(early) else "reversed")
    )
    actual_early = "pro" if early > 0.02 else "against" if early < -0.02 else "tied"
    actual_late = "pro" if late > 0.02 else "against" if late < -0.02 else "tied"

    score = 0.0
    parts = {
        "shape_match": resp.get("shape") == actual_shape,
        "early_match": resp.get("leading_side_early") == actual_early,
        "late_match": resp.get("leading_side_late") == actual_late,
    }
    if parts["shape_match"]:
        score += 0.4
    if parts["early_match"]:
        score += 0.3
    if parts["late_match"]:
        score += 0.3

    confidence = float(resp.get("confidence") or 0.0)
    parts["gt_shape"] = actual_shape
    parts["gt_early"] = actual_early
    parts["gt_late"] = actual_late
    return score * max(0.3, confidence), parts


def _grade_events(resp: dict, scen: EmpiricalScenario) -> tuple[float, dict]:
    if not resp.get("known"):
        return 0.0, {"reason": "declared unknown"}
    claimed = resp.get("events") or []
    if not claimed:
        return 0.0, {"reason": "no events claimed"}

    # grade by token overlap with ground-truth event descriptions
    gt_texts = [
        (ev.get("description") if isinstance(ev, dict) else "") or ""
        for ev in []  # placeholder; scen does not carry events in this dataclass
    ]
    # fall back: read events from raw source
    try:
        raw = json.loads(scen.source_path.read_text(encoding="utf-8"))
        gt_texts = [ev.get("description", "") for ev in raw.get("events") or []]
    except Exception:
        gt_texts = []

    if not gt_texts:
        # no gt events to compare against → we can only reward "known but empty list"
        return 0.1 * float(resp.get("confidence") or 0.0), {"reason": "no gt events"}

    def _toks(s: str) -> set[str]:
        return {t.lower().strip(".,;:\"'()") for t in s.split() if len(t) > 3}

    gt_tokens = [_toks(t) for t in gt_texts]
    matched = 0
    for item in claimed:
        desc = item.get("description") if isinstance(item, dict) else ""
        ctoks = _toks(desc or "")
        if any(len(ctoks & gt) >= 2 for gt in gt_tokens):
            matched += 1

    recall = min(1.0, matched / max(1, len(gt_texts)))
    confidence = float(resp.get("confidence") or 0.0)
    return recall * max(0.3, confidence), {
        "claimed_count": len(claimed),
        "gt_count": len(gt_texts),
        "matched": matched,
    }


def _grade_actors(resp: dict, scen: EmpiricalScenario) -> tuple[float, dict]:
    if not resp.get("known"):
        return 0.0, {"reason": "declared unknown"}
    try:
        raw = json.loads(scen.source_path.read_text(encoding="utf-8"))
        gt_agents = raw.get("agents") or []
    except Exception:
        return 0.0, {"reason": "could not load agents"}

    gt_pro = {
        (a.get("name") or "").lower()
        for a in gt_agents if (a.get("initial_position") or 0.0) > 0.2
    }
    gt_against = {
        (a.get("name") or "").lower()
        for a in gt_agents if (a.get("initial_position") or 0.0) < -0.2
    }

    def _hits(claimed: list, gt: set[str]) -> int:
        h = 0
        for c in claimed or []:
            name = (c or "").lower()
            for g in gt:
                # partial match: any gt token appears in claim or vice-versa
                g_toks = {t for t in g.split() if len(t) > 3}
                c_toks = {t for t in name.split() if len(t) > 3}
                if g_toks & c_toks:
                    h += 1
                    break
        return h

    claimed_pro = resp.get("pro_leaders") or []
    claimed_against = resp.get("against_leaders") or []
    pro_hits = _hits(claimed_pro, gt_pro)
    against_hits = _hits(claimed_against, gt_against)

    # normalize: up to 3 hits per side counted, /6 total
    raw_score = min(1.0, (pro_hits + against_hits) / 6.0) if (gt_pro or gt_against) else 0.0
    confidence = float(resp.get("confidence") or 0.0)
    return raw_score * max(0.3, confidence), {
        "pro_hits": pro_hits,
        "against_hits": against_hits,
        "gt_pro_count": len(gt_pro),
        "gt_against_count": len(gt_against),
    }


# ---------- Orchestration ----------


@dataclass
class AxisResult:
    axis: str
    score: float
    raw_response: dict
    detail: dict


@dataclass
class ScenarioProbe:
    id: str
    title: str
    country: str
    axes: list[AxisResult] = field(default_factory=list)

    @property
    def index(self) -> float:
        if not self.axes:
            return 0.0
        # outcome double-weighted
        w = {"outcome": 2.0, "trajectory": 1.0, "events": 1.0, "actors": 1.0}
        num = sum(w.get(a.axis, 1.0) * a.score for a in self.axes)
        den = sum(w.get(a.axis, 1.0) for a in self.axes)
        return num / den if den else 0.0


async def _probe_axis(
    client: GeminiClient,
    scen: EmpiricalScenario,
    axis: str,
    blinded_dir: Path | None = None,
) -> AxisResult:
    raw = json.loads(scen.source_path.read_text(encoding="utf-8"))
    title = raw.get("title") or scen.title
    country = raw.get("country") or scen.country
    date_start = raw.get("date_start") or ""
    date_end = raw.get("date_end") or ""

    if blinded_dir is not None:
        # substitute title/country/date fields with blinded values.
        # grader still uses `scen` (original) so ground-truth stays intact.
        bpath = blinded_dir / scen.source_path.name
        if bpath.is_file():
            try:
                braw = json.loads(bpath.read_text(encoding="utf-8"))
                title = braw.get("title") or title
                country = braw.get("country") or country
                date_start = braw.get("date_start_rel") or "T-?"
                date_end = braw.get("date_end_rel") or "T-0"
            except Exception as e:
                logger.warning(f"[{scen.id}] failed to load blinded file: {e}")

    prompts = {
        "outcome": _p_outcome(title, country, date_end),
        "trajectory": _p_trajectory(title, country, date_start, date_end),
        "events": _p_events(title, country, date_end),
        "actors": _p_actors(title, country, date_end),
    }
    prompt = prompts[axis]

    try:
        text = await client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROBE,
            temperature=0.0,
            max_output_tokens=700,
            component=f"probe_{axis}",
            response_json=True,
            retries=3,
        )
        resp = parse_json_response(text) or {}
        if isinstance(resp, list) and len(resp) == 1 and isinstance(resp[0], dict):
            resp = resp[0]
        if not isinstance(resp, dict):
            resp = {"known": False, "raw": str(resp)}
    except Exception as e:
        logger.warning(f"[{scen.id}/{axis}] probe failed: {e}")
        resp = {"known": False, "error": str(e)}

    grader = {
        "outcome": _grade_outcome,
        "trajectory": _grade_trajectory,
        "events": _grade_events,
        "actors": _grade_actors,
    }[axis]
    score, detail = grader(resp, scen)
    return AxisResult(axis=axis, score=float(score), raw_response=resp, detail=detail)


async def probe_scenario(
    client: GeminiClient,
    scen: EmpiricalScenario,
    axes: tuple[str, ...] = ("outcome", "trajectory", "events", "actors"),
    blinded_dir: Path | None = None,
) -> ScenarioProbe:
    tasks = [_probe_axis(client, scen, a, blinded_dir=blinded_dir) for a in axes]
    results = await asyncio.gather(*tasks)
    return ScenarioProbe(id=scen.id, title=scen.title, country=scen.country, axes=list(results))


async def run_probe(
    scenario_ids: list[str] | None,
    limit: int | None,
    budget: float,
    out_path: Path,
    blinded_dir: Path | None = None,
) -> dict:
    all_scens = load_empirical_scenarios()
    if scenario_ids:
        wanted = set(scenario_ids)
        scens = [s for s in all_scens if s.id in wanted]
    else:
        scens = all_scens
    if limit:
        scens = scens[:limit]

    if not scens:
        raise SystemExit("No empirical scenarios matched the filter.")

    client = GeminiClient(budget=budget)
    probes: list[ScenarioProbe] = []

    # sequential over scenarios but parallel per-axis inside each
    for i, scen in enumerate(scens, 1):
        logger.info(f"[{i}/{len(scens)}] probing {scen.id}")
        try:
            p = await probe_scenario(client, scen, blinded_dir=blinded_dir)
        except Exception as e:
            logger.error(f"[{scen.id}] scenario probe failed: {e}")
            continue
        probes.append(p)

    summary = {
        "n_scenarios": len(probes),
        "mean_index": (sum(p.index for p in probes) / len(probes)) if probes else 0.0,
        "by_axis_mean": {
            axis: (
                sum(a.score for p in probes for a in p.axes if a.axis == axis)
                / max(1, sum(1 for p in probes for a in p.axes if a.axis == axis))
            )
            for axis in ("outcome", "trajectory", "events", "actors")
        },
        "high_contamination": [
            {"id": p.id, "title": p.title, "index": round(p.index, 3)}
            for p in sorted(probes, key=lambda q: -q.index) if p.index >= 0.5
        ],
        "low_contamination": [
            {"id": p.id, "title": p.title, "index": round(p.index, 3)}
            for p in sorted(probes, key=lambda q: q.index) if p.index < 0.25
        ],
        "stats": {
            "total_cost": client.stats.total_cost,
            "calls": client.stats.call_count,
            "errors": client.stats.errors,
            "in_tokens": client.stats.total_input_tokens,
            "out_tokens": client.stats.total_output_tokens,
        },
    }

    payload = {
        "summary": summary,
        "probes": [
            {
                "id": p.id,
                "title": p.title,
                "country": p.country,
                "index": p.index,
                "axes": [
                    {
                        "axis": a.axis,
                        "score": a.score,
                        "response": a.raw_response,
                        "detail": a.detail,
                    }
                    for a in p.axes
                ],
            }
            for p in probes
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def _print_summary(summary: dict) -> None:
    print()
    print("=" * 72)
    print(f"CONTAMINATION PROBE — {summary['n_scenarios']} scenarios")
    print("=" * 72)
    print(f"mean contamination index: {summary['mean_index']:.3f}")
    print("by axis:")
    for axis, val in summary["by_axis_mean"].items():
        bar = "█" * int(round(val * 40))
        print(f"  {axis:12s} {val:.3f}  {bar}")
    print()
    if summary["high_contamination"]:
        print("HIGH contamination (index ≥ 0.50 — model remembers a lot):")
        for row in summary["high_contamination"]:
            print(f"  {row['index']:.3f}  {row['id']}  {row['title'][:60]}")
    if summary["low_contamination"]:
        print()
        print("LOW contamination (index < 0.25 — model likely safe to blind):")
        for row in summary["low_contamination"]:
            print(f"  {row['index']:.3f}  {row['id']}  {row['title'][:60]}")
    stats = summary.get("stats") or {}
    if stats:
        print()
        print(f"cost: ${stats.get('total_cost', 0):.4f}  "
              f"calls: {stats.get('calls', 0)}  "
              f"errors: {stats.get('errors', 0)}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="specific scenario ids to probe")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/contamination_probe.json"))
    parser.add_argument("--blinded-dir", type=Path, default=None,
                        help="If set, feed the LLM the blinded title/country/dates "
                             "from this dir instead of the originals.")
    args = parser.parse_args()

    summary = asyncio.run(run_probe(
        scenario_ids=args.scenarios,
        limit=args.limit,
        budget=args.budget,
        out_path=args.out,
        blinded_dir=args.blinded_dir,
    ))
    _print_summary(summary)


if __name__ == "__main__":
    # allow `python benchmarks/contamination_probe.py` in addition to `-m`
    if __package__ is None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from benchmarks.contamination_probe import main as _main  # type: ignore
        _main()
    else:
        main()
