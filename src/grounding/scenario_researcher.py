"""ScenarioResearcher: build verified fact sheets from web sources.

Pipeline:
1. Given a topic and timeframe, run structured searches
2. Extract a verified timeline with dates, numbers, sources
3. Identify key events with quantified impact
4. Produce a structured FactSheet that the LLM uses as constraint
   for round narrative generation

The FactSheet does NOT replace LLM generation — it CONSTRAINS it.
The LLM enriches a sequence of verified facts narratively,
rather than inventing a plausible story from scratch.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class VerifiedEvent:
    """A verified event with source."""
    date: str                          # ISO format or "YYYY-MM-DD approx"
    description: str                   # Factual description, max 2 sentences
    shock_magnitude: float             # 0-1, estimated impact on opinion
    shock_direction: int               # +1 (pro) or -1 (against)
    source_url: str = ""
    source_name: str = ""
    confidence: float = 0.8            # 0-1, how sure we are
    quantitative_detail: str = ""      # Specific numbers (e.g. "$91B HTM portfolio")


@dataclass
class VerifiedPoll:
    """A verified polling/sentiment data point."""
    date: str
    pro_pct: float                     # Percentage pro (0-100)
    sample_size: int = 0
    source: str = ""
    methodology: str = ""


@dataclass
class StakeholderInfo:
    """Information about a key stakeholder."""
    name: str
    role: str
    position: str                      # Public statement summarized
    date: str = ""
    source: str = ""


@dataclass
class FactSheet:
    """Complete fact sheet for a scenario.

    This is the final product of ScenarioResearcher.
    Passed to the LLM as constraint for narrative generation.
    """
    scenario_id: str = ""
    topic: str = ""
    domain: str = ""
    country: str = ""
    timeframe_start: str = ""          # ISO date
    timeframe_end: str = ""            # ISO date

    events: list[VerifiedEvent] = field(default_factory=list)
    polls: list[VerifiedPoll] = field(default_factory=list)
    stakeholders: list[StakeholderInfo] = field(default_factory=list)

    outcome_description: str = ""
    outcome_pro_pct: float = 0.0
    outcome_source: str = ""

    # Scenario-specific key numbers
    key_figures: dict[str, str] = field(default_factory=dict)

    # Metadata
    n_sources_consulted: int = 0
    research_timestamp: str = ""
    quality_score: int = 0             # 0-100

    def to_llm_context(self) -> str:
        """Format the FactSheet as LLM prompt context.

        This text is inserted in the prompt that generates round narratives.
        The LLM MUST respect these facts.
        """
        lines = [
            f"# VERIFIED FACTS — {self.topic}",
            f"Domain: {self.domain} | Country: {self.country}",
            f"Period: {self.timeframe_start} to {self.timeframe_end}",
            "",
            "## TIMELINE (verified events, chronological order):",
        ]
        for i, e in enumerate(self.events, 1):
            conf = f" [confidence: {e.confidence:.0%}]" if e.confidence < 1.0 else ""
            lines.append(f"{i}. [{e.date}] {e.description}{conf}")
            if e.quantitative_detail:
                lines.append(f"   Detail: {e.quantitative_detail}")
            if e.source_name:
                lines.append(f"   Source: {e.source_name}")

        if self.key_figures:
            lines.extend(["", "## KEY FIGURES:"])
            for k, v in self.key_figures.items():
                lines.append(f"- {k}: {v}")

        if self.polls:
            lines.extend(["", "## POLLING/SENTIMENT DATA:"])
            for p in self.polls:
                sz = f" (N={p.sample_size})" if p.sample_size else ""
                lines.append(f"- [{p.date}] {p.pro_pct}% pro{sz} — {p.source}")

        if self.stakeholders:
            lines.extend(["", "## KEY STAKEHOLDER POSITIONS:"])
            for s in self.stakeholders:
                lines.append(f"- {s.name} ({s.role}): {s.position}")

        lines.extend([
            "",
            "## FINAL OUTCOME:",
            f"{self.outcome_description}",
            f"Pro %: {self.outcome_pro_pct}% — Source: {self.outcome_source}",
            "",
            "INSTRUCTION: Your narrative for each round MUST be consistent with",
            "these verified facts. Do NOT contradict dates, numbers, or outcomes.",
            "You MAY add narrative color, agent reactions, and social dynamics",
            "that are plausible given these facts.",
        ])
        return "\n".join(lines)

    def to_scenario_events(self, n_rounds: int) -> list[dict]:
        """Convert verified events to simulator JSON format.

        Distributes events across simulation rounds.
        Uses `shock_magnitude` and `shock_direction` field names
        matching the simulator's EventInjector format.

        Returns:
            List of dicts with: round, description, shock_magnitude,
            shock_direction, source, _verified.
        """
        if not self.events or n_rounds <= 0:
            return []

        # Distribute events uniformly across rounds
        events_per_round: list[list[VerifiedEvent]] = [[] for _ in range(n_rounds)]
        for i, event in enumerate(self.events):
            round_idx = min(int(i * n_rounds / len(self.events)), n_rounds - 1)
            events_per_round[round_idx].append(event)

        result = []
        for round_idx, round_events in enumerate(events_per_round):
            if not round_events:
                continue

            # Multiple events in same round: compute net effect
            net_signed = sum(e.shock_magnitude * e.shock_direction for e in round_events)
            avg_magnitude = sum(e.shock_magnitude for e in round_events) / len(round_events)
            net_direction = 1 if net_signed >= 0 else -1
            description = " | ".join(e.description for e in round_events)
            sources = [e.source_url for e in round_events if e.source_url]

            result.append({
                "round": round_idx + 1,
                "description": description,
                "shock_magnitude": round(min(1.0, avg_magnitude), 3),
                "shock_direction": net_direction,
                "source": sources[0] if sources else "",
                "_sources": sources,
                "_verified": True,
            })

        return result

    def to_dict(self) -> dict:
        """Serialize to plain dict (JSON-safe)."""
        return asdict(self)


# ── ScenarioResearcher ──────────────────────────────────────────────────

class ScenarioResearcher:
    """Build verified FactSheets for opinion dynamics scenarios."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        search_fn: Optional[Callable[[str], list[dict]]] = None,
        fetch_fn: Optional[Callable[[str], str]] = None,
        max_search_rounds: int = 3,
        max_queries_per_round: int = 4,
    ):
        """
        Args:
            llm_fn: Function(prompt) → str. Wraps any LLM client.
            search_fn: Function(query) → list[dict] with 'url' and 'text' keys.
            fetch_fn: Function(url) → str full page text. Optional.
        """
        self.llm_fn = llm_fn
        self.search_fn = search_fn
        self.fetch_fn = fetch_fn
        self.max_search_rounds = max_search_rounds
        self.max_queries_per_round = max_queries_per_round

    def research(
        self,
        topic: str,
        domain: str,
        country: str = "",
        timeframe: str = "",
        scenario_id: str = "",
    ) -> FactSheet:
        """Full research pipeline.

        Round 1: Broad — timeline, outcome, key facts
        Round 2: Targeted — specific numbers, polling, stakeholder positions
        Round 3: Verification — cross-check critical facts

        Returns:
            Populated FactSheet. On failure, returns partial sheet
            with reduced quality_score.
        """
        sheet = FactSheet(
            scenario_id=scenario_id,
            topic=topic,
            domain=domain,
            country=country,
            research_timestamp=datetime.now().isoformat(),
        )

        # Round 1: Broad search
        logger.info(f"Research round 1: broad search for '{topic}'")
        try:
            broad_results = self._search_round_broad(topic, domain, country, timeframe)
            sheet = self._extract_timeline(sheet, broad_results)
            sheet = self._extract_outcome(sheet, broad_results)
        except Exception as e:
            logger.error(f"Broad search failed: {e}")

        # Round 2: Targeted search
        logger.info(f"Research round 2: targeted search")
        try:
            targeted_results = self._search_round_targeted(sheet)
            sheet = self._extract_key_figures(sheet, targeted_results)
            sheet = self._extract_polls(sheet, targeted_results)
            sheet = self._extract_stakeholders(sheet, targeted_results)
        except Exception as e:
            logger.error(f"Targeted search failed: {e}")

        # Round 3: Cross-verification
        logger.info(f"Research round 3: cross-verification")
        try:
            sheet = self._cross_verify(sheet)
        except Exception as e:
            logger.error(f"Cross-verification failed: {e}")

        # Compute quality score
        sheet.quality_score = self._compute_quality_score(sheet)

        # Derive timeframe from events
        if sheet.events:
            sheet.timeframe_start = sheet.events[0].date
            sheet.timeframe_end = sheet.events[-1].date

        return sheet

    # ── Search rounds ───────────────────────────────────────────────────

    def _search_round_broad(
        self, topic: str, domain: str, country: str, timeframe: str
    ) -> list[dict]:
        """Round 1: broad queries for overview."""
        if self.search_fn is None:
            return []

        queries = [
            f"{topic} timeline key events",
            f"{topic} outcome result {timeframe}",
            f"{topic} {domain} analysis impact",
        ]
        if country:
            queries.append(f"{topic} {country} public reaction")

        return self._run_searches(queries)

    def _search_round_targeted(self, sheet: FactSheet) -> list[dict]:
        """Round 2: targeted queries based on gaps."""
        if self.search_fn is None:
            return []

        queries = []
        if not sheet.polls:
            queries.append(f"{sheet.topic} public opinion poll survey data")
        if not sheet.key_figures:
            queries.append(f"{sheet.topic} key statistics numbers figures data")
        if not sheet.stakeholders:
            queries.append(f"{sheet.topic} key figures reactions statements")
        queries.append(f"{sheet.topic} official report primary source")

        return self._run_searches(queries)

    def _run_searches(self, queries: list[str]) -> list[dict]:
        """Execute a batch of search queries, collecting results."""
        all_results = []
        for q in queries[:self.max_queries_per_round]:
            try:
                results = self.search_fn(q)
                # Optionally fetch full page text
                if self.fetch_fn:
                    for r in results[:2]:
                        try:
                            r["full_text"] = self.fetch_fn(r.get("url", ""))[:3000]
                        except Exception:
                            pass
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{q}': {e}")
        return all_results

    # ── LLM extraction ──────────────────────────────────────────────────

    def _extract_timeline(self, sheet: FactSheet, results: list[dict]) -> FactSheet:
        """Use LLM to extract ordered timeline from search results."""
        search_text = self._results_to_text(results)
        if not search_text:
            return sheet

        prompt = f"""From these search results about "{sheet.topic}", extract a chronological
timeline of KEY EVENTS that shaped public opinion.

For each event, provide:
- date: YYYY-MM-DD (approximate if exact date unknown, mark with "approx")
- description: 1-2 factual sentences (no opinion, just facts)
- shock_magnitude: 0.0-1.0 (how impactful on public opinion)
  0.1 = minor news, 0.3 = notable, 0.5 = major, 0.8 = transformative, 1.0 = unprecedented
- shock_direction: +1 (increased public support/trust) or -1 (decreased support/trust)
- source_name: which source reported this
- quantitative_detail: any specific numbers mentioned (dollar amounts, percentages, counts)

Target: 5-10 events. Only include events that CHANGED public opinion, not background context.

Search results:
{search_text}

Return as JSON array of objects, nothing else."""

        response = self._call_llm(prompt)
        events_raw = _parse_json(response)
        if not isinstance(events_raw, list):
            return sheet

        for e in events_raw:
            if not isinstance(e, dict):
                continue
            try:
                mag = e.get("shock_magnitude")
                dirn = e.get("shock_direction")
                sheet.events.append(VerifiedEvent(
                    date=str(e.get("date") or ""),
                    description=str(e.get("description") or ""),
                    shock_magnitude=float(mag) if mag is not None else 0.5,
                    shock_direction=int(float(dirn)) if dirn is not None else -1,
                    source_name=str(e.get("source_name") or ""),
                    quantitative_detail=str(e.get("quantitative_detail") or ""),
                    confidence=0.8,
                ))
            except (ValueError, TypeError) as exc:
                logger.warning(f"Skipping malformed event: {exc}")

        sheet.events.sort(key=lambda ev: ev.date)
        return sheet

    def _extract_outcome(self, sheet: FactSheet, results: list[dict]) -> FactSheet:
        """Extract verified final outcome."""
        search_text = self._results_to_text(results)
        if not search_text:
            return sheet

        prompt = f"""From these search results about "{sheet.topic}", determine the FINAL OUTCOME
in terms of public opinion/support.

Provide:
- outcome_description: 1-2 sentence factual description of the final state
- outcome_pro_pct: percentage of public that supported/trusted (0-100)
- outcome_source: the most reliable source for this number

If there's an official result (election, referendum), use that.
If it's a sentiment/opinion measure, use the most cited survey.
If no quantitative data exists, estimate and mark confidence < 0.5.

Search results:
{search_text}

Return as JSON object, nothing else."""

        response = self._call_llm(prompt)
        outcome = _parse_json(response)
        if isinstance(outcome, dict):
            sheet.outcome_description = str(outcome.get("outcome_description") or "")
            pro = outcome.get("outcome_pro_pct")
            sheet.outcome_pro_pct = float(pro) if pro is not None else 50.0
            sheet.outcome_source = str(outcome.get("outcome_source") or "")

        return sheet

    def _extract_key_figures(self, sheet: FactSheet, results: list[dict]) -> FactSheet:
        """Extract scenario-specific quantitative figures."""
        search_text = self._results_to_text(results)
        if not search_text:
            return sheet

        prompt = f"""From these search results about "{sheet.topic}", extract KEY QUANTITATIVE
FIGURES relevant to understanding this scenario.

Examples: financial amounts, percentages, counts, before/after comparisons.

Return as JSON object with descriptive keys and string values.
Example: {{"stock_drop": "-60% in 24 hours", "affected_customers": "2.3 million"}}
Only include VERIFIED numbers from the search results. Max 8 entries.

Search results:
{search_text}

Return JSON object only, nothing else."""

        response = self._call_llm(prompt)
        figures = _parse_json(response)
        if isinstance(figures, dict):
            sheet.key_figures = {str(k): str(v) for k, v in figures.items()}

        return sheet

    def _extract_polls(self, sheet: FactSheet, results: list[dict]) -> FactSheet:
        """Extract polling/sentiment data."""
        search_text = self._results_to_text(results)
        if not search_text:
            return sheet

        prompt = f"""From these search results about "{sheet.topic}", extract any
PUBLIC OPINION or SENTIMENT DATA (polls, surveys, sentiment analysis).

For each data point:
- date: YYYY-MM-DD
- pro_pct: percentage supporting/trusting (0-100)
- sample_size: number of respondents (0 if unknown)
- source: pollster or data source name
- methodology: survey method if known

Return as JSON array. If no polling data exists, return empty array [].

Search results:
{search_text}

Return JSON array only, nothing else."""

        response = self._call_llm(prompt)
        polls_raw = _parse_json(response)
        if not isinstance(polls_raw, list):
            return sheet

        for p in polls_raw:
            if not isinstance(p, dict):
                continue
            try:
                sheet.polls.append(VerifiedPoll(
                    date=str(p.get("date", "")),
                    pro_pct=float(p.get("pro_pct", 50.0)),
                    sample_size=int(p.get("sample_size", 0)),
                    source=str(p.get("source", "")),
                    methodology=str(p.get("methodology", "")),
                ))
            except (ValueError, TypeError):
                pass

        return sheet

    def _extract_stakeholders(self, sheet: FactSheet, results: list[dict]) -> FactSheet:
        """Extract key stakeholder positions."""
        search_text = self._results_to_text(results)
        if not search_text:
            return sheet

        prompt = f"""From these search results about "{sheet.topic}", extract KEY STAKEHOLDER
POSITIONS — real people who made public statements about this event.

For each:
- name: full name
- role: title/position
- position: 1-2 sentence summary of their public stance
- date: when they made this statement (YYYY-MM-DD approx)
- source: where they said it

Return as JSON array. Max 8 stakeholders.

Search results:
{search_text}

Return JSON array only, nothing else."""

        response = self._call_llm(prompt)
        stakeholders_raw = _parse_json(response)
        if not isinstance(stakeholders_raw, list):
            return sheet

        for s in stakeholders_raw:
            if not isinstance(s, dict):
                continue
            sheet.stakeholders.append(StakeholderInfo(
                name=str(s.get("name", "")),
                role=str(s.get("role", "")),
                position=str(s.get("position", "")),
                date=str(s.get("date", "")),
                source=str(s.get("source", "")),
            ))

        return sheet

    # ── Verification ────────────────────────────────────────────────────

    def _cross_verify(self, sheet: FactSheet) -> FactSheet:
        """Round 3: cross-check high-impact facts with independent sources."""
        if self.search_fn is None:
            return sheet

        high_impact = [e for e in sheet.events if e.shock_magnitude > 0.5]

        for event in high_impact[:3]:
            try:
                results = self.search_fn(f"{event.description} verify fact check")
                if results:
                    event.confidence = 1.0
                    if results[0].get("url"):
                        event.source_url = results[0]["url"]
            except Exception:
                pass

        sheet.n_sources_consulted = sum(
            1 for e in sheet.events if e.source_url or e.source_name
        )
        return sheet

    # ── Quality scoring ─────────────────────────────────────────────────

    @staticmethod
    def _compute_quality_score(sheet: FactSheet) -> int:
        """Compute quality score 0-100 based on completeness."""
        score = 0

        # Events (0-30)
        score += min(30, len(sheet.events) * 4)

        # Outcome (0-20)
        if sheet.outcome_source:
            score += 20
        elif sheet.outcome_description:
            score += 10

        # Polls (0-15)
        score += min(15, len(sheet.polls) * 5)

        # Key figures (0-15)
        score += min(15, len(sheet.key_figures) * 3)

        # Stakeholders (0-10)
        score += min(10, len(sheet.stakeholders) * 2)

        # High-confidence events (0-10)
        high_conf = sum(1 for e in sheet.events if e.confidence >= 0.9)
        score += min(10, high_conf * 2)

        return min(100, score)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _results_to_text(self, results: list[dict], max_chars: int = 6000) -> str:
        """Concatenate search results into prompt text."""
        parts = []
        total = 0
        for r in results[:20]:
            text = r.get("full_text", r.get("text", r.get("snippet", "")))
            url = r.get("url", "")
            part = f"[Source: {url}]\n{text[:500]}\n---"
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM via the injected function."""
        if self.llm_fn is None:
            raise ValueError("LLM function not configured")
        return self.llm_fn(prompt)


# ── Utilities ───────────────────────────────────────────────────────────

def _parse_json(text: str) -> Any:
    """Extract JSON from LLM output, tolerating markdown fences and minor issues."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract first JSON array or object from the text
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Last resort: raise with original text for debugging
    return json.loads(text)


# ── Convenience function ────────────────────────────────────────────────

def research_scenario(
    topic: str,
    domain: str,
    country: str = "",
    timeframe: str = "",
    scenario_id: str = "",
    llm_fn: Optional[Callable[[str], str]] = None,
    search_fn: Optional[Callable[[str], list[dict]]] = None,
    fetch_fn: Optional[Callable[[str], str]] = None,
) -> FactSheet:
    """Convenience wrapper."""
    researcher = ScenarioResearcher(
        llm_fn=llm_fn,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
    )
    return researcher.research(
        topic=topic, domain=domain, country=country,
        timeframe=timeframe, scenario_id=scenario_id,
    )


# ── Standalone demo ─────────────────────────────────────────────────────

def _demo_svb():
    """Print a hardcoded SVB Collapse FactSheet to show expected output."""
    sheet = FactSheet(
        scenario_id="FIN-2023-SVB_COLLAPSE",
        topic="Silicon Valley Bank collapse",
        domain="financial",
        country="USA",
        timeframe_start="2023-03-08",
        timeframe_end="2023-03-13",
        events=[
            VerifiedEvent(
                date="2023-03-08",
                description="SVB announces $1.8B loss on bond portfolio sale and plans $2.25B capital raise",
                shock_magnitude=0.7,
                shock_direction=-1,
                source_name="SEC Filing",
                quantitative_detail="$1.8B loss on $21B AFS portfolio; $2.25B raise attempt",
                confidence=1.0,
            ),
            VerifiedEvent(
                date="2023-03-09",
                description="Customers withdraw $42B in a single day; SVB stock drops 60%",
                shock_magnitude=0.9,
                shock_direction=-1,
                source_name="FDIC, Bloomberg",
                quantitative_detail="$42B withdrawals; stock -60%; $91B HTM portfolio at risk",
                confidence=1.0,
            ),
            VerifiedEvent(
                date="2023-03-10",
                description="California DFPI closes SVB and appoints FDIC as receiver",
                shock_magnitude=1.0,
                shock_direction=-1,
                source_name="FDIC Press Release",
                quantitative_detail="$209B total assets; 2nd largest bank failure in US history",
                confidence=1.0,
            ),
            VerifiedEvent(
                date="2023-03-12",
                description="Fed, Treasury, FDIC announce all depositors will be made whole; Fed creates BTFP",
                shock_magnitude=0.8,
                shock_direction=1,
                source_name="Federal Reserve",
                quantitative_detail="Bank Term Funding Program (BTFP) for liquidity; 87% deposits were uninsured",
                confidence=1.0,
            ),
            VerifiedEvent(
                date="2023-03-13",
                description="Signature Bank closed by NYDFS; regional bank stocks plunge; contagion fears spread",
                shock_magnitude=0.7,
                shock_direction=-1,
                source_name="Reuters",
                quantitative_detail="Signature Bank: $110B assets; KBW Regional Bank Index -12%",
                confidence=1.0,
            ),
        ],
        polls=[
            VerifiedPoll(date="2023-03-15", pro_pct=19.0, sample_size=1500,
                         source="Gallup", methodology="Online survey"),
        ],
        stakeholders=[
            StakeholderInfo(name="Greg Becker", role="CEO, Silicon Valley Bank",
                           position="Urged calm, asked clients not to withdraw; resigned after closure",
                           date="2023-03-09", source="SVB internal memo"),
            StakeholderInfo(name="Janet Yellen", role="US Treasury Secretary",
                           position="Announced full depositor protection; emphasized banking system soundness",
                           date="2023-03-12", source="Treasury statement"),
            StakeholderInfo(name="Jerome Powell", role="Chair, Federal Reserve",
                           position="Created BTFP emergency facility; signaled continued rate path",
                           date="2023-03-12", source="Fed press release"),
        ],
        key_figures={
            "htm_portfolio": "$91B in held-to-maturity securities",
            "uninsured_deposits_pct": "87% of deposits uninsured",
            "capital_raise_attempt": "$2.25B (failed)",
            "single_day_withdrawals": "$42B on March 9",
            "stock_drop_day1": "-60% on March 9",
            "total_assets": "$209B at closure",
            "bank_rank": "2nd largest US bank failure (after WaMu 2008)",
        },
        outcome_description="SVB collapsed in 48 hours, triggering regional banking crisis and federal intervention",
        outcome_pro_pct=19.0,
        outcome_source="Gallup confidence in banking survey, March 2023",
        n_sources_consulted=5,
        quality_score=92,
    )

    print("=" * 70)
    print("ScenarioResearcher Demo: SVB Collapse (March 2023)")
    print("=" * 70)

    print(f"\nScenario: {sheet.scenario_id}")
    print(f"Quality Score: {sheet.quality_score}/100")
    print(f"Events: {len(sheet.events)} | Polls: {len(sheet.polls)} | "
          f"Stakeholders: {len(sheet.stakeholders)} | Key Figures: {len(sheet.key_figures)}")

    print(f"\n--- Timeline ({len(sheet.events)} events) ---")
    for e in sheet.events:
        arrow = "↑" if e.shock_direction > 0 else "↓"
        print(f"  [{e.date}] {arrow} mag={e.shock_magnitude:.1f} | {e.description[:80]}")
        if e.quantitative_detail:
            print(f"             {e.quantitative_detail}")

    print(f"\n--- Key Figures ---")
    for k, v in sheet.key_figures.items():
        print(f"  {k}: {v}")

    print(f"\n--- Outcome ---")
    print(f"  {sheet.outcome_description}")
    print(f"  Pro %: {sheet.outcome_pro_pct}% — Source: {sheet.outcome_source}")

    print(f"\n--- to_llm_context() output ---")
    print(sheet.to_llm_context())

    print(f"\n--- to_scenario_events(n_rounds=7) ---")
    for ev in sheet.to_scenario_events(7):
        print(f"  Round {ev['round']}: mag={ev['shock_magnitude']}, "
              f"dir={ev['shock_direction']:+d}, verified={ev['_verified']}")
        print(f"    {ev['description'][:90]}...")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    _demo_svb()
