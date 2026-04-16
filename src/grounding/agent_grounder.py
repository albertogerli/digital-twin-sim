"""AgentGrounder: generate scenario-specific elite agents based on real stakeholders.

Pipeline:
1. Given a scenario (topic, domain, timeframe), run structured web search
   to identify real key players
2. Filter by relevance (direct stake, public statements, sector involvement)
3. Generate agent profiles with initial positions informed by public record
4. Output: list of elite agents in simulator-compatible format

AgentGrounder does NOT modify the JAX simulator, opinion dynamics, or calibration.
It operates upstream: improving the QUALITY of inputs the simulator receives.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Maps stake_type → (rigidity_lo, rigidity_hi)
RIGIDITY_BY_STAKE = {
    "decision_maker": (0.75, 0.90),
    "regulator": (0.70, 0.85),
    "analyst": (0.40, 0.60),
    "activist": (0.80, 0.95),
    "affected_party": (0.50, 0.70),
}

# Celebrity names that should never appear as scenario-specific elite agents
CELEBRITY_BLOCKLIST = {
    "elon musk", "tim cook", "jeff bezos", "mark zuckerberg", "bill gates",
    "warren buffett", "sam altman", "sundar pichai", "satya nadella",
    "jensen huang", "larry page", "sergey brin", "jack dorsey",
}


@dataclass
class GroundedAgent:
    """An elite agent grounded on a real person/role."""
    name: str                          # Real name (e.g. "Martin Winterkorn")
    role: str                          # Role (e.g. "CEO Volkswagen AG")
    archetype: str = "politician"      # Archetype from domain plugin
    position: float = 0.0             # Initial position [-1, +1], informed
    influence: float = 0.5            # 0-1, how much they shape opinion
    rigidity: float = 0.7             # 0-1, resistance to position change
    relevance_score: float = 0.0      # 0-1, how relevant to the scenario
    evidence: list[str] = field(default_factory=list)
    stake_type: str = ""              # decision_maker|regulator|analyst|activist|affected_party
    bio: str = ""                     # 1-2 sentence bio
    communication_style: str = ""     # e.g. "formal, decisive"
    key_traits: list[str] = field(default_factory=list)

    def to_sim_format(self) -> dict:
        """Convert to the dict format that EliteAgent.from_spec() expects."""
        agent_id = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")
        return {
            "id": agent_id,
            "name": self.name,
            "role": self.role,
            "archetype": self.archetype,
            "position": max(-1.0, min(1.0, self.position)),
            "influence": max(0.0, min(1.0, self.influence)),
            "rigidity": max(0.0, min(1.0, self.rigidity)),
            "bio": self.bio or f"{self.role}. {self.evidence[0]}" if self.evidence else self.role,
            "communication_style": self.communication_style,
            "key_traits": self.key_traits,
            "platform_primary": "",
            "platform_secondary": "",
            # Metadata (prefixed with _ so simulator ignores them)
            "_relevance": self.relevance_score,
            "_stake": self.stake_type,
            "_evidence": self.evidence,
            "_grounded": True,
        }


@dataclass
class ScenarioContext:
    """Scenario context for grounding."""
    scenario_id: str = ""              # e.g. "CORP-2015-DIESELGATE_VW"
    topic: str = ""                    # e.g. "Volkswagen emissions scandal (Dieselgate)"
    domain: str = ""                   # e.g. "corporate"
    country: str = ""                  # e.g. "Germany/USA/EU"
    timeframe: str = ""                # e.g. "September 2015 - March 2016"
    key_question: str = ""             # e.g. "Public trust in VW after emissions fraud"
    n_elite_target: int = 5


class AgentGrounder:
    """Generate elite agents grounded on real stakeholders."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        search_fn: Optional[Callable[[str], list[dict]]] = None,
        max_search_queries: int = 5,
        min_relevance: float = 0.4,
    ):
        """
        Args:
            llm_fn: Function(prompt) → str. Wraps any LLM client.
            search_fn: Function(query) → list[dict] with 'url' and 'text' keys.
            max_search_queries: Max number of search queries to generate.
            min_relevance: Minimum relevance score to keep a candidate.
        """
        self.llm_fn = llm_fn
        self.search_fn = search_fn
        self.max_search_queries = max_search_queries
        self.min_relevance = min_relevance

    def ground(self, context: ScenarioContext) -> list[GroundedAgent]:
        """Full pipeline: search → identify → filter → profile.

        Returns:
            List of GroundedAgent sorted by relevance_score descending,
            filtered by min_relevance, truncated to n_elite_target.
        """
        # Step 1: Generate scenario-specific search queries
        queries = self._generate_search_queries(context)
        logger.info(f"Generated {len(queries)} search queries for {context.scenario_id}")

        # Step 2: Execute searches
        raw_results = self._execute_searches(queries)
        logger.info(f"Collected {len(raw_results)} search results")

        if not raw_results:
            logger.warning("No search results — returning empty agent list")
            return []

        # Step 3: Extract stakeholder candidates from results
        candidates = self._extract_stakeholders(context, raw_results)
        logger.info(f"Extracted {len(candidates)} stakeholder candidates")

        # Step 4: Filter by relevance
        relevant = [c for c in candidates if c.relevance_score >= self.min_relevance]

        # Step 5: Filter out celebrity agents
        relevant = [c for c in relevant if c.name.lower() not in CELEBRITY_BLOCKLIST]
        logger.info(f"{len(relevant)} candidates pass relevance + celebrity filter")

        # Step 6: Adjust rigidity/tolerance by role
        agents = self._profile_agents(relevant)

        # Step 7: Sort and truncate
        agents.sort(key=lambda a: a.relevance_score, reverse=True)
        return agents[:context.n_elite_target]

    def _generate_search_queries(self, context: ScenarioContext) -> list[str]:
        """Generate 3-5 targeted search queries to identify stakeholders."""
        prompt = f"""Given this scenario, generate {self.max_search_queries} specific web search queries
to identify the KEY STAKEHOLDERS — real people who had direct involvement,
decision-making power, or significant public voice in this event.

Scenario: {context.topic}
Domain: {context.domain}
Country: {context.country}
Timeframe: {context.timeframe}
Key question: {context.key_question}

Rules:
- Each query should target a DIFFERENT category of stakeholder
  (executives, regulators, analysts, activists, affected parties)
- Queries should be specific enough to find NAMES and ROLES
- Include the timeframe to get period-accurate results
- Do NOT generate generic queries about the event itself

Return as JSON array of strings, nothing else."""

        response = self._call_llm(prompt)
        return self._parse_json_array(response)

    def _execute_searches(self, queries: list[str]) -> list[dict]:
        """Execute web searches. Returns list of raw result dicts."""
        if self.search_fn is None:
            logger.warning("No search function configured — skipping search")
            return []

        all_results = []
        for q in queries:
            try:
                results = self.search_fn(q)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{q}': {e}")
        return all_results

    def _extract_stakeholders(
        self, context: ScenarioContext, search_results: list[dict]
    ) -> list[GroundedAgent]:
        """Use LLM to extract names/roles/positions from search results."""
        search_text = "\n---\n".join([
            f"Source: {r.get('url', 'unknown')}\n{r.get('text', '')[:500]}"
            for r in search_results[:15]
        ])

        prompt = f"""From these search results about "{context.topic}", extract ALL real people
who are key stakeholders. For each person, determine:

1. full_name: Their complete name
2. role: Their role/title at the time of the event
3. archetype: One of: politician, business_leader, scientist, journalist, regulator, activist, analyst
4. stake_type: One of: decision_maker, regulator, analyst, activist, affected_party
5. position_estimate: Their likely public stance on "{context.key_question}"
   as a number from -1.0 (strongly against) to +1.0 (strongly for)
6. influence: How much they shape public opinion on this (0.0-1.0)
   - Head of state or CEO of involved company: 0.8-0.95
   - Minister or regulator: 0.7-0.85
   - Prominent analyst or journalist: 0.5-0.7
7. rigidity_estimate: How rigid their position is (0.5=moderate, 0.9=very rigid)
8. relevance_score: How central they are to this scenario (0-1)
   - 1.0 = directly responsible / primary decision maker
   - 0.7 = key regulator or vocal public figure
   - 0.4 = relevant analyst or commentator
   - 0.2 = peripheral figure
9. evidence: 1-2 sentence summary of WHY this position/relevance
10. bio: 1-2 sentence factual bio
11. communication_style: How they communicate (e.g. "formal, decisive")
12. key_traits: 2-3 traits (e.g. ["strategic", "pragmatic"])

CRITICAL: Only include people with DIRECT stake in this specific scenario.
Do NOT include generic celebrities, tech CEOs, or public figures who had
no documented involvement.

Search results:
{search_text}

Return as JSON array of objects. Nothing else."""

        response = self._call_llm(prompt)
        candidates_raw = self._parse_json_array(response)

        agents = []
        for c in candidates_raw:
            if not isinstance(c, dict):
                continue
            try:
                agents.append(GroundedAgent(
                    name=c["full_name"],
                    role=c.get("role", ""),
                    archetype=c.get("archetype", "politician"),
                    position=float(c.get("position_estimate", 0.0)),
                    influence=float(c.get("influence", 0.5)),
                    rigidity=float(c.get("rigidity_estimate", 0.7)),
                    relevance_score=float(c.get("relevance_score", 0.5)),
                    evidence=[c.get("evidence", "")],
                    stake_type=c.get("stake_type", ""),
                    bio=c.get("bio", ""),
                    communication_style=c.get("communication_style", ""),
                    key_traits=c.get("key_traits", []),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed candidate: {e}")
        return agents

    def _profile_agents(self, candidates: list[GroundedAgent]) -> list[GroundedAgent]:
        """Refine profiles: adjust rigidity by role type."""
        for agent in candidates:
            if agent.stake_type in RIGIDITY_BY_STAKE:
                lo, hi = RIGIDITY_BY_STAKE[agent.stake_type]
                agent.rigidity = max(lo, min(hi, agent.rigidity))

        return candidates

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM via the injected function."""
        if self.llm_fn is None:
            raise ValueError("LLM function not configured")
        return self.llm_fn(prompt)

    @staticmethod
    def _parse_json_array(text: str) -> list:
        """Extract a JSON array from LLM output, tolerating markdown fences."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return json.loads(text)


# ── Convenience function for simulator integration ──────────────────────

def ground_scenario_from_config(
    scenario_config: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
    search_fn: Optional[Callable[[str], list[dict]]] = None,
    min_relevance: float = 0.4,
) -> list[dict]:
    """Given a scenario config dict, return grounded elite agents in simulator format.

    This is the interface the simulator will call.

    Args:
        scenario_config: dict with 'scenario_id', 'domain', 'topic', etc.
        llm_fn: Function(prompt) → str response.
        search_fn: Function(query) → list[dict] with 'url', 'text'.

    Returns:
        List of dicts compatible with EliteAgent.from_spec().
        Empty list if grounding fails (caller should fall back to LLM generation).
    """
    gt = scenario_config.get("ground_truth_outcome", {})
    context = ScenarioContext(
        scenario_id=scenario_config.get("scenario_id", "unknown"),
        topic=scenario_config.get("topic", scenario_config.get("title", "")),
        domain=scenario_config.get("domain", ""),
        country=scenario_config.get("country", ""),
        timeframe=scenario_config.get("timeframe", ""),
        key_question=scenario_config.get("key_question", gt.get("description", "")),
        n_elite_target=scenario_config.get("n_elite", 5),
    )

    grounder = AgentGrounder(
        llm_fn=llm_fn,
        search_fn=search_fn,
        min_relevance=min_relevance,
    )

    try:
        agents = grounder.ground(context)
    except Exception as e:
        logger.error(f"AgentGrounder failed for {context.scenario_id}: {e}")
        return []

    return [a.to_sim_format() for a in agents]


# ── Standalone demo ─────────────────────────────────────────────────────

def _demo_dieselgate():
    """Print a hardcoded Dieselgate example to show expected output."""
    agents = [
        GroundedAgent(
            name="Martin Winterkorn",
            role="CEO Volkswagen AG (resigned Sept 2015)",
            archetype="business_leader",
            position=0.6,
            influence=0.95,
            rigidity=0.85,
            relevance_score=1.0,
            stake_type="decision_maker",
            evidence=["CEO who resigned after EPA Notice of Violation; publicly denied personal knowledge"],
            bio="CEO of Volkswagen Group 2007-2015, resigned days after EPA revealed defeat devices.",
            communication_style="formal, defensive, corporate",
            key_traits=["strategic", "evasive", "institutional"],
        ),
        GroundedAgent(
            name="Gina McCarthy",
            role="Administrator, U.S. Environmental Protection Agency",
            archetype="regulator",
            position=-0.8,
            influence=0.85,
            rigidity=0.80,
            relevance_score=0.95,
            stake_type="regulator",
            evidence=["EPA under her leadership issued the original Notice of Violation to VW"],
            bio="EPA Administrator 2013-2017; led the investigation that exposed VW's defeat devices.",
            communication_style="authoritative, data-driven, firm",
            key_traits=["principled", "methodical"],
        ),
        GroundedAgent(
            name="Michael Horn",
            role="CEO Volkswagen Group of America",
            archetype="business_leader",
            position=0.3,
            influence=0.70,
            rigidity=0.75,
            relevance_score=0.85,
            stake_type="decision_maker",
            evidence=["Testified before US Congress; admitted 'we totally screwed up'"],
            bio="VW America CEO who faced Congressional hearings and admitted wrongdoing.",
            communication_style="direct, apologetic, pragmatic",
            key_traits=["pragmatic", "candid"],
        ),
        GroundedAgent(
            name="Alexander Dobrindt",
            role="German Federal Minister of Transport",
            archetype="politician",
            position=-0.3,
            influence=0.75,
            rigidity=0.70,
            relevance_score=0.80,
            stake_type="regulator",
            evidence=["Ordered independent testing of VW vehicles in Germany; criticized VW publicly"],
            bio="German Transport Minister who ordered domestic investigations into VW emissions.",
            communication_style="formal, politically cautious",
            key_traits=["cautious", "institutional"],
        ),
        GroundedAgent(
            name="Mary Nichols",
            role="Chair, California Air Resources Board (CARB)",
            archetype="regulator",
            position=-0.9,
            influence=0.80,
            rigidity=0.85,
            relevance_score=0.90,
            stake_type="regulator",
            evidence=["CARB's testing first detected anomalies; led technical investigation with ICCT"],
            bio="CARB Chair whose agency's testing first uncovered VW's emissions cheating.",
            communication_style="technical, authoritative, persistent",
            key_traits=["tenacious", "evidence-driven"],
        ),
    ]

    print("=" * 70)
    print("AgentGrounder Demo: Dieselgate VW (Sept 2015)")
    print("=" * 70)
    for a in agents:
        sim = a.to_sim_format()
        print(f"\n  {a.name}")
        print(f"    Role:      {a.role}")
        print(f"    Archetype: {a.archetype}")
        print(f"    Stake:     {a.stake_type}")
        print(f"    Position:  {a.position:+.2f}  (rigidity={a.rigidity:.2f}, influence={a.influence:.2f})")
        print(f"    Relevance: {a.relevance_score:.2f}")
        print(f"    Evidence:  {a.evidence[0]}")
        print(f"    Sim ID:    {sim['id']}")
    print(f"\n{'=' * 70}")
    print(f"Total: {len(agents)} grounded elite agents (0 celebrity agents)")
    print("=" * 70)


if __name__ == "__main__":
    _demo_dieselgate()
