"""Test AgentGrounder with mock LLM and mock search — no real API calls."""

import json
import pytest

from src.grounding.agent_grounder import (
    AgentGrounder,
    GroundedAgent,
    ScenarioContext,
    ground_scenario_from_config,
    CELEBRITY_BLOCKLIST,
    RIGIDITY_BY_STAKE,
)

# ── Fixtures ────────────────────────────────────────────────────────────

DIESELGATE_CONTEXT = ScenarioContext(
    scenario_id="CORP-2015-DIESELGATE_VW",
    topic="Volkswagen emissions scandal (Dieselgate)",
    domain="corporate",
    country="Germany/USA/EU",
    timeframe="September 2015 - March 2016",
    key_question="Public trust in Volkswagen after emissions fraud revelation",
    n_elite_target=5,
)

MOCK_SEARCH_QUERIES = json.dumps([
    "Dieselgate VW key executives Winterkorn Horn 2015",
    "VW emissions scandal EPA CARB regulators 2015",
    "Volkswagen Dieselgate congressional hearing witnesses",
    "Dieselgate analyst reactions automotive industry 2015",
    "VW emissions affected car owners class action 2015",
])

MOCK_SEARCH_RESULTS = [
    {
        "url": "https://example.com/dieselgate-timeline",
        "text": "Martin Winterkorn resigned as CEO of Volkswagen on September 23, 2015, "
                "days after the EPA issued a Notice of Violation. Michael Horn, CEO of VW America, "
                "testified before Congress admitting 'we totally screwed up'. Gina McCarthy, EPA "
                "Administrator, called VW's actions 'a threat to public health'.",
    },
    {
        "url": "https://example.com/carb-investigation",
        "text": "Mary Nichols, Chair of the California Air Resources Board (CARB), led the "
                "technical investigation alongside the International Council on Clean Transportation. "
                "Alexander Dobrindt, German Transport Minister, ordered independent testing of "
                "all VW diesel vehicles sold in Germany.",
    },
    {
        "url": "https://example.com/analyst-views",
        "text": "Ferdinand Dudenhöffer, automotive analyst at CAR Institute, called the scandal "
                "'the worst crisis in VW's history'. Arndt Ellinghorst, Evercore ISI analyst, "
                "downgraded VW stock and estimated $30B+ in total costs.",
    },
]

MOCK_STAKEHOLDERS = json.dumps([
    {
        "full_name": "Martin Winterkorn",
        "role": "CEO Volkswagen AG",
        "archetype": "business_leader",
        "stake_type": "decision_maker",
        "position_estimate": 0.6,
        "influence": 0.95,
        "rigidity_estimate": 0.85,
        "relevance_score": 1.0,
        "evidence": "CEO who resigned after EPA Notice of Violation",
        "bio": "CEO of Volkswagen Group 2007-2015.",
        "communication_style": "formal, defensive",
        "key_traits": ["strategic", "evasive"],
    },
    {
        "full_name": "Gina McCarthy",
        "role": "EPA Administrator",
        "archetype": "regulator",
        "stake_type": "regulator",
        "position_estimate": -0.8,
        "influence": 0.85,
        "rigidity_estimate": 0.80,
        "relevance_score": 0.95,
        "evidence": "Led the EPA investigation that exposed VW",
        "bio": "EPA Administrator 2013-2017.",
        "communication_style": "authoritative, firm",
        "key_traits": ["principled", "methodical"],
    },
    {
        "full_name": "Michael Horn",
        "role": "CEO Volkswagen America",
        "archetype": "business_leader",
        "stake_type": "decision_maker",
        "position_estimate": 0.3,
        "influence": 0.70,
        "rigidity_estimate": 0.75,
        "relevance_score": 0.85,
        "evidence": "Testified before US Congress",
        "bio": "VW America CEO during scandal.",
        "communication_style": "direct, apologetic",
        "key_traits": ["pragmatic", "candid"],
    },
    {
        "full_name": "Alexander Dobrindt",
        "role": "German Federal Minister of Transport",
        "archetype": "politician",
        "stake_type": "regulator",
        "position_estimate": -0.3,
        "influence": 0.75,
        "rigidity_estimate": 0.70,
        "relevance_score": 0.80,
        "evidence": "Ordered independent VW testing in Germany",
        "bio": "German Transport Minister 2013-2017.",
        "communication_style": "formal, cautious",
        "key_traits": ["cautious", "institutional"],
    },
    {
        "full_name": "Mary Nichols",
        "role": "Chair, California Air Resources Board",
        "archetype": "regulator",
        "stake_type": "regulator",
        "position_estimate": -0.9,
        "influence": 0.80,
        "rigidity_estimate": 0.85,
        "relevance_score": 0.90,
        "evidence": "CARB testing first uncovered the cheating",
        "bio": "CARB Chair, led technical investigation.",
        "communication_style": "technical, persistent",
        "key_traits": ["tenacious", "evidence-driven"],
    },
    {
        "full_name": "Elon Musk",
        "role": "CEO Tesla",
        "archetype": "business_leader",
        "stake_type": "analyst",
        "position_estimate": -0.5,
        "influence": 0.90,
        "rigidity_estimate": 0.60,
        "relevance_score": 0.3,
        "evidence": "Tweeted about Dieselgate but no direct involvement",
        "bio": "Tesla CEO, competitor.",
        "communication_style": "informal, provocative",
        "key_traits": ["contrarian"],
    },
])


def _make_mock_llm(call_log: list | None = None):
    """Return a mock LLM function that returns appropriate responses."""
    call_count = [0]
    if call_log is None:
        call_log = []

    def mock_llm(prompt: str) -> str:
        call_log.append(prompt)
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return MOCK_SEARCH_QUERIES
        else:
            return MOCK_STAKEHOLDERS

    return mock_llm


def _make_mock_search():
    """Return a mock search function."""
    def mock_search(query: str) -> list[dict]:
        return MOCK_SEARCH_RESULTS
    return mock_search


# ── Tests ───────────────────────────────────────────────────────────────

def test_dieselgate_grounding():
    """Verify Dieselgate produces VW stakeholders, not celebrity agents."""
    grounder = AgentGrounder(
        llm_fn=_make_mock_llm(),
        search_fn=_make_mock_search(),
        min_relevance=0.4,
    )

    agents = grounder.ground(DIESELGATE_CONTEXT)

    # Should produce 5 agents (n_elite_target)
    assert len(agents) == 5

    names_lower = {a.name.lower() for a in agents}

    # No celebrity agents
    for celeb in ["elon musk", "tim cook", "jeff bezos", "mark zuckerberg"]:
        assert celeb not in names_lower, f"Celebrity '{celeb}' should not appear"

    # At least 1 VW executive
    vw_execs = [a for a in agents if "volkswagen" in a.role.lower() or "vw" in a.role.lower()]
    assert len(vw_execs) >= 1, "Should have at least 1 VW executive"

    # At least 1 regulator
    regulators = [a for a in agents if a.stake_type == "regulator"]
    assert len(regulators) >= 1, "Should have at least 1 regulator"

    # All pass relevance threshold
    for a in agents:
        assert a.relevance_score >= 0.4, f"{a.name} relevance {a.relevance_score} below threshold"

    # Position coherence: VW executives should be defensive (position > 0)
    for a in vw_execs:
        assert a.position > 0, f"VW exec {a.name} should have positive position (defensive)"

    # Regulators should be critical (position < 0)
    for a in regulators:
        assert a.position < 0, f"Regulator {a.name} should have negative position (critical)"


def test_relevance_filtering():
    """Verify that agents below relevance threshold are excluded."""
    grounder = AgentGrounder(
        llm_fn=_make_mock_llm(),
        search_fn=_make_mock_search(),
        min_relevance=0.85,  # High threshold — should filter more
    )

    agents = grounder.ground(DIESELGATE_CONTEXT)

    # Only agents with relevance >= 0.85 should survive
    for a in agents:
        assert a.relevance_score >= 0.85, f"{a.name} should have been filtered (relevance={a.relevance_score})"

    # Elon Musk (0.3) and Dobrindt (0.80) should be filtered out
    names = {a.name for a in agents}
    assert "Alexander Dobrindt" not in names
    # Musk already filtered by celebrity blocklist + low relevance


def test_rigidity_by_role():
    """Verify rigidity is clamped to stake_type range."""
    grounder = AgentGrounder(
        llm_fn=_make_mock_llm(),
        search_fn=_make_mock_search(),
        min_relevance=0.4,
    )
    agents = grounder.ground(DIESELGATE_CONTEXT)

    for agent in agents:
        if agent.stake_type in RIGIDITY_BY_STAKE:
            lo, hi = RIGIDITY_BY_STAKE[agent.stake_type]
            assert lo <= agent.rigidity <= hi, (
                f"{agent.name} ({agent.stake_type}): rigidity {agent.rigidity} "
                f"outside [{lo}, {hi}]"
            )


def test_output_format_compatibility():
    """Verify to_sim_format() produces the dict EliteAgent.from_spec() expects."""
    agent = GroundedAgent(
        name="Martin Winterkorn",
        role="CEO Volkswagen AG",
        archetype="business_leader",
        position=0.6,
        influence=0.95,
        rigidity=0.85,
        relevance_score=1.0,
        stake_type="decision_maker",
        evidence=["Resigned after EPA notice"],
        bio="CEO of VW 2007-2015.",
        communication_style="formal",
        key_traits=["strategic"],
    )

    sim = agent.to_sim_format()

    # Required fields for EliteAgent.from_spec()
    assert "id" in sim and isinstance(sim["id"], str) and len(sim["id"]) > 0
    assert "name" in sim and sim["name"] == "Martin Winterkorn"
    assert "role" in sim and sim["role"] == "CEO Volkswagen AG"
    assert "archetype" in sim and sim["archetype"] == "business_leader"
    assert "position" in sim and -1.0 <= sim["position"] <= 1.0
    assert "influence" in sim and 0.0 <= sim["influence"] <= 1.0
    assert "rigidity" in sim and 0.0 <= sim["rigidity"] <= 1.0
    assert "bio" in sim
    assert "communication_style" in sim
    assert "key_traits" in sim and isinstance(sim["key_traits"], list)
    assert "platform_primary" in sim
    assert "platform_secondary" in sim

    # ID should be snake_case
    assert sim["id"] == "martin_winterkorn"

    # Metadata fields should be prefixed
    assert sim["_grounded"] is True


def test_fallback_no_search():
    """If search fails, AgentGrounder returns empty list without crashing."""
    def failing_search(query: str) -> list[dict]:
        raise ConnectionError("Network unreachable")

    grounder = AgentGrounder(
        llm_fn=_make_mock_llm(),
        search_fn=failing_search,
        min_relevance=0.4,
    )

    # Should not raise — returns empty list
    agents = grounder.ground(DIESELGATE_CONTEXT)
    assert agents == []


def test_fallback_no_search_fn():
    """If no search_fn is configured at all, returns empty list."""
    grounder = AgentGrounder(
        llm_fn=_make_mock_llm(),
        search_fn=None,
        min_relevance=0.4,
    )
    agents = grounder.ground(DIESELGATE_CONTEXT)
    assert agents == []


def test_ground_scenario_from_config():
    """Test the convenience function with a scenario config dict."""
    config = {
        "scenario_id": "CORP-2015-DIESELGATE_VW",
        "topic": "Volkswagen emissions scandal (Dieselgate)",
        "domain": "corporate",
        "country": "Germany/USA/EU",
        "timeframe": "September 2015 - March 2016",
        "key_question": "Public trust in Volkswagen after emissions fraud",
        "n_elite": 5,
    }

    result = ground_scenario_from_config(
        config,
        llm_fn=_make_mock_llm(),
        search_fn=_make_mock_search(),
    )

    assert isinstance(result, list)
    assert len(result) == 5
    # Each should be a dict with required keys
    for r in result:
        assert isinstance(r, dict)
        assert "id" in r
        assert "name" in r
        assert "position" in r


def test_celebrity_blocklist():
    """Verify the celebrity blocklist is populated."""
    assert "elon musk" in CELEBRITY_BLOCKLIST
    assert "tim cook" in CELEBRITY_BLOCKLIST
    assert "jeff bezos" in CELEBRITY_BLOCKLIST


def test_position_clamping():
    """Verify positions are clamped to [-1, +1] in sim format."""
    agent = GroundedAgent(name="Test", role="Test", position=1.5)
    sim = agent.to_sim_format()
    assert sim["position"] == 1.0

    agent2 = GroundedAgent(name="Test2", role="Test", position=-2.0)
    sim2 = agent2.to_sim_format()
    assert sim2["position"] == -1.0


def test_sorted_by_relevance():
    """Verify agents are returned sorted by relevance descending."""
    grounder = AgentGrounder(
        llm_fn=_make_mock_llm(),
        search_fn=_make_mock_search(),
        min_relevance=0.4,
    )
    agents = grounder.ground(DIESELGATE_CONTEXT)

    scores = [a.relevance_score for a in agents]
    assert scores == sorted(scores, reverse=True), "Agents should be sorted by relevance"
