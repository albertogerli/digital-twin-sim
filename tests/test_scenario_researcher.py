"""Test ScenarioResearcher with mocks — no real API calls."""

import json
import pytest

from src.grounding.scenario_researcher import (
    ScenarioResearcher,
    FactSheet,
    VerifiedEvent,
    VerifiedPoll,
    StakeholderInfo,
)

# ── Mock data ───────────────────────────────────────────────────────────

SVB_BROAD_SEARCH_RESULTS = [
    {
        "url": "https://example.com/svb-timeline",
        "text": (
            "Silicon Valley Bank collapsed in March 2023 after a bank run. "
            "On March 8, SVB announced a $1.8B loss on bond sales and a $2.25B capital raise. "
            "On March 9, customers withdrew $42B. On March 10, DFPI closed the bank. "
            "On March 12, the Fed announced all depositors would be made whole. "
            "SVB had $91B in HTM securities and 87% uninsured deposits."
        ),
    },
    {
        "url": "https://example.com/svb-outcome",
        "text": (
            "A Gallup poll in March 2023 found only 19% of Americans had confidence "
            "in the banking system, down from 27% in 2022. SVB collapse was the "
            "2nd largest US bank failure after Washington Mutual in 2008."
        ),
    },
]

SVB_TARGETED_SEARCH_RESULTS = [
    {
        "url": "https://example.com/svb-figures",
        "text": (
            "SVB total assets at closure: $209B. HTM portfolio: $91B in long-duration "
            "Treasuries and MBS. Uninsured deposits: 87% of $175B total. "
            "Stock drop on March 9: -60%. Single-day withdrawal: $42B."
        ),
    },
    {
        "url": "https://example.com/svb-stakeholders",
        "text": (
            "Greg Becker, SVB CEO, urged clients not to panic. Janet Yellen announced "
            "full depositor protection. Jerome Powell created the BTFP facility."
        ),
    },
]

MOCK_TIMELINE_RESPONSE = json.dumps([
    {
        "date": "2023-03-08",
        "description": "SVB announces $1.8B loss on bond portfolio sale and plans $2.25B capital raise",
        "shock_magnitude": 0.7,
        "shock_direction": -1,
        "source_name": "SEC Filing",
        "quantitative_detail": "$1.8B loss; $2.25B raise attempt",
    },
    {
        "date": "2023-03-09",
        "description": "Customers withdraw $42B in a single day; SVB stock drops 60%",
        "shock_magnitude": 0.9,
        "shock_direction": -1,
        "source_name": "Bloomberg",
        "quantitative_detail": "$42B withdrawals; stock -60%",
    },
    {
        "date": "2023-03-10",
        "description": "California DFPI closes SVB and appoints FDIC as receiver",
        "shock_magnitude": 1.0,
        "shock_direction": -1,
        "source_name": "FDIC",
        "quantitative_detail": "$209B total assets; 2nd largest bank failure in US history",
    },
    {
        "date": "2023-03-12",
        "description": "Fed, Treasury, FDIC announce all depositors will be made whole; Fed creates BTFP",
        "shock_magnitude": 0.8,
        "shock_direction": 1,
        "source_name": "Federal Reserve",
        "quantitative_detail": "Bank Term Funding Program; 87% deposits uninsured",
    },
    {
        "date": "2023-03-13",
        "description": "Signature Bank closed; regional bank stocks plunge; contagion fears spread",
        "shock_magnitude": 0.7,
        "shock_direction": -1,
        "source_name": "Reuters",
        "quantitative_detail": "Signature Bank: $110B assets; KBW Regional Bank Index -12%",
    },
])

MOCK_OUTCOME_RESPONSE = json.dumps({
    "outcome_description": "SVB collapsed triggering regional banking crisis and federal intervention",
    "outcome_pro_pct": 19.0,
    "outcome_source": "Gallup confidence in banking survey, March 2023",
})

MOCK_KEY_FIGURES_RESPONSE = json.dumps({
    "htm_portfolio": "$91B in held-to-maturity securities",
    "uninsured_deposits_pct": "87% of deposits uninsured",
    "capital_raise_attempt": "$2.25B (failed)",
    "single_day_withdrawals": "$42B on March 9",
    "stock_drop_day1": "-60% on March 9",
    "total_assets": "$209B at closure",
})

MOCK_POLLS_RESPONSE = json.dumps([
    {"date": "2023-03-15", "pro_pct": 19.0, "sample_size": 1500,
     "source": "Gallup", "methodology": "Online"},
])

MOCK_STAKEHOLDERS_RESPONSE = json.dumps([
    {"name": "Greg Becker", "role": "CEO, SVB",
     "position": "Urged calm, asked clients not to withdraw",
     "date": "2023-03-09", "source": "SVB memo"},
    {"name": "Janet Yellen", "role": "US Treasury Secretary",
     "position": "Announced full depositor protection",
     "date": "2023-03-12", "source": "Treasury statement"},
    {"name": "Jerome Powell", "role": "Chair, Federal Reserve",
     "position": "Created BTFP emergency facility",
     "date": "2023-03-12", "source": "Fed press release"},
])

# LLM responses in order of calls:
# 1. extract_timeline
# 2. extract_outcome
# 3. extract_key_figures
# 4. extract_polls
# 5. extract_stakeholders
ALL_LLM_RESPONSES = [
    MOCK_TIMELINE_RESPONSE,
    MOCK_OUTCOME_RESPONSE,
    MOCK_KEY_FIGURES_RESPONSE,
    MOCK_POLLS_RESPONSE,
    MOCK_STAKEHOLDERS_RESPONSE,
]


def _make_mock_llm():
    """Return a mock LLM that returns responses in order."""
    idx = [0]

    def mock_llm(prompt: str) -> str:
        i = idx[0]
        idx[0] += 1
        if i < len(ALL_LLM_RESPONSES):
            return ALL_LLM_RESPONSES[i]
        return "[]"

    return mock_llm


def _make_mock_search(broad_results=None, targeted_results=None):
    """Return a mock search function."""
    call_count = [0]

    def mock_search(query: str) -> list[dict]:
        call_count[0] += 1
        # First batch of calls → broad results; later → targeted
        if call_count[0] <= 4:
            return broad_results or SVB_BROAD_SEARCH_RESULTS
        return targeted_results or SVB_TARGETED_SEARCH_RESULTS

    return mock_search


# ── Tests ───────────────────────────────────────────────────────────────

def test_svb_collapse_research():
    """Verify SVB research extracts key facts."""
    researcher = ScenarioResearcher(
        llm_fn=_make_mock_llm(),
        search_fn=_make_mock_search(),
    )

    sheet = researcher.research(
        topic="Silicon Valley Bank collapse",
        domain="financial",
        country="USA",
        timeframe="March 2023",
        scenario_id="FIN-2023-SVB",
    )

    # Timeline has at least 5 events
    assert len(sheet.events) >= 5, f"Expected ≥5 events, got {len(sheet.events)}"

    # Key figures include specific numbers
    assert len(sheet.key_figures) >= 4, f"Expected ≥4 key figures, got {len(sheet.key_figures)}"

    # Outcome has a source
    assert sheet.outcome_source, "Outcome should have a source"
    assert sheet.outcome_pro_pct == 19.0

    # Quality score > 60
    assert sheet.quality_score > 60, f"Quality score {sheet.quality_score} too low"

    # Stakeholders extracted
    assert len(sheet.stakeholders) >= 2


def test_factsheet_to_llm_context():
    """Verify to_llm_context() produces structured text with constraints."""
    sheet = FactSheet(
        scenario_id="POL-2016-BREXIT",
        topic="Brexit referendum",
        domain="political",
        country="UK",
        timeframe_start="2016-02-20",
        timeframe_end="2016-06-23",
        events=[
            VerifiedEvent(
                date="2016-02-20",
                description="David Cameron announces referendum date",
                shock_magnitude=0.6,
                shock_direction=1,
                source_name="BBC",
            ),
            VerifiedEvent(
                date="2016-06-16",
                description="Jo Cox MP murdered",
                shock_magnitude=0.8,
                shock_direction=1,
                source_name="Guardian",
            ),
        ],
        polls=[
            VerifiedPoll(date="2016-06-20", pro_pct=44.0, sample_size=2000, source="YouGov"),
        ],
        stakeholders=[
            StakeholderInfo(name="Boris Johnson", role="Mayor of London",
                           position="Campaigned for Leave"),
        ],
        key_figures={"turnout": "72.2%"},
        outcome_pro_pct=51.89,
        outcome_source="Electoral Commission",
        outcome_description="Leave won with 51.89% of the vote",
    )

    context = sheet.to_llm_context()

    assert "VERIFIED FACTS" in context
    assert "INSTRUCTION" in context
    assert "51.89" in context
    assert "Electoral Commission" in context
    assert "David Cameron" in context
    assert "Jo Cox" in context
    assert "Boris Johnson" in context
    assert "YouGov" in context
    assert "72.2%" in context
    assert "Do NOT contradict" in context


def test_to_scenario_events_distribution():
    """Verify to_scenario_events() distributes events across rounds."""
    sheet = FactSheet(events=[
        VerifiedEvent(date=f"2023-03-{8+i:02d}", description=f"Event {i+1}",
                      shock_magnitude=0.5, shock_direction=-1)
        for i in range(8)
    ])

    # 8 events, 6 rounds → some rounds get multiple events
    events_6 = sheet.to_scenario_events(6)
    assert len(events_6) <= 6
    assert all(1 <= e["round"] <= 6 for e in events_6)
    # All events should use correct field names
    for e in events_6:
        assert "shock_magnitude" in e
        assert "shock_direction" in e
        assert "_verified" in e and e["_verified"] is True

    # 3 events, 6 rounds → some rounds have no events
    sheet2 = FactSheet(events=[
        VerifiedEvent(date=f"2023-03-{8+i:02d}", description=f"Event {i+1}",
                      shock_magnitude=0.5, shock_direction=-1)
        for i in range(3)
    ])
    events_sparse = sheet2.to_scenario_events(6)
    assert len(events_sparse) == 3  # Only 3 rounds have events

    # Empty events → empty list
    assert FactSheet().to_scenario_events(6) == []

    # Zero rounds → empty list
    assert sheet.to_scenario_events(0) == []


def test_quality_score_computation():
    """Verify quality score reflects completeness."""
    # Empty sheet: score ≈ 0
    empty = FactSheet()
    score_empty = ScenarioResearcher._compute_quality_score(empty)
    assert score_empty == 0

    # Only events (8 events × 4 = 32, capped at 30)
    events_only = FactSheet(events=[
        VerifiedEvent(date=f"2023-{i}", description=f"E{i}",
                      shock_magnitude=0.5, shock_direction=-1)
        for i in range(8)
    ])
    score_events = ScenarioResearcher._compute_quality_score(events_only)
    assert 25 <= score_events <= 35  # ~30 from events

    # Complete sheet: score > 70
    complete = FactSheet(
        events=[
            VerifiedEvent(date=f"2023-{i}", description=f"E{i}",
                          shock_magnitude=0.5, shock_direction=-1, confidence=1.0)
            for i in range(7)
        ],
        outcome_source="Official source",
        outcome_description="Outcome",
        polls=[VerifiedPoll(date="2023-01", pro_pct=50, source="Gallup")],
        key_figures={"a": "1", "b": "2", "c": "3"},
        stakeholders=[
            StakeholderInfo(name="A", role="R", position="P"),
            StakeholderInfo(name="B", role="R", position="P"),
        ],
    )
    score_complete = ScenarioResearcher._compute_quality_score(complete)
    assert score_complete > 70, f"Complete sheet score {score_complete} should be > 70"


def test_net_event_direction():
    """If two events in same round have opposing directions, net effect is computed."""
    sheet = FactSheet(events=[
        # Two events that will land in the same round (0 and 1 out of 2, both → round 1)
        VerifiedEvent(
            date="2023-03-10",
            description="Bank collapses",
            shock_magnitude=0.8,
            shock_direction=-1,
        ),
        VerifiedEvent(
            date="2023-03-10",
            description="Fed intervenes",
            shock_magnitude=0.6,
            shock_direction=1,
        ),
    ])

    events = sheet.to_scenario_events(1)  # 1 round → both events merge
    assert len(events) == 1

    e = events[0]
    # Net signed: 0.8*(-1) + 0.6*(+1) = -0.2 → direction -1
    assert e["shock_direction"] == -1
    # Avg magnitude: (0.8 + 0.6) / 2 = 0.7
    assert abs(e["shock_magnitude"] - 0.7) < 0.01


def test_cross_verify_raises_confidence():
    """After cross-verification, high-impact events should have higher confidence."""
    verify_called = []

    def mock_search(query: str) -> list[dict]:
        verify_called.append(query)
        return [{"url": "https://example.com/verified", "text": "Confirmed"}]

    researcher = ScenarioResearcher(
        llm_fn=_make_mock_llm(),
        search_fn=mock_search,
    )

    sheet = FactSheet(events=[
        VerifiedEvent(
            date="2023-03-10",
            description="Major event with high impact",
            shock_magnitude=0.9,
            shock_direction=-1,
            confidence=0.8,
        ),
        VerifiedEvent(
            date="2023-03-11",
            description="Minor event",
            shock_magnitude=0.2,
            shock_direction=-1,
            confidence=0.8,
        ),
    ])

    sheet = researcher._cross_verify(sheet)

    # High-impact event (0.9 > 0.5) should be verified → confidence 1.0
    assert sheet.events[0].confidence == 1.0
    assert sheet.events[0].source_url == "https://example.com/verified"

    # Low-impact event should NOT be verified (stays at 0.8)
    assert sheet.events[1].confidence == 0.8


def test_fallback_no_search():
    """If no search_fn, researcher returns empty sheet without crashing."""
    researcher = ScenarioResearcher(
        llm_fn=_make_mock_llm(),
        search_fn=None,
    )

    sheet = researcher.research(
        topic="Test scenario",
        domain="test",
    )

    assert isinstance(sheet, FactSheet)
    assert sheet.events == []
    assert sheet.quality_score == 0


def test_fallback_search_error():
    """If search raises, researcher returns partial sheet."""
    def failing_search(query: str) -> list[dict]:
        raise ConnectionError("Network error")

    researcher = ScenarioResearcher(
        llm_fn=_make_mock_llm(),
        search_fn=failing_search,
    )

    sheet = researcher.research(
        topic="Test scenario",
        domain="test",
    )

    assert isinstance(sheet, FactSheet)
    # Should not raise — gracefully handles errors


def test_factsheet_to_dict():
    """Verify to_dict() serialization is JSON-safe."""
    sheet = FactSheet(
        scenario_id="TEST",
        topic="Test",
        events=[
            VerifiedEvent(date="2023-01-01", description="E1",
                          shock_magnitude=0.5, shock_direction=-1),
        ],
        key_figures={"a": "1"},
    )
    d = sheet.to_dict()
    # Should be JSON-serializable
    serialized = json.dumps(d)
    assert "TEST" in serialized
    assert isinstance(d["events"], list)
    assert isinstance(d["key_figures"], dict)
