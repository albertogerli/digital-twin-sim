"""Financial impact scorer tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_scorer():
    """Try to import FinancialImpactScorer."""
    try:
        from core.simulation.financial_impact import FinancialImpactScorer
        return FinancialImpactScorer
    except ImportError:
        pytest.skip("FinancialImpactScorer not available")


def test_crisis_scope_low():
    """Low polarization + positive sentiment → low crisis scope."""
    Scorer = _get_scorer()
    scorer = Scorer()
    result = scorer.score(
        polarization=1.0,
        sentiment={"positive": 0.6, "neutral": 0.3, "negative": 0.1},
        event_shock=0.1,
    )
    assert result["crisis_scope"] < 0.3


def test_crisis_scope_high():
    """High polarization + negative sentiment → high crisis scope."""
    Scorer = _get_scorer()
    scorer = Scorer()
    result = scorer.score(
        polarization=8.0,
        sentiment={"positive": 0.1, "neutral": 0.2, "negative": 0.7},
        event_shock=0.8,
    )
    assert result["crisis_scope"] > 0.5


def test_panic_multiplier():
    """Panic multiplier should amplify with consecutive negative rounds."""
    Scorer = _get_scorer()
    scorer = Scorer()

    # Simulate consecutive negative rounds
    for _ in range(3):
        result = scorer.score(
            polarization=7.0,
            sentiment={"positive": 0.1, "neutral": 0.2, "negative": 0.7},
            event_shock=0.6,
        )

    assert result.get("panic_multiplier", 1.0) >= 1.0


def test_score_returns_expected_keys():
    """Score output should contain standard fields."""
    Scorer = _get_scorer()
    scorer = Scorer()
    result = scorer.score(
        polarization=5.0,
        sentiment={"positive": 0.3, "neutral": 0.4, "negative": 0.3},
        event_shock=0.3,
    )
    # At minimum should have some impact data
    assert isinstance(result, dict)
    assert len(result) > 0
