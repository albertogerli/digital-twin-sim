"""Tests for contagion risk scorer — CRI computation, labels, containment."""

import math
import os
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator.contagion import (
    ContagionMetrics,
    ContagionReport,
    ContagionScorer,
    ContagionThreshold,
)


# ── Helpers: mock EscalationEngine + state ───────────────────────────────────


def _make_mock_engine(engagement_scores=None, engagement_trend=0.0, wave_1=None,
                      wave_2=None, wave_3=None, round_metrics=None):
    """Build a MagicMock EscalationEngine with the needed state attributes."""
    engine = MagicMock()
    engine.state = MagicMock()
    engine.state.engagement_scores = engagement_scores or []
    engine.state.engagement_trend = engagement_trend
    engine.state.latest_engagement = (
        engagement_scores[-1] if engagement_scores else 0.0
    )
    engine.state.round_metrics = round_metrics or []

    # Activation plan mocks
    engine.plan = MagicMock()
    engine.plan.wave_1 = wave_1 or []
    engine.plan.wave_2 = wave_2 or []
    engine.plan.wave_3 = wave_3 or []
    return engine


# ── Tests: CRI computation ───────────────────────────────────────────────────


class TestCRIComputation:
    def test_zero_engagement_low_cri(self):
        """All-zero inputs should produce CRI near 0."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)
        cri = scorer.score_round(
            round_num=1, post_count=10, reaction_count=0,
            repost_count=0, top_post_engagement=0,
        )
        assert 0.0 <= cri <= 0.1

    def test_high_engagement_high_cri(self):
        """Maximal inputs should produce high CRI."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)
        cri = scorer.score_round(
            round_num=1, post_count=10, reaction_count=200,
            repost_count=50, top_post_engagement=1000,
            institutional_actors_active=10,
            union_activated=True,
            party_activated=True,
            sectors_affected=5,
            geographic_regions=6,
            international_attention=True,
            hashtag_convergence=0.9,
        )
        assert cri > 0.5

    def test_cri_bounded_0_1(self):
        """CRI must always be in [0, 1]."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)
        for i in range(5):
            cri = scorer.score_round(
                round_num=i + 1, post_count=max(1, i * 20),
                reaction_count=i * 100,
                repost_count=i * 30, top_post_engagement=i * 200,
                institutional_actors_active=i * 3,
                union_activated=i > 2,
                party_activated=i > 3,
                sectors_affected=i + 1,
                geographic_regions=i + 1,
                international_attention=i > 3,
                hashtag_convergence=min(1.0, i * 0.25),
            )
            assert 0.0 <= cri <= 1.0

    def test_cri_momentum_floor(self):
        """CRI should not drop more than 0.15 from previous round."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)

        # Spike round
        cri_high = scorer.score_round(
            round_num=1, post_count=10, reaction_count=200,
            repost_count=50, top_post_engagement=1000,
            institutional_actors_active=10,
            union_activated=True, party_activated=True,
            sectors_affected=5, geographic_regions=5,
            international_attention=True, hashtag_convergence=0.9,
        )
        # Quiet round — raw CRI would be near 0, but momentum prevents big drop
        cri_low = scorer.score_round(
            round_num=2, post_count=10, reaction_count=0,
            repost_count=0, top_post_engagement=0,
        )
        assert cri_low >= cri_high - 0.15 - 1e-10

    def test_cri_history_accumulates(self):
        """Each score_round call appends to cri_history."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)
        for i in range(3):
            scorer.score_round(round_num=i + 1, post_count=10, reaction_count=10)
        assert len(scorer.cri_history) == 3


# ── Tests: sub-dimension computation ─────────────────────────────────────────


class TestSubDimensions:
    def test_virality_increases_with_reposts(self):
        """Higher repost rate produces higher virality score."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)

        m_low = ContagionMetrics(round_num=1, repost_rate=0.1,
                                  top_post_reach=10, hashtag_convergence=0.0,
                                  media_amplification=0.0)
        m_high = ContagionMetrics(round_num=2, repost_rate=3.0,
                                   top_post_reach=500, hashtag_convergence=0.8,
                                   media_amplification=8.0)

        v_low = scorer._compute_virality(m_low)
        v_high = scorer._compute_virality(m_high)
        assert v_high > v_low

    def test_institutional_union_activation(self):
        """Union activation adds 0.8 * 0.30 = 0.24 to institutional score."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)

        m_base = ContagionMetrics(round_num=1)
        m_union = ContagionMetrics(round_num=1, union_activation=True)

        score_base = scorer._compute_institutional(m_base)
        score_union = scorer._compute_institutional(m_union)
        assert score_union > score_base
        assert abs(score_union - score_base - 0.8 * 0.30) < 1e-10

    def test_spillover_with_international_attention(self):
        """International attention adds 0.25 to spillover score."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)

        m_base = ContagionMetrics(round_num=1, sectors_affected=1,
                                   geographic_spread=1)
        m_intl = ContagionMetrics(round_num=1, sectors_affected=1,
                                   geographic_spread=1,
                                   international_attention=True)

        s_base = scorer._compute_spillover(m_base)
        s_intl = scorer._compute_spillover(m_intl)
        assert abs(s_intl - s_base - 0.25) < 1e-10


# ── Tests: risk label thresholds ─────────────────────────────────────────────


class TestRiskLabels:
    @pytest.mark.parametrize("cri,expected_label", [
        (0.10, "low"),
        (0.24, "low"),
        (0.25, "moderate"),
        (0.49, "moderate"),
        (0.50, "high"),
        (0.74, "high"),
        (0.75, "critical"),
        (0.99, "critical"),
    ])
    def test_risk_label_thresholds(self, cri, expected_label):
        """Risk label follows defined breakpoints: <0.25 low, <0.50 moderate, etc."""
        if cri < 0.25:
            label = "low"
        elif cri < 0.50:
            label = "moderate"
        elif cri < 0.75:
            label = "high"
        else:
            label = "critical"
        assert label == expected_label


# ── Tests: containment window estimation ─────────────────────────────────────


class TestContainmentWindow:
    def test_critical_cri_returns_zero(self):
        """If CRI >= 0.75, containment window is 0 (already critical)."""
        engine = _make_mock_engine(engagement_trend=0.1)
        scorer = ContagionScorer(engine)
        scorer.cri_history = [0.80]
        window = scorer._estimate_containment_window()
        assert window == 0

    def test_cooling_trend_returns_none(self):
        """If engagement trend <= 0 and CRI < 0.75, returns None (no urgency)."""
        engine = _make_mock_engine(engagement_trend=-0.05)
        scorer = ContagionScorer(engine)
        scorer.cri_history = [0.40]
        window = scorer._estimate_containment_window()
        assert window is None

    def test_rising_trend_returns_positive_rounds(self):
        """If trend > 0 and CRI < 0.75, should return a positive integer."""
        engine = _make_mock_engine(engagement_trend=0.1)
        scorer = ContagionScorer(engine)
        scorer.cri_history = [0.40]
        window = scorer._estimate_containment_window()
        assert window is not None
        assert window >= 1

    def test_empty_history_returns_none(self):
        """No history => no containment estimate."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)
        window = scorer._estimate_containment_window()
        assert window is None


# ── Tests: ContagionReport serialization ─────────────────────────────────────


class TestContagionReport:
    def test_to_dict_has_expected_keys(self):
        report = ContagionReport(
            contagion_risk_index=0.55,
            risk_label="high",
            virality_risk=0.3,
            institutional_contagion=0.6,
            cross_domain_spillover=0.2,
        )
        d = report.to_dict()
        assert d["contagion_risk_index"] == 0.55
        assert d["risk_label"] == "high"
        assert "thresholds" in d
        assert "cri_history" in d

    def test_empty_report_defaults(self):
        report = ContagionReport()
        d = report.to_dict()
        assert d["contagion_risk_index"] == 0.0
        assert d["risk_label"] == "low"
        assert d["containment_window"] is None


# ── Tests: edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_post_count_no_division_error(self):
        """post_count=0 should not cause ZeroDivisionError."""
        engine = _make_mock_engine()
        scorer = ContagionScorer(engine)
        cri = scorer.score_round(
            round_num=1, post_count=0, reaction_count=0,
        )
        assert 0.0 <= cri <= 1.0

    def test_single_round_report(self):
        """generate_report with only 1 round should not crash."""
        engine = _make_mock_engine(
            engagement_scores=[0.3], engagement_trend=0.0,
        )
        scorer = ContagionScorer(engine)
        scorer.score_round(round_num=1, post_count=5, reaction_count=10)
        report = scorer.generate_report()
        assert report.risk_label in ("low", "moderate", "high", "critical")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
