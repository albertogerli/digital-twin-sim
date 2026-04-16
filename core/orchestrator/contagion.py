"""Contagion Risk Scorer — measures crisis contagion and predicts escalation.

Designed for client-facing pitch:
  "If you cut 100 jobs, the contagion risk is 0.3 — local media only.
   If you cut 150, the risk jumps to 0.7 — Landini activates, PM responds."

The scorer computes three risk dimensions:
  1. Virality Risk — how likely is content to escape the initial audience?
  2. Institutional Contagion — will institutions (unions, parties) amplify?
  3. Cross-Domain Spillover — will the crisis bleed into other sectors?

These combine into a single Contagion Risk Index (CRI) on 0-1 scale,
plus a prediction of which escalation thresholds will be crossed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from core.orchestrator.escalation import EscalationEngine, EscalationState

logger = logging.getLogger(__name__)


@dataclass
class ContagionMetrics:
    """Snapshot of contagion indicators for a single round."""
    round_num: int

    # Virality
    repost_rate: float = 0.0       # avg reposts per original post
    top_post_reach: int = 0        # engagement on highest-performing post
    hashtag_convergence: float = 0.0  # 0-1: are agents using same hashtags?

    # Institutional
    institutional_response_count: int = 0  # how many institutions reacted
    union_activation: bool = False
    party_activation: bool = False
    media_amplification: float = 0.0  # mainstream media engagement ratio

    # Cross-domain
    sectors_affected: int = 1      # how many sectors are now discussing this
    geographic_spread: int = 1     # how many regions are now involved
    international_attention: bool = False


@dataclass
class ContagionThreshold:
    """A predicted threshold where crisis behavior changes."""
    name: str
    description: str
    engagement_level: float   # engagement_score needed to cross
    probability: float        # current probability of crossing (0-1)
    agents_activated: list[str] = field(default_factory=list)
    impact_description: str = ""


@dataclass
class ContagionReport:
    """Client-facing contagion risk assessment."""
    # Overall index
    contagion_risk_index: float = 0.0  # 0-1 composite
    risk_label: str = "low"            # low, moderate, high, critical

    # Sub-dimensions
    virality_risk: float = 0.0
    institutional_contagion: float = 0.0
    cross_domain_spillover: float = 0.0

    # Time series
    cri_history: list[float] = field(default_factory=list)

    # Thresholds
    thresholds: list[ContagionThreshold] = field(default_factory=list)

    # Narrative
    executive_summary: str = ""
    key_risk_factors: list[str] = field(default_factory=list)
    containment_window: Optional[int] = None  # rounds before escalation

    def to_dict(self) -> dict:
        return {
            "contagion_risk_index": round(self.contagion_risk_index, 3),
            "risk_label": self.risk_label,
            "virality_risk": round(self.virality_risk, 3),
            "institutional_contagion": round(self.institutional_contagion, 3),
            "cross_domain_spillover": round(self.cross_domain_spillover, 3),
            "cri_history": [round(c, 3) for c in self.cri_history],
            "thresholds": [
                {
                    "name": t.name,
                    "description": t.description,
                    "engagement_level": round(t.engagement_level, 3),
                    "probability": round(t.probability, 3),
                    "agents_activated": t.agents_activated,
                    "impact_description": t.impact_description,
                }
                for t in self.thresholds
            ],
            "executive_summary": self.executive_summary,
            "key_risk_factors": self.key_risk_factors,
            "containment_window": self.containment_window,
        }


class ContagionScorer:
    """Computes contagion risk and generates client-facing reports.

    Uses data from the EscalationEngine (engagement scores, round metrics)
    plus additional contagion-specific indicators to produce:
    - Per-round Contagion Risk Index (CRI)
    - Threshold predictions
    - Executive summaries for crisis consultants
    """

    # CRI sub-dimension weights
    W_VIRALITY = 0.35
    W_INSTITUTIONAL = 0.40
    W_SPILLOVER = 0.25

    # Threshold definitions
    THRESHOLDS = [
        {
            "name": "local_containment",
            "description": "Crisis remains in local media and sector-specific actors",
            "engagement_level": 0.0,
            "wave": 1,
        },
        {
            "name": "national_attention",
            "description": "National media picks up the story, union leaders react publicly",
            "engagement_level": 0.45,
            "wave": 2,
        },
        {
            "name": "political_escalation",
            "description": "Party leaders and ministers forced to take positions",
            "engagement_level": 0.65,
            "wave": 2,
        },
        {
            "name": "institutional_crisis",
            "description": "PM/President comments, parliamentary questions, potential policy response",
            "engagement_level": 0.80,
            "wave": 3,
        },
        {
            "name": "systemic_contagion",
            "description": "Crisis spreads to adjacent sectors, international attention, market impact",
            "engagement_level": 0.92,
            "wave": 3,
        },
    ]

    def __init__(self, escalation_engine: EscalationEngine):
        self.engine = escalation_engine
        self.contagion_history: list[ContagionMetrics] = []
        self.cri_history: list[float] = []

    def score_round(
        self,
        round_num: int,
        post_count: int,
        reaction_count: int,
        repost_count: int = 0,
        top_post_engagement: int = 0,
        institutional_actors_active: int = 0,
        union_activated: bool = False,
        party_activated: bool = False,
        sectors_affected: int = 1,
        geographic_regions: int = 1,
        international_attention: bool = False,
        hashtag_convergence: float = 0.0,
    ) -> float:
        """Score contagion risk for a completed round.

        Returns the Contagion Risk Index (0-1).
        """
        metrics = ContagionMetrics(
            round_num=round_num,
            repost_rate=repost_count / max(post_count, 1),
            top_post_reach=top_post_engagement,
            hashtag_convergence=hashtag_convergence,
            institutional_response_count=institutional_actors_active,
            union_activation=union_activated,
            party_activation=party_activated,
            media_amplification=reaction_count / max(post_count, 1),
            sectors_affected=sectors_affected,
            geographic_spread=geographic_regions,
            international_attention=international_attention,
        )
        self.contagion_history.append(metrics)

        # Compute sub-dimensions
        virality = self._compute_virality(metrics)
        institutional = self._compute_institutional(metrics)
        spillover = self._compute_spillover(metrics)

        # Weighted CRI
        cri = (
            self.W_VIRALITY * virality +
            self.W_INSTITUTIONAL * institutional +
            self.W_SPILLOVER * spillover
        )

        # Momentum: CRI can't drop more than 0.15 per round (crises have inertia)
        if self.cri_history:
            prev_cri = self.cri_history[-1]
            cri = max(cri, prev_cri - 0.15)

        cri = max(0.0, min(1.0, cri))
        self.cri_history.append(cri)

        logger.info(
            f"CRI round {round_num}: {cri:.3f} "
            f"(V={virality:.2f} I={institutional:.2f} S={spillover:.2f})"
        )
        return cri

    def generate_report(self) -> ContagionReport:
        """Generate a full contagion risk report from accumulated data.

        Call after scoring all rounds (or mid-simulation for live updates).
        """
        if not self.cri_history:
            return ContagionReport()

        current_cri = self.cri_history[-1]
        engagement = self.engine.state.latest_engagement

        # Risk label
        if current_cri < 0.25:
            label = "low"
        elif current_cri < 0.50:
            label = "moderate"
        elif current_cri < 0.75:
            label = "high"
        else:
            label = "critical"

        # Compute sub-dimensions from latest metrics
        latest = self.contagion_history[-1] if self.contagion_history else None
        virality = self._compute_virality(latest) if latest else 0.0
        institutional = self._compute_institutional(latest) if latest else 0.0
        spillover = self._compute_spillover(latest) if latest else 0.0

        # Build threshold predictions
        thresholds = self._predict_thresholds(engagement)

        # Key risk factors
        risk_factors = self._identify_risk_factors()

        # Containment window
        containment = self._estimate_containment_window()

        # Executive summary
        summary = self._build_executive_summary(
            current_cri, label, risk_factors, containment
        )

        return ContagionReport(
            contagion_risk_index=current_cri,
            risk_label=label,
            virality_risk=virality,
            institutional_contagion=institutional,
            cross_domain_spillover=spillover,
            cri_history=list(self.cri_history),
            thresholds=thresholds,
            executive_summary=summary,
            key_risk_factors=risk_factors,
            containment_window=containment,
        )

    def _compute_virality(self, m: ContagionMetrics) -> float:
        """Virality risk: how likely is content to escape initial audience."""
        # Repost rate: > 2.0 reposts/post is high virality
        repost_score = min(1.0, m.repost_rate / 3.0)

        # Top post reach: normalize against baseline
        reach_score = min(1.0, m.top_post_reach / 500)

        # Hashtag convergence: agents converging on same narrative frame
        convergence_score = m.hashtag_convergence

        # Media amplification
        amp_score = min(1.0, m.media_amplification / 10.0)

        return (
            0.30 * repost_score +
            0.25 * reach_score +
            0.25 * convergence_score +
            0.20 * amp_score
        )

    def _compute_institutional(self, m: ContagionMetrics) -> float:
        """Institutional contagion: are formal institutions amplifying?"""
        # Number of institutional actors that have reacted
        inst_score = min(1.0, m.institutional_response_count / 8)

        # Binary triggers with weights
        union_score = 0.8 if m.union_activation else 0.0
        party_score = 0.6 if m.party_activation else 0.0

        # Media amplification from institutional channels
        media_score = min(1.0, m.media_amplification / 8.0)

        return (
            0.30 * inst_score +
            0.30 * union_score +
            0.20 * party_score +
            0.20 * media_score
        )

    def _compute_spillover(self, m: ContagionMetrics) -> float:
        """Cross-domain spillover: is the crisis bleeding into other areas?"""
        # Sectors: 1=contained, 3+=spillover
        sector_score = min(1.0, (m.sectors_affected - 1) / 3.0)

        # Geographic spread: 1=local, 5+=national
        geo_score = min(1.0, (m.geographic_spread - 1) / 4.0)

        # International attention is a major escalation
        intl_score = 1.0 if m.international_attention else 0.0

        return (
            0.40 * sector_score +
            0.35 * geo_score +
            0.25 * intl_score
        )

    def _predict_thresholds(self, current_engagement: float) -> list[ContagionThreshold]:
        """Predict probability of crossing each escalation threshold."""
        thresholds = []
        trend = self.engine.state.engagement_trend

        for t_def in self.THRESHOLDS:
            level = t_def["engagement_level"]

            if current_engagement >= level:
                prob = 1.0  # Already crossed
            elif trend <= 0:
                # Cooling down — low probability of crossing higher thresholds
                gap = level - current_engagement
                prob = max(0.0, 0.2 * math.exp(-3 * min(gap, 100.0)))
            else:
                # Rising — extrapolate
                gap = level - current_engagement
                rounds_to_cross = gap / trend if trend > 0 else float("inf")
                # Probability decays with distance (sigmoid-like)
                exp_arg = 0.8 * (rounds_to_cross - 3)
                prob = 1.0 / (1.0 + math.exp(min(exp_arg, 500.0)))

            # Get agents in this wave from activation plan
            wave = t_def["wave"]
            wave_agents = []
            if wave == 1:
                wave_agents = [s.stakeholder_id for s in self.engine.plan.wave_1[:5]]
            elif wave == 2:
                wave_agents = [s.stakeholder_id for s in self.engine.plan.wave_2[:5]]
            elif wave == 3:
                wave_agents = [s.stakeholder_id for s in self.engine.plan.wave_3[:5]]

            thresholds.append(ContagionThreshold(
                name=t_def["name"],
                description=t_def["description"],
                engagement_level=level,
                probability=prob,
                agents_activated=wave_agents,
                impact_description=self._threshold_impact(t_def["name"]),
            ))

        return thresholds

    def _threshold_impact(self, threshold_name: str) -> str:
        """Describe the impact of crossing a threshold."""
        impacts = {
            "local_containment": "Crisis covered only by local outlets and sector actors. Minimal reputational damage.",
            "national_attention": "Major newspapers and TV channels cover the story. Union leaders make public statements. Social media trending.",
            "political_escalation": "Opposition parties use the crisis politically. Ministers forced to comment. Parliamentary questions possible.",
            "institutional_crisis": "Prime Minister addresses the issue. Government may announce emergency measures. Market impact likely.",
            "systemic_contagion": "Crisis extends beyond the original sector. International media coverage. Regulatory investigations. Potential policy overhaul.",
        }
        return impacts.get(threshold_name, "")

    def _identify_risk_factors(self) -> list[str]:
        """Identify the top risk factors driving contagion."""
        factors = []

        if not self.contagion_history:
            return factors

        latest = self.contagion_history[-1]
        state = self.engine.state

        # Check each dimension
        if latest.repost_rate > 2.0:
            factors.append(f"High content virality (repost rate: {latest.repost_rate:.1f}x)")

        if latest.union_activation:
            factors.append("Union leaders activated — amplifying worker grievances")

        if latest.party_activation:
            factors.append("Political parties engaged — crisis becoming partisan")

        if latest.sectors_affected > 2:
            factors.append(f"Cross-sector spillover ({latest.sectors_affected} sectors affected)")

        if latest.geographic_spread > 3:
            factors.append(f"Geographic spread ({latest.geographic_spread} regions involved)")

        if latest.international_attention:
            factors.append("International media attention")

        if state.engagement_trend > 0.1:
            factors.append(f"Accelerating engagement (trend: +{state.engagement_trend:.2f}/round)")

        if len(state.engagement_scores) >= 3:
            # Check for sustained high engagement
            recent = state.engagement_scores[-3:]
            if all(s > 0.5 for s in recent):
                factors.append("Sustained high engagement (3+ rounds above 0.50)")

        # Polarization spike
        if state.round_metrics and state.round_metrics[-1].polarization > 6.0:
            factors.append(f"High polarization ({state.round_metrics[-1].polarization:.1f}/10)")

        return factors[:6]  # Cap at 6 key factors

    def _estimate_containment_window(self) -> Optional[int]:
        """Estimate how many rounds before the crisis becomes uncontainable.

        Returns None if already uncontainable or if cooling down.
        """
        if not self.cri_history:
            return None

        current = self.cri_history[-1]
        if current >= 0.75:
            return 0  # Already critical

        trend = self.engine.state.engagement_trend
        if trend <= 0:
            return None  # Cooling down, no urgency

        # Rounds until CRI hits 0.75 (critical threshold)
        gap = 0.75 - current
        if gap <= 0:
            return 0

        # Conservative estimate using engagement trend as proxy
        rounds = math.ceil(gap / (trend * 0.8))  # 0.8 dampening factor
        return max(1, rounds)

    def _build_executive_summary(
        self,
        cri: float,
        label: str,
        risk_factors: list[str],
        containment: Optional[int],
    ) -> str:
        """Build a one-paragraph executive summary."""
        # CRI description
        if label == "low":
            outlook = "The crisis is contained within its initial scope."
            action = "Standard monitoring is sufficient."
        elif label == "moderate":
            outlook = "The crisis is gaining traction beyond local actors."
            if containment and containment > 0:
                action = f"Proactive communication within {containment} rounds can prevent escalation."
            else:
                action = "Proactive stakeholder engagement is recommended."
        elif label == "high":
            outlook = "The crisis has reached national attention with institutional amplification."
            action = "Immediate crisis communication strategy required. Key stakeholder briefing recommended."
        else:
            outlook = "The crisis has reached systemic proportions with cross-sector contagion."
            action = "Emergency response protocol. Direct leadership engagement required."

        # Build narrative
        parts = [
            f"Contagion Risk Index: {cri:.2f} ({label.upper()}).",
            outlook,
        ]

        if risk_factors:
            top_factors = "; ".join(risk_factors[:3])
            parts.append(f"Key drivers: {top_factors}.")

        parts.append(action)

        return " ".join(parts)

    def compare_scenarios(
        self,
        other: "ContagionScorer",
        scenario_a_name: str = "Scenario A",
        scenario_b_name: str = "Scenario B",
    ) -> dict:
        """Compare contagion risk between two scenario runs.

        Useful for "What happens if we cut 100 vs 150 jobs?" analysis.
        """
        report_a = self.generate_report()
        report_b = other.generate_report()

        return {
            scenario_a_name: {
                "cri": report_a.contagion_risk_index,
                "label": report_a.risk_label,
                "peak_cri": max(report_a.cri_history) if report_a.cri_history else 0,
                "thresholds_crossed": sum(
                    1 for t in report_a.thresholds if t.probability >= 0.95
                ),
            },
            scenario_b_name: {
                "cri": report_b.contagion_risk_index,
                "label": report_b.risk_label,
                "peak_cri": max(report_b.cri_history) if report_b.cri_history else 0,
                "thresholds_crossed": sum(
                    1 for t in report_b.thresholds if t.probability >= 0.95
                ),
            },
            "delta": {
                "cri_difference": round(
                    report_b.contagion_risk_index - report_a.contagion_risk_index, 3
                ),
                "risk_escalation": report_b.risk_label != report_a.risk_label,
                "additional_thresholds": sum(
                    1 for t in report_b.thresholds if t.probability >= 0.95
                ) - sum(
                    1 for t in report_a.thresholds if t.probability >= 0.95
                ),
            },
        }
