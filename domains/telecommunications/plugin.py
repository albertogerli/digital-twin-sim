"""Telecommunications domain plugin.

Frames simulations around telco-regulatory and industrial dynamics:
M&A consolidation operations, AGCom/AGCM/DG Comp rulings, spectrum
auctions, gigabit infrastructure act, ARPU/churn pressure, state-aid
review on FTTH wholesale subsidies.

Position axis intentionally NOT bullish/bearish (that's financial). Here
the axis is structural: pro-consolidation/incumbent ↔ pro-competition/
deregulation. This matches how telco-regulatory dossiers actually
polarise stakeholders (KKR/CDP/Confindustria push consolidation, DG Comp/
ARPCEPT/consumer groups push competition, sindacati push labor protection
orthogonally).
"""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class TelecommunicationsDomain(DomainPlugin):
    domain_id = "telecommunications"
    domain_label = "Telecom Regulatory / M&A Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Pro-competition / pro-deregulation / anti-consolidation",
            positive_label="Pro-consolidation / pro-incumbent / pro-strategic-protection",
            neutral_label="Technocratic / case-by-case",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "telecom_ceo", "regulator", "analyst",
                "union_leader", "eu_commissioner", "journalist",
            ],
            "optional_archetypes": [
                "policy_expert", "infrastructure_investor", "consumer_advocate",
                "telecom_engineer", "industry_lobbyist", "geopolitics_expert",
                "ceo", "politician",
            ],
            "position_distribution_hint": (
                "Telco CEOs and infrastructure investors (KKR, CDP) tend to lean "
                "pro-consolidation. EU DG Comp and consumer advocates lean "
                "pro-competition. Sindacati position orthogonally on labor protection. "
                "EU sovereignty commissioners (e.g. tech sovereignty VP) often "
                "argue pro-consolidation as response to US/China competition. "
                "Analysts and journalists span the axis with editorial bias."
            ),
            "elite_count_range": (8, 12),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (4, 7),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(
                id="regulatory_filing",
                description="AGCom/AGCM/DG Comp/BEREC/MIMIT official rulings — formal, legal language, citation-heavy",
                max_length=5000,
                channel_type="official",
            ),
            ChannelConfig(
                id="analyst_note",
                description="Mediobanca/Equita/Citi TLC research notes — data-driven, multiples + DCF, target prices",
                max_length=3000,
                channel_type="long_form",
            ),
            ChannelConfig(
                id="telecom_press",
                description="Sole 24 Ore TLC, MF, Reuters Telco, TelecomLive — journalistic, sector-focused",
                max_length=3000,
                channel_type="long_form",
            ),
            ChannelConfig(
                id="parliament",
                description="Camera/Senato + EU TRAN-ITRE audizioni — institutional speech, policy-framed",
                max_length=5000,
                channel_type="official",
            ),
            ChannelConfig(
                id="industry_forum",
                description="ASSTEL, Confindustria Digitale, Anitec-Assinform — B2B lobby positions",
                max_length=2000,
                channel_type="long_form",
            ),
            ChannelConfig(
                id="fintwit",
                description="Financial Twitter — short, punchy, market-moving telco takes",
                max_length=280,
                channel_type="short_form",
            ),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "telecom_ceo": ("telecom_press", "regulatory_filing"),
            "regulator": ("regulatory_filing", "telecom_press"),
            "analyst": ("analyst_note", "fintwit"),
            "union_leader": ("telecom_press", "industry_forum"),
            "eu_commissioner": ("regulatory_filing", "telecom_press"),
            "journalist": ("telecom_press", "fintwit"),
            "policy_expert": ("telecom_press", "fintwit"),
            "infrastructure_investor": ("analyst_note", "regulatory_filing"),
            "consumer_advocate": ("telecom_press", "industry_forum"),
            "telecom_engineer": ("industry_forum", "telecom_press"),
            "industry_lobbyist": ("industry_forum", "regulatory_filing"),
            "geopolitics_expert": ("telecom_press", "fintwit"),
            "ceo": ("telecom_press", "regulatory_filing"),
            "politician": ("parliament", "telecom_press"),
            "expert_commentator": ("telecom_press", "fintwit"),
            "lobbyist": ("industry_forum", "telecom_press"),
        }

    def get_channel_max_lengths(self) -> dict[str, int]:
        return {ch.id: ch.max_length for ch in self.get_channels()}

    def get_elite_prompt_template(self) -> str:
        return ELITE_ROUND_PROMPT

    def get_institutional_batch_prompt_template(self) -> str:
        return INSTITUTIONAL_BATCH_PROMPT

    def get_cluster_prompt_template(self) -> str:
        return CLUSTER_PROMPT

    def get_event_generation_prompt_template(self) -> str:
        return EVENT_GENERATION_PROMPT

    def get_elite_system_prompt_template(self) -> str:
        return ELITE_SYSTEM_PROMPT

    def get_report_system_prompt(self) -> str:
        return REPORT_SYSTEM

    def get_report_prompt_template(self) -> str:
        return REPORT_PROMPT

    def compute_domain_metrics(self, agents, clusters, platform, round_num) -> dict:
        """Telco-specific scoring on top of the generic position aggregate.

        Returns a dict that the frontend renders as custom_metrics chips.
        All values are 0-1 normalised (the frontend handles formatting).
        """
        all_positions = [a.position for a in agents] + [c.position for c in clusters]
        if not all_positions:
            return {}

        n = len(all_positions)
        avg = sum(all_positions) / n

        # Consolidation pressure: share of agents with pos > 0.2
        consolidation = sum(1 for p in all_positions if p > 0.2) / n
        # Competition pressure: share with pos < -0.2
        competition = sum(1 for p in all_positions if p < -0.2) / n

        # Regulatory burden index: weighted sum of regulator + EU commissioner
        # positive shift (positive = stricter, leaning toward intervention)
        regulator_shift = 0.0
        regulator_count = 0
        for a in agents:
            arch = getattr(a, "archetype", "")
            if arch in ("regulator", "eu_commissioner"):
                regulator_shift += abs(a.position)
                regulator_count += 1
        regulatory_burden = (regulator_shift / regulator_count) if regulator_count else 0.0

        # Labor friction: union leader negativity (negative pos = anti-deal in
        # the merger frame; for a generic telco brief, |pos| signals friction)
        union_friction = 0.0
        union_count = 0
        for a in agents:
            if getattr(a, "archetype", "") == "union_leader":
                union_friction += max(0.0, -a.position)
                union_count += 1
        labor_friction = (union_friction / union_count) if union_count else 0.0

        # Institutional confidence from cluster trust data (if available)
        trust_vals = [getattr(c, "trust_institutions", None) for c in clusters]
        trust_vals = [v for v in trust_vals if v is not None]
        institutional_confidence = sum(trust_vals) / len(trust_vals) if trust_vals else 0.5

        return {
            "industry_stance": round(avg, 3),
            "consolidation_pressure": round(consolidation, 3),
            "competition_pressure": round(competition, 3),
            "regulatory_burden": round(regulatory_burden, 3),
            "labor_friction": round(labor_friction, 3),
            "institutional_confidence": round(institutional_confidence, 3),
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        """Map average position + member archetypes to a telco-flavored label."""
        archetypes = [getattr(m, "archetype", "") for m in members]
        # Union-led coalition (sindacale) override
        if archetypes.count("union_leader") >= max(1, len(members) // 4):
            return "Blocco lavoratori (sindacale)"
        # EU-led sovereignty cluster
        if archetypes.count("eu_commissioner") + archetypes.count("geopolitics_expert") >= 2:
            return "Sovranità tech (UE / geopolitica)"
        # Position-based
        if avg_position > 0.35:
            return "Pro-consolidamento (incumbent + investitori)"
        if avg_position > 0.10:
            return "Pro-consolidamento moderato"
        if avg_position > -0.10:
            return "Tecnocratici / case-by-case"
        if avg_position > -0.35:
            return "Pro-competizione moderato"
        return "Pro-competizione (DG Comp + consumatori)"
