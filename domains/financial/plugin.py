"""Financial domain plugin — market events, regulation, IPOs, crypto."""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class FinancialDomain(DomainPlugin):
    domain_id = "financial"
    domain_label = "Financial Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Bearish/risk-averse/opposed to the change",
            positive_label="Bullish/risk-seeking/supportive of the change",
            neutral_label="Neutral/wait-and-see",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "central_banker", "analyst", "journalist", "regulator", "hedge_fund_manager",
            ],
            "optional_archetypes": [
                "retail_investor", "ceo", "cfo", "venture_capitalist", "economist",
                "trader", "crypto_influencer",
            ],
            "position_distribution_hint": (
                "Central bankers and regulators lean conservative/risk-averse. "
                "Hedge fund managers and traders span the spectrum. "
                "Include both institutional and retail investor perspectives."
            ),
            "elite_count_range": (8, 12),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(id="trading_desk", description="Trading floor chatter / Bloomberg terminal — terse, jargon-heavy", max_length=280, channel_type="short_form"),
            ChannelConfig(id="analyst_report", description="Research notes / analyst reports — data-driven, in-depth analysis", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="financial_news", description="Financial media / columns — journalistic style, market-focused", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="regulatory_filing", description="Official filings / regulatory statements — formal, legal language", max_length=5000, channel_type="official"),
            ChannelConfig(id="investor_forum", description="Investor discussion forum — analytical with strong opinions", max_length=2000, channel_type="long_form"),
            ChannelConfig(id="fintwit", description="Financial Twitter — short, punchy, market-moving takes", max_length=280, channel_type="short_form"),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "central_banker": ("regulatory_filing", "financial_news"),
            "hedge_fund_manager": ("trading_desk", "fintwit"),
            "retail_investor": ("investor_forum", "fintwit"),
            "analyst": ("analyst_report", "financial_news"),
            "journalist": ("financial_news", "fintwit"),
            "regulator": ("regulatory_filing", "financial_news"),
            "ceo": ("financial_news", "regulatory_filing"),
            "cfo": ("analyst_report", "regulatory_filing"),
            "venture_capitalist": ("fintwit", "investor_forum"),
            "economist": ("analyst_report", "financial_news"),
            "trader": ("trading_desk", "fintwit"),
            "crypto_influencer": ("fintwit", "investor_forum"),
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
        all_positions = [a.position for a in agents] + [c.position for c in clusters]
        if not all_positions:
            return {}

        n = len(all_positions)
        avg = sum(all_positions) / n

        # Market sentiment: proportion bullish vs bearish
        bullish = sum(1 for p in all_positions if p > 0.2) / n
        bearish = sum(1 for p in all_positions if p < -0.2) / n

        # Risk appetite: how far from neutral the average position is (0=risk-off, 1=risk-on)
        risk_appetite = max(0.0, min(1.0, (avg + 1) / 2))

        # FOMO/FUD index: measures clustering at extremes (high = herding behavior)
        extreme_count = sum(1 for p in all_positions if abs(p) > 0.6)
        fomo_fud_index = round(extreme_count / n, 3) if n > 0 else 0.0

        # Institutional confidence from cluster trust data
        trust_vals = [c.trust_institutions for c in clusters if hasattr(c, "trust_institutions")]
        avg_trust = sum(trust_vals) / len(trust_vals) if trust_vals else 0.5

        return {
            "market_sentiment": round(avg, 3),
            "risk_appetite": round(risk_appetite, 3),
            "fomo_fud_index": round(fomo_fud_index, 3),
            "institutional_confidence": round(avg_trust, 3),
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        if avg_position > 0.3:
            return "Bulls"
        elif avg_position < -0.3:
            return "Bears"
        elif avg_position > 0:
            return "Cautious optimists"
        else:
            return "Risk-averse"
