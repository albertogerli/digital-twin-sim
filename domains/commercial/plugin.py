"""Commercial domain plugin — product launches, pricing changes, competitive moves."""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class CommercialDomain(DomainPlugin):
    domain_id = "commercial"
    domain_label = "Commercial Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Against/reject the product or change",
            positive_label="Support/adopt the product or change",
            neutral_label="Undecided/wait-and-see",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "ceo", "journalist", "analyst", "competitor", "customer_advocate",
            ],
            "optional_archetypes": [
                "cfo", "product_manager", "marketing_director", "influencer",
                "investor", "regulator", "supply_chain",
            ],
            "position_distribution_hint": (
                "Company leadership supports the product/change. Competitors oppose. "
                "Analysts and journalists are split. Consumer advocates reflect real concerns."
            ),
            "elite_count_range": (8, 12),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(id="social_media", description="Social media (X/Twitter, LinkedIn) — short, punchy messages", max_length=280, channel_type="short_form"),
            ChannelConfig(id="review_platform", description="Review sites and forums — detailed consumer reviews and analysis", max_length=2000, channel_type="long_form"),
            ChannelConfig(id="trade_press", description="Industry publications — professional, analytical trade coverage", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="corporate_comms", description="Corporate communications — formal press releases, blog posts", max_length=5000, channel_type="official"),
            ChannelConfig(id="investor_call", description="Investor updates — earnings calls, shareholder letters, financial focus", max_length=2000, channel_type="official"),
            ChannelConfig(id="word_of_mouth", description="Informal channels — word-of-mouth, water cooler talk, community chatter", max_length=500, channel_type="short_form"),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "ceo": ("corporate_comms", "social_media"),
            "cfo": ("investor_call", "corporate_comms"),
            "product_manager": ("corporate_comms", "social_media"),
            "marketing_director": ("social_media", "corporate_comms"),
            "competitor": ("trade_press", "social_media"),
            "analyst": ("trade_press", "investor_call"),
            "journalist": ("trade_press", "social_media"),
            "influencer": ("social_media", "review_platform"),
            "investor": ("investor_call", "trade_press"),
            "customer_advocate": ("review_platform", "social_media"),
            "regulator": ("corporate_comms", "trade_press"),
            "supply_chain": ("trade_press", "corporate_comms"),
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

        # Adoption rate: fraction of actors with position > 0.2 (leaning toward adoption)
        adopters = sum(1 for p in all_positions if p > 0.2) / n
        # Opponents: fraction with position < -0.2
        opponents = sum(1 for p in all_positions if p < -0.2) / n
        # Wait-and-see: the rest
        undecided = 1 - adopters - opponents

        # Brand sentiment: weighted average shifted to 0-1 scale
        brand_sentiment = (avg + 1) / 2  # maps [-1,+1] to [0,1]

        # NPS proxy: promoters (>0.5) minus detractors (<-0.2), scaled to -100..+100
        promoters = sum(1 for p in all_positions if p > 0.5) / n
        detractors = sum(1 for p in all_positions if p < -0.2) / n
        nps_proxy = round((promoters - detractors) * 100, 1)

        # Market share shift: average position as proxy for directional movement
        market_share_shift = round(avg, 3)

        return {
            "adoption_rate": round(adopters, 3),
            "brand_sentiment": round(brand_sentiment, 3),
            "nps_proxy": nps_proxy,
            "market_share_shift": market_share_shift,
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        if avg_position > 0.3:
            return "Adopters"
        elif avg_position < -0.3:
            return "Opponents"
        elif avg_position > 0:
            return "Wait-and-see"
        else:
            return "Skeptics"
