"""Marketing domain plugin — campaign launches, brand crises, influencer strategies."""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class MarketingDomain(DomainPlugin):
    domain_id = "marketing"
    domain_label = "Marketing Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Negative brand perception/rejection",
            positive_label="Positive brand perception/advocacy",
            neutral_label="Brand-indifferent",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "brand_manager", "journalist", "influencer", "consumer_advocate", "competitor",
            ],
            "optional_archetypes": [
                "creative_director", "pr_agency", "media_buyer", "blogger",
                "community_manager", "celebrity", "crisis_manager",
            ],
            "position_distribution_hint": (
                "Brand team is positive. Competitors and consumer advocates lean negative. "
                "Influencers and journalists span the range. Include skeptics."
            ),
            "elite_count_range": (8, 12),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(id="social_media", description="Social media (Instagram/X/TikTok) — short, punchy messages", max_length=280, channel_type="short_form"),
            ChannelConfig(id="blog", description="Blog posts/articles — longer, in-depth analysis and reviews", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="news_media", description="Trade press/news outlets — journalistic coverage and industry analysis", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="brand_channel", description="Official brand communications — polished, on-brand messaging", max_length=2000, channel_type="official"),
            ChannelConfig(id="influencer_content", description="Influencer posts/stories — casual, engaging, personality-driven", max_length=500, channel_type="short_form"),
            ChannelConfig(id="community_forum", description="Community discussions/Reddit — authentic peer conversations and reviews", max_length=2000, channel_type="long_form"),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "brand_manager": ("brand_channel", "social_media"),
            "creative_director": ("brand_channel", "blog"),
            "influencer": ("influencer_content", "social_media"),
            "journalist": ("news_media", "blog"),
            "consumer_advocate": ("community_forum", "blog"),
            "competitor": ("brand_channel", "news_media"),
            "pr_agency": ("news_media", "brand_channel"),
            "media_buyer": ("social_media", "news_media"),
            "blogger": ("blog", "social_media"),
            "community_manager": ("community_forum", "social_media"),
            "celebrity": ("influencer_content", "social_media"),
            "crisis_manager": ("brand_channel", "news_media"),
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

        # Awareness: proportion of actors with strong positions (engaged, not indifferent)
        engaged = sum(1 for p in all_positions if abs(p) > 0.15) / n
        awareness = round(engaged, 3)

        # Engagement rate: average absolute position (higher = more engaged)
        engagement_rate = round(sum(abs(p) for p in all_positions) / n, 3)

        # Virality index: standard deviation of positions (higher spread = more viral debate)
        variance = sum((p - avg) ** 2 for p in all_positions) / n
        virality_index = round(variance ** 0.5, 3)

        # Sentiment shift: average position (positive = pro-brand, negative = anti-brand)
        sentiment_shift = round(avg, 3)

        return {
            "awareness": awareness,
            "engagement_rate": engagement_rate,
            "virality_index": virality_index,
            "sentiment_shift": sentiment_shift,
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        if avg_position > 0.3:
            return "Brand Advocates"
        elif avg_position < -0.3:
            return "Brand Critics"
        elif avg_position > 0:
            return "Neutral Observers"
        else:
            return "Skeptics"
