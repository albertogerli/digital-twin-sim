"""Political domain plugin — elections, referendums, policy changes."""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class PoliticalDomain(DomainPlugin):
    domain_id = "political"
    domain_label = "Political Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Against the policy/reform",
            positive_label="In favor of the policy/reform",
            neutral_label="Neutral / undecided",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "politician", "journalist", "union_leader", "business_leader", "magistrate",
            ],
            "optional_archetypes": [
                "activist", "academic", "influencer", "religious_leader", "lawyer", "bureaucrat",
            ],
            "position_distribution_hint": (
                "At least 2 agents per quadrant. Include both government and opposition politicians. "
                "Journalists should lean neutral. Union and business leaders typically oppose each other."
            ),
            "elite_count_range": (8, 14),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(id="social", description="Social media (X/Twitter-like) — short, punchy messages", max_length=280, channel_type="short_form"),
            ChannelConfig(id="forum", description="Discussion forum — longer, analytical content", max_length=2000, channel_type="long_form"),
            ChannelConfig(id="press", description="Newspaper/editorial — journalistic style, in-depth analysis", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="tv", description="TV statement/talk show — emotional impact, short phrases", max_length=1000, channel_type="short_form"),
            ChannelConfig(id="official", description="Official press release — formal, technical language", max_length=5000, channel_type="official"),
            ChannelConfig(id="street", description="Street talk/rallies — direct, colloquial language", max_length=500, channel_type="short_form"),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "politician": ("social", "tv"),
            "judge": ("official", "press"),
            "magistrate": ("official", "press"),
            "academic": ("press", "forum"),
            "journalist": ("press", "tv"),
            "media": ("tv", "social"),
            "business_leader": ("press", "forum"),
            "union_leader": ("street", "social"),
            "religious_leader": ("official", "press"),
            "influencer": ("social", "forum"),
            "lawyer": ("official", "press"),
            "activist": ("social", "street"),
            "bureaucrat": ("official", "press"),
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
        pro = sum(1 for p in all_positions if p > 0.2) / n
        against = sum(1 for p in all_positions if p < -0.2) / n
        neutral = 1 - pro - against

        # Trust proxy from cluster data
        trust_vals = [c.trust_institutions for c in clusters if hasattr(c, "trust_institutions")]
        avg_trust = sum(trust_vals) / len(trust_vals) if trust_vals else 0.5

        return {
            "voting_intention_pro": round(pro, 3),
            "voting_intention_against": round(against, 3),
            "voting_intention_neutral": round(neutral, 3),
            "avg_position": round(avg, 3),
            "institutional_trust": round(avg_trust, 3),
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        if avg_position > 0.3:
            return "Reform Supporters"
        elif avg_position < -0.3:
            return "Reform Opponents"
        elif avg_position > 0:
            return "Moderate Supporters"
        else:
            return "Moderate Opponents"
