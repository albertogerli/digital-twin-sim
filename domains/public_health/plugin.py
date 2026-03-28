"""Public health domain plugin — vaccination campaigns, health policy, pandemic response."""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class PublicHealthDomain(DomainPlugin):
    domain_id = "public_health"
    domain_label = "Public Health Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Opposes the health measure/policy",
            positive_label="Supports the health measure/policy",
            neutral_label="Undecided / hesitant",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "health_minister", "epidemiologist", "journalist", "doctor", "patient_advocate",
            ],
            "optional_archetypes": [
                "pharma_exec", "influencer", "activist", "religious_leader",
                "researcher", "nurse", "public_health_officer",
            ],
            "position_distribution_hint": (
                "Medical professionals generally support evidence-based measures. "
                "Activists and influencers span the range. Religious leaders vary. "
                "Include vaccine-hesitant and anti-measure voices for realism."
            ),
            "elite_count_range": (8, 12),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(id="social_media", description="Social media (X/Twitter-like) — short, punchy messages", max_length=280, channel_type="short_form"),
            ChannelConfig(id="health_forum", description="Health discussion forum — longer, analytical content, personal experiences", max_length=2000, channel_type="long_form"),
            ChannelConfig(id="news_media", description="News outlets/editorials — journalistic style, in-depth analysis", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="official_bulletin", description="Official government/institutional bulletin — formal, technical language", max_length=5000, channel_type="official"),
            ChannelConfig(id="medical_journal", description="Medical/scientific publication — evidence-based, peer-reviewed style", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="community_chat", description="Community groups/neighborhood chats — direct, colloquial, personal", max_length=500, channel_type="short_form"),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "health_minister": ("official_bulletin", "news_media"),
            "epidemiologist": ("medical_journal", "news_media"),
            "doctor": ("health_forum", "social_media"),
            "pharma_exec": ("official_bulletin", "news_media"),
            "journalist": ("news_media", "social_media"),
            "influencer": ("social_media", "community_chat"),
            "activist": ("social_media", "community_chat"),
            "patient_advocate": ("health_forum", "social_media"),
            "religious_leader": ("community_chat", "social_media"),
            "researcher": ("medical_journal", "health_forum"),
            "nurse": ("health_forum", "community_chat"),
            "public_health_officer": ("official_bulletin", "news_media"),
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

        # Compliance rate: fraction supporting the health measure (position > 0.2)
        compliance_rate = sum(1 for p in all_positions if p > 0.2) / n

        # Institutional trust proxy from cluster data
        trust_vals = [c.trust_institutions for c in clusters if hasattr(c, "trust_institutions")]
        institutional_trust = sum(trust_vals) / len(trust_vals) if trust_vals else 0.5

        # Misinformation spread: proxy from extreme anti-measure positions
        # Agents with very negative positions (< -0.5) are likely amplifying misinformation
        extreme_anti = sum(1 for p in all_positions if p < -0.5)
        misinfo_spread = round(extreme_anti / n, 3)

        # Behavioral intent: weighted average shifted to 0-1 scale
        # -1 → 0 (full non-compliance), +1 → 1 (full compliance)
        behavioral_intent = round((avg + 1) / 2, 3)

        return {
            "compliance_rate": round(compliance_rate, 3),
            "institutional_trust": round(institutional_trust, 3),
            "misinfo_spread": misinfo_spread,
            "behavioral_intent": behavioral_intent,
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        if avg_position > 0.3:
            return "Pro-measure"
        elif avg_position < -0.3:
            return "Anti-measure"
        elif abs(avg_position) <= 0.15:
            return "Hesitant"
        else:
            return "Science-aligned moderates"
