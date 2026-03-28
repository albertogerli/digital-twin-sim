"""Corporate domain plugin — M&A, restructuring, culture change, leadership transitions."""

from core.config.schema import AxisConfig, ChannelConfig
from domains.base_domain import DomainPlugin
from domains.domain_registry import DomainRegistry
from .prompts import (
    ELITE_ROUND_PROMPT, INSTITUTIONAL_BATCH_PROMPT, CLUSTER_PROMPT,
    EVENT_GENERATION_PROMPT, ELITE_SYSTEM_PROMPT, REPORT_SYSTEM,
    REPORT_PROMPT,
)


@DomainRegistry.register
class CorporateDomain(DomainPlugin):
    domain_id = "corporate"
    domain_label = "Corporate Simulation"

    def get_position_axis(self) -> AxisConfig:
        return AxisConfig(
            negative_label="Resist the change/initiative",
            positive_label="Support the change/initiative",
            neutral_label="Undecided / observing",
        )

    def get_agent_generation_guidance(self) -> dict:
        return {
            "required_archetypes": [
                "ceo", "board_member", "union_rep", "department_head", "journalist",
            ],
            "optional_archetypes": [
                "chro", "consultant", "investor", "middle_manager", "engineer",
                "employee_advocate", "hr_director",
            ],
            "position_distribution_hint": (
                "C-suite typically supports the initiative. Union reps and employee advocates "
                "lean against. Middle managers are split. Include at least one external voice (journalist/investor)."
            ),
            "elite_count_range": (8, 12),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_channels(self) -> list[ChannelConfig]:
        return [
            ChannelConfig(id="internal_memo", description="Official internal memos/announcements — formal, authoritative", max_length=3000, channel_type="official"),
            ChannelConfig(id="slack_channel", description="Slack/Teams messages — short, informal, quick reactions", max_length=500, channel_type="short_form"),
            ChannelConfig(id="town_hall", description="All-hands/town hall presentations — structured, persuasive", max_length=2000, channel_type="long_form"),
            ChannelConfig(id="media_coverage", description="Press/external media — journalistic, public-facing", max_length=3000, channel_type="long_form"),
            ChannelConfig(id="investor_relations", description="Investor communications/filings — formal, data-driven", max_length=5000, channel_type="official"),
            ChannelConfig(id="water_cooler", description="Informal hallway/break-room talk — casual, unfiltered", max_length=280, channel_type="short_form"),
        ]

    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        return {
            "ceo": ("internal_memo", "media_coverage"),
            "chro": ("internal_memo", "town_hall"),
            "board_member": ("investor_relations", "internal_memo"),
            "department_head": ("internal_memo", "slack_channel"),
            "union_rep": ("town_hall", "slack_channel"),
            "employee_advocate": ("slack_channel", "town_hall"),
            "consultant": ("internal_memo", "media_coverage"),
            "journalist": ("media_coverage", "slack_channel"),
            "investor": ("investor_relations", "media_coverage"),
            "middle_manager": ("slack_channel", "internal_memo"),
            "engineer": ("slack_channel", "water_cooler"),
            "hr_director": ("internal_memo", "town_hall"),
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

        supportive = sum(1 for p in all_positions if p > 0.2) / n
        resistant = sum(1 for p in all_positions if p < -0.2) / n
        neutral = 1 - supportive - resistant

        # Trust proxy from cluster data
        trust_vals = [c.trust_institutions for c in clusters if hasattr(c, "trust_institutions")]
        avg_trust = sum(trust_vals) / len(trust_vals) if trust_vals else 0.5

        # Employee sentiment: weighted average skewed toward clusters (the workforce)
        cluster_positions = [c.position for c in clusters]
        employee_sentiment = (
            sum(cluster_positions) / len(cluster_positions)
            if cluster_positions else avg
        )

        # Retention risk: higher when resistance is strong and trust is low
        retention_risk = round(min(1.0, resistant * 1.2 + (1 - avg_trust) * 0.3), 3)

        # Culture alignment: how unified the organization is (inverse of spread)
        spread = sum(abs(p - avg) for p in all_positions) / n
        culture_alignment = round(max(0.0, 1.0 - spread), 3)

        # Resistance index: proportion and intensity of resistors
        resistor_positions = [p for p in all_positions if p < -0.1]
        resistance_index = round(
            (len(resistor_positions) / n) *
            (abs(sum(resistor_positions) / len(resistor_positions)) if resistor_positions else 0),
            3,
        )

        return {
            "employee_sentiment": round(employee_sentiment, 3),
            "retention_risk": retention_risk,
            "culture_alignment": culture_alignment,
            "resistance_index": resistance_index,
            "support_ratio": round(supportive, 3),
            "resistance_ratio": round(resistant, 3),
            "neutral_ratio": round(neutral, 3),
            "avg_position": round(avg, 3),
            "institutional_trust": round(avg_trust, 3),
        }

    def label_coalition(self, avg_position: float, members: list) -> str:
        if avg_position > 0.3:
            return "Change Champions"
        elif avg_position < -0.3:
            return "Change Resistors"
        elif len(members) > 8:
            return "Silent Majority"
        else:
            return "Fence-sitters"
