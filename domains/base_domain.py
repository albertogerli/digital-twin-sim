"""Abstract domain plugin interface — all domains implement this."""

import json
import os
from abc import ABC, abstractmethod
from core.config.schema import AxisConfig, ChannelConfig


class DomainPlugin(ABC):
    """Base class for domain-specific simulation plugins."""

    domain_id: str = ""
    domain_label: str = ""

    @abstractmethod
    def get_position_axis(self) -> AxisConfig:
        """What the -1 to +1 scale means in this domain."""
        ...

    @abstractmethod
    def get_channels(self) -> list[ChannelConfig]:
        """Available communication channels for this domain."""
        ...

    @abstractmethod
    def get_archetype_channel_map(self) -> dict[str, tuple[str, str]]:
        """Maps agent archetype to (primary_channel, secondary_channel)."""
        ...

    @abstractmethod
    def get_channel_max_lengths(self) -> dict[str, int]:
        """Max post length per channel."""
        ...

    @abstractmethod
    def get_elite_prompt_template(self) -> str:
        """Prompt template for elite agent round generation."""
        ...

    @abstractmethod
    def get_institutional_batch_prompt_template(self) -> str:
        """Prompt template for institutional batch processing."""
        ...

    @abstractmethod
    def get_cluster_prompt_template(self) -> str:
        """Prompt template for citizen cluster round generation."""
        ...

    @abstractmethod
    def get_event_generation_prompt_template(self) -> str:
        """Prompt template for LLM-generated emergent events."""
        ...

    @abstractmethod
    def get_elite_system_prompt_template(self) -> str:
        """System prompt template for building elite agent personas."""
        ...

    @abstractmethod
    def get_report_system_prompt(self) -> str:
        """System prompt for report generation."""
        ...

    @abstractmethod
    def get_report_prompt_template(self) -> str:
        """Prompt template for generating the final report."""
        ...

    @abstractmethod
    def compute_domain_metrics(self, agents: list, clusters: list,
                                platform, round_num: int) -> dict:
        """Compute domain-specific metrics beyond basic polarization."""
        ...

    @abstractmethod
    def label_coalition(self, avg_position: float, members: list) -> str:
        """Generate a coalition label based on average position."""
        ...

    def get_position_descriptions(self) -> dict[str, str]:
        """Map position ranges to human-readable labels."""
        axis = self.get_position_axis()
        return {
            "strongly_positive": f"Strongly {axis.positive_label}",
            "moderately_positive": f"Moderately {axis.positive_label}",
            "neutral": axis.neutral_label,
            "moderately_negative": f"Moderately {axis.negative_label}",
            "strongly_negative": f"Strongly {axis.negative_label}",
        }

    def describe_position(self, position: float) -> str:
        """Convert numeric position to text description."""
        pd = self.get_position_descriptions()
        if position > 0.5:
            return pd["strongly_positive"]
        elif position > 0.2:
            return pd["moderately_positive"]
        elif position > -0.2:
            return pd["neutral"]
        elif position > -0.5:
            return pd["moderately_negative"]
        else:
            return pd["strongly_negative"]

    def get_agent_generation_guidance(self) -> dict:
        """Provide archetype checklist and position distribution hints for agent generation.

        Override in domain plugins to specify which agent roles are required/optional
        and how positions should be distributed across the axis.
        """
        return {
            "required_archetypes": [],
            "optional_archetypes": [],
            "position_distribution_hint": "Ensure agents span the full -1 to +1 range with at least 1 agent per quadrant.",
            "elite_count_range": (8, 14),
            "institutional_count_range": (6, 10),
            "cluster_count_range": (5, 8),
        }

    def get_fallback_strings(self) -> dict[str, str]:
        """Fallback strings for missing data."""
        return {
            "no_history": "No previous history.",
            "no_event": "Event not yet generated.",
            "no_coalition": "No coalition information available.",
            "no_viral": "No viral posts available.",
            "fallback_event": (
                "The dynamics in {timeline_label} continue along the trajectory "
                "of previous events. Tensions remain with polarization at {polarization:.1f}/10."
            ),
            "default_event": "The dynamics from the previous period continue.",
        }

    def get_memory_strings(self) -> dict[str, str]:
        """Strings for agent memory formatting."""
        return {
            "recent_rounds": "RECENT ROUNDS SUMMARY:\n",
            "recent_posts": "YOUR RECENT POSTS:\n",
            "current_alliances": "CURRENT ALLIANCES: ",
            "current_targets": "CURRENT TARGETS: ",
            "no_memory": "No previous memory.",
            "no_history": "No history.",
        }

    def get_mini_profile_template(self) -> dict[str, str]:
        """Template for institutional agent mini-profiles."""
        axis = self.get_position_axis()
        return {
            "position": f"Position ({axis.negative_label} ↔ {axis.positive_label}): {{pos:+.2f}}.",
            "trait": "Key trait: {trait}",
        }

    def _load_example(self, filename: str) -> str:
        """Load a JSON example file from the domain's examples/ directory."""
        # Resolve path relative to the concrete subclass's module
        domain_dir = os.path.dirname(os.path.abspath(
            getattr(type(self), '__module__', __name__).replace('.', '/') + '.py'
        ))
        # Fallback: use class file location
        import inspect
        try:
            domain_dir = os.path.dirname(inspect.getfile(type(self)))
        except (TypeError, OSError):
            pass
        example_path = os.path.join(domain_dir, "examples", filename)
        if os.path.exists(example_path):
            with open(example_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Remove meta fields
            data.pop("_description", None)
            return json.dumps(data, indent=2, ensure_ascii=False)
        return ""

    def get_elite_few_shot(self) -> str:
        """Return a few-shot example for elite agent generation. Override per domain."""
        return self._load_example("elite_round.json")

    def get_cluster_few_shot(self) -> str:
        """Return a few-shot example for cluster generation. Override per domain."""
        return self._load_example("cluster_round.json")

    def get_event_few_shot(self) -> str:
        """Return a few-shot example for event generation. Override per domain."""
        return self._load_example("event.json")
