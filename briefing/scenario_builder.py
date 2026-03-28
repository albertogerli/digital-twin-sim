"""Builds a validated ScenarioConfig from LLM analysis or YAML file."""

import json
import logging
import yaml
from typing import Optional

from core.config.schema import (
    ScenarioConfig, AxisConfig, ChannelConfig, AgentSpec, ClusterSpec,
)
from core.llm.base_client import BaseLLMClient
from .brief_analyzer import analyze_brief
from .web_research import research_context
from .entity_researcher import research_entities
from .agent_generator import generate_agents_multistep
from .agent_validator import validate_agents, critic_review
from .seed_builder import build_seed_from_entity_research

logger = logging.getLogger(__name__)


class ScenarioBuilder:
    """Builds ScenarioConfig from brief text or config file."""

    async def build_from_brief(
        self,
        brief_text: str,
        llm: BaseLLMClient,
        available_domains: list[str],
        interactive: bool = False,
        progress_callback=None,
        extra_context: str = "",
        seed_data_path: str = "",
    ) -> ScenarioConfig:
        """Analyze a free-text brief and build a ScenarioConfig.

        Uses multi-step pipeline:
        1. Web research (existing)
        2. Entity deep-dive (new — Phase 2)
        3. Multi-step agent generation (new — Phase 3)
        4. Validation & balancing (new — Phase 4)
        5. Auto seed data (new — Phase 5)

        Args:
            extra_context: Additional text from uploaded documents (RAG)
            seed_data_path: Path to structured seed data directory
        """
        # Step 0: Load seed data if available
        seed_data_bundle = None
        if seed_data_path:
            from seed_data import SeedDataLoader
            seed_data_bundle = SeedDataLoader(seed_data_path).load()
            if seed_data_bundle:
                print(f"  ├─ Loaded seed data: {len(seed_data_bundle.stakeholders)} stakeholders ✓")

        # Step 1: Web research for real-world context
        print("  Researching online context...")
        web_context = await research_context(brief_text, llm, progress_callback)
        if web_context:
            print(f"  ├─ Found {len(web_context)} chars of web context ✓")
        else:
            print(f"  ├─ No web context found, proceeding with brief only")

        # Merge web context with uploaded document context
        combined_context = web_context or ""
        if extra_context:
            print(f"  ├─ Injecting {len(extra_context)} chars from uploaded documents ✓")
            combined_context = (
                f"{combined_context}\n\n"
                f"=== DOCUMENTAZIONE FORNITA DAL CLIENTE ===\n"
                f"{extra_context[:8000]}\n"  # Cap at 8K chars to fit prompt
                f"=== FINE DOCUMENTAZIONE ==="
            ).strip()

        # Step 2: Entity deep-dive research (Phase 2)
        entity_context = ""
        if combined_context:
            print("  Entity deep-dive research...")
            entity_context = await research_entities(
                brief_text, combined_context, llm,
                progress_callback=progress_callback,
            )
            if entity_context:
                print(f"  ├─ Entity profiles: {len(entity_context)} chars ✓")
            else:
                print(f"  ├─ No entity profiles extracted")

        # Step 3: Multi-step agent generation (Phase 3)
        # Resolve domain plugin if possible (for archetype guidance)
        domain_plugin = None
        try:
            from domains.domain_registry import DomainRegistry
            # Try to detect domain from brief for early guidance
            for d in available_domains:
                plugin = DomainRegistry.get(d)
                if plugin:
                    domain_plugin = plugin
                    break  # Will be refined by scaffold step
        except Exception:
            pass

        print("  Multi-step agent generation...")
        analysis = await generate_agents_multistep(
            brief=brief_text,
            llm=llm,
            available_domains=available_domains,
            web_context=combined_context,
            seed_data_bundle=seed_data_bundle,
            entity_context=entity_context,
            domain_plugin=domain_plugin,
            progress_callback=progress_callback,
        )

        # Step 4: Validation & balancing (Phase 4)
        # Get domain guidance for validation
        domain_guidance = {}
        try:
            from domains.domain_registry import DomainRegistry
            resolved_plugin = DomainRegistry.get(analysis.get("domain", ""))
            if resolved_plugin:
                domain_guidance = resolved_plugin.get_agent_generation_guidance()
        except Exception:
            pass

        print("  Validating agent roster...")
        analysis, validation_result = validate_agents(analysis, domain_guidance)
        if validation_result.has_warnings or validation_result.has_errors:
            print(f"  ├─ Validation issues found:")
            print(validation_result.summary())
            # Run LLM critic for qualitative fixes
            print("  ├─ Running critic review...")
            analysis = await critic_review(analysis, validation_result, llm)
            print(f"  ├─ Critic review complete ✓")
        else:
            print(f"  ├─ All validation checks passed ✓")

        # Step 5: Auto seed data (Phase 5) — only if no manual seed data
        if not seed_data_bundle and entity_context:
            auto_seed = build_seed_from_entity_research(entity_context, analysis)
            if auto_seed:
                print(f"  ├─ Auto seed data: {len(auto_seed.stakeholders)} stakeholders ✓")
                # Store on analysis for potential future use
                analysis["_auto_seed_bundle"] = auto_seed

        config = self._analysis_to_config(analysis)

        # Set seed_data_path on config so engine can load it
        if seed_data_path:
            config.seed_data_path = seed_data_path

        if interactive:
            self._print_summary(config)

        return config

    def build_from_file(self, path: str) -> ScenarioConfig:
        """Load a ScenarioConfig from a YAML or JSON file."""
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return self._dict_to_config(data)

    def _analysis_to_config(self, analysis: dict) -> ScenarioConfig:
        """Convert LLM analysis output to ScenarioConfig."""
        # Build axis config
        axis_data = analysis.get("position_axis", {})
        axis = AxisConfig(
            negative_label=axis_data.get("negative_label", "Against"),
            positive_label=axis_data.get("positive_label", "In favor"),
            neutral_label=axis_data.get("neutral_label", "Neutral"),
        )

        # Build channels
        channels = []
        for ch in analysis.get("suggested_channels", []):
            channels.append(ChannelConfig(
                id=ch["id"],
                description=ch.get("description", ""),
                max_length=ch.get("max_length", 280),
                channel_type=ch.get("channel_type", "short_form"),
            ))

        # Build elite agents
        elite_agents = []
        for a in analysis.get("suggested_elite_agents", []):
            elite_agents.append(AgentSpec(
                id=a["id"],
                name=a["name"],
                role=a["role"],
                archetype=a.get("archetype", "unknown"),
                position=float(a.get("position", 0)),
                influence=float(a.get("influence", 0.5)),
                rigidity=float(a.get("rigidity", 0.5)),
                tier=1,
                bio=a.get("bio", ""),
                communication_style=a.get("communication_style", ""),
                key_traits=a.get("key_traits", []),
            ))

        # Build institutional agents
        inst_agents = []
        for a in analysis.get("suggested_institutional_agents", []):
            inst_agents.append(AgentSpec(
                id=a["id"],
                name=a["name"],
                role=a["role"],
                archetype=a.get("category", "unknown"),
                position=float(a.get("position", 0)),
                influence=float(a.get("influence", 0.3)),
                rigidity=float(a.get("rigidity", 0.5)),
                tier=2,
                key_trait=a.get("key_trait", ""),
                category=a.get("category", ""),
            ))

        # Build citizen clusters
        clusters = []
        for c in analysis.get("suggested_citizen_clusters", []):
            clusters.append(ClusterSpec(
                id=c["id"],
                name=c["name"],
                description=c.get("description", ""),
                size=int(c.get("size", 1000)),
                position=float(c.get("position", 0)),
                engagement_base=float(c.get("engagement_base", 0.5)),
                info_channel=c.get("info_channel", ""),
                demographic_attributes=c.get("demographic_attributes", {}),
            ))

        return ScenarioConfig(
            name=analysis.get("scenario_name", "Untitled Scenario"),
            description=analysis.get("description", ""),
            domain=analysis.get("domain", "political"),
            language=analysis.get("language", "en"),
            num_rounds=int(analysis.get("num_rounds", 9)),
            timeline_unit=analysis.get("timeline_unit", "month"),
            timeline_labels=analysis.get("timeline_labels", []),
            position_axis=axis,
            channels=channels,
            elite_agents=elite_agents,
            institutional_agents=inst_agents,
            citizen_clusters=clusters,
            initial_event=analysis.get("initial_event", ""),
            scenario_context=analysis.get("scenario_context", ""),
            metrics_to_track=analysis.get("metrics_to_track", []),
        )

    def _dict_to_config(self, data: dict) -> ScenarioConfig:
        """Convert a raw dict (from YAML/JSON) to ScenarioConfig."""
        return self._analysis_to_config(data)

    def _print_summary(self, config: ScenarioConfig):
        """Print a summary of the generated config for review."""
        print(f"""
  ┌─ Scenario: {config.name}
  ├─ Domain: {config.domain}
  ├─ Rounds: {config.num_rounds} ({config.timeline_unit}s)
  ├─ Axis: [{config.position_axis.negative_label}] ← → [{config.position_axis.positive_label}]
  ├─ Elite agents: {len(config.elite_agents)}
  ├─ Institutional agents: {len(config.institutional_agents)}
  ├─ Citizen clusters: {len(config.citizen_clusters)}
  └─ Channels: {', '.join(ch.id for ch in config.channels)}
""")

    def save_config(self, config: ScenarioConfig, path: str):
        """Save a ScenarioConfig to YAML for review/reuse."""
        data = config.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"  Config saved to: {path}")
