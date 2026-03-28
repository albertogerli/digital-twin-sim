#!/usr/bin/env python3
"""DigitalTwinSim — Universal Digital Twin Simulation Platform.

Usage:
    python run.py --brief "Your scenario description here"
    python run.py --config configs/my_scenario.yaml
    python run.py --brief "scenario" --domain political --rounds 6
"""

import argparse
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulation.engine import SimulationEngine
from core.config.schema import ScenarioConfig
from domains.domain_registry import DomainRegistry
from briefing.scenario_builder import ScenarioBuilder


# Provider → (default model, env var)
LLM_PROVIDERS = {
    "gemini": ("gemini-3.1-flash-lite-preview", "GOOGLE_API_KEY"),
    "openai": ("gpt-5.4-mini", "OPENAI_API_KEY"),
}


def create_llm(provider: str, model: str | None = None, budget: float = 5.0):
    """Create the appropriate LLM client based on provider."""
    if provider == "openai":
        from core.llm.openai_client import OpenAIClient
        return OpenAIClient(model=model or "gpt-5.4-mini", budget=budget)
    else:
        from core.llm.gemini_client import GeminiClient
        return GeminiClient(model=model or "gemini-3.1-flash-lite-preview", budget=budget)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DigitalTwinSim — Universal Digital Twin Simulation Platform"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--brief", type=str,
        help="Free-text scenario description (LLM will design the simulation)"
    )
    group.add_argument(
        "--config", type=str,
        help="Path to YAML/JSON scenario config file"
    )

    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "openai"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Force domain (political, commercial, marketing, etc.)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of rounds")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="Max LLM spend in USD (default: 5.0)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override (default: provider's default)")
    parser.add_argument("--elite-only", action="store_true",
                        help="Run only elite agents (faster, cheaper)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--save-config", type=str, default=None,
                        help="Save generated config to YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate config only, don't run simulation")

    return parser.parse_args()


async def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Discover domain plugins
    DomainRegistry.discover()
    available_domains = DomainRegistry.list_domains()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  DigitalTwinSim v1.0                                    ║
║  Universal Digital Twin Simulation Platform              ║
║  Available domains: {', '.join(available_domains):<36} ║
╚══════════════════════════════════════════════════════════╝
""")

    # Initialize LLM
    default_model = LLM_PROVIDERS[args.provider][0]
    model = args.model or default_model
    print(f"  LLM: {args.provider} / {model}")
    llm = create_llm(args.provider, model, args.budget)

    # Build scenario config
    builder = ScenarioBuilder()

    if args.brief:
        config = await builder.build_from_brief(
            brief_text=args.brief,
            llm=llm,
            available_domains=available_domains,
            interactive=True,
        )
    else:
        config = builder.build_from_file(args.config)

    # Apply overrides
    if args.domain:
        config.domain = args.domain
    if args.rounds:
        config.num_rounds = args.rounds
    config.budget_usd = args.budget

    # Save config if requested
    if args.save_config:
        builder.save_config(config, args.save_config)

    if args.dry_run:
        print("  Dry run — config generated, simulation not started.")
        return

    # Get domain plugin
    domain = DomainRegistry.get(config.domain)
    print(f"  Using domain plugin: {domain.domain_label}")

    # Run simulation
    engine = SimulationEngine(
        llm=llm,
        config=config,
        domain=domain,
        output_dir=args.output_dir,
        elite_only=args.elite_only,
        verbose=args.verbose,
    )

    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
