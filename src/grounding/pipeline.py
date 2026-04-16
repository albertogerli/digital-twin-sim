"""Complete grounding pipeline: research + agent generation.

Usage:
    from src.grounding.pipeline import ground_scenario

    fact_sheet, agents, events = ground_scenario(
        topic="SVB Collapse",
        domain="financial",
        country="USA",
        timeframe="March 2023",
        llm_fn=my_llm_fn,
        search_fn=my_search_fn,
    )

    # fact_sheet.to_llm_context() → context for LLM narrative
    # agents → list of elite agents in simulator format
    # events → list of events in simulator format
"""

import logging
from typing import Callable, Optional

from .scenario_researcher import ScenarioResearcher, FactSheet
from .agent_grounder import AgentGrounder, ScenarioContext

logger = logging.getLogger(__name__)


def ground_scenario(
    topic: str,
    domain: str,
    country: str = "",
    timeframe: str = "",
    scenario_id: str = "",
    n_elite: int = 5,
    n_rounds: int = 7,
    llm_fn: Optional[Callable[[str], str]] = None,
    search_fn: Optional[Callable[[str], list[dict]]] = None,
    fetch_fn: Optional[Callable[[str], str]] = None,
) -> tuple[FactSheet, list[dict], list[dict]]:
    """Complete grounding pipeline.

    Returns:
        (fact_sheet, grounded_agents, scenario_events)
        On failure, returns partial results with empty lists.
    """
    # Step 1: Research
    researcher = ScenarioResearcher(
        llm_fn=llm_fn, search_fn=search_fn, fetch_fn=fetch_fn,
    )
    try:
        fact_sheet = researcher.research(
            topic=topic, domain=domain, country=country,
            timeframe=timeframe, scenario_id=scenario_id,
        )
    except Exception as e:
        logger.error(f"Research failed for '{topic}': {e}")
        fact_sheet = FactSheet(scenario_id=scenario_id, topic=topic, domain=domain)

    # Step 2: Agent grounding
    context = ScenarioContext(
        scenario_id=scenario_id,
        topic=topic,
        domain=domain,
        country=country,
        timeframe=timeframe,
        key_question=fact_sheet.outcome_description or topic,
        n_elite_target=n_elite,
    )
    grounder = AgentGrounder(llm_fn=llm_fn, search_fn=search_fn)
    try:
        agents = grounder.ground(context)
        agent_dicts = [a.to_sim_format() for a in agents]
    except Exception as e:
        logger.error(f"Agent grounding failed for '{topic}': {e}")
        agent_dicts = []

    # Step 3: Convert verified events to simulator format
    scenario_events = fact_sheet.to_scenario_events(n_rounds)

    return fact_sheet, agent_dicts, scenario_events
