"""Agent grounding: generate scenario-specific elite agents and verified fact sheets."""

from .agent_grounder import AgentGrounder, GroundedAgent, ScenarioContext, ground_scenario_from_config
from .scenario_researcher import ScenarioResearcher, FactSheet, VerifiedEvent, VerifiedPoll, StakeholderInfo, research_scenario
from .pipeline import ground_scenario
