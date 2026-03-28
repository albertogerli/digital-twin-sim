"""Checkpoint save/load for simulation state."""

import json
import os
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str,
    scenario_name: str,
    round_num: int,
    elite_agents: list,
    institutional_agents: list,
    citizen_swarm,
    coalition_history: list,
    cost: float,
    elite_only: bool = False,
    domain: str = "",
) -> str:
    """Save full state checkpoint and return filename."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Sanitize scenario name for filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in scenario_name)
    filename = f"state_{safe_name}_r{round_num}.json"
    filepath = os.path.join(checkpoint_dir, filename)

    state = {
        "scenario": scenario_name,
        "domain": domain,
        "round": round_num,
        "elite_agents": [a.to_dict() for a in elite_agents],
        "institutional_agents": [
            {
                "id": a.id, "name": a.name, "position": a.position,
                "emotional_state": a.emotional_state,
            }
            for a in institutional_agents
        ] if not elite_only else [],
        "citizen_clusters": [
            c.to_dict() for c in citizen_swarm.clusters.values()
        ] if not elite_only else [],
        "coalition_history": coalition_history,
        "cost": cost,
    }

    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return filename
