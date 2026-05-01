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
    confidence_interval: dict = None,
    regime_info: dict = None,
    params_used: dict = None,
    orchestrator_state: dict = None,
    financial_twin_state: dict = None,
    financial_feedback: dict = None,
) -> str:
    """Save full state checkpoint and return filename.

    Args:
        orchestrator_state: Optional dict with escalation/contagion/financial state
            for wargame rollback (save scumming).
    """
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
                "id": a.id, "name": a.name, "role": a.role,
                "position": a.position, "original_position": a.original_position,
                "influence": a.influence, "rigidity": a.rigidity,
                "emotional_state": a.emotional_state,
                "key_trait": getattr(a, "key_trait", ""),
                "category": getattr(a, "category", ""),
            }
            for a in institutional_agents
        ] if not elite_only else [],
        "citizen_clusters": [
            c.to_dict() for c in citizen_swarm.clusters.values()
        ] if not elite_only else [],
        "coalition_history": coalition_history,
        "cost": cost,
    }

    # v2/v3 calibration metadata (optional, backward-compatible)
    if confidence_interval:
        state["confidence_interval"] = confidence_interval
    if regime_info:
        state["regime_info"] = regime_info
    if params_used:
        state["params_used"] = params_used

    # Orchestrator state for wargame rollback
    if orchestrator_state:
        state["orchestrator_state"] = orchestrator_state

    # Financial twin (Sprint 1+) — exposes ALM KPIs to export pipeline
    if financial_twin_state:
        state["financial_twin"] = financial_twin_state
    if financial_feedback:
        state["financial_feedback"] = financial_feedback

    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return filename


def load_checkpoint(filepath: str) -> dict:
    """Load a checkpoint from disk."""
    with open(filepath, "r") as f:
        return json.load(f)


def find_checkpoint(checkpoint_dir: str, scenario_name: str, round_num: int) -> str:
    """Find checkpoint file path for a given scenario and round."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in scenario_name)
    filepath = os.path.join(checkpoint_dir, f"state_{safe_name}_r{round_num}.json")
    if os.path.exists(filepath):
        return filepath
    raise FileNotFoundError(f"No checkpoint for {scenario_name} round {round_num}")


def restore_agents(checkpoint: dict):
    """Reconstruct agents and clusters from a checkpoint dict.

    Returns (elite_agents, institutional_agents, citizen_clusters_list).
    """
    from ..agents.elite_agent import EliteAgent
    from ..agents.institutional_agent import InstitutionalAgent
    from ..agents.citizen_cluster import CitizenCluster

    elite_agents = []
    for d in checkpoint.get("elite_agents", []):
        agent = EliteAgent(
            id=d["id"], name=d["name"], role=d["role"],
            archetype=d.get("archetype", "unknown"),
            position=d["position"],
            original_position=d.get("original_position", d["position"]),
            influence=d.get("influence", 0.5),
            rigidity=d.get("rigidity", 0.5),
            system_prompt=d.get("system_prompt", ""),
            emotional_state=d.get("emotional_state", "neutral"),
            engagement_level=d.get("engagement_level", 0.5),
        )
        elite_agents.append(agent)

    inst_agents = []
    for d in checkpoint.get("institutional_agents", []):
        agent = InstitutionalAgent.from_spec(d)
        agent.position = d["position"]
        agent.original_position = d.get("original_position", d["position"])
        agent.emotional_state = d.get("emotional_state", "neutral")
        inst_agents.append(agent)

    clusters = []
    for d in checkpoint.get("citizen_clusters", []):
        cluster = CitizenCluster.from_spec(d)
        cluster.position = d["position"]
        cluster.original_position = d.get("original_position", d["position"])
        cluster.engagement_level = d.get("engagement_level", 0.5)
        cluster.trust_institutions = d.get("trust_institutions", 0.5)
        cluster.dominant_sentiment = d.get("dominant_sentiment", "indifferent")
        cluster.sentiment_distribution = d.get("sentiment_distribution", {})
        cluster.emergent_narrative = d.get("emergent_narrative", "")
        cluster.key_concerns = d.get("key_concerns", [])
        clusters.append(cluster)

    return elite_agents, inst_agents, clusters
