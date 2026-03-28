"""Synthetic simulation runner for calibration.

Runs OpinionDynamics math WITHOUT any LLM calls. Creates synthetic agents
from historical polling data and simulates 9 rounds of opinion evolution.
Cost: $0. Runtime: ~10ms per scenario per parameter combo.

Key design choices for realism:
- Elite agents broadcast to all agents (simulating media coverage)
- Events directly shift susceptible agents (undecided, low rigidity)
- Agent "anchor" drifts over time (once convinced, new position becomes baseline)
- Homophily decreases during shock rounds (events break filter bubbles)
"""

import random
from dataclasses import dataclass

from .historical_scenario import GroundTruth, PollingDataPoint


@dataclass
class SyntheticAgent:
    """Lightweight agent for calibration — no LLM, just position tracking."""
    id: str
    position: float
    original_position: float
    rigidity: float
    tolerance: float
    influence: float
    tier: int  # 1=elite, 2=institutional, 3=citizen


def _create_agents_from_polling(
    initial_polling: PollingDataPoint,
    n_agents: int = 100,
    seed: int = 42,
) -> list[SyntheticAgent]:
    """Create synthetic agents whose position distribution matches initial polling.

    Maps polling to positions:
      pro agents  → uniform in [+0.15, +0.9]
      against     → uniform in [-0.9, -0.15]
      undecided   → uniform in [-0.15, +0.15]

    Agent types: 5% elite (tier 1), 10% institutional (tier 2), 85% citizen (tier 3).
    """
    rng = random.Random(seed)

    n_pro = round(n_agents * initial_polling.pro_pct / 100)
    n_against = round(n_agents * initial_polling.against_pct / 100)
    n_undecided = n_agents - n_pro - n_against

    agents: list[SyntheticAgent] = []
    idx = 0

    def _make_agent(i: int, pos: float) -> SyntheticAgent:
        r = i / n_agents
        if r < 0.05:
            tier, influence = 1, rng.uniform(0.7, 1.0)
            rigidity = rng.uniform(0.5, 0.8)
        elif r < 0.15:
            tier, influence = 2, rng.uniform(0.4, 0.7)
            rigidity = rng.uniform(0.3, 0.6)
        else:
            tier, influence = 3, rng.uniform(0.1, 0.4)
            rigidity = rng.uniform(0.15, 0.45)
        return SyntheticAgent(
            id=f"agent_{i}", position=pos, original_position=pos,
            rigidity=rigidity, tolerance=rng.uniform(0.3, 0.7),
            influence=influence, tier=tier,
        )

    for _ in range(n_pro):
        agents.append(_make_agent(idx, rng.uniform(0.15, 0.9)))
        idx += 1
    for _ in range(n_against):
        agents.append(_make_agent(idx, rng.uniform(-0.9, -0.15)))
        idx += 1
    for _ in range(n_undecided):
        agents.append(_make_agent(idx, rng.uniform(-0.15, 0.15)))
        idx += 1

    rng.shuffle(agents)
    return agents


def _get_event_for_round(
    round_num: int,
    ground_truth: GroundTruth,
    polling: list[PollingDataPoint],
) -> tuple[float, float]:
    """Extract event shock from ground truth.

    Uses the ACTUAL polling delta to derive shock direction and magnitude.
    Key events get amplified shocks.

    Returns (shock_magnitude, shock_direction).
    """
    has_event = any(
        e.get("round_equivalent") == round_num
        for e in ground_truth.key_events
    )

    current = next((p for p in polling if p.round_equivalent == round_num), None)
    previous = next((p for p in polling if p.round_equivalent == round_num - 1), None)

    if current and previous:
        pro_delta = current.pro_pct - previous.pro_pct
        # Direction: positive = pro-shift, negative = against-shift
        direction = max(-1.0, min(1.0, pro_delta / 4.0))
    else:
        direction = 0.0

    if has_event:
        magnitude = min(0.6, max(0.25, abs(direction) * 2.5))
    else:
        magnitude = min(0.35, max(0.05, abs(direction) * 1.5))

    return magnitude, direction


def _build_feed(
    agent: SyntheticAgent,
    all_agents: list[SyntheticAgent],
    elites: list[SyntheticAgent],
    rng: random.Random,
    shock_magnitude: float,
    feed_size: int = 10,
) -> list[tuple[float, float, float]]:
    """Build a synthetic feed for an agent.

    Key improvements over naive homophily:
    - Elite agents always appear (media broadcast effect)
    - During shocks, homophily decreases (events break filter bubbles)
    - Higher-influence agents get higher engagement
    """
    others = [a for a in all_agents if a.id != agent.id]
    if not others:
        return []

    # Elite broadcast: always include 1-2 elite agents with high engagement
    feed: list[tuple[float, float, float]] = []
    elite_sample = rng.sample(elites, min(2, len(elites))) if elites else []
    for e in elite_sample:
        if e.id != agent.id:
            feed.append((e.position, e.influence, rng.uniform(0.5, 1.0)))

    remaining_slots = feed_size - len(feed)

    # Homophily ratio: lower during shocks (events break bubbles)
    homophily_ratio = max(0.3, 0.6 - shock_magnitude)
    n_homophily = int(remaining_slots * homophily_ratio)
    n_random = remaining_slots - n_homophily

    nearby = sorted(others, key=lambda a: abs(a.position - agent.position))
    used_ids = {e.id for _, _, _ in feed}  # Track elites already added

    for a in nearby[:n_homophily]:
        if a.id not in used_ids:
            feed.append((a.position, a.influence, rng.uniform(0.1, 0.6)))
            used_ids.add(a.id)

    pool = [a for a in others if a.id not in used_ids]
    if pool:
        for a in rng.sample(pool, min(n_random, len(pool))):
            feed.append((a.position, a.influence, rng.uniform(0.1, 0.5)))

    return feed[:feed_size]


def _apply_direct_event_shift(
    agents: list[SyntheticAgent],
    shock_magnitude: float,
    shock_direction: float,
    direct_shift_weight: float,
    rng: random.Random,
):
    """Directly shift susceptible agents during major events.

    In reality, big events don't just nudge — they convert undecided voters
    and can flip weakly-held positions. This models that direct effect.

    Susceptibility = (1 - rigidity) × (1 - abs(position))
    Agents near center with low rigidity are most susceptible.
    """
    if abs(shock_direction) < 0.05:
        return

    for agent in agents:
        susceptibility = (1 - agent.rigidity) * max(0, 1 - abs(agent.position))
        shift = shock_magnitude * shock_direction * susceptibility * direct_shift_weight
        agent.position = max(-1.0, min(1.0, agent.position + shift))


def _drift_anchors(agents: list[SyntheticAgent], drift_rate: float = 0.15):
    """Slowly update original_position toward current position.

    Models the psychological reality that once you change your mind,
    your new position becomes your baseline. Without this, anchor_weight
    always pulls agents back to round 1, preventing opinion reversals.
    """
    for agent in agents:
        agent.original_position += drift_rate * (agent.position - agent.original_position)


def _agents_to_avg_position(agents: list[SyntheticAgent]) -> float:
    """Average position across all agents."""
    return sum(a.position for a in agents) / len(agents) if agents else 0.0


def _agents_to_pro_pct(agents: list[SyntheticAgent]) -> float:
    """Convert agent positions to pro percentage (decided voters only)."""
    pro = sum(1 for a in agents if a.position > 0.05)
    against = sum(1 for a in agents if a.position < -0.05)
    total_decided = pro + against
    if total_decided == 0:
        return 50.0
    return (pro / total_decided) * 100.0


def run_synthetic_simulation(
    ground_truth: GroundTruth,
    params: dict,
    n_agents: int = 100,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Run a synthetic simulation using only OpinionDynamics math.

    Improvements over naive approach:
    1. Elite broadcast: top agents appear in everyone's feed
    2. Direct event shifts: major events directly convert susceptible agents
    3. Anchor drift: original_position slowly updates, enabling reversals
    4. Dynamic homophily: filter bubbles break during shock events

    Returns:
        (final_pro_pct, positions_per_round) — for comparison with ground truth.
    """
    from core.simulation.opinion_dynamics import OpinionDynamics

    # Extract calibratable synthetic params
    direct_shift_weight = params.get("direct_shift_weight", 0.4)
    anchor_drift_rate = params.get("anchor_drift_rate", 0.2)

    od = OpinionDynamics(
        anchor_weight=params.get("anchor_weight", 0.1),
        social_weight=params.get("social_weight", 0.15),
        event_weight=params.get("event_weight", 0.05),
        herd_weight=params.get("herd_weight", 0.05),
        herd_threshold=params.get("herd_threshold", 0.2),
    )

    polling = ground_truth.polling_trajectory
    if not polling:
        return 50.0, []

    agents = _create_agents_from_polling(polling[0], n_agents=n_agents, seed=seed)
    elites = [a for a in agents if a.tier == 1]
    rng = random.Random(seed + 1)

    positions_per_round: list[float] = []
    # Round 1: use actual polling as ground truth (not sim output)
    positions_per_round.append(
        (polling[0].pro_pct - polling[0].against_pct) / 100.0
    )

    num_rounds = len(polling)

    for round_idx in range(1, num_rounds):
        round_num = round_idx + 1
        shock_mag, shock_dir = _get_event_for_round(round_num, ground_truth, polling)

        # Phase 1: Direct event impact on susceptible agents
        if shock_mag > 0.15:
            _apply_direct_event_shift(agents, shock_mag, shock_dir, direct_shift_weight, rng)

        # Phase 2: Elite agents shift toward event direction (opinion leaders react)
        for elite in elites:
            if shock_mag > 0.2:
                elite_shift = shock_dir * shock_mag * (1 - elite.rigidity) * 0.2
                elite.position = max(-1.0, min(1.0, elite.position + elite_shift))

        # Phase 3: Social influence via OpinionDynamics
        for agent in agents:
            feed = _build_feed(agent, agents, elites, rng, shock_mag)
            new_pos = od.update_position(
                agent_position=agent.position,
                agent_original_position=agent.original_position,
                agent_rigidity=agent.rigidity,
                agent_tolerance=agent.tolerance,
                feed_authors_positions=feed,
                event_shock_magnitude=shock_mag,
                event_shock_direction=shock_dir,
            )
            agent.position = new_pos

        # Phase 4: Anchor drift (enables opinion reversals over time)
        _drift_anchors(agents, drift_rate=anchor_drift_rate)

        # Record position for trajectory comparison
        avg_pos = _agents_to_avg_position(agents)
        positions_per_round.append(avg_pos)

    final_pro_pct = _agents_to_pro_pct(agents)
    return final_pro_pct, positions_per_round
