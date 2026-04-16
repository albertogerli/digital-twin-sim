#!/usr/bin/env python3
"""Grounding PoC: measure impact of grounded agents + events on 3 scenarios.

Does NOT recalibrate — uses existing v2 posteriors.
Hypothesis: grounded inputs → lower raw error → less discrepancy needed.
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp

from src.dynamics.opinion_dynamics_jax import simulate_scenario, ScenarioData
from src.dynamics.param_utils import get_default_frozen_params
from src.observation.observation_model import build_scenario_data_from_json, build_sparse_interaction
from src.grounding.pipeline import ground_scenario
from src.grounding.scenario_researcher import FactSheet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grounding_poc")

# ── Paths ───────────────────────────────────────────────────────────────

SCENARIOS_DIR = ROOT / "calibration" / "empirical" / "scenarios"
POSTERIORS_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2_discrepancy" / "posteriors_v2.json"
OUTPUT_DIR = ROOT / "calibration" / "results" / "grounding_poc"

SCENARIO_FILES = {
    "CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW": SCENARIOS_DIR / "CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW.json",
    "FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI": SCENARIOS_DIR / "FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI.json",
    "POL-2016-BREXIT": SCENARIOS_DIR / "POL-2016-BREXIT.json",
}

# ── Step 0: Setup LLM and Search ────────────────────────────────────────

def setup_llm_fn():
    """Create llm_fn using Google Gemini (project's default LLM)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    from google import genai

    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-lite-preview"

    def llm_fn(prompt: str) -> str:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": 0.2, "max_output_tokens": 4000},
                )
                return response.text
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"LLM retry {attempt+1} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
        return ""

    print(f"✓ LLM: Google Gemini ({model})")
    return llm_fn


def setup_search_fn(llm_fn):
    """Create search_fn using Gemini's native Google Search grounding."""
    import re

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-lite-preview"

    def grounded_search(query: str) -> list[dict]:
        """Use Gemini + Google Search grounding to get real web results."""
        prompt = (
            f"Search the web for: {query}\n\n"
            "Return a detailed factual summary with specific dates, names, numbers, "
            "and events. Include as many concrete facts as possible."
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(
                        google_search=types.GoogleSearch()
                    )],
                    temperature=0.1,
                    max_output_tokens=4000,
                ),
            )

            results = []

            # Extract grounding metadata (real URLs + snippets)
            for candidate in response.candidates or []:
                gm = getattr(candidate, "grounding_metadata", None)
                if not gm:
                    continue
                # grounding_chunks contain the actual search results
                for chunk in getattr(gm, "grounding_chunks", []) or []:
                    web = getattr(chunk, "web", None)
                    if web:
                        url = getattr(web, "uri", "") or ""
                        title = getattr(web, "title", "") or ""
                        results.append({"url": url, "text": title})
                # search_entry_point may have rendered content
                # grounding_supports link text spans to sources
                for support in getattr(gm, "grounding_supports", []) or []:
                    seg = getattr(support, "segment", None)
                    if seg:
                        text = getattr(seg, "text", "") or ""
                        if text and len(text) > 30:
                            # Get source URLs from indices
                            urls = []
                            for idx_obj in getattr(support, "grounding_chunk_indices", []) or []:
                                idx = int(idx_obj) if not isinstance(idx_obj, int) else idx_obj
                                if idx < len(results):
                                    urls.append(results[idx]["url"])
                            url = urls[0] if urls else ""
                            results.append({"url": url, "text": text})

            # Always include the full response text as a "result"
            if response.text:
                results.append({"url": "gemini-grounded-search", "text": response.text})

            time.sleep(0.3)  # Rate limit
            logger.info(f"Grounded search: '{query[:50]}' → {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Grounded search failed for '{query[:50]}': {e}")
            return []

    print(f"✓ Search: Gemini Google Search Grounding (real web search)")
    return grounded_search


# ── Load posteriors ─────────────────────────────────────────────────────

def load_domain_params(domain: str) -> dict:
    """Load domain-specific posterior means + frozen params."""
    with open(POSTERIORS_PATH) as f:
        post = json.load(f)

    # Use domain-specific if available, else global
    if domain in post.get("domains", {}):
        mu = post["domains"][domain]["mu_d"]["mean"]
        logger.info(f"Using domain-specific params for '{domain}'")
    else:
        mu = post["global"]["mu_global"]["mean"]
        logger.info(f"Using global params (domain '{domain}' not found)")

    params = {
        "alpha_herd": mu[0],
        "alpha_anchor": mu[1],
        "alpha_social": mu[2],
        "alpha_event": mu[3],
    }
    frozen = get_default_frozen_params()
    # Convert JAX arrays to floats for consistency
    return {k: float(v) if hasattr(v, "item") else v for k, v in {**params, **frozen}.items()}


# ── Build ScenarioData from grounded inputs ─────────────────────────────

def build_grounded_scenario_data(
    original_scenario: dict,
    grounded_agents: list[dict],
    grounded_events: list[dict],
    seed: int = 42,
) -> ScenarioData:
    """Build ScenarioData replacing elite agents and events with grounded versions.

    Keeps citizen/institutional agents from original, replaces elite with grounded.
    """
    original_agents = original_scenario["agents"]
    n_rounds = original_scenario["n_rounds"]

    # Keep non-elite agents from original
    non_elite = [a for a in original_agents if a.get("type") != "elite"]

    # Convert grounded agents to scenario format
    grounded_as_scenario = []
    for ga in grounded_agents:
        grounded_as_scenario.append({
            "type": "elite",
            "name": ga.get("name", "unknown"),
            "initial_position": ga.get("position", 0.0),
            "influence": ga.get("influence", 0.5),
        })

    # Merge: grounded elite + original non-elite
    all_agents = grounded_as_scenario + non_elite
    n_agents = len(all_agents)

    # Build arrays
    positions = []
    agent_types = []
    rigidities = []
    tolerances = []
    influences = []

    type_map = {"elite": 0, "institutional": 0, "citizen_cluster": 1}
    rigidity_map = {"elite": 0.7, "institutional": 0.8, "citizen_cluster": 0.3}
    tolerance_map = {"elite": 0.3, "institutional": 0.4, "citizen_cluster": 0.6}

    for a in all_agents:
        atype = a["type"]
        positions.append(a["initial_position"])
        agent_types.append(type_map.get(atype, 1))
        rigidities.append(rigidity_map.get(atype, 0.5))
        tolerances.append(tolerance_map.get(atype, 0.5))
        influences.append(a.get("influence", 0.5))

    initial_positions = jnp.array(positions, dtype=jnp.float32)
    agent_types_arr = jnp.array(agent_types, dtype=jnp.int32)
    agent_rigidities = jnp.array(rigidities, dtype=jnp.float32)
    agent_tolerances = jnp.array(tolerances, dtype=jnp.float32)
    influences_arr = jnp.array(influences, dtype=jnp.float32)

    # Events from grounded
    events_arr = jnp.zeros((n_rounds, 2), dtype=jnp.float32)
    for evt in grounded_events:
        r = evt["round"] - 1
        if 0 <= r < n_rounds:
            mag = evt.get("shock_magnitude", 0.0)
            dir_ = evt.get("shock_direction", 0.0)
            events_arr = events_arr.at[r, 0].add(float(mag))
            events_arr = events_arr.at[r, 1].set(float(dir_))

    llm_shifts = jnp.zeros((n_rounds, n_agents), dtype=jnp.float32)
    interaction_matrix = build_sparse_interaction(influences_arr, seed=seed)

    return ScenarioData(
        initial_positions=initial_positions,
        agent_types=agent_types_arr,
        agent_rigidities=agent_rigidities,
        agent_tolerances=agent_tolerances,
        events=events_arr,
        llm_shifts=llm_shifts,
        interaction_matrix=interaction_matrix,
    )


# ── Logit helpers ───────────────────────────────────────────────────────

def logit(p):
    """Logit: log(p / (1-p)), clamped to avoid ±inf."""
    p = max(0.001, min(0.999, p / 100.0))
    return math.log(p / (1 - p))


# ── Visualization helpers ─────────────────────────────────────────────

# Custom diverging colormap: red (anti) → white (neutral) → green (pro)
SENTIMENT_CMAP = LinearSegmentedColormap.from_list(
    "sentiment", ["#c0392b", "#e74c3c", "#ffffff", "#2ecc71", "#27ae60"]
)

# Coalition palette
COAL_COLORS = {
    "Strong Anti": "#c0392b",
    "Lean Anti": "#e67e73",
    "Neutral": "#95a5a6",
    "Lean Pro": "#7dcea0",
    "Strong Pro": "#27ae60",
}
COAL_ORDER = ["Strong Anti", "Lean Anti", "Neutral", "Lean Pro", "Strong Pro"]


def _classify_coalition(pos: float) -> str:
    """Classify agent position into a coalition bucket."""
    if pos < -0.5:
        return "Strong Anti"
    elif pos < -0.1:
        return "Lean Anti"
    elif pos <= 0.1:
        return "Neutral"
    elif pos <= 0.5:
        return "Lean Pro"
    else:
        return "Strong Pro"


def _build_agent_labels(scenario_json: dict, grounded_agents: list[dict], is_grounded: bool) -> list[str]:
    """Build human-readable cluster labels for each agent."""
    if is_grounded:
        labels = []
        for ga in grounded_agents:
            labels.append(f"Elite: {ga.get('name', '?')[:20]}")
        non_elite = [a for a in scenario_json["agents"] if a.get("type") != "elite"]
        for a in non_elite:
            atype = a["type"]
            if atype == "institutional":
                labels.append(f"Inst: {a.get('name', 'Institution')[:20]}")
            else:
                labels.append(f"Citizen: {a.get('name', 'Cluster')[:20]}")
        return labels
    else:
        labels = []
        for a in scenario_json["agents"]:
            atype = a["type"]
            if atype == "elite":
                labels.append(f"Elite: {a.get('name', '?')[:20]}")
            elif atype == "institutional":
                labels.append(f"Inst: {a.get('name', 'Institution')[:20]}")
            else:
                labels.append(f"Citizen: {a.get('name', 'Cluster')[:20]}")
        return labels


def _agent_type_groups(scenario_json: dict, grounded_agents: list[dict], is_grounded: bool) -> dict:
    """Group agent indices by demographic type. Returns {group_name: [indices]}."""
    groups = {}
    if is_grounded:
        n_elite = len(grounded_agents)
        non_elite = [a for a in scenario_json["agents"] if a.get("type") != "elite"]
        for i in range(n_elite):
            groups.setdefault("Elite", []).append(i)
        for j, a in enumerate(non_elite):
            atype = a["type"]
            key = "Institutional" if atype == "institutional" else "Citizens"
            groups.setdefault(key, []).append(n_elite + j)
    else:
        for i, a in enumerate(scenario_json["agents"]):
            atype = a["type"]
            if atype == "elite":
                groups.setdefault("Elite", []).append(i)
            elif atype == "institutional":
                groups.setdefault("Institutional", []).append(i)
            else:
                groups.setdefault("Citizens", []).append(i)
    return groups


def plot_sentiment_heatmap(
    trajectories: np.ndarray,
    scenario_json: dict,
    grounded_agents: list[dict],
    is_grounded: bool,
    title: str,
    save_path: Path,
):
    """Heatmap: sentiment per agent cluster per round.

    Rows = agent labels, columns = rounds.
    Color = position in [-1, +1].
    """
    n_rounds, n_agents = trajectories.shape
    labels = _build_agent_labels(scenario_json, grounded_agents, is_grounded)
    labels = labels[:n_agents]  # safety

    fig_h = max(4, 0.45 * n_agents + 1.5)
    fig, ax = plt.subplots(figsize=(max(6, n_rounds * 0.9 + 2), fig_h))

    im = ax.imshow(
        trajectories.T,  # agents on y-axis, rounds on x-axis
        aspect="auto", cmap=SENTIMENT_CMAP, vmin=-1, vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(n_rounds))
    ax.set_xticklabels([f"R{r+1}" for r in range(n_rounds)], fontsize=9)
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Round", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Annotate cells with position value
    for r in range(n_rounds):
        for a in range(n_agents):
            val = trajectories[r, a]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(r, a, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Position (Anti ← → Pro)", fontsize=9)

    # Add group separators
    groups = _agent_type_groups(scenario_json, grounded_agents, is_grounded)
    y_pos = 0
    for gname, indices in groups.items():
        if y_pos > 0:
            ax.axhline(y=y_pos - 0.5, color="black", linewidth=1.5, linestyle="--")
        # Group label on the right
        mid = y_pos + len(indices) / 2 - 0.5
        ax.annotate(gname, xy=(n_rounds - 0.3, mid), fontsize=8,
                    fontweight="bold", color="#2c3e50", ha="left", va="center",
                    annotation_clip=False)
        y_pos += len(indices)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Heatmap saved: {save_path.name}")


def plot_coalition_flow(
    trajectories: np.ndarray,
    scenario_json: dict,
    grounded_agents: list[dict],
    is_grounded: bool,
    title: str,
    save_path: Path,
):
    """Alluvial / stacked-area chart showing coalition sizes across rounds.

    At each round, agents are classified into 5 coalition buckets based on position.
    Shows how the population shifts between coalitions over time.
    """
    n_rounds, n_agents = trajectories.shape

    # Compute coalition sizes per round
    coalition_data = {c: [] for c in COAL_ORDER}
    for r in range(n_rounds):
        counts = {c: 0 for c in COAL_ORDER}
        for a in range(n_agents):
            bucket = _classify_coalition(float(trajectories[r, a]))
            counts[bucket] += 1
        for c in COAL_ORDER:
            coalition_data[c].append(counts[c])

    rounds = list(range(1, n_rounds + 1))

    # ── Figure with 2 subplots: stacked area + transition matrix ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: Stacked area chart
    bottoms = np.zeros(n_rounds)
    for c in COAL_ORDER:
        values = np.array(coalition_data[c], dtype=float)
        ax1.fill_between(rounds, bottoms, bottoms + values,
                         label=c, color=COAL_COLORS[c], alpha=0.85, linewidth=0)
        # Label inside bands if large enough
        for r in range(n_rounds):
            if values[r] >= 1:
                ax1.text(rounds[r], bottoms[r] + values[r] / 2,
                         str(int(values[r])), ha="center", va="center",
                         fontsize=7, fontweight="bold", color="white" if c != "Neutral" else "black")
        bottoms += values

    ax1.set_xlim(1, n_rounds)
    ax1.set_ylim(0, n_agents)
    ax1.set_xticks(rounds)
    ax1.set_xticklabels([f"R{r}" for r in rounds])
    ax1.set_xlabel("Round", fontsize=10)
    ax1.set_ylabel("Number of Agents", fontsize=10)
    ax1.set_title(f"Coalition Flow — {title}", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Right: Round-to-round transition heatmap (last transition)
    # Show transitions from R1→R_last
    if n_rounds >= 2:
        start_coalitions = [_classify_coalition(float(trajectories[0, a])) for a in range(n_agents)]
        end_coalitions = [_classify_coalition(float(trajectories[-1, a])) for a in range(n_agents)]

        # Build transition matrix
        trans = np.zeros((len(COAL_ORDER), len(COAL_ORDER)), dtype=int)
        for a in range(n_agents):
            i_from = COAL_ORDER.index(start_coalitions[a])
            i_to = COAL_ORDER.index(end_coalitions[a])
            trans[i_from, i_to] += 1

        im2 = ax2.imshow(trans, cmap="YlOrRd", interpolation="nearest")
        ax2.set_xticks(range(len(COAL_ORDER)))
        ax2.set_xticklabels([c.replace("Strong ", "S.").replace("Lean ", "L.") for c in COAL_ORDER],
                            fontsize=8, rotation=45, ha="right")
        ax2.set_yticks(range(len(COAL_ORDER)))
        ax2.set_yticklabels([c.replace("Strong ", "S.").replace("Lean ", "L.") for c in COAL_ORDER],
                            fontsize=8)
        ax2.set_xlabel("Final Coalition (R" + str(n_rounds) + ")", fontsize=9)
        ax2.set_ylabel("Initial Coalition (R1)", fontsize=9)
        ax2.set_title("Transition Matrix R1 → R" + str(n_rounds), fontsize=10, fontweight="bold")

        # Annotate
        for i in range(len(COAL_ORDER)):
            for j in range(len(COAL_ORDER)):
                if trans[i, j] > 0:
                    ax2.text(j, i, str(trans[i, j]), ha="center", va="center",
                             fontsize=10, fontweight="bold",
                             color="white" if trans[i, j] > trans.max() * 0.5 else "black")

        fig.colorbar(im2, ax=ax2, shrink=0.8)
    else:
        ax2.text(0.5, 0.5, "Need ≥2 rounds", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Coalition flow saved: {save_path.name}")


def plot_agent_trajectories(
    trajectories: np.ndarray,
    scenario_json: dict,
    grounded_agents: list[dict],
    is_grounded: bool,
    title: str,
    save_path: Path,
):
    """Spaghetti plot: each agent's position trajectory over rounds."""
    n_rounds, n_agents = trajectories.shape
    labels = _build_agent_labels(scenario_json, grounded_agents, is_grounded)
    labels = labels[:n_agents]
    groups = _agent_type_groups(scenario_json, grounded_agents, is_grounded)

    group_colors = {"Elite": "#e74c3c", "Institutional": "#3498db", "Citizens": "#27ae60"}
    fig, ax = plt.subplots(figsize=(max(6, n_rounds * 1.2), 5))

    rounds = range(1, n_rounds + 1)
    for gname, indices in groups.items():
        color = group_colors.get(gname, "#7f8c8d")
        for idx in indices:
            if idx < n_agents:
                lw = 2.5 if gname == "Elite" else 1.5
                ls = "-" if gname == "Elite" else ("--" if gname == "Institutional" else ":")
                ax.plot(rounds, trajectories[:, idx], color=color, linewidth=lw,
                        linestyle=ls, alpha=0.8, label=labels[idx])
                # End label
                ax.annotate(labels[idx][:15], xy=(n_rounds, trajectories[-1, idx]),
                            fontsize=6, color=color, va="center",
                            xytext=(5, 0), textcoords="offset points")

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax.axhspan(-0.1, 0.1, alpha=0.1, color="gray", label="Neutral zone")
    ax.set_xlim(1, n_rounds + 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(list(rounds))
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Position (Anti ← → Pro)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Custom legend by group
    handles = [mpatches.Patch(color=group_colors.get(g, "#7f8c8d"), label=g) for g in groups]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Trajectories saved: {save_path.name}")


def generate_all_charts(
    sid: str,
    scenario_json: dict,
    orig_result: dict,
    grounded_result: dict | None,
    grounded_agents: list[dict],
    out_dir: Path,
):
    """Generate all charts for a scenario (both baseline and grounded)."""
    short = sid.split("-", 2)[-1].replace("_", " ")[:25]

    # Baseline charts
    orig_traj = np.array(orig_result["trajectories"])
    plot_sentiment_heatmap(
        orig_traj, scenario_json, [], False,
        f"Sentiment Heatmap — {short} (Baseline)",
        out_dir / "heatmap_baseline.png",
    )
    plot_coalition_flow(
        orig_traj, scenario_json, [], False,
        f"{short} (Baseline)",
        out_dir / "coalition_flow_baseline.png",
    )
    plot_agent_trajectories(
        orig_traj, scenario_json, [], False,
        f"Agent Trajectories — {short} (Baseline)",
        out_dir / "trajectories_baseline.png",
    )

    # Grounded charts
    if grounded_result is not None:
        gnd_traj = np.array(grounded_result["trajectories"])
        plot_sentiment_heatmap(
            gnd_traj, scenario_json, grounded_agents, True,
            f"Sentiment Heatmap — {short} (Grounded)",
            out_dir / "heatmap_grounded.png",
        )
        plot_coalition_flow(
            gnd_traj, scenario_json, grounded_agents, True,
            f"{short} (Grounded)",
            out_dir / "coalition_flow_grounded.png",
        )
        plot_agent_trajectories(
            gnd_traj, scenario_json, grounded_agents, True,
            f"Agent Trajectories — {short} (Grounded)",
            out_dir / "trajectories_grounded.png",
        )


# ── Main PoC Pipeline ──────────────────────────────────────────────────

def run_poc():
    print("=" * 70)
    print("GROUNDING PoC — Impact on 3 Scenarios")
    print("=" * 70)

    # Step 0: Setup
    print("\n── Step 0: Setup ──")
    llm_fn = setup_llm_fn()
    search_fn = setup_search_fn(llm_fn)

    # Scenario configs
    scenarios = {
        "CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW": {
            "topic": "Volkswagen emissions scandal (Dieselgate) — public trust in VW",
            "domain": "corporate",
            "country": "Germany/USA/EU",
            "timeframe": "September 2015 - March 2016",
        },
        "FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI": {
            "topic": "Silicon Valley Bank collapse — public confidence in US banking",
            "domain": "financial",
            "country": "USA",
            "timeframe": "March 2023",
        },
        "POL-2016-BREXIT": {
            "topic": "2016 Brexit referendum — UK EU membership vote",
            "domain": "political",
            "country": "United Kingdom",
            "timeframe": "January 2016 - June 2016",
        },
    }

    results = {}

    for sid, info in scenarios.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {sid}")
        print(f"{'='*70}")

        scenario_path = SCENARIO_FILES[sid]
        with open(scenario_path) as f:
            scenario_json = json.load(f)

        gt_pct = scenario_json["ground_truth_outcome"]["pro_pct"]
        n_rounds = scenario_json["n_rounds"]
        domain = scenario_json["domain"]

        # Load params for this domain
        params = load_domain_params(domain)

        # ── Step 1: Ground ──
        print(f"\n── Step 1: Grounding {sid} ──")
        out_dir = OUTPUT_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            fact_sheet, grounded_agents, grounded_events = ground_scenario(
                topic=info["topic"],
                domain=domain,
                country=info["country"],
                timeframe=info["timeframe"],
                scenario_id=sid,
                n_elite=5,
                n_rounds=n_rounds,
                llm_fn=llm_fn,
                search_fn=search_fn,
            )

            # Save intermediates
            with open(out_dir / "fact_sheet.json", "w") as f:
                json.dump(fact_sheet.to_dict(), f, indent=2, default=str)
            with open(out_dir / "agents.json", "w") as f:
                json.dump(grounded_agents, f, indent=2)
            with open(out_dir / "events.json", "w") as f:
                json.dump(grounded_events, f, indent=2)
            with open(out_dir / "llm_context.txt", "w") as f:
                f.write(fact_sheet.to_llm_context())

            print(f"  FactSheet: {len(fact_sheet.events)} events, quality={fact_sheet.quality_score}")
            print(f"  Grounded agents: {len(grounded_agents)}")
            for a in grounded_agents:
                print(f"    {a['name']} ({a['role'][:40]}) pos={a['position']:+.2f} inf={a['influence']:.2f}")
            print(f"  Grounded events: {len(grounded_events)}")

            # Compare agents
            orig_elite = [a for a in scenario_json["agents"] if a.get("type") == "elite"]
            print(f"\n  Original elite agents:")
            for a in orig_elite:
                print(f"    {a.get('name','?')} pos={a['initial_position']:+.2f}")
            print(f"  Grounded elite agents:")
            for a in grounded_agents:
                print(f"    {a['name']} pos={a['position']:+.2f} [stake={a.get('_stake','?')}]")

            grounding_ok = True

        except Exception as e:
            print(f"  ⚠ GROUNDING FAILED: {e}")
            grounding_ok = False
            fact_sheet = None
            grounded_agents = []
            grounded_events = []

        # ── Step 2: Simulate with original inputs (baseline) ──
        print(f"\n── Step 2: Baseline simulation ──")
        orig_result = None
        try:
            orig_data = build_scenario_data_from_json(scenario_json, seed=42)
            orig_result = simulate_scenario(params, orig_data)
            orig_pct = float(orig_result["final_pro_pct"])
            print(f"  Original sim: {orig_pct:.1f}% (GT={gt_pct}%, error={abs(orig_pct-gt_pct):.1f}pp)")
        except Exception as e:
            print(f"  ⚠ BASELINE FAILED: {e}")
            orig_pct = None

        # ── Step 3: Simulate with grounded inputs ──
        print(f"\n── Step 3: Grounded simulation ──")
        grounded_result = None
        if grounding_ok and grounded_agents:
            # If no grounded events, fall back to original events
            events_to_use = grounded_events if grounded_events else scenario_json.get("events", [])
            if not grounded_events:
                print(f"  (Using original events — grounded events empty)")
            try:
                grounded_data = build_grounded_scenario_data(
                    scenario_json, grounded_agents, events_to_use, seed=42,
                )
                grounded_result = simulate_scenario(params, grounded_data)
                grounded_pct = float(grounded_result["final_pro_pct"])
                print(f"  Grounded sim: {grounded_pct:.1f}% (GT={gt_pct}%, error={abs(grounded_pct-gt_pct):.1f}pp)")
            except Exception as e:
                print(f"  ⚠ GROUNDED SIM FAILED: {e}")
                grounded_pct = None
        else:
            grounded_pct = None
            print(f"  Skipped (no grounded agents)")

        # ── Generate charts ──
        print(f"\n── Charts ──")
        if orig_result is not None:
            try:
                generate_all_charts(
                    sid, scenario_json, orig_result, grounded_result,
                    grounded_agents, out_dir,
                )
            except Exception as e:
                print(f"  ⚠ Chart generation failed: {e}")

        # Store results
        results[sid] = {
            "gt_pct": gt_pct,
            "orig_pct": orig_pct,
            "grounded_pct": grounded_pct,
            "n_grounded_agents": len(grounded_agents),
            "n_grounded_events": len(grounded_events),
            "quality_score": fact_sheet.quality_score if fact_sheet else 0,
            "grounding_ok": grounding_ok,
        }

    # ── Step 4: Comparison table ──
    print(f"\n{'='*70}")
    print("STEP 4: COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"\n{'Scenario':<20} {'GT%':>6} {'Orig%':>7} {'Ground%':>8} {'Err_O':>6} {'Err_G':>6} {'Δ':>6} {'Improv':>8}")
    print("-" * 75)

    deltas = []
    for sid, r in results.items():
        short = sid.split("-")[-1][:15]
        gt = r["gt_pct"]
        orig = r["orig_pct"]
        grounded = r["grounded_pct"]

        if orig is not None and grounded is not None:
            err_o = abs(orig - gt)
            err_g = abs(grounded - gt)
            delta = err_o - err_g  # positive = improvement
            improv = f"{delta/err_o*100:+.0f}%" if err_o > 0 else "N/A"
            deltas.append(delta)

            # b_s implicit
            bs_orig = logit(gt) - logit(orig)
            bs_grounded = logit(gt) - logit(grounded)

            print(f"{short:<20} {gt:>6.1f} {orig:>7.1f} {grounded:>8.1f} {err_o:>6.1f} {err_g:>6.1f} {delta:>+6.1f} {improv:>8}")
            print(f"  b_s: original={bs_orig:+.3f}  grounded={bs_grounded:+.3f}  |Δb_s|={abs(bs_orig)-abs(bs_grounded):+.3f}")
        else:
            print(f"{short:<20} {gt:>6.1f} {'FAIL':>7} {'FAIL':>8}")

    # ── Verdict ──
    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        print(f"\nAverage Δ error: {avg_delta:+.1f} pp")

        if avg_delta > 5:
            verdict = "STRONG SIGNAL — full recalibration justified"
        elif avg_delta > 2:
            verdict = "MODERATE SIGNAL — consider targeted recalibration on worst domains"
        elif avg_delta > 0:
            verdict = "WEAK SIGNAL — grounding alone insufficient, root cause is elsewhere"
        else:
            verdict = "NEGATIVE — grounding introduces noise, investigate"
        print(f"Verdict: {verdict}")
    else:
        verdict = "INCOMPLETE — not enough data"
        avg_delta = 0

    # ── Step 5: Qualitative analysis (Dieselgate) ──
    print(f"\n{'='*70}")
    print("STEP 5: QUALITATIVE ANALYSIS — DIESELGATE")
    print(f"{'='*70}")

    diesel_sid = "CORP-2015-DIESELGATE_PUBLIC_TRUST_IN_VW"
    diesel_dir = OUTPUT_DIR / diesel_sid

    with open(SCENARIO_FILES[diesel_sid]) as f:
        diesel_json = json.load(f)

    print("\nAGENTI ELITE ORIGINALI:")
    for a in diesel_json["agents"]:
        if a["type"] == "elite":
            print(f"  {a.get('name','?')} — pos={a['initial_position']:+.2f} inf={a.get('influence',0):.2f}")

    if (diesel_dir / "agents.json").exists():
        with open(diesel_dir / "agents.json") as f:
            diesel_grounded = json.load(f)
        print("\nAGENTI ELITE GROUNDED:")
        for a in diesel_grounded:
            print(f"  {a['name']} — {a['role'][:50]}")
            print(f"    pos={a['position']:+.2f} inf={a['influence']:.2f} stake={a.get('_stake','?')} rel={a.get('_relevance',0):.2f}")

    print("\nEVENTI ORIGINALI:")
    for e in diesel_json.get("events", []):
        print(f"  R{e['round']}: mag={e.get('shock_magnitude',0):.2f} dir={e.get('shock_direction',0):+.1f}")
        print(f"       {e['description'][:80]}")

    if (diesel_dir / "events.json").exists():
        with open(diesel_dir / "events.json") as f:
            diesel_gevents = json.load(f)
        print("\nEVENTI GROUNDED:")
        for e in diesel_gevents:
            print(f"  R{e['round']}: mag={e.get('shock_magnitude',0):.3f} dir={e.get('shock_direction',0):+d}")
            print(f"       {e['description'][:80]}")
            if e.get("source"):
                print(f"       source: {e['source'][:60]}")

    if (diesel_dir / "fact_sheet.json").exists():
        with open(diesel_dir / "fact_sheet.json") as f:
            diesel_fs = json.load(f)
        if diesel_fs.get("key_figures"):
            print("\nFACT SHEET KEY FIGURES:")
            for k, v in diesel_fs["key_figures"].items():
                print(f"  {k}: {v}")

    # ── Step 6: Save report ──
    report_lines = [
        "# Grounding PoC Report",
        "",
        "## 1. Setup",
        "",
        "- **LLM**: Google Gemini (gemini-3.1-flash-lite-preview), temperature=0.2",
        "- **Search**: DuckDuckGo HTML scraping (real web search)",
        "- **Posteriors**: v2 domain-specific means (NOT recalibrated)",
        "- **Scenarios**: 3 (Dieselgate, SVB, Brexit)",
        "",
        "## 2. Comparison Table",
        "",
        "| Scenario | GT% | Orig% | Ground% | Err_O | Err_G | Δ (pp) | Improvement |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for sid, r in results.items():
        short = sid.split("-", 2)[-1].replace("_", " ")[:30]
        gt = r["gt_pct"]
        orig = r["orig_pct"]
        grounded = r["grounded_pct"]
        if orig is not None and grounded is not None:
            err_o = abs(orig - gt)
            err_g = abs(grounded - gt)
            delta = err_o - err_g
            improv = f"{delta/err_o*100:+.0f}%" if err_o > 0 else "N/A"
            bs_o = logit(gt) - logit(orig)
            bs_g = logit(gt) - logit(grounded)
            report_lines.append(
                f"| {short} | {gt:.1f} | {orig:.1f} | {grounded:.1f} | {err_o:.1f} | {err_g:.1f} | {delta:+.1f} | {improv} |"
            )
        else:
            report_lines.append(f"| {short} | {gt:.1f} | FAIL | FAIL | - | - | - | - |")

    report_lines.extend([
        "",
        "### Implicit discrepancy (b_s)",
        "",
        "| Scenario | b_s original | b_s grounded | |Δb_s| |",
        "|---|---|---|---|",
    ])
    for sid, r in results.items():
        short = sid.split("-", 2)[-1].replace("_", " ")[:30]
        gt = r["gt_pct"]
        orig = r["orig_pct"]
        grounded = r["grounded_pct"]
        if orig is not None and grounded is not None:
            bs_o = logit(gt) - logit(orig)
            bs_g = logit(gt) - logit(grounded)
            report_lines.append(
                f"| {short} | {bs_o:+.3f} | {bs_g:+.3f} | {abs(bs_o)-abs(bs_g):+.3f} |"
            )

    report_lines.extend([
        "",
        f"## 3. Verdict",
        "",
        f"**Average Δ error: {avg_delta:+.1f} pp**",
        "",
        f"**{verdict}**",
        "",
        "## 4. Qualitative Notes",
        "",
    ])

    # Add qualitative notes
    diesel_r = results.get(diesel_sid, {})
    if diesel_r.get("grounding_ok"):
        report_lines.extend([
            "### Dieselgate",
            f"- Original elite: 2 agents (Winterkorn + generic 'Financial Analysts')",
            f"- Grounded elite: {diesel_r['n_grounded_agents']} agents (scenario-specific stakeholders)",
            f"- Grounded events: {diesel_r['n_grounded_events']} (from {diesel_r['quality_score']}/100 FactSheet)",
            "",
        ])

    report_lines.extend([
        "## 5. Recommendation",
        "",
        "Based on these results:",
        "",
    ])

    if avg_delta > 2:
        report_lines.extend([
            "- **Recalibrate**: Run full SVI with grounded inputs on all 42 scenarios",
            "- **Priority**: Financial and corporate domains (highest b_s reduction)",
            "- **Expected outcome**: Lower σ_b,within as simulator needs less discrepancy correction",
        ])
    elif avg_delta > 0:
        report_lines.extend([
            "- **Targeted**: Apply grounding to worst-performing domains (financial, corporate)",
            "- **Monitor**: Brexit-like scenarios should not degrade",
            "- **Root cause**: σ_b,within=0.558 may be partly structural, not just input quality",
        ])
    else:
        report_lines.extend([
            "- **Investigate**: Grounding may be introducing noise via search quality",
            "- **Try**: Use higher-quality search (Google API, Perplexity) before dismissing",
            "- **Root cause**: Error likely structural (model misspecification), not input quality",
        ])

    report_lines.extend([
        "",
        "---",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
    ])

    report_path = OUTPUT_DIR / "poc_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    run_poc()
