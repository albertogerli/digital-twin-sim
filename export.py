#!/usr/bin/env python3
"""Export simulation results to frontend-compatible JSON.

Universal exporter: works with ANY domain plugin output.
Converts checkpoint JSON + SQLite database into replay_meta.json
and replay_round_N.json files.

Usage:
    python export.py --scenario "Luxury_Leather_Transition_Debate"
    python export.py --scenario "Luxury_Leather_Transition_Debate" --output-dir exports/
"""

import argparse
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUTS = os.path.join(BASE_DIR, "outputs")

AVATAR_COLORS = [
    "#22c55e", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
    "#14b8a6", "#e11d48", "#0ea5e9", "#a855f7", "#10b981",
]

COALITION_COLORS = ["#1e40af", "#059669", "#dc2626", "#f59e0b", "#8b5cf6",
                    "#ec4899", "#06b6d4", "#84cc16"]


def agent_to_handle(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return f"@{parts[0][0]}{parts[-1]}".replace(" ", "")
    return f"@{name.replace(' ', '')}"


def extract_hashtags(text: str) -> list[str]:
    return re.findall(r"#\w+", text)


# Common stop words to filter when generating synthetic hashtags
_STOP_WORDS = frozenset(
    "il lo la le gli i un una uno di da in con su per tra fra del dello della "
    "dei degli delle al allo alla ai agli alle che è e o non si ma ho ha come "
    "se ci più anche già molto questo quella questi quelle sono nel nella nei "
    "the a an and or but is are was were to of for in on at by with from its "
    "it be as this that which who what not can do does did".split()
)


def extract_keywords_as_hashtags(text: str, top_n: int = 3) -> list[str]:
    """Extract significant words from text and convert to hashtags."""
    words = re.findall(r"[A-Za-zÀ-ÿ]{4,}", text)
    filtered = [w for w in words if w.lower() not in _STOP_WORDS and not w.isupper()]
    # Prefer capitalized words (proper nouns, concepts)
    capitalized = [w for w in filtered if w[0].isupper()]
    if capitalized:
        filtered = capitalized
    counts = Counter(filtered)
    return [f"#{w}" for w, _ in counts.most_common(top_n)]


# ── Data loading ──────────────────────────────────────────────

def discover_checkpoints(outputs_dir: str, scenario: str) -> list[str]:
    """Find all checkpoint files for a scenario, sorted by round."""
    files = []
    for f in os.listdir(outputs_dir):
        if f.startswith(f"state_{scenario}_r") and f.endswith(".json"):
            files.append(f)
    files.sort(key=lambda f: int(re.search(r"_r(\d+)\.json", f).group(1)))
    return files


def load_checkpoint(outputs_dir: str, scenario: str, round_num: int) -> dict:
    path = os.path.join(outputs_dir, f"state_{scenario}_r{round_num}.json")
    with open(path) as f:
        return json.load(f)


def load_db(outputs_dir: str, scenario: str) -> sqlite3.Connection:
    db_path = os.path.join(outputs_dir, f"social_{scenario}.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_posts_for_round(conn: sqlite3.Connection, round_num: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM posts WHERE round = ? ORDER BY id", (round_num,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_reactions_for_post(conn: sqlite3.Connection, post_id: int) -> dict:
    rows = conn.execute(
        "SELECT reaction_type, COUNT(*) as cnt FROM reactions WHERE post_id = ? GROUP BY reaction_type",
        (post_id,)
    ).fetchall()
    result = {"like": 0, "repost": 0, "reply": 0}
    for r in rows:
        result[r["reaction_type"]] = r["cnt"]
    return result


def _infer_domain(checkpoints: list[dict], scenario: str) -> str:
    """Infer domain from checkpoint data or scenario name keywords."""
    # 1. Check if checkpoint has domain field (new format)
    if checkpoints:
        domain = checkpoints[0].get("domain", "")
        if domain:
            return domain

    # 2. Infer from scenario name keywords
    name = scenario.lower()
    keyword_map = {
        "sport": ["ronaldo", "azzurri", "figb", "world_cup", "football", "tifosi", "challenge_in_fig"],
        "political": ["referendum", "election", "vote", "campaign", "civic", "padova", "corsa", "city_green"],
        "financial": ["bank", "bitcoin", "stock", "trading", "fund", "reserve"],
        "corporate": ["restructuring", "org", "flat_org", "cree", "newco", "transfer", "appointment",
                       "arianna", "spa", "lighting", "expert", "ai_generative"],
        "commercial": ["iphone", "price", "brand", "luxury", "leather", "fashion"],
        "marketing": ["nike", "deepfake", "ad_backlash"],
        "public_health": ["vaccine", "mrna", "booster", "mandate", "health"],
    }
    for domain, keywords in keyword_map.items():
        if any(kw in name for kw in keywords):
            return domain

    return "unknown"


def load_config(outputs_dir: str, scenario: str) -> dict | None:
    """Try to load the scenario YAML config if saved."""
    configs_dir = os.path.join(BASE_DIR, "configs")
    for f in os.listdir(configs_dir) if os.path.isdir(configs_dir) else []:
        if f.endswith(".yaml") or f.endswith(".yml"):
            import yaml
            path = os.path.join(configs_dir, f)
            with open(path) as fh:
                cfg = yaml.safe_load(fh)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cfg.get("name", ""))
            if safe == scenario:
                return cfg
    return None


# ── Agent name/info map ───────────────────────────────────────

def build_agent_name_map(checkpoints: list[dict]) -> dict:
    names = {}
    cp = checkpoints[0]
    for a in cp.get("elite_agents", []):
        names[a["id"]] = {
            "name": a.get("name", a["id"]),
            "role": a.get("role", ""),
            "tier": 1,
            "archetype": a.get("archetype", "unknown"),
            "influence": a.get("influence", 0.7),
        }
    for a in cp.get("institutional_agents", []):
        names[a["id"]] = {
            "name": a.get("name", a["id"]),
            "role": a.get("role", ""),
            "tier": 2,
            "archetype": a.get("archetype", "unknown"),
            "influence": a.get("influence", 0.4),
        }
    for c in cp.get("citizen_clusters", []):
        names[c["id"]] = {
            "name": c.get("name", c["id"]),
            "role": "Citizen cluster",
            "tier": 3,
            "archetype": "citizen",
            "size": c.get("size", 20),
            "influence": 0.3,
        }
    return names


# ── Meta builder ──────────────────────────────────────────────

def build_meta(scenario: str, checkpoints: list[dict], name_map: dict,
               config: dict | None = None) -> dict:
    agents = []
    color_idx = 0

    # All agents from first checkpoint (elites + institutional)
    all_agents_cp = (checkpoints[0].get("elite_agents", []) +
                     checkpoints[0].get("institutional_agents", []))
    pos_map = {a["id"]: a.get("position", 0) for a in all_agents_cp}
    inf_map = {a["id"]: a.get("influence", 0.5) for a in all_agents_cp}

    for aid, info in name_map.items():
        if info["tier"] == 3:
            continue
        agents.append({
            "id": aid,
            "name": info["name"],
            "role": info["role"],
            "handle": agent_to_handle(info["name"]),
            "avatarColor": AVATAR_COLORS[color_idx % len(AVATAR_COLORS)],
            "position": round(pos_map.get(aid, 0), 3),
            "influence": round(inf_map.get(aid, info.get("influence", 0.5)), 2),
            "archetype": info.get("archetype", "unknown"),
            "tier": info["tier"],
        })
        color_idx += 1

    # Build scenario title
    title = scenario.replace("_", " ")
    if config:
        title = config.get("name", title)

    return {
        "scenario": scenario,
        "title": title,
        "totalRounds": len(checkpoints),
        "totalAgents": len(agents),
        "agents": agents,
    }


# ── Graph snapshot builder ────────────────────────────────────

def build_graph_snapshot(round_num: int, checkpoint: dict,
                         prev_checkpoint: dict | None,
                         name_map: dict, timeline_label: str) -> dict:
    nodes = []
    edges = []

    all_agents = checkpoint.get("elite_agents", []) + checkpoint.get("institutional_agents", [])
    prev_agents = {}
    if prev_checkpoint:
        for a in (prev_checkpoint.get("elite_agents", []) +
                  prev_checkpoint.get("institutional_agents", [])):
            prev_agents[a["id"]] = a

    for a in all_agents:
        info = name_map.get(a["id"], {})
        prev_pos = prev_agents.get(a["id"], {}).get("position", a["position"])
        delta = round(a["position"] - prev_pos, 3)

        nodes.append({
            "id": a["id"],
            "name": info.get("name", a["id"]),
            "type": "persona",
            "description": info.get("role", ""),
            "power_level": round(a.get("influence", 0.5), 2),
            "position": round(a["position"], 3),
            "position_history": [round(a["position"], 3)],
            "delta": delta,
            "sentiment": a.get("emotional_state", "neutral"),
            "category": info.get("archetype", "unknown"),
        })

    # Cluster nodes
    clusters = checkpoint.get("citizen_clusters", [])
    prev_clusters = {}
    if prev_checkpoint:
        for c in prev_checkpoint.get("citizen_clusters", []):
            prev_clusters[c["id"]] = c

    for c in clusters:
        prev_pos = prev_clusters.get(c["id"], {}).get("position", c["position"])
        delta = round(c["position"] - prev_pos, 3)
        nodes.append({
            "id": c["id"],
            "name": c.get("name", c["id"]),
            "type": "cluster",
            "description": f"Cluster: {c.get('name', c['id'])}",
            "power_level": 0.5,
            "position": round(c["position"], 3),
            "position_history": [round(c["position"], 3)],
            "delta": delta,
            "sentiment": c.get("dominant_sentiment", "neutral"),
            "category": "public_opinion",
            "clusterSize": c.get("size", 20),
        })

    # Agent-agent edges (similar positions)
    for i, a1 in enumerate(all_agents):
        for a2 in all_agents[i+1:]:
            dist = abs(a1["position"] - a2["position"])
            if dist < 0.4 and random.random() < 0.3:
                edges.append({
                    "source": a1["id"],
                    "target": a2["id"],
                    "weight": round(1 - dist, 2),
                    "type": "influence",
                })

    # Cluster-agent edges
    for c in clusters:
        dists = [(a["id"], abs(c["position"] - a["position"])) for a in all_agents]
        dists.sort(key=lambda x: x[1])
        for aid, d in dists[:4]:
            edges.append({
                "source": c["id"],
                "target": aid,
                "weight": round(max(0.1, 1 - d * 2), 2),
                "type": "cluster_influence",
            })

    return {
        "round": round_num,
        "month": timeline_label,
        "event_label": timeline_label,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "avg_position": round(sum(n["position"] for n in nodes) / max(len(nodes), 1), 3),
        },
    }


# ── Indicators ────────────────────────────────────────────────

_prev_hashtag_counts: dict[str, int] = {}  # module-level for trend tracking


def compute_indicators(conn: sqlite3.Connection, round_num: int,
                       checkpoint: dict) -> dict:
    global _prev_hashtag_counts
    posts = get_posts_for_round(conn, round_num)

    # Hashtags — extract explicit #tags first, fall back to keyword extraction
    all_tags: list[str] = []
    for p in posts:
        explicit = extract_hashtags(p["content"])
        if explicit:
            all_tags.extend(explicit)
        else:
            all_tags.extend(extract_keywords_as_hashtags(p["content"], top_n=2))

    tag_counts = Counter(all_tags).most_common(10)

    # Compute trend for each hashtag
    hashtags_with_trend = []
    for tag, count in tag_counts:
        prev = _prev_hashtag_counts.get(tag, 0)
        if prev == 0:
            trend = "new"
        elif count > prev:
            trend = "up"
        else:
            trend = "down"
        hashtags_with_trend.append({"tag": tag, "count": count, "trend": trend})

    # Store for next round comparison
    _prev_hashtag_counts = dict(tag_counts)

    # Polarization
    all_positions = []
    for a in checkpoint.get("elite_agents", []):
        all_positions.append(a["position"])
    for a in checkpoint.get("institutional_agents", []):
        all_positions.append(a["position"])
    for c in checkpoint.get("citizen_clusters", []):
        all_positions.append(c["position"])

    if all_positions:
        mean_pos = sum(all_positions) / len(all_positions)
        variance = sum((p - mean_pos) ** 2 for p in all_positions) / len(all_positions)
        polarization = min(9, max(1, 1 + variance * 10))
    else:
        polarization = 5.5

    # Sentiment from agent emotional states
    sentiments = [a.get("emotional_state", "neutral") for a in checkpoint.get("elite_agents", [])]
    sentiments += [a.get("emotional_state", "neutral") for a in checkpoint.get("institutional_agents", [])]
    sent_counter = Counter(sentiments)
    total_s = max(sum(sent_counter.values()), 1)

    positive_keys = {"satisfied", "triumphant", "optimistic", "hopeful"}
    negative_keys = {"furious", "combative", "worried", "angry"}
    pos_ratio = sum(sent_counter.get(k, 0) for k in positive_keys) / total_s
    neg_ratio = sum(sent_counter.get(k, 0) for k in negative_keys) / total_s
    neu_ratio = max(0, 1 - pos_ratio - neg_ratio)

    return {
        "polarization": round(polarization, 1),
        "engagement": len(posts),
        "sentiment": {
            "positive": round(pos_ratio, 2),
            "neutral": round(neu_ratio, 2),
            "negative": round(neg_ratio, 2),
        },
        "trendingHashtags": hashtags_with_trend,
    }


# ── Coalitions ────────────────────────────────────────────────

def build_coalitions(checkpoint: dict) -> dict:
    coalitions_raw = []
    if checkpoint.get("coalition_history"):
        coalitions_raw = checkpoint["coalition_history"][-1].get("coalitions", [])

    coalitions = []
    for i, c in enumerate(coalitions_raw):
        members = c.get("member_ids", c.get("members", []))
        if not members:
            members = c.get("top_members", [])

        coalitions.append({
            "label": c.get("label", f"Coalition {i+1}"),
            "color": COALITION_COLORS[i % len(COALITION_COLORS)],
            "members": members,
            "avg_position": c.get("avg_position", 0),
            "size": c.get("size", len(members)),
        })

    return {"coalitions": coalitions}


# ── Post impacts ──────────────────────────────────────────────

def compute_post_impacts(posts_data: list[dict], checkpoint: dict,
                         name_map: dict) -> list[dict]:
    impacts = []
    all_agents = checkpoint.get("elite_agents", []) + checkpoint.get("institutional_agents", [])

    for post in posts_data[:50]:
        author_id = post["author_id"]
        author_pos = 0
        for a in all_agents:
            if a["id"] == author_id:
                author_pos = a["position"]
                break

        influenced = []
        edge_effects = []
        for a in random.sample(all_agents, min(6, len(all_agents))):
            if a["id"] == author_id:
                continue
            dist = abs(a["position"] - author_pos)
            if dist < 0.6:
                shift = round(random.uniform(-0.03, 0.03), 3)
                info = name_map.get(a["id"], {})
                influenced.append({
                    "agentId": a["id"],
                    "agentName": info.get("name", a["id"]),
                    "positionBefore": round(a["position"], 3),
                    "positionAfter": round(a["position"] + shift, 3),
                    "shift": shift,
                })
                edge_effects.append({
                    "source": author_id,
                    "target": a["id"],
                    "weightDelta": round(random.uniform(-0.05, 0.05), 3),
                })

        if influenced:
            total_shift = sum(i["shift"] for i in influenced)
            impacts.append({
                "postId": post["frontend_id"],
                "authorId": author_id,
                "influencedAgents": influenced,
                "reach": len(influenced),
                "aggregateShift": round(total_shift, 3),
                "edgeEffects": edge_effects,
            })

    return impacts


# ── Real-world effects (domain-generic) ──────────────────────

def compute_real_world_effects(round_num: int, polarization: float,
                               checkpoint: dict) -> dict:
    """Domain-agnostic real-world effects estimation."""
    all_positions = [a["position"] for a in checkpoint.get("elite_agents", [])]
    all_positions += [a["position"] for a in checkpoint.get("institutional_agents", [])]
    all_positions += [c["position"] for c in checkpoint.get("citizen_clusters", [])]

    if all_positions:
        avg_pos = sum(all_positions) / len(all_positions)
        support = sum(1 for p in all_positions if p > 0.2) / len(all_positions)
        opposition = sum(1 for p in all_positions if p < -0.2) / len(all_positions)
    else:
        avg_pos, support, opposition = 0, 0.5, 0.5

    # Generic indicators that work for any domain
    stability = max(20, 70 - (polarization - 5) * 4 - round_num * 1.5 + random.gauss(0, 2))
    tension = min(100, max(10, int(30 + (polarization - 5) * 8 + round_num * 1.5 + random.gauss(0, 3))))
    media_intensity = min(100, max(20, int(50 + (polarization - 5) * 5 + random.gauss(0, 5))))
    confidence = max(20, int(60 - (polarization - 5) * 3 + avg_pos * 10 + random.gauss(0, 3)))

    return {
        "overview": {
            "stability_index": round(stability),
            "tension_index": tension,
            "media_intensity": media_intensity,
            "stakeholder_confidence": confidence,
        },
        "opinion": {
            "support_rate": round(support * 100),
            "opposition_rate": round(opposition * 100),
            "undecided_rate": round((1 - support - opposition) * 100),
            "avg_position": round(avg_pos, 3),
        },
        "engagement": {
            "active_discussions": int(max(0, 50 + (polarization - 5) * 10 + random.gauss(0, 5))),
            "viral_content_pieces": int(max(0, 5 + polarization * 1.5 + random.gauss(0, 2))),
            "coalition_count": len(checkpoint.get("coalition_history", [{}])[-1].get("coalitions", []))
                if checkpoint.get("coalition_history") else 0,
        },
    }


# ── Round data builder ────────────────────────────────────────

def build_round_data(scenario: str, round_num: int, conn: sqlite3.Connection,
                     checkpoint: dict, prev_checkpoint: dict | None,
                     name_map: dict, timeline_label: str,
                     event_data: dict | None = None) -> dict:
    # Event
    if event_data:
        event_obj = {
            "event": event_data.get("event", event_data.get("description", "")),
            "shock_magnitude": event_data.get("shock_magnitude", 0.3),
            "shock_direction": event_data.get("shock_direction", 0),
        }
        key_insight = event_data.get("institutional_impact", event_data.get("public_perception", ""))
    else:
        event_obj = {"event": f"Round {round_num} dynamics continue", "shock_magnitude": 0.3, "shock_direction": 0}
        key_insight = ""

    # Posts from DB
    raw_posts = get_posts_for_round(conn, round_num)

    posts = []
    for i, p in enumerate(raw_posts):
        reactions = get_reactions_for_post(conn, p["id"])
        likes = reactions["like"]
        reposts = reactions["repost"]
        replies = reactions["reply"]
        total_eng = likes + reposts * 2 + replies * 3

        info = name_map.get(p["author_id"], {})
        frontend_id = f"scenario_{scenario}_rp_{i:05d}_r{round_num}"

        posts.append({
            "id": frontend_id,
            "frontend_id": frontend_id,
            "author_id": p["author_id"],
            "author_name": info.get("name", p["author_id"].replace("_", " ").title()),
            "author_role": info.get("role", ""),
            "tier": p.get("author_tier", info.get("tier", 1)),
            "platform": p["platform"],
            "text": p["content"],
            "round": round_num,
            "likes": likes,
            "reposts": reposts,
            "replies": replies,
            "engagement_score": round(total_eng / max(1, max(likes, 1) * 3), 2),
            "virality_tier": 3 if total_eng > 20 else (2 if total_eng > 5 else 1),
        })

    # Sort by engagement
    posts.sort(key=lambda p: p["likes"] + p["reposts"] * 2 + p["replies"] * 3, reverse=True)

    # Clean internal field
    for p in posts:
        p.pop("frontend_id", None)

    # Re-assign IDs post-sort
    for i, p in enumerate(posts):
        p["id"] = f"scenario_{scenario}_rp_{i:05d}_r{round_num}"

    # Build sections
    indicators = compute_indicators(conn, round_num, checkpoint)
    graph = build_graph_snapshot(round_num, checkpoint, prev_checkpoint, name_map, timeline_label)
    coalitions = build_coalitions(checkpoint)
    real_world = compute_real_world_effects(round_num, indicators["polarization"], checkpoint)

    # Post impacts
    posts_for_impact = [{"frontend_id": p["id"], "author_id": p["author_id"]} for p in posts[:50]]
    post_impacts = compute_post_impacts(posts_for_impact, checkpoint, name_map)

    result = {
        "round": round_num,
        "month": timeline_label,
        "event": event_obj,
        "posts": posts,
        "graphSnapshot": graph,
        "indicators": indicators,
        "coalitions": coalitions,
        "key_insight": key_insight or f"Round {round_num} analysis",
        "postImpacts": post_impacts,
        "realWorldEffects": real_world,
    }

    # v2/v3 calibration data (backward-compatible — only present if checkpoint has it)
    if checkpoint.get("confidence_interval"):
        result["confidence_interval"] = checkpoint["confidence_interval"]
    if checkpoint.get("regime_info"):
        result["regime_info"] = checkpoint["regime_info"]

    return result


# ── Editorial export (agents, polarization, etc.) ─────────────

def build_editorial_data(scenario: str, checkpoints: list[dict],
                         name_map: dict, conn: sqlite3.Connection,
                         config: dict | None = None) -> dict:
    """Build editorial-style JSON files for the frontend."""
    num_rounds = len(checkpoints)
    title = scenario.replace("_", " ")
    if config:
        title = config.get("name", title)

    # metadata.json
    # Extract calibration info from last checkpoint (v2+ data, backward-compatible)
    last_cp = checkpoints[-1]
    calibration_meta = {}
    if last_cp.get("params_used"):
        pu = last_cp["params_used"]
        calibration_meta["model_version"] = pu.get("_model_version", "v1")
        calibration_meta["params_source"] = pu.get("_source", "unknown")
        calibration_meta["params"] = {
            k: v for k, v in pu.items() if not k.startswith("_")
        }
    if last_cp.get("confidence_interval"):
        calibration_meta["final_confidence_interval"] = last_cp["confidence_interval"]
    if last_cp.get("regime_info"):
        calibration_meta["final_regime_info"] = last_cp["regime_info"]

    metadata = {
        "scenario_id": scenario,
        "scenario_name": title,
        "num_rounds": num_rounds,
        "domain": config.get("domain") if config and config.get("domain") else _infer_domain(checkpoints, scenario),
        "description": config.get("description", "") if config else "",
    }
    if calibration_meta:
        metadata["calibration"] = calibration_meta

    # agents.json
    agents = []
    cp0 = checkpoints[0]
    cp_last = checkpoints[-1]
    for a in cp0.get("elite_agents", []):
        final = next((x for x in cp_last.get("elite_agents", []) if x["id"] == a["id"]), a)
        agents.append({
            "id": a["id"],
            "name": a.get("name", a["id"]),
            "role": a.get("role", ""),
            "archetype": a.get("archetype", "unknown"),
            "tier": 1,
            "initial_position": round(a.get("original_position", a["position"]), 3),
            "final_position": round(final["position"], 3),
            "position_delta": round(final["position"] - a.get("original_position", a["position"]), 3),
            "influence": a.get("influence", 0.5),
            "emotional_state": final.get("emotional_state", "neutral"),
        })

    # polarization.json — per-round polarization curve
    polarization_data = []
    for i, cp in enumerate(checkpoints):
        all_pos = [a["position"] for a in cp.get("elite_agents", [])]
        all_pos += [a["position"] for a in cp.get("institutional_agents", [])]
        all_pos += [c["position"] for c in cp.get("citizen_clusters", [])]
        if all_pos:
            mean_p = sum(all_pos) / len(all_pos)
            var = sum((p - mean_p) ** 2 for p in all_pos) / len(all_pos)
            pol = min(9, max(1, 1 + var * 10))
        else:
            pol = 5
        polarization_data.append({
            "round": i + 1,
            "polarization": round(pol, 2),
            "avg_position": round(mean_p if all_pos else 0, 3),
            "num_agents": len(all_pos),
        })

    # coalitions.json — per-round coalition data
    coalitions_data = []
    for i, cp in enumerate(checkpoints):
        coalitions_data.append({
            "round": i + 1,
            **build_coalitions(cp),
        })

    # evolving_graph.json — per-round graph snapshots
    graph_snapshots = []
    for i, cp in enumerate(checkpoints):
        prev = checkpoints[i - 1] if i > 0 else None
        label = f"Round {i + 1}"
        if config and config.get("timeline_labels"):
            labels = config["timeline_labels"]
            if i < len(labels):
                label = labels[i]
        graph_snapshots.append(
            build_graph_snapshot(i + 1, cp, prev, name_map, label)
        )

    # posts.json — top viral posts across all rounds
    top_posts = []
    for r in range(1, num_rounds + 1):
        raw = get_posts_for_round(conn, r)
        for p in raw:
            reactions = get_reactions_for_post(conn, p["id"])
            total_eng = reactions["like"] + reactions["repost"] * 2 + reactions["reply"] * 3
            info = name_map.get(p["author_id"], {})
            top_posts.append({
                "author_id": p["author_id"],
                "author_name": info.get("name", p["author_id"]),
                "platform": p["platform"],
                "text": p["content"],
                "round": r,
                "likes": reactions["like"],
                "reposts": reactions["repost"],
                "replies": reactions["reply"],
                "total_engagement": total_eng,
            })
    top_posts.sort(key=lambda p: p["total_engagement"], reverse=True)
    top_posts = top_posts[:30]  # Top 30 viral posts

    return {
        "metadata": metadata,
        "agents": agents,
        "polarization": polarization_data,
        "coalitions": coalitions_data,
        "evolving_graph": graph_snapshots,
        "top_posts": top_posts,
    }


# ── Main export function ──────────────────────────────────────

def export_scenario(scenario: str, outputs_dir: str, export_dir: str):
    print(f"\n[{scenario}] Exporting...")

    # Discover rounds
    cp_files = discover_checkpoints(outputs_dir, scenario)
    if not cp_files:
        print(f"  ERROR: No checkpoints found for scenario '{scenario}' in {outputs_dir}")
        return

    num_rounds = len(cp_files)
    print(f"  Found {num_rounds} rounds")

    # Load checkpoints
    checkpoints = []
    for r in range(1, num_rounds + 1):
        checkpoints.append(load_checkpoint(outputs_dir, scenario, r))

    # Load DB
    conn = load_db(outputs_dir, scenario)

    # Load config if available
    config = load_config(outputs_dir, scenario)

    # Build name map
    name_map = build_agent_name_map(checkpoints)

    # Create output directory
    scenario_dir = os.path.join(export_dir, f"scenario_{scenario}")
    os.makedirs(scenario_dir, exist_ok=True)

    # ── Replay export ──
    # Meta
    meta = build_meta(scenario, checkpoints, name_map, config)
    meta_path = os.path.join(scenario_dir, "replay_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Written: replay_meta.json ({meta['totalAgents']} agents)")

    # Per-round
    for r in range(1, num_rounds + 1):
        prev_cp = checkpoints[r - 2] if r > 1 else None

        # Get timeline label
        timeline_label = f"Round {r}"
        if config and config.get("timeline_labels"):
            labels = config["timeline_labels"]
            if r - 1 < len(labels):
                timeline_label = labels[r - 1]

        # Try to get event data from checkpoint
        event_data = None
        if checkpoints[r - 1].get("events"):
            event_data = checkpoints[r - 1]["events"][-1]

        round_data = build_round_data(
            scenario, r, conn, checkpoints[r - 1], prev_cp,
            name_map, timeline_label, event_data,
        )
        round_path = os.path.join(scenario_dir, f"replay_round_{r}.json")
        with open(round_path, "w") as f:
            json.dump(round_data, f, indent=2, ensure_ascii=False)
        print(f"  Written: replay_round_{r}.json ({len(round_data['posts'])} posts)")

    # ── Editorial export ──
    editorial = build_editorial_data(scenario, checkpoints, name_map, conn, config)

    for key, data in editorial.items():
        path = os.path.join(scenario_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if isinstance(data, list):
            print(f"  Written: {key}.json ({len(data)} entries)")
        elif isinstance(data, dict):
            print(f"  Written: {key}.json")

    # Copy report if exists
    report_path = os.path.join(outputs_dir, f"{scenario}_report.md")
    if os.path.exists(report_path):
        import shutil
        dest = os.path.join(scenario_dir, "report.md")
        shutil.copy2(report_path, dest)
        print(f"  Written: report.md")

    conn.close()
    print(f"[{scenario}] Export complete! → {scenario_dir}")


def discover_scenarios(outputs_dir: str) -> list[str]:
    """Auto-discover all scenarios from checkpoint files."""
    scenarios = set()
    for f in os.listdir(outputs_dir):
        m = re.match(r"state_(.+)_r\d+\.json", f)
        if m:
            scenarios.add(m.group(1))
    return sorted(scenarios)


def main():
    parser = argparse.ArgumentParser(description="Export simulation results to frontend JSON")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario name (auto-discovered from outputs if not specified)")
    parser.add_argument("--outputs-dir", type=str, default=DEFAULT_OUTPUTS,
                        help="Directory with simulation outputs")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Export target directory (default: outputs/exports/)")
    parser.add_argument("--all", action="store_true",
                        help="Export all discovered scenarios")
    args = parser.parse_args()

    export_dir = args.export_dir or os.path.join(args.outputs_dir, "exports")

    if args.all or not args.scenario:
        scenarios = discover_scenarios(args.outputs_dir)
        if not scenarios:
            print("No scenarios found in outputs directory.")
            return
        print(f"Discovered {len(scenarios)} scenario(s): {', '.join(scenarios)}")
        for s in scenarios:
            export_scenario(s, args.outputs_dir, export_dir)
    else:
        export_scenario(args.scenario, args.outputs_dir, export_dir)

    print(f"\nAll exports complete! → {export_dir}")


if __name__ == "__main__":
    main()
