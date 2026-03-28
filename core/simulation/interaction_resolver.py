"""Resolves platform interactions: engagement simulation, follow updates, coalition detection."""

import logging
import random
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class InteractionResolver:
    """Handles post-generation platform dynamics."""

    def __init__(self, platform_engine, domain_plugin=None):
        self.platform = platform_engine
        self.domain = domain_plugin
        self.engagement_scales = {
            1: (50, 500),   # Elite
            2: (10, 100),   # Institutional
            3: (2, 20),     # Citizen
        }

    def resolve_round(self, round_num: int, all_agents: list,
                      posts_this_round: list[dict]):
        """Run all interaction resolution phases for a round."""
        self._simulate_engagement(round_num, all_agents, posts_this_round)
        self._update_follow_graph(round_num, all_agents)
        coalitions = self._detect_coalitions(all_agents)
        return coalitions

    def _simulate_engagement(self, round_num: int, all_agents: list,
                              posts_this_round: list[dict]):
        """Simulate likes, reposts, and replies based on position alignment."""
        agent_positions = {a.id: a.position for a in all_agents}
        db_posts = self.platform.get_posts_by_round(round_num)

        for post in db_posts:
            post_id = post["id"]
            author_id = post["author_id"]
            author_tier = post["author_tier"]
            author_pos = agent_positions.get(author_id, 0.0)

            min_eng, max_eng = self.engagement_scales.get(author_tier, (5, 50))

            reacting_agents = random.sample(
                all_agents,
                min(len(all_agents), random.randint(min_eng, max_eng))
            )

            for agent in reacting_agents:
                if agent.id == author_id:
                    continue

                distance = abs(agent.position - author_pos)

                if distance < 0.3:
                    roll = random.random()
                    if roll < 0.70:
                        self.platform.add_reaction(post_id, agent.id, "like", round_num)
                    elif roll < 0.85:
                        self.platform.add_reaction(post_id, agent.id, "repost", round_num)
                elif distance > 0.5:
                    roll = random.random()
                    if roll < 0.20:
                        self.platform.add_reaction(post_id, agent.id, "downvote", round_num)
                else:
                    if random.random() < 0.3:
                        self.platform.add_reaction(post_id, agent.id, "like", round_num)

    def _update_follow_graph(self, round_num: int, all_agents: list):
        """10% of agents add 1-2 new follows toward aligned accounts each round."""
        agents_to_update = random.sample(
            all_agents, max(1, len(all_agents) // 10)
        )

        for agent in agents_to_update:
            candidates = [
                a for a in all_agents
                if a.id != agent.id and abs(a.position - agent.position) < 0.4
            ]
            if not candidates:
                continue

            new_follows = random.sample(candidates, min(2, len(candidates)))
            for followed in new_follows:
                # Use first available channel
                channel = "social"
                if self.domain:
                    channels = self.domain.get_channels()
                    if channels:
                        channel = channels[0].id
                self.platform.add_follow(agent.id, followed.id, channel, round_num)

    def _detect_coalitions(self, all_agents: list, n_clusters: int = 4) -> list[dict]:
        """Detect coalitions using k-means on agent positions."""
        if len(all_agents) < n_clusters:
            return []

        positions = np.array([[a.position] for a in all_agents])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(positions)

        coalitions = []
        used_labels: set[str] = set()

        for i in range(n_clusters):
            members = [all_agents[j] for j in range(len(all_agents)) if labels[j] == i]
            if not members:
                continue

            avg_pos = sum(m.position for m in members) / len(members)

            # Use domain plugin for labeling if available
            if self.domain:
                label = self.domain.label_coalition(avg_pos, members)
            else:
                if avg_pos > 0.3:
                    label = "Supporters"
                elif avg_pos < -0.3:
                    label = "Opponents"
                elif avg_pos > 0:
                    label = "Moderate Supporters"
                else:
                    label = "Moderate Opponents"

            # Deduplicate labels — append position qualifier if collision
            if label in used_labels:
                if avg_pos > 0.15:
                    label = f"{label} (Pro)"
                elif avg_pos < -0.15:
                    label = f"{label} (Anti)"
                elif avg_pos >= 0:
                    label = f"{label} (Leaning Pro)"
                else:
                    label = f"{label} (Leaning Anti)"
            # If still duplicate after qualifier, add numeric suffix
            if label in used_labels:
                suffix = 2
                base = label
                while label in used_labels:
                    label = f"{base} {suffix}"
                    suffix += 1
            used_labels.add(label)

            coalitions.append({
                "id": i,
                "label": label,
                "avg_position": round(avg_pos, 3),
                "size": len(members),
                "members": [m.id for m in members],
                "top_members": [
                    m.id for m in sorted(
                        members, key=lambda x: x.influence, reverse=True
                    )[:5]
                ],
            })

        return coalitions
