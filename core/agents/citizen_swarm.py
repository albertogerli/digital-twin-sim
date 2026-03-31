"""Citizen swarm manager — orchestrates parallel cluster simulation."""

import asyncio
import logging
from collections import Counter
from typing import Optional

from .citizen_cluster import CitizenCluster
from ..llm.base_client import BaseLLMClient
from ..llm.json_parser import JSONParseError

logger = logging.getLogger(__name__)


class CitizenSwarm:
    """Manages all citizen clusters and their simulation."""

    def __init__(self, clusters: list[CitizenCluster]):
        self.clusters = {c.id: c for c in clusters}

    async def simulate_round(
        self,
        llm: BaseLLMClient,
        round_number: int,
        timeline_label: str,
        round_event: str,
        viral_posts: str,
        prompt_template: str,
        channel_map: dict[str, tuple[str, str]],
        channel_descriptions: dict[str, str],
        channel_max_lengths: dict[str, int],
    ) -> list[dict]:
        """Simulate all clusters for a round. One LLM call per cluster."""
        tasks = [
            self._simulate_cluster(
                cluster, llm, round_number, timeline_label, round_event,
                viral_posts, prompt_template, channel_map,
                channel_descriptions, channel_max_lengths,
            )
            for cluster in self.clusters.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Cluster simulation failed: {r}")
            elif r:
                processed.append(r)
        return processed

    async def _simulate_cluster(
        self,
        cluster: CitizenCluster,
        llm: BaseLLMClient,
        round_number: int,
        timeline_label: str,
        round_event: str,
        viral_posts: str,
        prompt_template: str,
        channel_map: dict[str, tuple[str, str]],
        channel_descriptions: dict[str, str],
        channel_max_lengths: dict[str, int],
    ) -> Optional[dict]:
        primary, secondary = channel_map.get(
            cluster.info_channel,
            (list(channel_map.values())[0] if channel_map else ("social", "forum"))
        )
        info_desc = channel_descriptions.get(cluster.info_channel, "General media")

        prompt = prompt_template.format(
            size=cluster.size,
            cluster_description=cluster.get_description(),
            round_number=round_number,
            timeline_label=timeline_label,
            round_event=round_event,
            viral_posts=viral_posts,
            previous_state=cluster.get_previous_state(),
            cluster_id=cluster.id,
            info_channel_desc=info_desc,
            primary_platform=primary,
            secondary_platform=secondary,
        )

        try:
            result = await llm.generate_json(
                prompt=prompt,
                temperature=0.7,
                max_output_tokens=1500,
                component=f"citizen_{cluster.id}",
            )

            # Guard: unwrap list if LLM returned array
            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}

            # Update cluster state with delta cap
            from ..simulation.validators import clamp_position_delta, CLUSTER_DELTA_CAP, normalize_sentiment_distribution
            shift = float(result.get("shift_from_last_month", 0))
            new_pos = cluster.position + shift
            cluster.position = clamp_position_delta(cluster.position, new_pos, CLUSTER_DELTA_CAP)
            cluster.engagement_level = float(
                result.get("engagement_level", cluster.engagement_level)
            )
            cluster.trust_institutions = float(
                result.get("trust_in_institutions", cluster.trust_institutions)
            )
            cluster.dominant_sentiment = result.get(
                "dominant_sentiment", "indifferent"
            )
            cluster.emergent_narrative = result.get("emergent_narrative", "")
            cluster.key_concerns = result.get("key_concerns", [])

            dist = result.get("sentiment_distribution", {})
            if dist:
                cluster.sentiment_distribution = normalize_sentiment_distribution(dist)

            # Extract sample posts
            posts = []
            for sp in result.get("sample_posts", []):
                raw_plat = sp.get("platform", primary)
                if raw_plat not in channel_max_lengths:
                    raw_plat = primary
                max_len = channel_max_lengths.get(raw_plat, 280)
                posts.append({
                    "platform": raw_plat,
                    "text": str(sp.get("text", ""))[:max_len],
                    "author_id": f"{cluster.id}_{sp.get('author_archetype', 'citizen')}",
                    "author_tier": 3,
                })

            # Save round history
            round_data = {
                "round": round_number,
                "position": cluster.position,
                "dominant_sentiment": cluster.dominant_sentiment,
                "engagement_level": cluster.engagement_level,
                "emergent_narrative": cluster.emergent_narrative,
                "key_concerns": cluster.key_concerns,
                "sentiment_distribution": cluster.sentiment_distribution,
            }
            cluster.round_history.append(round_data)

            return {
                "cluster_id": cluster.id,
                "posts": posts,
                "position": cluster.position,
                "sentiment": cluster.dominant_sentiment,
                "engagement": cluster.engagement_level,
                "narrative": cluster.emergent_narrative,
                "concerns": cluster.key_concerns,
                "trust": cluster.trust_institutions,
            }

        except (JSONParseError, Exception) as e:
            logger.error(f"Cluster {cluster.id} failed in round {round_number}: {e}")
            print(f"    ⚠ Cluster {cluster.name}: {type(e).__name__}: {str(e)[:100]}")
            return None

    def get_all_positions(self) -> dict[str, float]:
        return {cid: c.position for cid, c in self.clusters.items()}

    def get_avg_sentiment(self) -> str:
        sentiments = [c.dominant_sentiment for c in self.clusters.values()]
        counter = Counter(sentiments)
        if counter:
            return counter.most_common(1)[0][0]
        return "indifferent"
