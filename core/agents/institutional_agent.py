"""Tier 2: Institutional agents processed in batches."""

import logging
from typing import Optional

from .base_agent import BaseAgent
from ..llm.base_client import BaseLLMClient
from ..llm.json_parser import JSONParseError

logger = logging.getLogger(__name__)


class InstitutionalAgent(BaseAgent):
    """Tier 2 agent: batch-processed with lighter persona."""

    def __init__(self, **kwargs):
        self.key_trait: str = kwargs.pop("key_trait", "")
        self.category: str = kwargs.pop("category", "")
        self.graph_connections: list[str] = kwargs.pop("graph_connections", [])
        kwargs["tier"] = 2
        super().__init__(**kwargs)

    def mini_profile(self, profile_template: dict[str, str] | None = None) -> str:
        t = profile_template or {
            "position": "Position: {pos:+.2f}.",
            "trait": "Key trait: {trait}",
        }
        return (
            f"- {self.id}: {self.name}, {self.role}. "
            f"{t['position'].format(pos=self.position)} "
            f"{t['trait'].format(trait=self.key_trait)}"
        )

    @classmethod
    def from_spec(cls, spec) -> "InstitutionalAgent":
        if hasattr(spec, 'model_dump'):
            d = spec.model_dump()
        else:
            d = spec
        return cls(
            id=d["id"],
            name=d["name"],
            role=d["role"],
            archetype=d.get("category", d.get("archetype", "unknown")),
            position=d.get("position", 0.0),
            original_position=d.get("position", 0.0),
            influence=d.get("influence", 0.3),
            rigidity=d.get("rigidity", 0.5),
            key_trait=d.get("key_trait", ""),
            category=d.get("category", ""),
            graph_connections=d.get("graph_connections", []),
        )


async def process_institutional_batch(
    agents: list[InstitutionalAgent],
    llm: BaseLLMClient,
    round_number: int,
    timeline_label: str,
    round_event: str,
    viral_posts: str,
    prompt_template: str,
    channel_max_lengths: dict[str, int],
    profile_template: dict[str, str] | None = None,
) -> list[dict]:
    """Process a batch of up to 10 institutional agents in a single LLM call."""
    agents_list = "\n".join(a.mini_profile(profile_template) for a in agents)

    prompt = prompt_template.format(
        count=len(agents),
        round_number=round_number,
        timeline_label=timeline_label,
        round_event=round_event,
        viral_posts=viral_posts,
        agents_list=agents_list,
    )

    try:
        results = await llm.generate_json(
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=6000,
            component="institutional_batch",
        )

        # OpenAI JSON mode often wraps arrays in an object
        if isinstance(results, dict):
            # Try to find the array inside the dict
            for key in ("agents", "results", "responses", "data"):
                if key in results and isinstance(results[key], list):
                    results = results[key]
                    break
            else:
                # Single agent result wrapped as dict
                results = [results]

        results_by_id = {r.get("id", ""): r for r in results}
        processed = []

        for agent in agents:
            r = results_by_id.get(agent.id)
            if not r:
                logger.warning(f"No result for institutional agent {agent.id}")
                continue

            new_pos = float(r.get("position_shift", agent.position))
            agent.position = max(-1.0, min(1.0, new_pos))
            agent.emotional_state = r.get("sentiment", "neutral")

            posts = []
            raw_posts = r.get("posts", [])
            if not raw_posts:
                tweet = r.get("post_twitter", "")
                if tweet:
                    raw_posts.append({"platform": "social", "text": str(tweet)})
                forum = r.get("post_forum", "")
                if forum:
                    raw_posts.append({"platform": "forum", "text": str(forum)})

            for rp in raw_posts:
                text = str(rp.get("text", ""))
                if not text or text == "null":
                    continue
                plat = rp.get("platform", "social")
                max_len = channel_max_lengths.get(plat, 500)
                posts.append({
                    "platform": plat,
                    "text": text[:max_len],
                    "author_id": agent.id,
                    "author_tier": 2,
                })

            agent.memory.add_round(
                round_num=round_number,
                summary=r.get("key_action", ""),
                posts=[{"platform": p["platform"], "text": p["text"]} for p in posts],
                engagement={},
            )

            processed.append({
                "agent_id": agent.id,
                "posts": posts,
                "position": agent.position,
                "sentiment": agent.emotional_state,
                "key_action": r.get("key_action", ""),
            })

        return processed

    except (JSONParseError, Exception) as e:
        logger.error(f"Institutional batch failed in round {round_number}: {e}")
        return []
