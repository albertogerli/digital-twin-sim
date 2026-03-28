"""Tier 1: Elite agents with individual LLM calls and full persona."""

import logging
from typing import Optional

from .base_agent import BaseAgent
from ..llm.base_client import BaseLLMClient
from ..llm.json_parser import JSONParseError

logger = logging.getLogger(__name__)


class EliteAgent(BaseAgent):
    """Tier 1 agent: individual LLM calls, full persona, multi-platform."""

    def __init__(self, **kwargs):
        kwargs["tier"] = 1
        self.platform_primary: str = kwargs.pop("platform_primary", "")
        self.platform_secondary: str = kwargs.pop("platform_secondary", "")
        super().__init__(**kwargs)

    async def generate_round(
        self,
        llm: BaseLLMClient,
        round_number: int,
        timeline_label: str,
        round_event: str,
        viral_posts: str,
        polarization: float,
        avg_sentiment: str,
        top_narratives: str,
        prompt_template: str,
        channel_descriptions: dict[str, str],
        channel_max_lengths: dict[str, int],
    ) -> Optional[dict]:
        """Generate this agent's content and position update for a round."""
        primary = self.platform_primary
        secondary = self.platform_secondary
        primary_max = channel_max_lengths.get(primary, 280)
        secondary_max = channel_max_lengths.get(secondary, 500)

        primary_desc = channel_descriptions.get(primary, primary)
        secondary_desc = channel_descriptions.get(secondary, secondary)
        platforms_desc = f"{primary} ({primary_desc}), {secondary} ({secondary_desc})"

        prompt = prompt_template.format(
            system_prompt=self.system_prompt,
            round_number=round_number,
            timeline_label=timeline_label,
            round_event=round_event,
            agent_memory=self.memory.get_context(),
            viral_posts=viral_posts,
            polarization=polarization,
            avg_sentiment=avg_sentiment,
            top_narratives=top_narratives,
            platforms_description=platforms_desc,
            primary_platform=primary,
            secondary_platform=secondary,
            primary_max_len=primary_max,
            secondary_max_len=secondary_max,
        )

        try:
            result = await llm.generate_json(
                prompt=prompt,
                temperature=0.8,
                max_output_tokens=1500,
                component=f"elite_{self.id}",
            )

            # Update agent state with delta cap
            from ..simulation.validators import clamp_position_delta, ELITE_DELTA_CAP
            new_position = float(result.get("position", self.position))
            self.position = clamp_position_delta(self.position, new_position, ELITE_DELTA_CAP)
            self.emotional_state = result.get("emotional_state", self.emotional_state)

            # Build posts list
            posts = []
            raw_posts = result.get("posts", [])
            if not raw_posts:
                tweet = result.get("tweet", "")
                if tweet:
                    raw_posts.append({"platform": primary, "text": tweet})
                forum = result.get("forum_post")
                if forum and forum != "null":
                    raw_posts.append({"platform": secondary, "text": str(forum)})

            for rp in raw_posts:
                text = str(rp.get("text", ""))
                if not text or text == "null":
                    continue
                plat = rp.get("platform", primary)
                max_len = channel_max_lengths.get(plat, 280)
                posts.append({
                    "platform": plat,
                    "text": text[:max_len],
                    "author_id": self.id,
                    "author_tier": 1,
                })

            # Reaction to viral post
            reaction = result.get("reaction_to_viral")
            if reaction and reaction != "null":
                posts.append({
                    "platform": primary,
                    "text": str(reaction)[:channel_max_lengths.get(primary, 280)],
                    "author_id": self.id,
                    "author_tier": 1,
                    "is_reply": True,
                })

            # Validate alliance/target references
            from ..simulation.validators import validate_agent_references
            alliances = result.get("alliances", [])
            targets = result.get("targets", [])

            # Update memory
            self.memory.add_round(
                round_num=round_number,
                summary=result.get("position_reasoning", ""),
                posts=[{"platform": p["platform"], "text": p["text"]} for p in posts],
                engagement={},
                alliances=alliances,
                targets=targets,
            )

            return {
                "agent_id": self.id,
                "posts": posts,
                "position": self.position,
                "emotional_state": self.emotional_state,
                "strategic_move": result.get("strategic_move", ""),
                "alliances": result.get("alliances", []),
                "targets": result.get("targets", []),
                "position_reasoning": result.get("position_reasoning", ""),
            }

        except (JSONParseError, Exception) as e:
            logger.error(f"Elite agent {self.id} failed in round {round_number}: {e}")
            return None

    @classmethod
    def from_spec(cls, spec, system_prompt: str = "") -> "EliteAgent":
        """Create an EliteAgent from an AgentSpec."""
        if hasattr(spec, 'model_dump'):
            d = spec.model_dump()
        else:
            d = spec
        return cls(
            id=d["id"],
            name=d["name"],
            role=d["role"],
            archetype=d.get("archetype", "unknown"),
            position=d["position"],
            original_position=d["position"],
            influence=d.get("influence", 0.5),
            rigidity=d.get("rigidity", 0.5),
            system_prompt=system_prompt or d.get("system_prompt", ""),
            platform_primary=d.get("platform_primary", ""),
            platform_secondary=d.get("platform_secondary", ""),
        )
