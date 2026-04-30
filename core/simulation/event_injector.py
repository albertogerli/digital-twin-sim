"""Emergent event generation — LLM creates events based on simulation dynamics."""

import logging
from typing import Optional

from ..llm.base_client import BaseLLMClient
from .validators import clamp_shock_magnitude, check_event_plausibility

logger = logging.getLogger(__name__)


class EventInjector:
    """Generates emergent events via LLM based on simulation dynamics."""

    def __init__(
        self,
        llm: BaseLLMClient,
        scenario_context: str,
        initial_event: str,
        event_prompt_template: str,
        timeline_labels: list[str],
        fallback_strings: dict[str, str],
        language: str = "en",
        historical_context: str = "",
        few_shot_example: str = "",
    ):
        self.llm = llm
        self.scenario_context = scenario_context
        self.initial_event = initial_event
        self.event_prompt_template = event_prompt_template
        self.timeline_labels = timeline_labels
        self.fallbacks = fallback_strings
        self.language = language
        self.historical_context = historical_context
        self.few_shot_example = few_shot_example
        self.event_history: list[dict] = []

        # Blinded-mode detection: the sim-lift adapter anonymizes scenarios
        # with tokens like "Country_NN" and "PRO_LEADER_N" and states
        # "Ground truth withheld". When we detect those, we disable the
        # domain few-shot example (which may be culturally specific, e.g.
        # Italian) and prepend a hard-guard instruction telling the LLM to
        # stay within the anonymized schema. See sim_adapter._blind().
        self.blinded = self._detect_blinded(scenario_context)
        if self.blinded:
            self.few_shot_example = ""  # drop potentially-contaminating example
            blind_guard = (
                "\nBLINDED MODE: the scenario is anonymized. You MUST NOT invent or "
                "reference real-world named institutions, parties, politicians, media "
                "outlets, or legal bodies from any country. Use only the generic "
                "placeholder labels already present in the scenario context (e.g. "
                "PRO_LEADER_1, AGAINST_LEADER_3, COUNTRY, Country_NN) or purely "
                "role-based descriptions (e.g. 'the central bank', 'a major daily "
                "newspaper', 'the finance ministry'). This overrides any example "
                "event style.\n"
            )
            self.event_prompt_template = blind_guard + self.event_prompt_template

        # Inject language instruction
        if language and language != "en":
            lang_map = {"it": "Italian", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese"}
            lang_name = lang_map.get(language, language)
            self.event_prompt_template = f"\nIMPORTANT: Write ALL text content in {lang_name}. Only JSON keys in English.\n" + self.event_prompt_template

    @staticmethod
    def _detect_blinded(scenario_context: str) -> bool:
        if not scenario_context:
            return False
        markers = ("Ground truth withheld", "Country_", "PRO_LEADER_", "AGAINST_LEADER_")
        return any(m in scenario_context for m in markers)

    def _get_timeline_label(self, round_num: int) -> str:
        if self.timeline_labels and round_num <= len(self.timeline_labels):
            return self.timeline_labels[round_num - 1]
        return f"Round {round_num}"

    async def generate_event(
        self,
        round_num: int,
        elite_agents: list,
        polarization: float,
        dominant_sentiment: str,
        top_narratives: str,
        coalition_info: str,
        viral_posts: str,
    ) -> dict:
        """Generate an emergent event for this round via LLM."""
        timeline_label = self._get_timeline_label(round_num)

        # Round 1: use initial event from config
        if round_num == 1:
            event = {
                "round": 1,
                "timeline_label": timeline_label,
                "event": self.initial_event,
                "shock_magnitude": 0.7,
                "shock_direction": 0.0,
                "key_actors": [],
                "institutional_impact": "",
                "public_perception": "",
            }
            self.event_history.append(event)
            return event

        # Rounds 2+: emergent generation via LLM
        history = self._format_history()

        agent_states = "\n".join(
            f"- {a.name} ({a.role}): position {a.position:+.2f}, state {a.emotional_state}"
            for a in elite_agents[:15]
        )

        # Build enhanced prompt with historical context and few-shot
        extra_context = ""
        if self.historical_context:
            extra_context += f"\n\n{self.historical_context}\n"
        if self.few_shot_example:
            extra_context += f"\n\nEXAMPLE OF A REALISTIC EVENT (use as quality reference, do NOT copy):\n{self.few_shot_example}\n"

        prompt = self.event_prompt_template.format(
            timeline_label=timeline_label,
            scenario_context=self.scenario_context + extra_context,
            initial_context=self.initial_event,
            history=history,
            agent_states=agent_states,
            polarization=polarization,
            dominant_sentiment=dominant_sentiment,
            top_narratives=top_narratives,
            coalition_info=coalition_info or self.fallbacks.get("no_coalition", ""),
            viral_posts=viral_posts or self.fallbacks.get("no_viral", ""),
        )

        try:
            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.85,
                max_output_tokens=1000,
                component="event_generation",
            )

            # Guard: if LLM returned a list, take the first dict
            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}

            event_text = result.get("event", self.fallbacks.get("default_event", ""))
            shock_mag = clamp_shock_magnitude(float(result.get("shock_magnitude", 0.3)))

            event = {
                "round": round_num,
                "timeline_label": timeline_label,
                "event": event_text,
                "shock_magnitude": shock_mag,
                "shock_direction": max(-1.0, min(1.0, float(result.get("shock_direction", 0.0)))),
                "key_actors": result.get("key_actors_affected", []),
                "institutional_impact": result.get("institutional_impact", ""),
                "public_perception": result.get("public_perception", ""),
            }

            # Plausibility check — regenerate once if score < 4
            is_plausible, score, reason = await check_event_plausibility(
                self.llm, event_text, self.scenario_context
            )
            if not is_plausible:
                logger.warning(f"Round {round_num} event failed plausibility (score={score}), regenerating...")
                retry = await self.llm.generate_json(
                    prompt=prompt + f"\n\nPREVIOUS ATTEMPT WAS TOO IMPLAUSIBLE ({reason}). Generate a more realistic, grounded event.",
                    temperature=0.6,
                    max_output_tokens=1000,
                    component="event_generation_retry",
                )
                if isinstance(retry, list):
                    retry = retry[0] if retry and isinstance(retry[0], dict) else {}
                event["event"] = retry.get("event", event_text)
                event["shock_magnitude"] = clamp_shock_magnitude(float(retry.get("shock_magnitude", shock_mag)))
                event["shock_direction"] = max(-1.0, min(1.0, float(retry.get("shock_direction", event["shock_direction"]))))

            self.event_history.append(event)
            return event

        except Exception as e:
            logger.error(f"Event generation failed for round {round_num}: {e}")
            fallback_template = self.fallbacks.get(
                "fallback_event",
                "Dynamics continue. Polarization at {polarization:.1f}/10."
            )
            fallback = {
                "round": round_num,
                "timeline_label": timeline_label,
                "event": fallback_template.format(
                    timeline_label=timeline_label, polarization=polarization
                ),
                "shock_magnitude": 0.2,
                "shock_direction": 0.0,
            }
            self.event_history.append(fallback)
            return fallback

    def _format_history(self) -> str:
        if not self.event_history:
            return self.fallbacks.get("no_history", "No previous history.")
        lines = []
        for e in self.event_history:
            lines.append(
                f"[{e.get('timeline_label', '?')}] {e['event'][:300]}"
                f" (shock: {e.get('shock_magnitude', 0):.1f}, "
                f"dir: {e.get('shock_direction', 0):+.1f})"
            )
        return "\n".join(lines)
