"""Phase-by-phase round execution manager — domain-agnostic."""

import asyncio
import logging
import os
import sys

from ..llm.base_client import BaseLLMClient
from ..agents.elite_agent import EliteAgent
from ..agents.institutional_agent import InstitutionalAgent, process_institutional_batch
from ..agents.citizen_swarm import CitizenSwarm
from ..platform.platform_engine import PlatformEngine
from ..platform.metrics import EngagementMetrics
from .event_injector import EventInjector
from .interaction_resolver import InteractionResolver
from .opinion_dynamics import OpinionDynamics
from .param_loader import CalibratedParamLoader
from .checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class RoundManager:
    """Executes all 7 phases of a simulation round."""

    def __init__(
        self,
        llm: BaseLLMClient,
        platform: PlatformEngine,
        event_injector: EventInjector,
        elite_agents: list[EliteAgent],
        institutional_agents: list[InstitutionalAgent],
        citizen_swarm: CitizenSwarm,
        domain_plugin,
        scenario_name: str,
        checkpoint_dir: str = "outputs",
        elite_only: bool = False,
        verbose: bool = False,
        progress_callback=None,
        language: str = "en",
        scenario_context: str = "",
        metrics_to_track: list[str] = None,
    ):
        self.llm = llm
        self.platform = platform
        self.event_injector = event_injector
        self.elite_agents = elite_agents
        self.institutional_agents = institutional_agents
        self.citizen_swarm = citizen_swarm
        self.domain = domain_plugin
        self.scenario_name = scenario_name
        self.checkpoint_dir = checkpoint_dir
        self.elite_only = elite_only
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.language = language
        self.scenario_context = scenario_context
        self.metrics_to_track = metrics_to_track or []
        self.interaction_resolver = InteractionResolver(platform, domain_plugin)

        # Load calibrated params from v2 posterior (with v1 fallback)
        self.param_loader = CalibratedParamLoader()
        domain_id = getattr(domain_plugin, "domain_id", "") if domain_plugin else ""
        params = self.param_loader.get_params(domain=domain_id, include_uncertainty=True)
        # Log param source
        source = params.get("_source", "unknown")
        model_ver = params.get("_model_version", "?")
        logger.info(f"OpinionDynamics params: source={source}, model={model_ver}")
        if params.get("_ci95"):
            for k, (lo, hi) in params["_ci95"].items():
                logger.info(f"  {k}: {params.get(k, '?'):.4f} [{lo:.3f}, {hi:.3f}]")
        # Print comparison for debugging
        self.param_loader.print_comparison(domain=domain_id)
        # Create OpinionDynamics with v1-compatible params (strip metadata keys)
        od_params = {k: v for k, v in params.items() if not k.startswith("_")}
        self.opinion_dynamics = OpinionDynamics(**od_params)
        self._params_info = params  # Keep for CI propagation
        self.metrics = EngagementMetrics(platform)

        # Regime switching: activate for financial/corporate domains
        self._use_regime_switching = domain_id in ("financial", "corporate")
        self._regime_prob = 0.0  # Track current regime probability
        self.coalition_history = []
        self.custom_metrics_history = []

    def _all_agents_flat(self) -> list:
        agents = list(self.elite_agents)
        if not self.elite_only:
            agents.extend(self.institutional_agents)
        return agents

    def _all_positions(self) -> list[float]:
        positions = [a.position for a in self.elite_agents]
        if not self.elite_only:
            positions.extend(a.position for a in self.institutional_agents)
            positions.extend(c.position for c in self.citizen_swarm.clusters.values())
        return positions

    async def _notify(self, event_type: str, data: dict):
        if self.progress_callback:
            await self.progress_callback(event_type, data)

    async def execute_round(self, round_num: int) -> dict:
        """Execute all phases of a round and return summary."""
        print(f"\n  [{self.scenario_name}] ━━ Round {round_num} ━━━━━━━━━━━━━")
        sys.stdout.flush()
        await self._notify("round_start", {"round": round_num, "cost": self.llm.stats.total_cost})

        # Get viral posts from previous round
        viral_posts_text = ""
        if round_num > 1:
            viral_posts_text = self.platform.format_viral_posts(round_num - 1, top_n=5)

        # Current stats
        all_positions = self._all_positions()
        polarization = self.metrics.polarization_index(all_positions)
        avg_sentiment = (
            self.citizen_swarm.get_avg_sentiment()
            if not self.elite_only else "neutral"
        )
        top_narratives = ", ".join(
            self.metrics.extract_narratives(
                round_num - 1 if round_num > 1 else None, top_n=3
            )
        )

        # Coalition info from previous round
        coalition_info = ""
        if self.coalition_history:
            last_coalitions = self.coalition_history[-1].get("coalitions", [])
            parts = []
            if isinstance(last_coalitions, list):
                for c in last_coalitions:
                    label = c.get("label", "?")
                    size = c.get("size", 0)
                    avg_pos = c.get("avg_position", 0)
                    parts.append(f"  {label}: {size} members, pos {avg_pos:+.2f}")
            coalition_info = "\n".join(parts) if parts else ""

        # === Phase 1: Event Generation ===
        print(f"  ├─ Phase 1: Event generation   ", end="", flush=True)

        # What-If: use event override for the first branched round
        branch_override = getattr(self, "_branch_event_override", None)
        if branch_override:
            timeline_label = self.event_injector._get_timeline_label(round_num)
            event = {
                "round": round_num,
                "timeline_label": timeline_label,
                "event": branch_override,
                "shock_magnitude": 0.5,
                "shock_direction": 0.0,
                "key_actors": [],
                "institutional_impact": "",
                "public_perception": "",
            }
            self.event_injector.event_history.append(event)
            self._branch_event_override = None  # only override once
            print(f"✓ ({timeline_label}) [WHAT-IF OVERRIDE]")
        else:
            event = await self.event_injector.generate_event(
                round_num=round_num,
                elite_agents=self.elite_agents,
                polarization=polarization,
                dominant_sentiment=avg_sentiment,
                top_narratives=top_narratives or "No emerging narrative.",
                coalition_info=coalition_info,
                viral_posts=viral_posts_text,
            )
            timeline_label = event["timeline_label"]
            print(f"✓ ({timeline_label})")
        round_event = event["event"]
        await self._notify("round_phase", {"round": round_num, "phase": "event_generation", "message": f"Evento: {timeline_label}", "phase_index": 1, "total_phases": 7})
        if self.verbose:
            print(f"     Event: {round_event[:120]}...")

        # Get domain-specific prompt templates and channel info
        channel_descs = {
            ch.id: ch.description for ch in self.domain.get_channels()
        }
        channel_max_lens = self.domain.get_channel_max_lengths()
        elite_prompt = self.domain.get_elite_prompt_template()
        inst_prompt = self.domain.get_institutional_batch_prompt_template()
        cluster_prompt = self.domain.get_cluster_prompt_template()
        channel_map = self.domain.get_archetype_channel_map()
        profile_template = self.domain.get_mini_profile_template()

        # Inject few-shot examples into prompts
        elite_few_shot = self.domain.get_elite_few_shot()
        cluster_few_shot = self.domain.get_cluster_few_shot()
        if elite_few_shot:
            elite_prompt = elite_prompt + f"\n\nEXAMPLE OF A HIGH-QUALITY RESPONSE (use as format/quality reference, do NOT copy content):\n{elite_few_shot}\n"
        if cluster_few_shot:
            cluster_prompt = cluster_prompt + f"\n\nEXAMPLE OF A HIGH-QUALITY RESPONSE (use as format/quality reference, do NOT copy content):\n{cluster_few_shot}\n"

        # Inject language instruction into prompts
        if self.language and self.language != "en":
            lang_map = {"it": "Italian", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ar": "Arabic", "ru": "Russian"}
            lang_name = lang_map.get(self.language, self.language)
            lang_instruction = f"\n\nIMPORTANT: Write ALL text content (posts, reactions, descriptions, reasoning, narratives, events) in {lang_name}. Only JSON keys must remain in English.\n"
            elite_prompt = lang_instruction + elite_prompt
            inst_prompt = lang_instruction + inst_prompt
            cluster_prompt = lang_instruction + cluster_prompt

        # === Phase 2: Elite Agent Generation ===
        print(f"  ├─ Phase 2: Elite reactions     ", end="", flush=True)
        elite_results = await self._run_elite_agents(
            round_num, timeline_label, round_event, viral_posts_text,
            polarization, avg_sentiment, top_narratives,
            elite_prompt, channel_descs, channel_max_lens,
        )
        for r in elite_results:
            if r:
                for p in r.get("posts", []):
                    self.platform.add_post(p, round_num)
        cost_str = f"${self.llm.stats.total_cost:.2f}"
        print(f"✓ {len([r for r in elite_results if r])}/{len(self.elite_agents)}  ({cost_str})")
        await self._notify("round_phase", {"round": round_num, "phase": "elite_reactions", "message": f"Elite: {len([r for r in elite_results if r])}/{len(self.elite_agents)}", "phase_index": 2, "total_phases": 7})

        # === Phase 3: Institutional Agent Batch ===
        if not self.elite_only:
            print(f"  ├─ Phase 3: Institutional batch ", end="", flush=True)
            inst_results = await self._run_institutional_agents(
                round_num, timeline_label, round_event, viral_posts_text,
                inst_prompt, channel_max_lens, profile_template,
            )
            for batch in inst_results:
                if isinstance(batch, list):
                    for r in batch:
                        for p in r.get("posts", []):
                            self.platform.add_post(p, round_num)
            print(f"✓ ({cost_str})")
            await self._notify("round_phase", {"round": round_num, "phase": "institutional_batch", "message": f"Institutional: {len(self.institutional_agents)} agents", "phase_index": 3, "total_phases": 7})

            # === Phase 4: Citizen Swarm ===
            print(f"  ├─ Phase 4: Citizen swarm       ", end="", flush=True)
            cluster_results = await self.citizen_swarm.simulate_round(
                self.llm, round_num, timeline_label, round_event,
                viral_posts_text, cluster_prompt, channel_map,
                channel_descs, channel_max_lens,
            )
            for cr in cluster_results:
                for p in cr.get("posts", []):
                    self.platform.add_post(p, round_num)
            print(f"✓ {len(cluster_results)} clusters ({cost_str})")
            await self._notify("round_phase", {"round": round_num, "phase": "citizen_swarm", "message": f"Citizens: {len(cluster_results)} clusters", "phase_index": 4, "total_phases": 7})
        else:
            print(f"  ├─ Phase 3-4: Skipped (elite-only mode)")

        # === Phase 5: Platform Dynamics ===
        await self._notify("round_phase", {"round": round_num, "phase": "platform_dynamics", "message": "Resolving engagement, feeds, follows", "phase_index": 5, "total_phases": 7})
        print(f"  ├─ Phase 5: Platform dynamics   [resolving engagement, feeds, follows]")
        all_agents = self._all_agents_flat()
        all_round_posts = self.platform.get_posts_by_round(round_num)
        coalitions = self.interaction_resolver.resolve_round(
            round_num, all_agents, all_round_posts
        )
        self.coalition_history.append({"round": round_num, "coalitions": coalitions})

        # === Phase 6: Opinion Dynamics ===
        await self._notify("round_phase", {"round": round_num, "phase": "opinion_dynamics", "message": f"Updating {len(all_agents)} positions", "phase_index": 6, "total_phases": 7})
        print(f"  ├─ Phase 6: Opinion dynamics    [updating {len(all_agents)} positions]")
        self.opinion_dynamics.update_all_agents(all_agents, self.platform, event)

        # Save agent states
        for agent in all_agents:
            self.platform.save_agent_state(
                agent.id, round_num, agent.position,
                agent.emotional_state, agent.engagement_level, ""
            )
        if not self.elite_only:
            for cluster in self.citizen_swarm.clusters.values():
                self.platform.save_agent_state(
                    cluster.id, round_num, cluster.position,
                    cluster.dominant_sentiment, cluster.engagement_level, ""
                )

        # === Phase 7: Checkpoint ===
        await self._notify("round_phase", {"round": round_num, "phase": "checkpoint", "message": "Saving checkpoint", "phase_index": 7, "total_phases": 7})
        checkpoint = save_checkpoint(
            self.checkpoint_dir, self.scenario_name, round_num,
            self.elite_agents, self.institutional_agents,
            self.citizen_swarm, self.coalition_history,
            self.llm.stats.total_cost, self.elite_only,
            domain=getattr(self.domain, "domain_id", ""),
            confidence_interval=getattr(self, "_last_ci", None),
            regime_info=getattr(self, "_last_regime", None),
            params_used={
                k: v for k, v in self._params_info.items()
                if not k.startswith("_") or k in ("_source", "_model_version")
            } if hasattr(self, "_params_info") else None,
        )
        print(f"  ├─ Phase 7: Checkpoint [saved: {checkpoint}]")

        # Domain-specific metrics
        domain_metrics = {}
        if self.domain:
            domain_metrics = self.domain.compute_domain_metrics(
                all_agents,
                list(self.citizen_swarm.clusters.values()),
                self.platform, round_num,
            )

        # LLM-evaluated custom metrics
        custom_metrics = {}
        if self.metrics_to_track:
            custom_metrics = await self._evaluate_custom_metrics(
                round_num, timeline_label, round_event,
                all_agents, coalitions, domain_metrics,
            )
            self.custom_metrics_history.append({"round": round_num, **custom_metrics})

        # Round stats
        round_stats = self.platform.get_round_stats(round_num)
        all_positions = self._all_positions()
        polarization = self.metrics.polarization_index(all_positions)
        print(f"  ├─ Stats: {round_stats['posts']} posts | "
              f"{round_stats['reactions']} reactions | "
              f"polarization: {polarization:.1f}/10")
        if custom_metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in custom_metrics.items())
            print(f"  ├─ Metrics: {metrics_str}")
        print(f"  └─ Running cost: ${self.llm.stats.total_cost:.2f}")

        # Build enriched round data for live dashboard
        top_posts_raw = self.platform.get_top_posts(round_num, top_n=10)
        top_posts_data = []
        for p in top_posts_raw:
            top_posts_data.append({
                "id": p.get("id"),
                "author_id": p.get("author_id", ""),
                "author_name": p.get("author_name", p.get("author_id", "?")),
                "platform": p.get("platform", "social"),
                "text": p.get("content", ""),
                "likes": p.get("likes", 0),
                "reposts": p.get("reposts", 0),
                "replies": p.get("reply_count", 0),
                "total_engagement": p.get("likes", 0) + p.get("reposts", 0) * 2 + p.get("reply_count", 0) * 3,
            })

        # Agent snapshot
        agents_snapshot = []
        for a in all_agents:
            agents_snapshot.append({
                "id": a.id,
                "name": a.name,
                "role": getattr(a, "role", ""),
                "position": round(a.position, 3),
                "emotional_state": getattr(a, "emotional_state", "neutral"),
                "tier": getattr(a, "tier", 1),
            })
        # Add citizen clusters
        if not self.elite_only:
            for c in self.citizen_swarm.clusters.values():
                agents_snapshot.append({
                    "id": c.id,
                    "name": c.name,
                    "role": "citizen_cluster",
                    "position": round(c.position, 3),
                    "emotional_state": getattr(c, "emotional_state", "neutral"),
                    "tier": 3,
                    "cluster_size": getattr(c, "population", 0),
                })

        # Sentiment distribution
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for a in agents_snapshot:
            es = a.get("emotional_state", "neutral").lower()
            if es in ("satisfied", "triumphant", "optimistic", "hopeful"):
                sentiments["positive"] += 1
            elif es in ("combative", "furious", "worried", "frustrated"):
                sentiments["negative"] += 1
            else:
                sentiments["neutral"] += 1
        total_s = max(sum(sentiments.values()), 1)
        sentiment_pcts = {k: round(v / total_s, 2) for k, v in sentiments.items()}

        # Coalition data for frontend
        coalitions_data = []
        if isinstance(coalitions, list):
            coalitions_data = coalitions
        elif isinstance(coalitions, dict):
            coalitions_data = coalitions.get("coalitions", [])

        # Build confidence interval from posterior uncertainty
        ci95 = self._params_info.get("_ci95", {}) if hasattr(self, "_params_info") else {}
        confidence_interval = None
        if ci95:
            # Approximate pro_pct CI from parameter uncertainty
            # Use sigma_delta (model discrepancy) as primary uncertainty source
            disc = self.param_loader.get_discrepancy(
                domain=getattr(self.domain, "domain_id", None)
            )
            sigma_pp = disc["sigma_delta"] * 25  # logit → pp rough conversion
            avg_pos = sum(p for p in all_positions) / max(len(all_positions), 1) if all_positions else 0
            pro_est = (1 + avg_pos) / 2 * 100  # rough position → pro% mapping
            confidence_interval = {
                "pro_pct_mean": round(pro_est, 1),
                "pro_pct_ci95_lo": round(max(0, pro_est - 1.96 * sigma_pp), 1),
                "pro_pct_ci95_hi": round(min(100, pro_est + 1.96 * sigma_pp), 1),
                "sigma_pp": round(sigma_pp, 1),
            }

        # Regime info (defaults to normal)
        regime_info = {
            "regime_prob": round(self._regime_prob, 3),
            "regime_label": "crisis" if self._regime_prob > 0.5 else "normal",
        }

        # Estimate regime probability from shock magnitude
        shock_mag = event.get("shock_magnitude", 0)
        if self._use_regime_switching and shock_mag > 0.4:
            # Simple heuristic: large shocks push toward crisis
            self._regime_prob = min(1.0, self._regime_prob * 0.9 + 0.6 * (shock_mag - 0.3))
        else:
            self._regime_prob = max(0.0, self._regime_prob * 0.7)
        regime_info["regime_prob"] = round(self._regime_prob, 3)
        regime_info["regime_label"] = "crisis" if self._regime_prob > 0.5 else "normal"

        # Store for checkpoint (will be saved in next round's checkpoint)
        self._last_ci = confidence_interval
        self._last_regime = regime_info

        result = {
            "round": round_num,
            "timeline_label": timeline_label,
            "event": round_event,
            "posts": round_stats["posts"],
            "reactions": round_stats["reactions"],
            "polarization": polarization,
            "coalitions": coalitions,
            "domain_metrics": domain_metrics,
            "custom_metrics": custom_metrics,
            "cost": self.llm.stats.total_cost,
            "confidence_interval": confidence_interval,
            "regime_info": regime_info,
        }

        await self._notify("round_complete", {
            "round": round_num,
            "cost": self.llm.stats.total_cost,
            "timeline_label": timeline_label,
            "event": round_event,
            "posts_count": round_stats["posts"],
            "reactions_count": round_stats["reactions"],
            "polarization": polarization,
            "top_posts": top_posts_data,
            "agents": agents_snapshot,
            "sentiment": sentiment_pcts,
            "coalitions": coalitions_data,
            "domain_metrics": domain_metrics,
            "custom_metrics": custom_metrics,
            "shock_magnitude": event.get("shock_magnitude", 0),
            "shock_direction": event.get("shock_direction", 0),
            "confidence_interval": confidence_interval,
            "regime_info": regime_info,
            "calibration_source": self._params_info.get("_model_version", "v1") + "_" + self._params_info.get("_source", "unknown") if hasattr(self, "_params_info") else "v1_default",
        })
        return result

    async def _evaluate_custom_metrics(self, round_num, timeline_label, round_event,
                                          all_agents, coalitions, domain_metrics):
        """Use LLM to evaluate scenario-specific metrics for this round."""
        if not self.metrics_to_track:
            return {}

        # Build state summary for LLM
        agent_summary = []
        for a in all_agents[:15]:  # Top 15 agents
            agent_summary.append(f"- {a.name} ({getattr(a, 'role', '?')}): pos={a.position:+.2f}, mood={getattr(a, 'emotional_state', '?')}")

        viral_text = self.platform.format_viral_posts(round_num, top_n=5)

        coalition_text = ""
        coal_list = coalitions if isinstance(coalitions, list) else coalitions.get("coalitions", []) if isinstance(coalitions, dict) else []
        for c in coal_list:
            coalition_text += f"- {c.get('label', '?')}: {c.get('size', 0)} members, avg_pos={c.get('avg_position', 0):+.2f}\n"

        metrics_list = "\n".join(f'- "{m}"' for m in self.metrics_to_track)
        prev_metrics_text = ""
        if self.custom_metrics_history:
            prev = self.custom_metrics_history[-1]
            prev_metrics_text = "Previous round metrics: " + ", ".join(
                f"{k}={v}" for k, v in prev.items() if k != "round"
            )

        prompt = f"""You are evaluating a simulation scenario: "{self.scenario_name}".
Context: {self.scenario_context[:500]}

Round {round_num} ({timeline_label}):
Event: {round_event}

Key agents:
{chr(10).join(agent_summary)}

Coalitions:
{coalition_text or 'None detected'}

Viral posts this round:
{viral_text or 'None'}

{prev_metrics_text}

Evaluate these scenario-specific metrics for the current round state.
Each metric should be a number from 0 to 100 (where 0=minimum, 100=maximum).

Metrics to evaluate:
{metrics_list}

Respond with JSON only:
{{
{', '.join(f'  "{m}": <0-100>' for m in self.metrics_to_track)}
}}"""

        lang_instruction = ""
        if self.language and self.language != "en":
            lang_instruction = f"\nNote: The scenario is in {self.language}. Understand the content but respond with the JSON metrics only."
            prompt += lang_instruction

        try:
            result = await self.llm.generate_json(
                prompt=prompt,
                temperature=0.3,
                max_output_tokens=500,
                component="metrics_evaluation",
            )
            # Ensure all values are numbers 0-100
            clean = {}
            for m in self.metrics_to_track:
                val = result.get(m, 50)
                if isinstance(val, (int, float)):
                    clean[m] = max(0, min(100, round(val)))
                else:
                    clean[m] = 50
            return clean
        except Exception as e:
            logger.warning(f"Custom metrics evaluation failed: {e}")
            return {m: 50 for m in self.metrics_to_track}

    async def _run_elite_agents(self, round_num, timeline_label, round_event,
                                 viral_posts, polarization, avg_sentiment,
                                 top_narratives, prompt_template,
                                 channel_descs, channel_max_lens):
        tasks = [
            agent.generate_round(
                self.llm, round_num, timeline_label, round_event,
                viral_posts, polarization, avg_sentiment, top_narratives,
                prompt_template, channel_descs, channel_max_lens,
            )
            for agent in self.elite_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed = []
        errors = 0
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                errors += 1
                print(f"    ⚠ Elite agent {self.elite_agents[i].name} failed: {r}")
                logger.error(f"Elite agent {self.elite_agents[i].id} error: {r}")
                processed.append(None)
            else:
                posts_count = len(r.get("posts", [])) if r else 0
                if r and posts_count == 0:
                    print(f"    ⚠ Elite agent {self.elite_agents[i].name} returned 0 posts")
                processed.append(r)
        if errors:
            print(f"    ⚠ {errors}/{len(self.elite_agents)} elite agents failed!")
        return processed

    async def _run_institutional_agents(self, round_num, timeline_label,
                                         round_event, viral_posts,
                                         prompt_template, channel_max_lens,
                                         profile_template):
        batches = []
        batch_size = 10
        for i in range(0, len(self.institutional_agents), batch_size):
            batch = self.institutional_agents[i:i + batch_size]
            batches.append(batch)

        tasks = [
            process_institutional_batch(
                batch, self.llm, round_num, timeline_label, round_event,
                viral_posts, prompt_template, channel_max_lens, profile_template,
            )
            for batch in batches
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
