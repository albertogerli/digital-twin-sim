"""Reporting service — generates final Markdown report + financial JSON.

Extracted from `engine.py`. Owns:
  - `generate_report` — builds the per-round narrative via the domain's
    report prompt template and writes `<scenario>_report.md`.
  - `save_financial_impact` — serialises per-round orchestrator output to
    `<scenario>_financial_impact.json` for the frontend bridge.

The engine hands us the final round results + agent collections; we never
touch the simulation state directly.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..llm.base_client import BaseLLMClient
    from ..config.schema import ScenarioConfig
    from ..agents.elite_agent import EliteAgent
    from ..agents.citizen_swarm import CitizenSwarm
    from domains.base_domain import DomainPlugin

logger = logging.getLogger(__name__)


_LANG_MAP = {
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
}


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


class ReportingService:
    """Writes the Markdown report and financial-impact JSON."""

    def __init__(
        self,
        llm: "BaseLLMClient",
        config: "ScenarioConfig",
        domain: "DomainPlugin",
        output_dir: str,
        elite_only: bool = False,
    ):
        self.llm = llm
        self.config = config
        self.domain = domain
        self.output_dir = output_dir
        self.elite_only = elite_only

    # ── Markdown report ───────────────────────────────────────────────────

    async def generate_report(
        self,
        round_results: list[dict],
        elite_agents: list["EliteAgent"],
        citizen_swarm: Optional["CitizenSwarm"],
    ) -> str:
        """Generate the final report. Returns the path it was written to."""
        print(f"\n  Generating report...")

        system_prompt, user_prompt = self._build_prompts(
            round_results, elite_agents, citizen_swarm
        )

        try:
            report_text = await self.llm.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_output_tokens=8000,
                component="report",
            )
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report_text = f"# {self.config.name}\n\nReport generation failed: {e}"

        path = os.path.join(
            self.output_dir, f"{_safe_name(self.config.name)}_report.md"
        )
        with open(path, "w") as f:
            f.write(report_text)
        print(f"  └─ Report: {path} ✓")
        return path

    def _build_prompts(
        self,
        round_results: list[dict],
        elite_agents: list["EliteAgent"],
        citizen_swarm: Optional["CitizenSwarm"],
    ) -> tuple[str, str]:
        report_system = self.domain.get_report_system_prompt()
        report_template = self.domain.get_report_prompt_template()

        round_summaries = "\n".join(
            f"Round {r['round']} ({r.get('timeline_label', '?')}): "
            f"{r['posts']} posts, {r['reactions']} reactions, "
            f"polarization {r['polarization']:.1f}/10"
            for r in round_results
        )

        elite_summary = "\n".join(
            f"- {a.name} ({a.role}): pos {a.position:+.2f}, state {a.emotional_state}"
            for a in elite_agents
        )

        cluster_summary = ""
        if not self.elite_only and citizen_swarm is not None and citizen_swarm.clusters:
            cluster_lines = [
                f"- {c.name}: pos {c.position:+.2f}, "
                f"sentiment {c.dominant_sentiment}, "
                f"engagement {c.engagement_level:.1f}"
                for c in citizen_swarm.clusters.values()
            ]
            cluster_summary = "CITIZEN CLUSTERS:\n" + "\n".join(cluster_lines)

        user_prompt = report_template.format(
            scenario_title=self.config.name,
            num_rounds=self.config.num_rounds,
            round_summaries=round_summaries,
            num_elite=len(elite_agents),
            elite_summary=elite_summary,
            cluster_summary=cluster_summary,
        )

        lang = getattr(self.config, "language", "en")
        if lang and lang != "en":
            lang_name = _LANG_MAP.get(lang, lang)
            lang_instruction = (
                f"\n\nCRITICAL LANGUAGE REQUIREMENT: Write the ENTIRE report in "
                f"{lang_name}. Every heading, paragraph, analysis, conclusion, and "
                f"narrative MUST be in {lang_name}. Do NOT use English for any part "
                f"of the report content."
            )
            report_system += lang_instruction
            user_prompt = lang_instruction + "\n\n" + user_prompt

        return report_system, user_prompt

    # ── Financial impact JSON ─────────────────────────────────────────────

    def save_financial_impact(
        self, round_results: list[dict], financial_scorer_active: bool,
    ) -> Optional[str]:
        """Serialise per-round financial-impact output. Returns path or None."""
        if not financial_scorer_active:
            return None

        from core.orchestrator.financial_impact import FIN_SCHEMA_VERSION

        rounds_payload = []
        for r in round_results:
            orch = r.get("orchestrator") or {}
            fin = orch.get("financial_impact")
            if not fin:
                continue
            rounds_payload.append({
                "round": r["round"],
                "timeline_label": r.get("timeline_label", ""),
                **fin,
            })

        if not rounds_payload:
            return None

        out_path = os.path.join(
            self.output_dir,
            f"{_safe_name(self.config.name)}_financial_impact.json",
        )
        payload = {
            "schema_version": FIN_SCHEMA_VERSION,
            "scenario": self.config.name,
            "domain": self.config.domain,
            "num_rounds": self.config.num_rounds,
            "provenance": "backend-simulated",
            "rounds": rounds_payload,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  └─ Financial impact: {out_path} ✓ ({len(rounds_payload)} rounds)")
        return out_path
