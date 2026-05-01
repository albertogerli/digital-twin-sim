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

import datetime as _dt
import html as _html
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

# Concrete heading translations for languages where the domain prompt
# templates leak English headings (LLMs tend to copy template wording even
# when instructed to translate). For unlisted languages, the LLM is left
# to translate on its own.
_HEADING_OVERRIDES = {
    "it": {
        "Executive Summary": "Sintesi Esecutiva",
        "Simulated Timeline": "Cronologia Simulata",
        "Coalition Map": "Mappa delle Coalizioni",
        "Market Sentiment Dynamics": "Dinamiche del Sentiment di Mercato",
        "Polarization and Herding": "Polarizzazione ed Effetto Gregge",
        "The Viral Posts That Moved the Market": "Post Virali che Hanno Mosso il Mercato",
        "Dominant Emerging Narrative": "Narrazione Dominante Emergente",
        "Institutional and Regulatory Impact": "Impatto Istituzionale e Regolamentare",
        "Scenarios Forward": "Scenari Futuri",
        "Methodological Note": "Nota Metodologica",
    },
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
                f"of the report content, INCLUDING markdown section headings "
                f"(##, ###). The template below uses English headings only as "
                f"a structural guide — translate them when you write."
            )
            overrides = _HEADING_OVERRIDES.get(lang)
            if overrides:
                mapping_lines = "\n".join(
                    f"  '{en}' → '{tr}'" for en, tr in overrides.items()
                )
                lang_instruction += (
                    f"\n\nUse these EXACT translations for the section headings:\n"
                    f"{mapping_lines}"
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

    # ── HTML report (printable to PDF) ────────────────────────────────────

    def generate_html_report(
        self,
        round_results: list[dict],
        elite_agents: list["EliteAgent"],
        citizen_swarm: Optional["CitizenSwarm"],
        markdown_report_path: Optional[str] = None,
        cost: Optional[float] = None,
    ) -> str:
        """Render a self-contained HTML report (CSS + inline SVG charts) that
        prints cleanly to PDF via the browser. Returns the output path.

        Designed to be the single deliverable handed to a client: cover,
        executive summary, KPI dashboard, polarization & sentiment charts,
        per-round breakdown with top viral posts, agent panel.
        """
        md_text = ""
        if markdown_report_path and os.path.exists(markdown_report_path):
            with open(markdown_report_path, "r") as f:
                md_text = f.read()

        html_body = _build_html_body(
            config=self.config,
            domain=self.domain,
            round_results=round_results,
            elite_agents=elite_agents,
            citizen_swarm=citizen_swarm,
            md_text=md_text,
            cost=cost,
        )

        out_path = os.path.join(
            self.output_dir, f"{_safe_name(self.config.name)}_report.html"
        )
        with open(out_path, "w") as f:
            f.write(html_body)
        print(f"  └─ HTML report: {out_path} ✓")
        return out_path


# ── HTML builder helpers (stateless, easier to test) ─────────────────────

_CSS = """
:root {
  --ink: #0f172a; --muted: #475569; --line: #e2e8f0;
  --accent: #2563eb; --pos: #16a34a; --neg: #dc2626; --neu: #94a3b8;
  --bg: #ffffff; --bg-soft: #f8fafc;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--ink); background: var(--bg); margin: 0;
  font-size: 11pt; line-height: 1.45;
}
.page { max-width: 980px; margin: 0 auto; padding: 28px 36px; }
header.cover {
  border-bottom: 3px solid var(--ink);
  padding-bottom: 18px; margin-bottom: 22px;
}
header.cover .eyebrow {
  font-size: 9pt; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); font-weight: 600;
}
header.cover h1 {
  font-size: 26pt; line-height: 1.15; margin: 4px 0 6px;
  font-weight: 800; color: var(--ink);
}
header.cover .brief { color: var(--muted); font-size: 11pt; }
header.cover .meta {
  display: flex; gap: 18px; margin-top: 12px;
  font-size: 9pt; color: var(--muted);
}
header.cover .meta span strong { color: var(--ink); font-weight: 600; }
h2 {
  font-size: 14pt; font-weight: 700; margin: 26px 0 10px;
  padding-bottom: 4px; border-bottom: 1px solid var(--line);
}
h3 { font-size: 11.5pt; font-weight: 700; margin: 16px 0 6px; }
.grid { display: grid; gap: 12px; }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.kpi {
  border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px;
  background: var(--bg-soft);
}
.kpi .label {
  font-size: 8.5pt; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); font-weight: 600;
}
.kpi .value { font-size: 20pt; font-weight: 800; color: var(--ink); }
.kpi .delta { font-size: 9pt; margin-left: 6px; }
.kpi .delta.up { color: var(--pos); }
.kpi .delta.down { color: var(--neg); }
.chart {
  border: 1px solid var(--line); border-radius: 6px; padding: 12px;
  background: var(--bg);
}
.chart svg { width: 100%; height: auto; }
.round { border: 1px solid var(--line); border-radius: 6px;
         padding: 12px 14px; margin-bottom: 12px; page-break-inside: avoid; }
.round-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.round-num {
  background: var(--accent); color: #fff; border-radius: 3px;
  padding: 2px 7px; font-size: 9pt; font-weight: 700;
}
.round-title { font-weight: 700; font-size: 11.5pt; }
.round-meta { color: var(--muted); font-size: 9pt; }
.event { color: var(--muted); margin: 4px 0 10px; font-size: 10pt; }
.posts { margin-top: 8px; }
.post {
  border-left: 3px solid var(--line); padding: 4px 10px; margin: 6px 0;
  background: var(--bg-soft); page-break-inside: avoid;
}
.post .hdr {
  display: flex; gap: 8px; align-items: baseline; font-size: 9pt;
  color: var(--muted); margin-bottom: 2px;
}
.post .hdr .author { color: var(--ink); font-weight: 600; }
.post .hdr .platform {
  text-transform: uppercase; letter-spacing: 0.08em; font-size: 8pt;
  background: var(--line); padding: 1px 5px; border-radius: 2px;
}
.post .hdr .eng {
  margin-left: auto; color: var(--accent); font-weight: 600;
}
.post .text { font-size: 10pt; color: var(--ink); }
.sent-bar { display: flex; height: 8px; border-radius: 2px; overflow: hidden; }
.sent-bar > div { transition: width 0.3s; }
.md-block { font-size: 10.5pt; }
.md-block h1, .md-block h2, .md-block h3 { font-weight: 700; margin-top: 12px; }
.md-block h1 { font-size: 16pt; }
.md-block h2 { font-size: 13pt; border: none; padding: 0; }
.md-block h3 { font-size: 11.5pt; }
.md-block p { margin: 6px 0; }
.md-block ul, .md-block ol { padding-left: 20px; margin: 6px 0; }
.agent-grid { display: grid; grid-template-columns: repeat(2, 1fr);
              gap: 8px; font-size: 9.5pt; }
.agent { display: flex; gap: 8px; align-items: center; }
.agent .dot {
  width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
}
.agent .name { font-weight: 600; }
.agent .role { color: var(--muted); margin-left: auto; font-size: 9pt; }
footer.colofon {
  margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--line);
  font-size: 8.5pt; color: var(--muted); text-align: center;
}
.print-bar {
  position: fixed; top: 12px; right: 12px;
  background: var(--ink); color: #fff; padding: 8px 14px;
  border-radius: 4px; font-size: 10pt; font-weight: 600;
  cursor: pointer; border: none; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
@media print {
  .print-bar { display: none; }
  .page { max-width: none; padding: 0; }
  header.cover { page-break-after: avoid; }
  h2 { page-break-after: avoid; }
  .round { break-inside: avoid; }
  body { font-size: 10pt; }
  @page { size: A4; margin: 18mm 14mm; }
}
"""


def _esc(s) -> str:
    return _html.escape(str(s) if s is not None else "")


def _build_polarization_svg(rounds: list[dict]) -> str:
    """Inline SVG line chart for polarization 0-10 across rounds."""
    if not rounds:
        return ""
    w, h = 720, 180
    pad_l, pad_r, pad_t, pad_b = 36, 16, 14, 24
    n = len(rounds)
    if n < 2:
        n = 2
    points = []
    for i, r in enumerate(rounds):
        x = pad_l + (w - pad_l - pad_r) * (i / max(1, len(rounds) - 1))
        y = pad_t + (h - pad_t - pad_b) * (1 - max(0, min(10, r.get("polarization", 0))) / 10)
        points.append((x, y, r.get("round", i + 1), r.get("polarization", 0)))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
    grid = ""
    for v in (0, 2.5, 5, 7.5, 10):
        gy = pad_t + (h - pad_t - pad_b) * (1 - v / 10)
        grid += (f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w-pad_r}" y2="{gy:.1f}" '
                 f'stroke="#e2e8f0" stroke-width="0.5"/>'
                 f'<text x="{pad_l-6}" y="{gy+3:.1f}" text-anchor="end" '
                 f'font-size="9" fill="#94a3b8">{v}</text>')
    dots = ""
    for x, y, rnd, val in points:
        dots += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2563eb"/>'
                 f'<text x="{x:.1f}" y="{h-pad_b+14}" text-anchor="middle" '
                 f'font-size="9" fill="#475569">R{rnd}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'{grid}<polyline points="{poly}" fill="none" stroke="#2563eb" '
            f'stroke-width="2"/>{dots}</svg>')


def _build_sentiment_svg(rounds: list[dict]) -> str:
    """Stacked area-ish bars for positive/neutral/negative sentiment per round."""
    if not rounds:
        return ""
    w, h = 720, 160
    pad_l, pad_r, pad_t, pad_b = 36, 16, 14, 24
    bar_area = h - pad_t - pad_b
    n = max(1, len(rounds))
    bw = (w - pad_l - pad_r) / n
    bars = ""
    for i, r in enumerate(rounds):
        s = r.get("sentiment") or {"positive": 0, "neutral": 1, "negative": 0}
        pos = float(s.get("positive", 0)); neu = float(s.get("neutral", 0)); neg = float(s.get("negative", 0))
        tot = pos + neu + neg or 1
        pos_h = bar_area * (pos / tot); neu_h = bar_area * (neu / tot); neg_h = bar_area * (neg / tot)
        x = pad_l + i * bw + 3
        bw_inner = max(2, bw - 6)
        y = pad_t
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw_inner:.1f}" height="{pos_h:.1f}" fill="#16a34a"/>'
        y += pos_h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw_inner:.1f}" height="{neu_h:.1f}" fill="#94a3b8"/>'
        y += neu_h
        bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw_inner:.1f}" height="{neg_h:.1f}" fill="#dc2626"/>'
        bars += (f'<text x="{x + bw_inner/2:.1f}" y="{h-pad_b+14}" text-anchor="middle" '
                 f'font-size="9" fill="#475569">R{r.get("round", i+1)}</text>')
    legend = ('<g font-size="9" fill="#475569">'
              '<rect x="36" y="2" width="9" height="9" fill="#16a34a"/><text x="48" y="10">Positivo</text>'
              '<rect x="110" y="2" width="9" height="9" fill="#94a3b8"/><text x="122" y="10">Neutro</text>'
              '<rect x="170" y="2" width="9" height="9" fill="#dc2626"/><text x="182" y="10">Negativo</text>'
              '</g>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'{legend}{bars}</svg>')


def _markdown_to_html_lite(md: str) -> str:
    """Minimal markdown → HTML (headings, bold, italic, lists, paragraphs).
    Avoids a runtime dependency on markdown libs in the report path."""
    if not md:
        return ""
    lines = md.split("\n")
    out: list[str] = []
    in_ul = False
    for raw in lines:
        line = raw.rstrip()
        if not line:
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append("")
            continue
        if line.startswith("### "):
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(f"<h3>{_esc(line[4:])}</h3>")
        elif line.startswith("## "):
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(f"<h2>{_esc(line[3:])}</h2>")
        elif line.startswith("# "):
            if in_ul: out.append("</ul>"); in_ul = False
            out.append(f"<h1>{_esc(line[2:])}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            content = _esc(line[2:])
            content = content.replace("**", "")
            out.append(f"<li>{content}</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            content = _esc(line)
            out.append(f"<p>{content}</p>")
    if in_ul: out.append("</ul>")
    # Inline bold/italic — applied after escape.
    text = "\n".join(out)
    import re as _re
    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = _re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def _build_html_body(
    *,
    config,
    domain,
    round_results: list[dict],
    elite_agents: list,
    citizen_swarm,
    md_text: str,
    cost: Optional[float],
) -> str:
    name = _esc(getattr(config, "name", "Scenario"))
    brief = _esc(getattr(config, "brief", "") or "")
    domain_id = _esc(getattr(config, "domain", ""))
    num_rounds = len(round_results)
    today = _dt.datetime.now().strftime("%d %B %Y")
    final = round_results[-1] if round_results else {}
    final_pol = float(final.get("polarization", 0)) if final else 0.0
    initial_pol = float(round_results[0].get("polarization", 0)) if round_results else 0.0
    pol_delta = final_pol - initial_pol
    total_posts = sum(int(r.get("posts", 0)) for r in round_results)
    total_reactions = sum(int(r.get("reactions", 0)) for r in round_results)
    total_cost = cost if cost is not None else float(final.get("cost", 0))

    # KPIs (custom_metrics last vs first)
    custom_kpis: list[tuple[str, float, Optional[float]]] = []
    if round_results:
        last_m = (round_results[-1].get("custom_metrics") or {})
        first_m = (round_results[0].get("custom_metrics") or {})
        for k, v in last_m.items():
            try:
                v_f = float(v)
                first_v = first_m.get(k)
                d = (v_f - float(first_v)) if first_v is not None else None
                custom_kpis.append((k, v_f, d))
            except (TypeError, ValueError):
                continue

    pol_chart = _build_polarization_svg(round_results)
    sent_chart = _build_sentiment_svg(round_results)
    md_html = _markdown_to_html_lite(md_text)

    # Round cards
    round_blocks: list[str] = []
    for r in round_results:
        rnd = _esc(r.get("round", "?"))
        label = _esc(r.get("timeline_label", ""))
        ev = _esc((r.get("event") or "")[:600])
        pol = float(r.get("polarization", 0))
        posts_n = int(r.get("posts", 0))
        sent = r.get("sentiment") or {"positive": 0, "neutral": 1, "negative": 0}
        pos_pct = float(sent.get("positive", 0)) * 100
        neu_pct = float(sent.get("neutral", 0)) * 100
        neg_pct = float(sent.get("negative", 0)) * 100
        top_posts_html = ""
        for p in (r.get("top_posts") or [])[:5]:
            author = _esc(p.get("author_name") or p.get("author_id") or "?")
            platform = _esc((p.get("platform") or "social").upper())
            text = _esc((p.get("text") or "")[:600])
            eng = int(p.get("total_engagement", 0))
            top_posts_html += (
                f'<div class="post">'
                f'<div class="hdr"><span class="author">{author}</span>'
                f'<span class="platform">{platform}</span>'
                f'<span class="eng" title="Engagement totale">★ {eng:,}</span></div>'
                f'<div class="text">{text}</div></div>'
            )
        if not top_posts_html:
            top_posts_html = '<div class="event"><em>Nessun post in questo round.</em></div>'
        custom_html = ""
        cm = r.get("custom_metrics") or {}
        if cm:
            chips = " ".join(
                f'<span class="round-meta"><strong>{_esc(k)}</strong>: {_esc(v)}</span>'
                for k, v in cm.items()
            )
            custom_html = f'<div class="round-meta">{chips}</div>'
        round_blocks.append(
            f'<div class="round">'
            f'<div class="round-head">'
            f'<span class="round-num">R{rnd}</span>'
            f'<span class="round-title">{label}</span>'
            f'<span class="round-meta">polar. {pol:.1f}/10 · {posts_n} post</span>'
            f'</div>'
            f'<div class="event">{ev}</div>'
            f'<div class="sent-bar">'
            f'<div style="width:{pos_pct:.1f}%;background:#16a34a"></div>'
            f'<div style="width:{neu_pct:.1f}%;background:#94a3b8"></div>'
            f'<div style="width:{neg_pct:.1f}%;background:#dc2626"></div>'
            f'</div>'
            f'{custom_html}'
            f'<h3>Top post virali</h3>'
            f'<div class="posts">{top_posts_html}</div>'
            f'</div>'
        )

    # Agents panel
    agent_html = ""
    for a in elite_agents[:24]:
        pos = getattr(a, "position", 0.0)
        color = "#16a34a" if pos > 0.2 else ("#dc2626" if pos < -0.2 else "#94a3b8")
        agent_html += (
            f'<div class="agent">'
            f'<span class="dot" style="background:{color}"></span>'
            f'<span class="name">{_esc(getattr(a, "name", "?"))}</span>'
            f'<span class="role">{_esc(getattr(a, "role", ""))} · pos {pos:+.2f}</span>'
            f'</div>'
        )

    kpi_blocks = (
        f'<div class="kpi"><div class="label">Polarizzazione finale</div>'
        f'<div class="value">{final_pol:.1f}<span class="delta {"up" if pol_delta>=0 else "down"}">'
        f'{pol_delta:+.1f}</span></div></div>'
        f'<div class="kpi"><div class="label">Post totali</div>'
        f'<div class="value">{total_posts:,}</div></div>'
        f'<div class="kpi"><div class="label">Reazioni totali</div>'
        f'<div class="value">{total_reactions:,}</div></div>'
    )
    for k, v, d in custom_kpis[:6]:
        delta_html = ""
        if d is not None and abs(d) > 0.01:
            cls = "up" if d > 0 else "down"
            delta_html = f'<span class="delta {cls}">{d:+.1f}</span>'
        kpi_blocks += (
            f'<div class="kpi"><div class="label">{_esc(k)}</div>'
            f'<div class="value">{v:.0f}{delta_html}</div></div>'
        )

    md_section = f'<section><h2>Sintesi narrativa</h2><div class="md-block">{md_html}</div></section>' if md_html else ""

    # ── ALM section (banking domain only) ─────────────────────────────
    alm_html = ""
    twin_states = [r.get("financial_twin") for r in round_results if r.get("financial_twin")]
    if twin_states:
        baseline_nim = twin_states[0]["nim_pct"]
        baseline_cet1 = twin_states[0]["cet1_pct"]
        baseline_lcr = twin_states[0]["lcr_pct"]
        baseline_dep = twin_states[0]["deposit_balance"]
        last = twin_states[-1]
        rows = ""
        for st in twin_states:
            br = ", ".join(st.get("breaches", []))
            br_html = f'<span style="color:#dc2626;font-weight:600">{_esc(br)}</span>' if br else "—"
            rows += (
                f'<tr><td style="padding:4px 8px">R{st["round"]}</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["nim_pct"]:.2f}%</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["cet1_pct"]:.1f}%</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["lcr_pct"]:.0f}%</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["deposit_balance"]:.3f}</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["loan_demand_index"]:.2f}</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["deposit_runoff_round_pct"]*100:.2f}%</td>'
                f'<td style="padding:4px 8px;text-align:right">{st["policy_rate_pct"]:.2f}%</td>'
                f'<td style="padding:4px 8px">{br_html}</td></tr>'
            )
        cumulative_dep_drop = (1 - last["deposit_balance"] / max(baseline_dep, 1e-9)) * 100
        alm_html = (
            f'<section><h2>Stato ALM (Asset-Liability Management)</h2>'
            f'<p style="color:#475569;font-size:10pt;margin-bottom:8px">'
            f'Modello deterministico parametrato su benchmark italiani 2025 '
            f'(deposit β EBA, elasticità credito al consumo Bonaccorsi/Magri, '
            f'NIM/CET1/LCR EBA Risk Dashboard). I numeri stanno entro vincoli '
            f'ALM realistici per banca commerciale italiana media.</p>'
            f'<table style="width:100%;border-collapse:collapse;font-size:10pt;">'
            f'<thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">'
            f'<th style="padding:6px 8px;text-align:left">Round</th>'
            f'<th style="padding:6px 8px;text-align:right">NIM</th>'
            f'<th style="padding:6px 8px;text-align:right">CET1</th>'
            f'<th style="padding:6px 8px;text-align:right">LCR</th>'
            f'<th style="padding:6px 8px;text-align:right">Depositi</th>'
            f'<th style="padding:6px 8px;text-align:right">Domanda credito</th>'
            f'<th style="padding:6px 8px;text-align:right">Runoff/round</th>'
            f'<th style="padding:6px 8px;text-align:right">Policy rate</th>'
            f'<th style="padding:6px 8px;text-align:left">Breach</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            f'<p style="margin-top:8px;font-size:10pt;color:#475569">'
            f'<strong>Variazione cumulativa depositi:</strong> {-cumulative_dep_drop:+.2f}% '
            f'(baseline normalizzato 1.000). '
            f'<strong>NIM Δ:</strong> {(last["nim_pct"] - baseline_nim)*100:+.0f}bp · '
            f'<strong>CET1 Δ:</strong> {last["cet1_pct"] - baseline_cet1:+.1f}pp · '
            f'<strong>LCR Δ:</strong> {last["lcr_pct"] - baseline_lcr:+.0f}pp.</p>'
            f'</section>'
        )

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>{name} — Report</title>
<style>{_CSS}</style>
</head>
<body>
<button class="print-bar" onclick="window.print()">📄 Stampa / Salva PDF</button>
<div class="page">
  <header class="cover">
    <div class="eyebrow">DigitalTwinSim · Report di simulazione</div>
    <h1>{name}</h1>
    <div class="brief">{brief}</div>
    <div class="meta">
      <span><strong>Dominio:</strong> {domain_id or '—'}</span>
      <span><strong>Round:</strong> {num_rounds}</span>
      <span><strong>Costo LLM:</strong> ${total_cost:.3f}</span>
      <span><strong>Data:</strong> {today}</span>
    </div>
  </header>

  <section>
    <h2>KPI principali</h2>
    <div class="grid grid-3">{kpi_blocks}</div>
  </section>

  <section>
    <h2>Polarizzazione (0–10)</h2>
    <div class="chart">{pol_chart}</div>
  </section>

  <section>
    <h2>Sentiment per round</h2>
    <div class="chart">{sent_chart}</div>
  </section>

  {alm_html}

  {md_section}

  <section>
    <h2>Cronaca per round</h2>
    {''.join(round_blocks)}
  </section>

  <section>
    <h2>Agenti élite</h2>
    <div class="agent-grid">{agent_html}</div>
  </section>

  <footer class="colofon">
    Generato da DigitalTwinSim · {today} · Premi <kbd>⌘P</kbd> / <kbd>Ctrl+P</kbd> per esportare in PDF
  </footer>
</div>
</body>
</html>
"""
