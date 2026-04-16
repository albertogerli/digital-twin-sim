"""Update Pipeline — orchestrates the full nightly update cycle.

    fetch → match → analyze → update → validate → persist

Each stage is independent and testable. The pipeline ties them together
with logging, budget controls, and error recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from stakeholder_graph.db import StakeholderDB
from stakeholder_graph.updater.analysis.article_analyzer import ArticleAnalyzer
from stakeholder_graph.updater.config import UpdaterConfig
from stakeholder_graph.updater.matching.entity_matcher import EntityMatcher
from stakeholder_graph.updater.persistence.changelog import Changelog
from stakeholder_graph.updater.persistence.writer import Writer
from stakeholder_graph.updater.sources.rss_source import GoogleNewsSource, RSSSource, RawArticle
from stakeholder_graph.updater.update.position_updater import PositionUpdater
from stakeholder_graph.updater.update.validator import Validator

logger = logging.getLogger(__name__)


@dataclass
class UpdateReport:
    """Summary of a pipeline run."""
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    # Stage metrics
    articles_fetched: int = 0
    articles_deduplicated: int = 0
    stakeholders_matched: int = 0
    total_matches: int = 0
    llm_calls: int = 0
    signals_extracted: int = 0

    # Update metrics
    updates_computed: int = 0
    updates_approved: int = 0
    updates_flagged: int = 0
    updates_rejected: int = 0
    position_changes: int = 0
    influence_changes: int = 0
    quotes_added: int = 0

    # Budget
    estimated_cost_usd: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class UpdatePipeline:
    """Main orchestrator for the nightly graph update.

    Usage:
        config = UpdaterConfig()
        db = StakeholderDB()
        pipeline = UpdatePipeline(config, db, llm_client)
        report = await pipeline.run()
    """

    def __init__(
        self,
        config: UpdaterConfig,
        db: StakeholderDB,
        llm_client=None,  # GeminiClient or compatible
    ):
        self.config = config
        self.db = db
        self.llm = llm_client

        # State
        self._state_path = Path(db.data_dir) / config.state_path.replace("data/", "")
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        since: Optional[datetime] = None,
        stakeholder_ids: Optional[list[str]] = None,
    ) -> UpdateReport:
        """Execute the full update pipeline.

        Args:
            since: Only process articles published after this time.
                   Defaults to last successful run time or 24h ago.
            stakeholder_ids: If set, only update these specific stakeholders.
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        started = datetime.now(timezone.utc)
        report = UpdateReport(
            run_id=run_id,
            started_at=started.isoformat(),
        )

        since = since or self._last_run_time()
        logger.info(f"=== Pipeline run {run_id} | since={since.isoformat()} ===")

        try:
            # ── Stage 1: Fetch ─────────────────────────────────────────
            articles = await self._fetch(since, report)

            # ── Stage 2: Match ─────────────────────────────────────────
            all_stakeholders = self.db.all()
            if stakeholder_ids:
                all_stakeholders = [s for s in all_stakeholders if s.id in stakeholder_ids]

            matcher = EntityMatcher(all_stakeholders, min_body_mentions=self.config.min_name_occurrences)
            matches = matcher.match(articles)

            # Cap articles per stakeholder
            for sid in list(matches.keys()):
                matches[sid] = matches[sid][:self.config.max_articles_per_stakeholder]

            report.stakeholders_matched = len(matches)
            report.total_matches = sum(len(v) for v in matches.values())

            if not matches:
                logger.info("No matches found, pipeline complete")
                self._save_report(report, started)
                return report

            # ── Stage 3: Analyze (LLM) ─────────────────────────────────
            if not self.llm:
                logger.warning("No LLM client provided, skipping analysis")
                self._save_report(report, started)
                return report

            analyzer = ArticleAnalyzer(
                self.llm,
                concurrency=self.config.llm_concurrency,
                rate_limit_delay=self.config.llm_rate_limit_delay,
                max_article_chars=self.config.max_article_chars,
                max_calls_per_run=self.config.max_llm_calls_per_run,
            )

            stakeholder_lookup = {s.id: s for s in all_stakeholders}
            analyses = await analyzer.analyze_batch(
                matches, stakeholder_lookup,
                max_articles_per_stakeholder=self.config.max_articles_per_stakeholder,
            )

            report.llm_calls = analyzer.calls_made
            report.signals_extracted = sum(len(a.position_signals) for a in analyses)

            # ── Stage 4: Compute Updates ───────────────────────────────
            updater = PositionUpdater(self.config)
            updates = updater.compute_all(analyses, stakeholder_lookup)
            report.updates_computed = len(updates)

            # ── Stage 5: Validate ──────────────────────────────────────
            validator = Validator(
                max_drift=self.config.max_drift_per_run,
            )
            result = validator.check(updates)

            report.updates_approved = len(result.approved)
            report.updates_flagged = len(result.flagged)
            report.updates_rejected = len(result.rejected)
            report.position_changes = sum(len(u.position_deltas) for u in result.approved)
            report.influence_changes = sum(1 for u in result.approved if u.influence_delta)
            report.quotes_added = sum(len(u.quote_deltas) for u in result.approved)

            # ── Stage 6: Persist ───────────────────────────────────────
            if self.config.dry_run:
                logger.info("DRY RUN — skipping persistence")
            elif result.approved:
                changelog = Changelog(Path(self.db.data_dir) / ".changelog.jsonl")
                writer = Writer(
                    Path(self.db.data_dir),
                    backup=self.config.backup_on_write,
                )

                writer.commit(result.approved, run_id)
                changelog.append_all(result.approved, run_id)
                changelog.append_run_summary(run_id, report.to_dict())

            self._save_state(run_id, started)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            report.errors.append(str(e))

        self._save_report(report, started)
        return report

    async def _fetch(self, since: datetime, report: UpdateReport) -> list[RawArticle]:
        """Stage 1: Fetch articles from all sources."""
        all_articles: list[RawArticle] = []

        # RSS feeds
        rss_tasks = []
        for feed_cfg in self.config.rss_feeds:
            if not feed_cfg.get("enabled", True):
                continue
            source = RSSSource(
                name=feed_cfg["name"],
                url=feed_cfg["url"],
                language=feed_cfg.get("language", "it"),
            )
            rss_tasks.append(source.fetch(since))

        rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
        for result in rss_results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"RSS fetch error: {result}")

        # Google News for tier-1 stakeholders
        if self.config.google_news_enabled:
            tier1 = self.db.query(min_tier=1, min_influence=0.5)
            if tier1:
                gn = GoogleNewsSource(
                    language=self.config.google_news_language,
                    region=self.config.google_news_region,
                )
                gn_articles = await gn.fetch_batch(
                    [s.name for s in tier1[:30]],  # top 30 tier-1
                    since=since,
                    concurrency=3,
                )
                all_articles.extend(gn_articles)

        report.articles_fetched = len(all_articles)

        # Deduplicate by URL
        seen_urls = set()
        deduped = []
        for a in all_articles:
            if a.url not in seen_urls:
                seen_urls.add(a.url)
                deduped.append(a)

        report.articles_deduplicated = len(deduped)
        logger.info(f"Fetched {report.articles_fetched} articles, {len(deduped)} after dedup")
        return deduped

    def _last_run_time(self) -> datetime:
        """Get the timestamp of the last successful run, or 24h ago."""
        try:
            if self._state_path.exists():
                state = json.loads(self._state_path.read_text())
                return datetime.fromisoformat(state["last_run"])
        except Exception:
            pass
        return datetime.now(timezone.utc) - timedelta(hours=24)

    def _save_state(self, run_id: str, started: datetime):
        """Save pipeline state for next run."""
        state = {
            "last_run": started.isoformat(),
            "last_run_id": run_id,
        }
        self._state_path.write_text(json.dumps(state, indent=2))

    def _save_report(self, report: UpdateReport, started: datetime):
        """Finalize and save the report."""
        finished = datetime.now(timezone.utc)
        report.finished_at = finished.isoformat()
        report.duration_seconds = (finished - started).total_seconds()

        # Estimate cost (Gemini flash-lite pricing)
        # ~1500 input tokens + ~500 output tokens per call
        input_cost = report.llm_calls * 1500 * 0.075 / 1_000_000  # $0.075/1M
        output_cost = report.llm_calls * 500 * 0.30 / 1_000_000   # $0.30/1M
        report.estimated_cost_usd = round(input_cost + output_cost, 4)

        # Save report to file
        report_path = self._state_path.parent / ".last_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

        logger.info(
            f"=== Pipeline {report.run_id} complete in {report.duration_seconds:.1f}s | "
            f"{report.articles_fetched} articles → {report.stakeholders_matched} stakeholders → "
            f"{report.position_changes} position changes | "
            f"~${report.estimated_cost_usd:.4f} ===\n"
        )
