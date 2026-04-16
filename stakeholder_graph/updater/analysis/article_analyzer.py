"""Article Analyzer — LLM-based extraction of position signals from articles."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from stakeholder_graph.schema import Stakeholder
from stakeholder_graph.updater.analysis.signal import (
    ArticleAnalysis,
    InfluenceSignal,
    PositionSignal,
    QuoteSignal,
    RelationshipSignal,
)
from stakeholder_graph.updater.analysis.prompts import (
    ARTICLE_ANALYSIS_PROMPT,
    STANDARD_TOPICS,
    SYSTEM_PROMPT,
)
from stakeholder_graph.updater.sources.rss_source import RawArticle

logger = logging.getLogger(__name__)


class ArticleAnalyzer:
    """Analyzes articles using LLM to extract stakeholder signals."""

    def __init__(
        self,
        llm_client,  # GeminiClient or compatible
        concurrency: int = 5,
        rate_limit_delay: float = 0.5,
        max_article_chars: int = 2000,
        max_calls_per_run: int = 500,
    ):
        self.llm = llm_client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_limit_delay = rate_limit_delay
        self.max_article_chars = max_article_chars
        self.max_calls_per_run = max_calls_per_run
        self._call_count = 0

    async def analyze_one(
        self,
        stakeholder: Stakeholder,
        article: RawArticle,
    ) -> Optional[ArticleAnalysis]:
        """Analyze a single article for signals about one stakeholder."""
        if self._call_count >= self.max_calls_per_run:
            logger.warning("Max LLM calls reached, skipping")
            return None

        # Format current positions
        positions_json = json.dumps(
            [{"topic": p.topic_tag, "value": p.value, "confidence": p.confidence}
             for p in stakeholder.positions],
            ensure_ascii=False,
        )

        prompt = ARTICLE_ANALYSIS_PROMPT.format(
            name=stakeholder.name,
            role=stakeholder.role,
            party_or_org=stakeholder.party_or_org,
            positions_json=positions_json,
            title=article.title,
            source_name=article.source_name,
            published=article.published.isoformat() if article.published else "unknown",
            body=article.body[:self.max_article_chars],
            available_topics=", ".join(STANDARD_TOPICS),
        )

        async with self.semaphore:
            try:
                self._call_count += 1
                response = await self.llm.generate_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    schema_hint="ArticleAnalysis",
                )
                await asyncio.sleep(self.rate_limit_delay)

                if not response:
                    return None

                return self._parse_response(response, stakeholder, article)

            except Exception as e:
                logger.error(f"LLM analysis error for {stakeholder.name}: {e}")
                return None

    def _parse_response(
        self,
        response: dict,
        stakeholder: Stakeholder,
        article: RawArticle,
    ) -> Optional[ArticleAnalysis]:
        """Parse LLM JSON response into typed signals."""
        if not response.get("relevant", False):
            return None

        analysis = ArticleAnalysis(relevant=True)

        # Position signals
        for sig in response.get("position_signals", []):
            try:
                # Validate evidence is somewhat present in article
                evidence = sig.get("evidence", "")
                if evidence and len(evidence) > 10:
                    # Soft check: at least some words overlap with article
                    evidence_words = set(evidence.lower().split()[:5])
                    article_words = set(article.body.lower().split())
                    if len(evidence_words & article_words) < 2:
                        logger.debug(f"Evidence doesn't match article for {stakeholder.name}, skipping signal")
                        continue

                analysis.position_signals.append(PositionSignal(
                    stakeholder_id=stakeholder.id,
                    topic_tag=sig["topic_tag"],
                    direction=max(-1.0, min(1.0, float(sig["direction"]))),
                    strength=sig.get("strength", "moderate"),
                    evidence=evidence,
                    source_url=article.url,
                    source_name=article.source_name,
                    published=article.published,
                    is_new_topic=sig.get("is_new_topic", False),
                ))
            except (KeyError, ValueError) as e:
                logger.debug(f"Invalid position signal: {e}")

        # Quotes
        for quote in response.get("quotes", []):
            if isinstance(quote, str) and len(quote) > 10:
                analysis.quotes.append(QuoteSignal(
                    stakeholder_id=stakeholder.id,
                    quote=quote,
                    source_url=article.url,
                    source_name=article.source_name,
                    published=article.published,
                ))

        # Influence signal
        inf = response.get("influence_signal")
        if inf and isinstance(inf, dict):
            try:
                analysis.influence_signal = InfluenceSignal(
                    stakeholder_id=stakeholder.id,
                    direction=inf.get("direction", "stable"),
                    magnitude=float(inf.get("magnitude", 0.0)),
                    reason=inf.get("reason", ""),
                    source_url=article.url,
                )
            except (ValueError, KeyError):
                pass

        # Relationship signals
        for rel in response.get("relationship_signals", []):
            try:
                analysis.relationship_signals.append(RelationshipSignal(
                    source_id=stakeholder.id,
                    target_name=rel["target_name"],
                    relation_type=rel.get("relation_type", "neutral"),
                    evidence=rel.get("evidence", ""),
                    source_url=article.url,
                ))
            except KeyError:
                pass

        return analysis if analysis.position_signals or analysis.quotes else None

    async def analyze_batch(
        self,
        matches: dict[str, list[RawArticle]],
        stakeholders: dict[str, Stakeholder],
        max_articles_per_stakeholder: int = 5,
    ) -> list[ArticleAnalysis]:
        """Analyze all matched articles for all stakeholders.

        Args:
            matches: stakeholder_id -> list of matched articles
            stakeholders: stakeholder_id -> Stakeholder lookup
            max_articles_per_stakeholder: cap per stakeholder to limit costs

        Returns:
            List of non-None ArticleAnalysis results.
        """
        tasks = []
        for sid, articles in matches.items():
            stakeholder = stakeholders.get(sid)
            if not stakeholder:
                continue
            # Cap articles per stakeholder, prefer most recent
            sorted_articles = sorted(
                articles,
                key=lambda a: a.published or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )[:max_articles_per_stakeholder]

            for article in sorted_articles:
                tasks.append(self.analyze_one(stakeholder, article))

        logger.info(f"Analyzing {len(tasks)} stakeholder-article pairs")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        analyses = []
        for r in results:
            if isinstance(r, ArticleAnalysis) and r.relevant:
                analyses.append(r)
            elif isinstance(r, Exception):
                logger.debug(f"Analysis task error: {r}")

        logger.info(f"Extracted {sum(len(a.position_signals) for a in analyses)} position signals from {len(analyses)} relevant analyses")
        return analyses

    @property
    def calls_made(self) -> int:
        return self._call_count
