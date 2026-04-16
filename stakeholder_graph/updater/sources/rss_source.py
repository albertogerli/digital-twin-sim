"""RSS feed source — fetches articles from configured RSS feeds."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import ssl

import aiohttp
import certifi
import feedparser
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RawArticle(BaseModel):
    """A raw article fetched from a news source."""
    url: str
    title: str
    body: str = ""              # extracted text (first N chars)
    source_name: str = ""
    published: Optional[datetime] = None
    language: str = "it"
    fetch_timestamp: datetime = None

    class Config:
        # allow datetime default
        validate_default = True

    def __init__(self, **data):
        if data.get("fetch_timestamp") is None:
            data["fetch_timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


class RSSSource:
    """Fetches articles from a single RSS feed."""

    def __init__(self, name: str, url: str, language: str = "it", rate_limit: float = 1.0):
        self.name = name
        self.url = url
        self.language = language
        self.rate_limit = rate_limit

    async def fetch(self, since: Optional[datetime] = None) -> list[RawArticle]:
        """Fetch articles from RSS feed, optionally filtering by date."""
        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning(f"RSS {self.name}: HTTP {resp.status}")
                        return []
                    text = await resp.text()
        except Exception as e:
            logger.error(f"RSS {self.name} fetch error: {e}")
            return []

        feed = feedparser.parse(text)
        articles = []

        for entry in feed.entries:
            # Parse published date
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass

            # Filter by date
            if since and pub_date and pub_date < since:
                continue

            # Extract body from summary/content
            body = ""
            if hasattr(entry, "content") and entry.content:
                body = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                body = entry.summary or ""

            # Strip HTML tags (basic)
            import re
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()

            url = entry.get("link", "")
            if not url:
                continue

            articles.append(RawArticle(
                url=url,
                title=entry.get("title", ""),
                body=body[:3000],  # cap body length
                source_name=self.name,
                published=pub_date,
                language=self.language,
            ))

        logger.info(f"RSS {self.name}: {len(articles)} articles")
        return articles


class GoogleNewsSource:
    """Fetches Google News RSS for specific stakeholder queries."""

    BASE_URL = "https://news.google.com/rss/search"

    def __init__(self, language: str = "it", region: str = "IT", rate_limit: float = 2.0):
        self.language = language
        self.region = region
        self.rate_limit = rate_limit

    async def fetch_for_stakeholder(
        self, name: str, since: Optional[datetime] = None
    ) -> list[RawArticle]:
        """Fetch recent Google News articles mentioning a stakeholder."""
        import urllib.parse
        query = urllib.parse.quote(f'"{name}"')
        url = f"{self.BASE_URL}?q={query}&hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}"

        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; StakeholderGraph/1.0)"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        return []
                    text = await resp.text()
        except Exception as e:
            logger.debug(f"Google News for '{name}': {e}")
            return []

        feed = feedparser.parse(text)
        articles = []

        for entry in feed.entries[:10]:  # cap at 10 per stakeholder
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass

            if since and pub_date and pub_date < since:
                continue

            articles.append(RawArticle(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                body=entry.get("summary", "")[:1000],
                source_name=f"Google News ({name})",
                published=pub_date,
                language=self.language,
            ))

        return articles

    async def fetch_batch(
        self,
        names: list[str],
        since: Optional[datetime] = None,
        concurrency: int = 5,
    ) -> list[RawArticle]:
        """Fetch Google News for multiple stakeholders with rate limiting."""
        semaphore = asyncio.Semaphore(concurrency)
        all_articles = []

        async def _fetch_one(name: str):
            async with semaphore:
                articles = await self.fetch_for_stakeholder(name, since)
                await asyncio.sleep(self.rate_limit)
                return articles

        tasks = [_fetch_one(n) for n in names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"Google News batch error: {result}")

        logger.info(f"Google News batch: {len(all_articles)} articles for {len(names)} stakeholders")
        return all_articles
