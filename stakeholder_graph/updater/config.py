"""Configuration for the Graph Updater pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceConfig:
    """Configuration for a single news source."""
    name: str
    url: str
    language: str = "it"
    enabled: bool = True
    rate_limit_seconds: float = 1.0


# ── Default Italian RSS feeds ──────────────────────────────────────────────
DEFAULT_RSS_FEEDS: list[dict] = [
    {"name": "ANSA - Politica", "url": "https://www.ansa.it/sito/notizie/politica/politica_rss.xml"},
    {"name": "ANSA - Economia", "url": "https://www.ansa.it/sito/notizie/economia/economia_rss.xml"},
    {"name": "ANSA - Cronaca", "url": "https://www.ansa.it/sito/notizie/cronaca/cronaca_rss.xml"},
    {"name": "Repubblica - Politica", "url": "https://www.repubblica.it/rss/politica/rss2.0.xml"},
    {"name": "Repubblica - Economia", "url": "https://www.repubblica.it/rss/economia/rss2.0.xml"},
    {"name": "Corriere - Politica", "url": "https://xml2.corriereobjects.it/rss/politica.xml"},
    {"name": "Corriere - Economia", "url": "https://xml2.corriereobjects.it/rss/economia.xml"},
    {"name": "Il Sole 24 Ore", "url": "https://www.ilsole24ore.com/rss/italia.xml"},
    {"name": "Il Sole 24 Ore - Economia", "url": "https://www.ilsole24ore.com/rss/economia.xml"},
    {"name": "AGI - Politica", "url": "https://www.agi.it/politica/rss"},
    {"name": "Il Fatto Quotidiano", "url": "https://www.ilfattoquotidiano.it/feed/"},
    {"name": "Il Post", "url": "https://www.ilpost.it/feed/"},
    # EU / International
    {"name": "Politico EU", "url": "https://www.politico.eu/feed/", "language": "en"},
    {"name": "Reuters - World", "url": "https://feeds.reuters.com/reuters/worldNews", "language": "en"},
    {"name": "BBC - World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "language": "en"},
]


@dataclass
class UpdaterConfig:
    """Master configuration for the nightly update pipeline."""

    # ── Sources ──
    rss_feeds: list[dict] = field(default_factory=lambda: list(DEFAULT_RSS_FEEDS))
    google_news_enabled: bool = True
    google_news_language: str = "it"
    google_news_region: str = "IT"
    twitter_enabled: bool = False  # requires TWITTER_BEARER_TOKEN

    # ── Matching ──
    min_name_occurrences: int = 2          # surname must appear 2x in body (or 1x in title)
    match_title_boost: bool = True         # name in title = automatic match

    # ── LLM Analysis ──
    llm_model: str = "gemini-2.0-flash"
    llm_concurrency: int = 5
    llm_rate_limit_delay: float = 0.5
    max_articles_per_stakeholder: int = 5  # cap to avoid budget blowout
    max_article_chars: int = 2000          # truncate article body for LLM

    # ── EMA Update Algorithm ──
    ema_alpha: float = 0.15                # smoothing factor (lower = more conservative)
    max_drift_per_run: float = 0.10        # max position change in a single night
    rigidity_dampening: bool = True        # high-rigidity stakeholders get extra dampening
    strength_weights: dict = field(default_factory=lambda: {
        "strong": 0.30,
        "moderate": 0.15,
        "weak": 0.05,
    })

    # ── Confidence Upgrade Thresholds ──
    confidence_upgrade_signals: dict = field(default_factory=lambda: {
        "low_to_medium": 3,    # 3 concordant signals upgrade low → medium
        "medium_to_high": 7,   # 7 concordant signals upgrade medium → high
    })

    # ── Tier Scheduling ──
    tier1_frequency: str = "daily"         # Tier 1 checked every night
    tier2_frequency: str = "daily"         # Tier 2 checked daily via RSS, weekly via Google News
    tier2_google_news_batch_size: int = 80 # ~80 Tier 2 per night via Google News (7-day rotation)

    # ── Staleness Detection ──
    staleness_days: int = 90               # flag stakeholder if no articles in 90 days
    auto_deactivate_days: int = 180        # auto set active=false after 180 days silence

    # ── Persistence ──
    changelog_path: str = "data/.changelog.jsonl"
    state_path: str = "data/.updater_state.json"
    lockfile_path: str = "data/.updater.lock"
    backup_on_write: bool = True

    # ── Budget ──
    max_llm_calls_per_run: int = 500       # safety cap
    max_cost_per_run_usd: float = 2.0      # emergency stop

    # ── Dry Run ──
    dry_run: bool = False
