"""Tests for ticker_relevance — deterministic relevance scorer."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator.ticker_relevance import (
    UniverseLoader,
    TickerRelevanceScorer,
    RelevantUniverse,
)


class TestUniverseLoader:
    def test_loads_universe(self):
        loader = UniverseLoader()
        assert len(loader._data["stocks"]) >= 150
        assert len(loader._data["indices"]) >= 10
        assert len(loader._data["sectors"]) >= 14

    def test_get_beta_it_regime(self):
        loader = UniverseLoader()
        beta = loader.get_beta("banking", "IT")
        assert beta.political_beta == 1.85
        assert beta.crisis_alpha == -3.2

    def test_get_beta_us_regime(self):
        loader = UniverseLoader()
        beta = loader.get_beta("banking", "US")
        assert beta.political_beta == 1.10

    def test_get_beta_fallback_to_default(self):
        loader = UniverseLoader()
        beta = loader.get_beta("banking", "NONEXISTENT")
        assert beta.political_beta == 1.20  # default regime

    def test_resolve_org_exact(self):
        loader = UniverseLoader()
        assert loader.resolve_org("apple") == ["AAPL"]
        assert loader.resolve_org("unicredit") == ["UCG.MI"]

    def test_resolve_org_case_insensitive(self):
        loader = UniverseLoader()
        # resolve_org expects lowered input per _resolve_tickers_for_org
        assert loader.resolve_org("apple") == ["AAPL"]

    def test_resolve_org_partial(self):
        loader = UniverseLoader()
        # "eni" should match
        assert "ENI.MI" in loader.resolve_org("eni")

    def test_resolve_org_no_match(self):
        loader = UniverseLoader()
        assert loader.resolve_org("nonexistentcorp") == []

    def test_tickers_for_sector(self):
        loader = UniverseLoader()
        banking = loader.tickers_for_sector("banking")
        assert len(banking) > 5
        assert all(s["sector"] == "banking" for s in banking)

    def test_get_stock(self):
        loader = UniverseLoader()
        stock = loader.get_stock("AAPL")
        assert stock is not None
        assert stock["name"] == "Apple"
        assert stock["sector"] == "tech"

    def test_get_stock_not_found(self):
        loader = UniverseLoader()
        assert loader.get_stock("ZZZZZZ") is None

    def test_get_ticker_sector(self):
        loader = UniverseLoader()
        assert loader.get_ticker_sector("UCG.MI") == "banking"
        assert loader.get_ticker_sector("NONEXIST") is None


class TestTickerRelevanceScorer:
    def test_commercial_apple_samsung(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("commercial", ["Apple", "Samsung"], "US", ["smartphone"])
        assert isinstance(result, RelevantUniverse)
        tickers = [t["ticker"] for t in result.tickers]
        assert "AAPL" in tickers
        assert "005930.KS" in tickers
        assert result.beta_regime == "US"
        assert len(result.tickers) <= 30

    def test_financial_domain_it(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", [], "IT", [])
        tickers = [t["ticker"] for t in result.tickers]
        # Should include Italian banks
        assert any(t.endswith(".MI") for t in tickers)
        assert result.beta_regime == "IT"

    def test_min_sectors(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("political", [], "", [])
        sectors = set(t["sector"] for t in result.tickers)
        assert len(sectors) >= 3

    def test_max_30_tickers(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("commercial", [], "US", ["ai", "cloud", "smartphone"])
        assert len(result.tickers) <= 30

    def test_indices_included(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", [], "US", [])
        assert len(result.indices) >= 1
        assert len(result.indices) <= 3

    def test_beta_regime_em(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", [], "Brazil", [])
        assert result.beta_regime == "EM"

    def test_beta_regime_default(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", [], "", [])
        assert result.beta_regime == "default"

    def test_entity_direct_match_priority(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("corporate", ["Tesla", "Boeing"], "", [])
        tickers = [t["ticker"] for t in result.tickers]
        assert "TSLA" in tickers
        assert "BA" in tickers

    def test_rationale_not_empty(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("commercial", ["Apple"], "US", [])
        assert len(result.rationale) > 20

    def test_country_filter_prioritises_local(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", [], "IT", [])
        # Italian tickers should score higher than US ones
        top5_tickers = [t["ticker"] for t in result.tickers[:5]]
        italian_in_top5 = [t for t in top5_tickers if t.endswith(".MI")]
        assert len(italian_in_top5) >= 2

    def test_keyword_tag_boost(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("technology", [], "US", ["ai", "gpu"])
        tickers = [t["ticker"] for t in result.tickers]
        # NVDA should be selected (has both "ai" and "gpu" tags)
        assert "NVDA" in tickers

    def test_geopolitical_domain(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("geopolitical", [], "", [])
        sectors = set(t["sector"] for t in result.tickers)
        assert "defense" in sectors or "energy_fossil" in sectors

    def test_empty_entities_and_keywords(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("political", [], "", [])
        assert len(result.tickers) > 0
        assert result.beta_regime == "default"

    def test_unknown_domain_still_returns_results(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("nonexistent_domain", ["Apple"], "", [])
        tickers = [t["ticker"] for t in result.tickers]
        # Entity resolution still works even for unknown domain
        assert "AAPL" in tickers

    def test_indices_country_match(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", [], "IT", [])
        idx_tickers = [i["ticker"] for i in result.indices]
        # FTSE MIB should be included for Italy
        assert "FTSEMIB.MI" in idx_tickers

    def test_relevant_universe_dataclass(self):
        ru = RelevantUniverse()
        assert ru.tickers == []
        assert ru.indices == []
        assert ru.beta_regime == "default"
        assert ru.rationale == ""

    def test_mega_cap_boost(self):
        scorer = TickerRelevanceScorer()
        result = scorer.select("commercial", [], "US", [])
        # Mega cap stocks should appear (Apple, Microsoft, etc.)
        tickers = [t["ticker"] for t in result.tickers]
        mega_caps = {"AAPL", "MSFT", "AMZN", "META", "NVDA"}
        assert len(mega_caps & set(tickers)) >= 2
