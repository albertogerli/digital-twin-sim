"""Ticker Relevance — deterministic relevance scorer.

Given a crisis domain, entities, country, and keywords, selects the most
relevant subset of the stock universe (max 30 tickers + up to 3 indices).

This is a pure, deterministic scorer — no LLM calls. It replaces the old
approach of hardcoding ticker lists per scenario.

Selection pipeline:
1. **Entity resolution** — resolve named orgs to tickers via UniverseLoader
2. **Domain→sector mapping** — map crisis domain to relevant sectors
3. **Country/region filter** — prioritise tickers from the affected country/region
4. **Keyword tag match** — boost tickers whose tags overlap with keywords
5. **Beta regime** — determine which beta regime (IT, US, EM, default) applies
6. **Cap at 30** — rank by relevance score, take top 30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.orchestrator.financial_impact import UniverseLoader

logger = logging.getLogger(__name__)

# ── Domain → Sector Mapping ──────────────────────────────────────────────────

DOMAIN_SECTORS: dict[str, list[str]] = {
    "political": ["banking", "sovereign_debt", "insurance", "utilities", "defense"],
    "financial": ["banking", "insurance", "sovereign_debt", "real_estate", "energy_fossil", "utilities", "tech", "automotive"],
    "commercial": ["tech", "food_consumer", "luxury", "automotive", "media"],
    "corporate": ["tech", "automotive", "energy_fossil", "infrastructure", "luxury"],
    "geopolitical": ["defense", "energy_fossil", "banking", "sovereign_debt"],
    "social": ["media", "telecom", "healthcare", "food_consumer"],
    "environmental": ["energy_fossil", "energy_renewable", "utilities", "automotive"],
    "health": ["healthcare", "food_consumer", "insurance"],
    "technology": ["tech", "telecom", "media"],
}

# ── Country → Beta Regime ────────────────────────────────────────────────────

COUNTRY_TO_REGIME: dict[str, str] = {
    "IT": "IT", "US": "US",
    # Emerging markets
    "BR": "EM", "IN": "EM", "CN": "EM", "MX": "EM", "AR": "EM",
    "TW": "EM", "KR": "EM", "CL": "EM", "HK": "EM",
    # Developed Europe → default (no special regime)
    "DE": "default", "FR": "default", "GB": "default",
    "NL": "default", "CH": "default", "DK": "default",
    "SE": "default", "FI": "default", "JP": "default", "AU": "default",
}

# Broad country text → regime mapping for free-form input
_COUNTRY_TEXT_TO_REGIME: dict[str, str] = {
    "italy": "IT", "italia": "IT", "italian": "IT",
    "united states": "US", "usa": "US", "america": "US", "american": "US",
    "brazil": "EM", "brasil": "EM", "india": "EM", "china": "EM",
    "mexico": "EM", "argentina": "EM", "taiwan": "EM", "korea": "EM",
    "chile": "EM", "hong kong": "EM",
    "germany": "default", "france": "default", "uk": "default",
    "united kingdom": "default", "japan": "default", "australia": "default",
    "netherlands": "default", "switzerland": "default", "denmark": "default",
    "sweden": "default", "finland": "default",
}

# ── Country → Region ─────────────────────────────────────────────────────────

COUNTRY_TO_REGION: dict[str, str] = {
    "IT": "europe", "DE": "europe", "FR": "europe", "GB": "europe",
    "NL": "europe", "CH": "europe", "DK": "europe", "SE": "europe",
    "FI": "europe",
    "US": "north_america",
    "JP": "asia_pacific", "KR": "asia_pacific", "CN": "asia_pacific",
    "HK": "asia_pacific", "TW": "asia_pacific", "IN": "asia_pacific",
    "AU": "asia_pacific",
    "BR": "latam", "MX": "latam", "AR": "latam", "CL": "latam",
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RelevantUniverse:
    """The selected subset of tickers + indices for a scenario."""
    tickers: list[dict] = field(default_factory=list)   # [{ticker, name, sector, score, ...}]
    indices: list[dict] = field(default_factory=list)    # [{ticker, name, country}]
    beta_regime: str = "default"
    rationale: str = ""


# ── Scorer ───────────────────────────────────────────────────────────────────

class TickerRelevanceScorer:
    """Deterministic ticker relevance scorer.

    Usage:
        scorer = TickerRelevanceScorer()
        result = scorer.select("financial", ["UniCredit"], "IT", ["banking"])
    """

    def __init__(self):
        self._loader = UniverseLoader()

    def _resolve_regime(self, country: str) -> str:
        """Map a country string to a beta regime."""
        if not country:
            return "default"
        # Try direct ISO code
        upper = country.strip().upper()
        if upper in COUNTRY_TO_REGIME:
            return COUNTRY_TO_REGIME[upper]
        # Try free-form text
        lower = country.strip().lower()
        if lower in _COUNTRY_TEXT_TO_REGIME:
            return _COUNTRY_TEXT_TO_REGIME[lower]
        return "default"

    def _resolve_country_code(self, country: str) -> str | None:
        """Map free-form country to ISO code for region filtering."""
        if not country:
            return None
        upper = country.strip().upper()
        if upper in COUNTRY_TO_REGIME:
            return upper
        lower = country.strip().lower()
        # Reverse lookup
        text_to_code = {
            "italy": "IT", "italia": "IT", "italian": "IT",
            "united states": "US", "usa": "US", "america": "US", "american": "US",
            "brazil": "BR", "brasil": "BR", "india": "IN", "china": "CN",
            "mexico": "MX", "argentina": "AR", "taiwan": "TW", "korea": "KR",
            "chile": "CL", "hong kong": "HK",
            "germany": "DE", "france": "FR", "uk": "GB",
            "united kingdom": "GB", "japan": "JP", "australia": "AU",
            "netherlands": "NL", "switzerland": "CH", "denmark": "DK",
            "sweden": "SE", "finland": "FI",
        }
        return text_to_code.get(lower)

    def select(
        self,
        domain: str,
        entities: list[str],
        country: str,
        keywords: list[str],
        *,
        max_tickers: int = 30,
        max_indices: int = 3,
    ) -> RelevantUniverse:
        """Select the most relevant tickers for a scenario.

        Args:
            domain: Crisis domain (political, financial, commercial, etc.)
            entities: Named organisations (e.g. ["Apple", "Samsung"])
            country: Country code or name (e.g. "IT", "Brazil")
            keywords: Topic keywords (e.g. ["smartphone", "ai"])
            max_tickers: Cap on returned tickers (default 30)
            max_indices: Cap on returned indices (default 3)

        Returns:
            RelevantUniverse with scored tickers, indices, beta regime, rationale.
        """
        regime = self._resolve_regime(country)
        country_code = self._resolve_country_code(country)
        region = COUNTRY_TO_REGION.get(country_code, "") if country_code else ""

        # 1. Entity resolution → direct matches (highest priority)
        direct_tickers: dict[str, float] = {}  # ticker → score
        for entity in entities:
            resolved = self._loader.resolve_org(entity.lower())
            for t in resolved:
                direct_tickers[t] = direct_tickers.get(t, 0) + 10.0

        # 2. Domain → sector expansion
        domain_lower = domain.lower().strip()
        relevant_sectors = DOMAIN_SECTORS.get(domain_lower, [])

        # 3. Score all stocks
        scored: dict[str, float] = dict(direct_tickers)
        all_stocks = self._loader._data.get("stocks", [])
        keyword_set = {k.lower() for k in keywords}

        for stock in all_stocks:
            ticker = stock["ticker"]
            score = scored.get(ticker, 0.0)

            # Sector relevance
            if stock.get("sector") in relevant_sectors:
                sector_idx = relevant_sectors.index(stock["sector"])
                score += 5.0 - (sector_idx * 0.3)  # primary sectors score higher

            # Country match
            if country_code and stock.get("country") == country_code:
                score += 3.0
            elif region and stock.get("region") == region:
                score += 1.0

            # Keyword tag match
            tags = set(stock.get("tags", []))
            overlap = tags & keyword_set
            score += len(overlap) * 2.0

            # Market cap boost (mega > large > mid)
            cap_tier = stock.get("market_cap_tier", "mid")
            if cap_tier == "mega":
                score += 1.0
            elif cap_tier == "large":
                score += 0.5

            if score > 0:
                scored[ticker] = score

        # 4. Ensure minimum sector diversity (at least 3 sectors)
        if relevant_sectors and len(relevant_sectors) < 3:
            # Add fallback sectors
            fallback = ["banking", "tech", "energy_fossil", "healthcare", "utilities"]
            for fb in fallback:
                if fb not in relevant_sectors:
                    relevant_sectors.append(fb)
                    if len(relevant_sectors) >= 5:
                        break
            # Re-score with fallback sectors (lower weight)
            for stock in all_stocks:
                ticker = stock["ticker"]
                if ticker not in scored and stock.get("sector") in relevant_sectors:
                    scored[ticker] = 1.0

        # Ensure at least 3 sectors represented in output.
        # Strategy: first pick best-in-sector for each relevant sector (diversity),
        # then fill remaining slots by score.
        sorted_tickers = sorted(scored.items(), key=lambda x: -x[1])
        min_sectors = 3

        selected: list[dict] = []
        selected_set: set[str] = set()
        sectors_seen: set[str] = set()

        def _make_entry(ticker: str, score: float) -> dict | None:
            stock = self._loader.get_stock(ticker)
            if stock is None:
                return None
            return {
                "ticker": ticker,
                "name": stock.get("name", ""),
                "sector": stock.get("sector", ""),
                "country": stock.get("country", ""),
                "score": round(score, 2),
            }

        # Phase 1: ensure diversity — pick top ticker per sector
        sector_best: dict[str, tuple[str, float]] = {}
        for ticker, score in sorted_tickers:
            stock = self._loader.get_stock(ticker)
            if stock is None:
                continue
            sector = stock.get("sector", "")
            if sector and sector not in sector_best:
                sector_best[sector] = (ticker, score)

        # Add best from each relevant sector first
        for sector in relevant_sectors:
            if sector in sector_best and len(selected) < max_tickers:
                ticker, score = sector_best[sector]
                if ticker not in selected_set:
                    entry = _make_entry(ticker, score)
                    if entry:
                        selected.append(entry)
                        selected_set.add(ticker)
                        sectors_seen.add(sector)

        # Phase 2: fill remaining slots by score
        for ticker, score in sorted_tickers:
            if len(selected) >= max_tickers:
                break
            if ticker in selected_set:
                continue
            entry = _make_entry(ticker, score)
            if entry:
                selected.append(entry)
                selected_set.add(ticker)
                sectors_seen.add(entry["sector"])

        # Phase 3: if still fewer than 3 sectors, add top stocks from other sectors
        if len(sectors_seen) < min_sectors:
            all_sector_names = set(s.get("sector") for s in all_stocks)
            missing = all_sector_names - sectors_seen
            for sector in sorted(missing):
                if len(sectors_seen) >= min_sectors:
                    break
                sector_stocks = self._loader.tickers_for_sector(sector)
                if sector_stocks:
                    top = sector_stocks[0]
                    if top["ticker"] not in selected_set:
                        selected.append({
                            "ticker": top["ticker"],
                            "name": top.get("name", ""),
                            "sector": sector,
                            "country": top.get("country", ""),
                            "score": 0.5,
                        })
                        selected_set.add(top["ticker"])
                        sectors_seen.add(sector)

        # Trim to max
        selected = selected[:max_tickers]

        # 5. Select indices
        indices = self._select_indices(country_code, region, max_indices)

        # 6. Build rationale
        entity_str = ", ".join(entities) if entities else "none"
        keyword_str = ", ".join(keywords) if keywords else "none"
        sector_str = ", ".join(sorted(sectors_seen))
        rationale = (
            f"Domain: {domain}. Entities: {entity_str}. "
            f"Country: {country or 'unspecified'} (regime: {regime}). "
            f"Keywords: {keyword_str}. "
            f"Sectors covered: {sector_str}. "
            f"Selected {len(selected)} tickers and {len(indices)} indices."
        )

        return RelevantUniverse(
            tickers=selected,
            indices=indices,
            beta_regime=regime,
            rationale=rationale,
        )

    def _select_indices(
        self, country_code: str | None, region: str, max_indices: int
    ) -> list[dict]:
        """Pick the most relevant market indices."""
        all_indices = self._loader._data.get("indices", [])
        selected = []

        # Priority: country-specific index first
        for idx in all_indices:
            if country_code and idx.get("country") == country_code:
                selected.append(idx)

        # Then regional
        for idx in all_indices:
            if idx not in selected and region and idx.get("region") == region:
                selected.append(idx)

        # Then global benchmarks (S&P 500 always useful)
        global_benchmarks = {"^GSPC", "^STOXX50E"}
        for idx in all_indices:
            if idx not in selected and idx.get("ticker") in global_benchmarks:
                selected.append(idx)

        return selected[:max_indices]
