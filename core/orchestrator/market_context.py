"""MarketContext — per-scenario financial reference frame.

Replaces the former `UniverseLoader` singleton. A `MarketContext` binds a
geography (IT, US, EU, EM, …) to a `MarketDataProvider` and exposes the
sector betas, ticker lookups, sovereign-spread calibration, and local-index
metadata the scorer needs — all parameterised rather than hardcoded to
Italian markets.

The two specialised sub-objects (`SovereignSpreadModel`, `LocalIndexModel`)
are views over the same geography-specific macro config; they carry no
state beyond their constructor args so instances are trivially shareable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .market_data import MarketDataProvider
    from .financial_impact import SectorBeta

logger = logging.getLogger(__name__)


_REGRESSION_CACHE: object | None = None


def _get_regression():
    """Lazily load the event-study regression. Soft-fails if sklearn is absent."""
    global _REGRESSION_CACHE
    if _REGRESSION_CACHE is not None:
        return _REGRESSION_CACHE
    try:
        from .sovereign_model import load_or_fit
        _REGRESSION_CACHE = load_or_fit()
    except Exception as e:
        logger.warning(
            "Event-study regression unavailable (%s); falling back to parametric "
            "SovereignSpreadModel. Install sklearn to enable the non-linear model.",
            e,
        )
        _REGRESSION_CACHE = False
    return _REGRESSION_CACHE or None


@dataclass(frozen=True)
class LocalIndexModel:
    """Benchmark index for a geography (FTSE MIB for IT, S&P 500 for US, …).

    `base_move_pct_per_unit` is the first-order impact on the broad index per
    unit of crisis intensity; `cap_abs_pct` bounds the absolute move (prevents
    unrealistic runaway numbers at extreme intensity)."""

    ticker: str
    label: str
    base_move_pct_per_unit: float = 0.5
    cap_abs_pct: float = 8.0

    def estimate_impact_pct(
        self, short_moves: list[float], long_moves: list[float],
    ) -> float:
        """Aggregate ticker-level moves into an index-level estimate.

        Weighted avg of shorts (they dominate crisis moves) with a small
        offset from the long leg. Mirrors the legacy `_compute_ftse_impact`
        but the weights are now explicit and the cap is driven by config.
        """
        if not short_moves:
            return 0.0
        avg_short = sum(short_moves) / len(short_moves)
        idx = avg_short * 0.3
        if long_moves:
            avg_long = sum(long_moves) / len(long_moves)
            idx += avg_long * 0.1
        return max(-self.cap_abs_pct, min(0.0, idx))


# Topic keyword → model category. Keys matched as substrings (lower-cased)
# against elements of `detected_topics`. First match wins.
_TOPIC_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("fiscal", "fiscal"), ("budget", "fiscal"), ("debt", "fiscal"), ("tax", "fiscal"),
    ("monetary", "monetary"), ("rate", "monetary"), ("ecb", "monetary"), ("fed", "monetary"),
    ("geopolit", "geopolitical"), ("war", "geopolitical"), ("sanction", "geopolitical"),
    ("invasion", "geopolitical"), ("nato", "geopolitical"),
)


def _classify_topic(detected_topics: list[str] | None, political_topics: frozenset[str]) -> str:
    """Map a topic list to one of: political | fiscal | monetary | geopolitical | macro."""
    topics = [t.lower() for t in (detected_topics or []) if isinstance(t, str)]
    if not topics:
        return "macro"
    for t in topics:
        for kw, cat in _TOPIC_CATEGORY_RULES:
            if kw in t:
                return cat
    if set(topics) & {t.lower() for t in political_topics}:
        return "political"
    return "macro"


@dataclass(frozen=True)
class SovereignSpreadModel:
    """Sovereign yield-spread sensitivity for a geography.

    Two modes:

    * **Regression mode** (default when a `SovereignSpreadRegression` is
      attached): Ridge + quantile regression trained on a 28-event
      historical corpus (2011 BTP crisis, 2018 Italy budget, 2022 Truss,
      2018 Argentina, 2020 COVID, etc.). Returns both median and p95 bps,
      with no artificial cap — a 2011-style shock can produce 400+ bps.

    * **Parametric fallback** (used when the regression is absent, e.g.
      in unit tests that construct the model directly): the old linear
      `sensitivity_bps_per_unit * intensity` with `cap_bps` ceiling.
    """

    spread_name: str
    sensitivity_bps_per_unit: float
    non_political_sensitivity_bps_per_unit: float
    cap_bps: int
    political_topics: frozenset[str]
    spread_unit: str = "bps"
    # Regression-mode context (all optional for backward compat)
    country: str = ""
    base_spread_bps: float = 100.0
    vix_level: float = 20.0
    regression: object | None = None  # SovereignSpreadRegression | None

    def impact_bps(self, intensity: float, detected_topics: list[str] | None) -> int:
        """Return the median spread impact in bps (signed positive = widening)."""
        band = self.impact_band(intensity, detected_topics)
        return int(round(band[0]))

    def impact_band(
        self, intensity: float, detected_topics: list[str] | None,
    ) -> tuple[float, float]:
        """Return (median_bps, p95_bps). Falls back to parametric when no regression."""
        if self.regression is not None and self.country:
            topic = _classify_topic(detected_topics, self.political_topics)
            band = self.regression.predict(
                intensity=intensity, topic=topic, country=self.country,
                base_spread_bps=self.base_spread_bps, vix_level=self.vix_level,
            )
            return (band.median_bps, band.p95_bps)
        # Parametric fallback — preserved for tests and minimal configs
        topics = set(detected_topics or [])
        is_political = bool(topics & self.political_topics)
        if not is_political:
            median = intensity * self.non_political_sensitivity_bps_per_unit
        else:
            median = min(self.cap_bps, self.sensitivity_bps_per_unit * intensity)
        # Simple tail heuristic for the parametric mode: +50% on severe topics
        p95 = median * 1.5 if is_political else median * 1.2
        return (median, p95)


class MarketContext:
    """Per-scenario market reference frame.

    Instantiated with a `geography` (e.g. "IT", "US") and an optional
    `MarketDataProvider`. Falls back to the process-wide static provider
    when none is supplied, which keeps backward-compat for callers that
    just want "the default Italian regime".

    The context owns:
      - a sector→beta lookup resolved through the geography's `beta_regime`
        (so `MarketContext(geography="US")` returns US-calibrated betas
        without passing a regime arg everywhere)
      - a `SovereignSpreadModel` (replaces the hardcoded BTP constants)
      - a `LocalIndexModel` (replaces the hardcoded FTSE MIB aggregation)
      - org/ticker lookups that previously went through `UniverseLoader()`
    """

    def __init__(
        self,
        geography: str = "IT",
        provider: "MarketDataProvider | None" = None,
    ):
        from .market_data import get_default_provider

        self.geography = geography
        self.provider: MarketDataProvider = provider or get_default_provider()

        macro = self.provider.macro()
        # Geography resolution: exact match → explicit default → hard default
        macro_cfg = macro.get(geography) or macro.get("default") or {}
        if not macro_cfg:
            logger.warning(
                "No macro config for geography=%s; using minimal defaults. "
                "Extend stock_universe.json:macro to calibrate this region.",
                geography,
            )

        self._beta_regime: str = macro_cfg.get("beta_regime", geography)

        idx_cfg = macro_cfg.get("local_index", {})
        self.local_index = LocalIndexModel(
            ticker=idx_cfg.get("ticker", "FTSEMIB.MI"),
            label=idx_cfg.get("label", "FTSE MIB"),
            base_move_pct_per_unit=float(idx_cfg.get("base_move_pct_per_unit", 0.5)),
            cap_abs_pct=float(idx_cfg.get("cap_abs_pct", 8.0)),
        )

        sov_cfg = macro_cfg.get("sovereign", {})
        regression = _get_regression()
        country_code = sov_cfg.get("country") or geography
        self.sovereign = SovereignSpreadModel(
            spread_name=sov_cfg.get("spread_name", "Sovereign"),
            spread_unit=sov_cfg.get("spread_unit", "bps"),
            sensitivity_bps_per_unit=float(sov_cfg.get("sensitivity_bps_per_unit", 20)),
            non_political_sensitivity_bps_per_unit=float(
                sov_cfg.get("non_political_sensitivity_bps_per_unit", 3)
            ),
            cap_bps=int(sov_cfg.get("cap_bps", 100)),
            political_topics=frozenset(sov_cfg.get("political_topics", [])),
            country=country_code,
            base_spread_bps=float(sov_cfg.get("base_spread_bps", 100.0)),
            vix_level=float(sov_cfg.get("vix_level", 20.0)),
            regression=regression,
        )

    @classmethod
    def with_live_data(
        cls, geography: str = "IT", refresh: bool = True,
    ) -> "MarketContext":
        """Construct a MarketContext backed by the live YFinanceProvider.

        The simulation entrypoint calls this so every scenario inherits a
        fresh VIX / UST-10Y snapshot. TTL-gated (5 min): `refresh=True`
        triggers a fetch only if the cache is stale.

        Gracefully degrades to the static provider if yfinance is missing
        or the network is unreachable — the caller always gets a working
        context, just without the live overlay.
        """
        try:
            from .providers.yfinance_provider import get_live_provider
            provider = get_live_provider()
            if refresh:
                provider.refresh()
            return cls(geography=geography, provider=provider)
        except Exception as e:
            logger.warning(
                "Live provider unavailable (%s); falling back to static provider. "
                "The sovereign regression will run with stale base_spread / VIX priors.",
                e,
            )
            return cls(geography=geography)

    # ── Beta + ticker lookups (previously on UniverseLoader) ─────────────

    @property
    def beta_regime(self) -> str:
        return self._beta_regime

    def get_beta(self, sector: str, regime: str | None = None) -> "SectorBeta":
        """Return the SectorBeta for `sector` under this context's regime.

        Passing `regime` explicitly overrides the context's default regime
        (useful for stress-testing what-if scenarios: "how would this crisis
        look under EM betas?").
        """
        from .financial_impact import SectorBeta

        effective_regime = regime or self._beta_regime
        sectors = self.provider.sectors()
        s_data = sectors.get(sector, {})
        betas = s_data.get("betas", {})
        coeffs = betas.get(effective_regime) or betas.get("default", {})
        return SectorBeta(
            sector=sector,
            political_beta=coeffs.get("political_beta", 1.0),
            spread_beta=coeffs.get("spread_beta", 0.0),
            crisis_alpha=coeffs.get("crisis_alpha", 0.0),
            volatility_multiplier=coeffs.get("volatility_multiplier", 1.0),
        )

    def resolve_org(self, name: str) -> list[str]:
        if not name:
            return []
        name_lower = name.lower().strip()
        aliases = self.provider.org_aliases()
        for alias, tickers in aliases.items():
            if alias in name_lower or name_lower in alias:
                return tickers
        return []

    def get_stock(self, ticker: str) -> dict | None:
        for s in self.provider.stocks():
            if s.get("ticker") == ticker:
                return s
        return None

    def get_ticker_sector(self, ticker: str) -> str | None:
        stock = self.get_stock(ticker)
        return stock["sector"] if stock else None

    def tickers_for_sector(self, sector: str) -> list[dict]:
        return [s for s in self.provider.stocks() if s.get("sector") == sector]

    def sector_label(self, sector: str) -> str:
        sectors = self.provider.sectors()
        return sectors.get(sector, {}).get("label", sector.replace("_", " ").title())

    # ── Convenience wrappers for scorer ──────────────────────────────────

    def sovereign_spread_bps(
        self, intensity: float, detected_topics: list[str] | None,
    ) -> int:
        return self.sovereign.impact_bps(intensity, detected_topics)

    def sovereign_spread_band(
        self, intensity: float, detected_topics: list[str] | None,
    ) -> tuple[int, int]:
        """Return (median_bps, p95_bps) — tail estimate for crisis scenarios."""
        median, p95 = self.sovereign.impact_band(intensity, detected_topics)
        return int(round(median)), int(round(p95))

    def local_index_impact_pct(
        self, short_moves: list[float], long_moves: list[float],
    ) -> float:
        return self.local_index.estimate_impact_pct(short_moves, long_moves)
