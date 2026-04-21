"""Financial Impact Scorer — parametric, bidirectional, agent-driven.

Core principles:
1. **No hardcoded tickers** — tickers are extracted dynamically from the active
   agents' `affiliated_tickers` or resolved via ORG_TICKER_MAP from party_or_org.
2. **Bidirectional** — every crisis has losers AND winners. The module outputs
   Pair Trades: short the directly exposed, long the structural beneficiaries.
3. **Historical Beta** — sector sensitivity is calibrated on real empirical
   data (political beta, spread beta), not subjective 0-1 scores.
4. **LLM Flash Note** — Gemini generates a Goldman-Sachs-style analyst note
   from the raw crisis metrics, replacing string concatenation.

Input: active agents + crisis metrics from escalation/contagion engines.
Output: FinancialImpactReport with pair trades, betas, LLM note.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .market_context import MarketContext

logger = logging.getLogger(__name__)

# Schema version for the frontend bridge. Bump whenever to_dict() fields change.
# Frontend lib/types/financial-impact.ts must import and check this.
FIN_SCHEMA_VERSION = "2.0.0"


# ── Organisation → Ticker Resolution ────────────────────────────────────────
# Fallback when stakeholder.affiliated_tickers is empty.
# Maps party_or_org (case-insensitive) → tickers on Borsa Italiana / major exchanges.



def _resolve_tickers_for_org(org: str, market: "MarketContext") -> list[str]:
    """Resolve tickers from an organisation name via the injected MarketContext.

    The market context carries the active provider (static JSON, yfinance,
    a test fixture, etc.) — so callers in different geographies or data-
    source regimes resolve against the right alias table.
    """
    if not org:
        return []
    return market.resolve_org(org.lower().strip())

# ── Historical Beta Coefficients ─────────────────────────────────────────────
# Empirical sector sensitivity to Italian political/crisis events.
#
# Sources: Bloomberg sector correlation analysis, BTP-Bund spread regressions,
# event studies on Italian political crises (2011 sovereign, 2018 budget,
# 2022 Draghi resignation).
#
# political_beta: sector return / FTSE MIB return during political shocks
#   >1.0 = amplifies market moves, <1.0 = defensive
# spread_beta: sector return sensitivity to 10bps BTP-Bund widening (in %)
#   negative = loses when spread widens
# crisis_alpha: average sector excess return during Italian political crises (%)
#   negative = underperforms during crises
# volatility_multiplier: realized vol / FTSE MIB vol during crises

@dataclass
class SectorBeta:
    """Empirical sector sensitivity coefficients."""
    sector: str
    political_beta: float      # vs FTSE MIB during political events
    spread_beta: float         # % return per 10bps spread widening
    crisis_alpha: float        # excess return during IT political crises (%)
    volatility_multiplier: float  # realized vol amplification
    notes: str = ""

    @property
    def is_defensive(self) -> bool:
        return self.political_beta < 0.8 and self.crisis_alpha > -0.5


# Calibrated from event studies on Italian market:
# - 2011 sovereign crisis (Berlusconi resignation / Monti)
# - 2018 budget standoff (Conte I / Tria vs EU)
# - 2022 Draghi resignation (Jul 14-21)
# - 2023 bank windfall tax (Aug 7-10)


from .market_context import MarketContext as _MarketContext


# ── Pair Trade Logic ─────────────────────────────────────────────────────────
# For each crisis type: which sectors lose (SHORT) and which gain (LONG).
# The insight: every crisis creates a pair trade opportunity.

CRISIS_PAIR_TRADES: dict[str, dict] = {
    "geopolitical": {
        "short": ['banking', 'energy_fossil'],
        "long": ['defense', 'utilities'],
        "rationale": "War risks.",
    },

    "supply_chain": {
        "short": ['automotive', 'tech'],
        "long": ['infrastructure'],
        "rationale": "Logistics bottlenecks.",
    },

    "currency_crisis": {
        "short": ['banking', 'sovereign_debt'],
        "long": ['luxury', 'tech'],
        "rationale": "Flight to global earners.",
    },

    "commodity_shock": {
        "short": ['food_consumer', 'automotive'],
        "long": ['energy_fossil'],
        "rationale": "Input costs up.",
    },

    "pandemic": {
        "short": ['luxury', 'automotive'],
        "long": ['healthcare', 'tech'],
        "rationale": "Lockdowns help tech and health.",
    },

    "tech_regulation": {
        "short": ['tech'],
        "long": ['telecom', 'banking'],
        "rationale": "Big tech regulated, incumbents gain.",
    },

    "central_bank_hawkish": {
        "short": ['real_estate', 'tech'],
        "long": ['banking', 'insurance'],
        "rationale": "Rates up, NIM up.",
    },

    "trade_war": {
        "short": ['automotive', 'tech'],
        "long": ['defense', 'utilities'],
        "rationale": "Trade barriers hit supply chains.",
    },

    "labor_reform": {
        "short": ["automotive", "real_estate"],
        "long": ["utilities", "healthcare", "defense"],
        "rationale": "Labor disputes hit cyclicals; defensives/regulated outperform",
    },
    "industrial_policy": {
        "short": ["automotive", "energy_fossil"],
        "long": ["energy_renewable", "defense", "tech"],
        "rationale": "Industrial policy shifts penalize incumbents, reward transition sectors",
    },
    "fiscal_policy": {
        "short": ["banking", "sovereign_debt", "real_estate", "insurance"],
        "long": ["utilities", "healthcare", "luxury"],
        "rationale": "Fiscal instability → spread widening → banks & RE hammered; global earners safe",
    },
    "environment": {
        "short": ["energy_fossil", "automotive"],
        "long": ["energy_renewable", "utilities", "infrastructure"],
        "rationale": "Green regulation penalizes fossil/ICE, benefits renewables and grid",
    },
    "eu_integration": {
        "short": ["banking", "sovereign_debt"],
        "long": ["defense", "luxury", "food_consumer"],
        "rationale": "EU tension → Italexit risk → banks bleed; global earners hedge",
    },
    "premierato": {
        "short": ["banking", "sovereign_debt"],
        "long": ["utilities", "healthcare"],
        "rationale": "Constitutional uncertainty → political risk premium; defensives outperform",
    },
    "defense_spending": {
        "short": [],
        "long": ["defense"],
        "rationale": "NATO spending debates → direct LDO/FCT order book catalyst",
    },
    "healthcare": {
        "short": ["healthcare"],
        "long": ["insurance"],
        "rationale": "Public health crisis → SSN pressure; private insurance demand rises",
    },
    "media_freedom": {
        "short": ["media"],
        "long": [],
        "rationale": "Press restrictions → media sector derating",
    },
    "immigration": {
        "short": [],
        "long": [],
        "rationale": "Limited direct market impact; second-order through political risk",
    },
    "autonomia_differenziata": {
        "short": ["utilities", "banking"],
        "long": [],
        "rationale": "Fiscal decentralization creates regional credit differentiation",
    },
    "judiciary_reform": {
        "short": [],
        "long": [],
        "rationale": "Rule-of-law perception → marginal FDI impact, not directly tradeable",
    },
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TickerImpact:
    """Impact estimate for a single ticker with temporal curve."""
    ticker: str
    sector: str
    direction: str              # "short" or "long"
    # Temporal curve: T+1 (Day 1 panic), T+3 (Day 3 resolution), T+7 (Day 7 outcome)
    t1_pct: float               # Day 1 — initial panic / euphoria
    t3_pct: float               # Day 3 — partial resolution / continuation
    t7_pct: float               # Day 7 — structural outcome
    beta: float                 # Political beta used
    confidence: float           # 0-1
    source: str                 # "agent:carlo_messina" or "pair_trade:fiscal_policy"

    # Back-compat aliases
    @property
    def short_term_pct(self) -> float:
        return self.t1_pct

    @property
    def medium_term_pct(self) -> float:
        return self.t7_pct


@dataclass
class PairTrade:
    """A long/short pair recommendation."""
    short_leg: list[TickerImpact]
    long_leg: list[TickerImpact]
    topic: str
    rationale: str


@dataclass
class FinancialImpactReport:
    """Client-facing financial impact assessment."""
    # Core
    market_volatility_warning: str  # "LOW" / "MODERATE" / "HIGH" / "CRITICAL"
    headline: str                   # LLM-generated analyst flash note

    # Crisis scope
    crisis_scope: str = "macro_systematic"  # "macro_systematic" or "micro_idiosyncratic"
    scope_confidence: float = 0.0           # 0-1, how confident the classifier is
    scope_disclaimer: str = ""              # Non-empty when model not applicable

    # Pair trades (the key output)
    pair_trades: list[PairTrade] = field(default_factory=list)

    # All individual ticker impacts (flattened)
    ticker_impacts: list[TickerImpact] = field(default_factory=list)

    # Aggregate
    ftse_mib_impact_pct: float = 0.0
    btp_spread_impact_bps: int = 0

    # Geography metadata — populated by the scorer from its MarketContext so
    # the frontend can label the sovereign spread and local index correctly
    # (BTP-Bund + FTSE MIB for IT, UST + S&P 500 for US, …).
    geography: str = "IT"
    beta_regime: str = "IT"
    local_index_ticker: str = "FTSEMIB.MI"
    local_index_label: str = "FTSE MIB"
    sovereign_spread_name: str = "BTP-Bund"

    # Crisis metadata
    crisis_wave: int = 1
    contagion_risk: float = 0.0
    engagement_score: float = 0.0
    institutional_actors_count: int = 0

    # Time-series
    impact_history: list[dict] = field(default_factory=list)

    # Non-serialised: the MarketContext that produced this report. Used by
    # `_enrich_ticker` to look up ticker name + sector label. Populated by
    # the scorer; reports constructed directly in tests can leave it None
    # (enrichment will fall back to raw ticker symbols).
    market: "_MarketContext | None" = field(default=None, repr=False, compare=False)

    # Back-compat: sector_impacts for frontend
    @property
    def sector_impacts(self) -> list[dict]:
        """Backward-compatible sector view (grouped from ticker_impacts)."""
        sectors: dict[str, dict] = {}
        for t in self.ticker_impacts:
            if t.sector not in sectors:
                sectors[t.sector] = {
                    "sector": t.sector, "tickers": [],
                    "t1_pct": 0.0, "t3_pct": 0.0, "t7_pct": 0.0,
                    "short_term_pct": 0.0, "medium_term_pct": 0.0,
                    "confidence": 0.0, "direction": t.direction,
                    "rationale": t.source,
                }
            s = sectors[t.sector]
            s["tickers"].append(t.ticker)
            if t.direction == "short":
                s["t1_pct"] = min(s["t1_pct"], t.t1_pct)
                s["t3_pct"] = min(s["t3_pct"], t.t3_pct)
                s["t7_pct"] = min(s["t7_pct"], t.t7_pct)
            else:
                s["t1_pct"] = max(s["t1_pct"], t.t1_pct)
                s["t3_pct"] = max(s["t3_pct"], t.t3_pct)
                s["t7_pct"] = max(s["t7_pct"], t.t7_pct)
            s["short_term_pct"] = s["t1_pct"]
            s["medium_term_pct"] = s["t7_pct"]
            s["confidence"] = max(s["confidence"], t.confidence)
        return list(sectors.values())

    def _enrich_ticker(self, t: "TickerImpact") -> dict:
        """Enrich a TickerImpact with name + sectorLabel from the universe."""
        stock = self.market.get_stock(t.ticker) if self.market else None
        sectors = self.market.provider.sectors() if self.market else {}
        sector_def = sectors.get(t.sector, {})
        return {
            "ticker": t.ticker,
            "name": stock.get("name") if stock else t.ticker,
            "sector": t.sector,
            "sectorLabel": sector_def.get("label", t.sector.replace("_", " ").title()),
            "direction": t.direction,
            "t1_pct": round(t.t1_pct, 2),
            "t3_pct": round(t.t3_pct, 2),
            "t7_pct": round(t.t7_pct, 2),
            "beta": round(t.beta, 2),
            "confidence": round(t.confidence, 2),
            "source": t.source,
        }

    def to_dict(self) -> dict:
        """Frontend-aligned schema.

        This is the SINGLE source of truth for financial-impact JSON.
        Frontend TypeScript types must mirror these fields exactly.
        Schema version is bumped when fields change (see FIN_SCHEMA_VERSION).
        """
        enriched = [self._enrich_ticker(t) for t in self.ticker_impacts]
        return {
            "schema_version": FIN_SCHEMA_VERSION,
            "provenance": "backend-simulated",  # see F1.3 — frontend uses this for UI badge
            # Renamed to match TS RoundFinancial
            "volatility_warning": self.market_volatility_warning,
            # Legacy alias kept for backward compat with existing consumers
            "market_volatility_warning": self.market_volatility_warning,
            "headline": self.headline,
            "crisis_scope": self.crisis_scope,
            "scope_confidence": round(self.scope_confidence, 2),
            "scope_disclaimer": self.scope_disclaimer,
            "pair_trades": [
                {
                    "topic": pt.topic,
                    "rationale": pt.rationale,
                    "short_leg": [self._enrich_ticker(t) for t in pt.short_leg],
                    "long_leg": [self._enrich_ticker(t) for t in pt.long_leg],
                }
                for pt in self.pair_trades
            ],
            "tickers": enriched,            # new TS-aligned key
            "ticker_impacts": enriched,     # legacy alias
            "sector_impacts": self.sector_impacts,
            "ftse_mib_impact_pct": round(self.ftse_mib_impact_pct, 2),
            "local_index_impact_pct": round(self.ftse_mib_impact_pct, 2),  # geo-agnostic alias
            "local_index_ticker": self.local_index_ticker,
            "local_index_label": self.local_index_label,
            "btp_spread_bps": self.btp_spread_impact_bps,               # TS-aligned
            "btp_spread_impact_bps": self.btp_spread_impact_bps,        # legacy alias
            "sovereign_spread_bps": self.btp_spread_impact_bps,         # geo-agnostic alias
            "sovereign_spread_name": self.sovereign_spread_name,
            "geography": self.geography,
            "beta_regime": self.beta_regime,
            "crisis_wave": self.crisis_wave,
            "contagion_risk": round(self.contagion_risk, 3),
            "engagement_score": round(self.engagement_score, 3),
            "institutional_actors_count": self.institutional_actors_count,
            "impact_history": self.impact_history,
        }


# ── Scorer ────────────────────────────────────────────────────────────────────

class FinancialImpactScorer:
    """Parametric, agent-driven financial impact scorer.

    Key differences from v1:
    - Tickers extracted dynamically from active agents (not hardcoded maps)
    - Historical sector betas replace subjective sensitivity scores
    - Bidirectional output: short AND long legs per crisis topic
    - LLM analyst report replaces string concatenation
    """

    # BTP spread: empirical sensitivity (bps per unit composite crisis score)
    # Calibrated from: 2018 budget crisis (+250bps over 3 months at peak)
    # and 2022 Draghi (+30bps in 1 week at moderate intensity)
    BTP_SENSITIVITY_BPS_PER_UNIT = 35

    # ── Crisis scope classification ──────────────────────────────────────
    # Topics that indicate macro/systematic risk (model's strength: 56-75%)
    MACRO_TOPICS = {
        "fiscal_policy", "eu_integration", "premierato",
        "autonomia_differenziata", "defense_spending", "environment",
        "labor_reform", "industrial_policy", "immigration",
        "judiciary_reform", "media_freedom",
    }
    # Sectors that indicate idiosyncratic/micro risk (model's weakness: 25-32%)
    IDIOSYNCRATIC_SECTORS = {
        "corporate_governance", "m_and_a", "earnings",
        "product_recall", "fraud", "accounting",
    }

    def __init__(
        self,
        detected_topics: list[str] = None,
        detected_sectors: list[str] = None,
        llm=None,  # Optional BaseLLMClient for Flash Note generation
        market: "MarketContext | None" = None,
        relevant_universe=None,
    ):
        self.detected_topics = detected_topics or []
        self.detected_sectors = detected_sectors or []
        self.llm = llm
        self.impact_history: list[dict] = []
        self.relevant_universe = relevant_universe
        # Market context resolution order:
        #   1. explicit `market` arg (new code, tests)
        #   2. regime derived from `relevant_universe.beta_regime` (orchestrator)
        #   3. default IT context (legacy calls, single-scenario scripts)
        if market is not None:
            self.market: _MarketContext = market
        elif relevant_universe is not None and getattr(relevant_universe, "beta_regime", None):
            self.market = _MarketContext(geography=relevant_universe.beta_regime)
        else:
            self.market = _MarketContext(geography="IT")

    def classify_crisis_scope(self) -> tuple[str, float, str]:
        """Classify whether this crisis is macro/systematic or micro/idiosyncratic.

        Returns (scope, confidence, disclaimer).
        - macro_systematic: regulatory, political, fiscal — model is calibrated for these
        - micro_idiosyncratic: single-company events — model not applicable

        Logic:
        1. If ANY detected topic is in MACRO_TOPICS → macro (high confidence)
        2. If detected sectors overlap IDIOSYNCRATIC_SECTORS → idiosyncratic
        3. If NO macro topics AND few sectors → idiosyncratic (low confidence)
        """
        topics = set(self.detected_topics)
        sectors = set(self.detected_sectors)

        macro_overlap = topics & self.MACRO_TOPICS
        idio_overlap = sectors & self.IDIOSYNCRATIC_SECTORS

        if macro_overlap:
            # Strong macro signal
            confidence = min(0.95, 0.6 + len(macro_overlap) * 0.1)
            return "macro_systematic", confidence, ""

        if idio_overlap:
            return (
                "micro_idiosyncratic", 0.85,
                "Crisis Scope: Idiosyncratic. Il modello è calibrato su rischio "
                "politico/regolamentare sistemico. Le previsioni per eventi "
                "micro-aziendali hanno accuratezza limitata (~30%). "
                "Usare con cautela.",
            )

        # Ambiguous: no macro topics, no idiosyncratic sectors
        # Check if we have any topics at all
        if not topics:
            return (
                "micro_idiosyncratic", 0.6,
                "Crisis Scope: Non classificabile. Nessun topic macro rilevato. "
                "Il modello potrebbe non essere applicabile.",
            )

        # Has topics but none are macro — likely idiosyncratic
        return (
            "micro_idiosyncratic", 0.7,
            "Crisis Scope: Idiosyncratic. Topics rilevati non rientrano nel "
            "perimetro macro/regolamentare del modello.",
        )

    def score_round(
        self,
        round_num: int,
        engagement_score: float,
        contagion_risk: float,
        active_wave: int,
        polarization: float,
        polarization_velocity: float = 0.0,
        negative_institutional_pct: float = 0.0,
        negative_ceo_count: int = 0,
        active_agents: list = None,
        relevant_universe=None,
    ) -> FinancialImpactReport:
        """Compute financial impact for a completed round.

        Args:
            active_agents: List of agent objects with .party_or_org and
                          optionally .affiliated_tickers for dynamic ticker resolution.
        """
        # Cache for dashboard/summary — no behavioural effect yet.
        self.relevant_universe = relevant_universe

        # ── Step 0: Crisis scope classification ────────────────────────
        crisis_scope, scope_confidence, scope_disclaimer = self.classify_crisis_scope()

        # ── Step 1: Composite crisis intensity ─────────────────────────
        base_intensity = self._compute_intensity(
            engagement_score, contagion_risk, active_wave,
            polarization_velocity, negative_institutional_pct, negative_ceo_count,
        )

        # ── Step 2: Extract tickers from active agents ─────────────────
        agent_tickers = self._extract_agent_tickers(active_agents or [])

        # ── Step 3: Build pair trades per detected topic ───────────────
        all_ticker_impacts: list[TickerImpact] = []
        pair_trades: list[PairTrade] = []

        for topic in self.detected_topics:
            pt = self._build_pair_trade(topic, base_intensity, agent_tickers)
            if pt:
                pair_trades.append(pt)
                all_ticker_impacts.extend(pt.short_leg)
                all_ticker_impacts.extend(pt.long_leg)

        # Add direct agent-ticker impacts (agents explicitly involved)
        for ticker, agent_source in agent_tickers.items():
            if not any(t.ticker == ticker for t in all_ticker_impacts):
                sector = self.market.get_ticker_sector(ticker) or "unknown"
                beta = self.market.get_beta(sector)
                impact = self._compute_ticker_impact(
                    ticker, sector, "short", base_intensity, beta, agent_source,
                )
                all_ticker_impacts.append(impact)

        # ── Step 3b: Idiosyncratic confidence penalty ──────────────────
        if crisis_scope == "micro_idiosyncratic":
            for ti in all_ticker_impacts:
                # Slash confidence for idiosyncratic events
                ti.confidence = round(ti.confidence * 0.4, 2)

        # ── Step 4: Aggregate market impact ────────────────────────────
        ftse_impact = self._compute_ftse_impact(all_ticker_impacts, base_intensity)
        btp_spread = self._compute_btp_spread(base_intensity)

        # ── Step 5: Warning level ──────────────────────────────────────
        warning = self._classify_warning(base_intensity, contagion_risk)

        # ── Step 6: Headline (sync fallback — LLM is async) ───────────
        headline = self._build_fallback_headline(
            warning, pair_trades, all_ticker_impacts, active_wave, btp_spread,
        )
        if scope_disclaimer:
            headline = f"⚠ {scope_disclaimer}\n{headline}"

        # ── Step 7: Track history ──────────────────────────────────────
        round_data = {
            "round": round_num,
            "intensity": round(base_intensity, 3),
            "warning": warning,
            "crisis_scope": crisis_scope,
            "ftse_impact": round(ftse_impact, 2),
            "btp_spread": btp_spread,
            "n_tickers_affected": len(all_ticker_impacts),
            "n_pair_trades": len(pair_trades),
        }
        self.impact_history.append(round_data)

        return FinancialImpactReport(
            market_volatility_warning=warning,
            headline=headline,
            crisis_scope=crisis_scope,
            scope_confidence=scope_confidence,
            scope_disclaimer=scope_disclaimer,
            pair_trades=pair_trades,
            ticker_impacts=all_ticker_impacts,
            ftse_mib_impact_pct=round(ftse_impact, 2),
            btp_spread_impact_bps=btp_spread,
            geography=self.market.geography,
            beta_regime=self.market.beta_regime,
            local_index_ticker=self.market.local_index.ticker,
            local_index_label=self.market.local_index.label,
            sovereign_spread_name=self.market.sovereign.spread_name,
            crisis_wave=active_wave,
            contagion_risk=contagion_risk,
            engagement_score=engagement_score,
            institutional_actors_count=int(negative_institutional_pct * 20),
            impact_history=list(self.impact_history),
            market=self.market,
        )

    # ── Core computations ─────────────────────────────────────────────────

    def _compute_intensity(
        self,
        engagement: float,
        cri: float,
        wave: int,
        polar_vel: float,
        neg_inst_pct: float,
        neg_ceo: int,
    ) -> float:
        """Composite crisis intensity (0 to ~8 with panic multiplier).

        Uses exponential engagement scaling: low engagement barely registers,
        high engagement triggers non-linear amplification.

        **Panic multiplier** (Fat Tails):
        When Wave 3 + CRI > 0.8, markets enter panic selling / margin call
        dynamics. The intensity gets an exponential boost to model fat-tailed
        moves like -20/30% on banks during sovereign crises.
        Calibrated from: 2011 UCG -70%, 2018 UCG -30%, 2023 bank tax -8%.
        """
        # Exponential engagement: 0.3→0.09, 0.5→0.25, 0.7→0.49, 0.9→0.81
        eng_factor = engagement ** 2

        # Wave multiplier (calibrated from event studies)
        # Wave 1 = local containment, Wave 2 = national attention, Wave 3 = institutional crisis
        wave_mult = {1: 0.4, 2: 1.0, 3: 2.2}.get(wave, 1.0)

        # Contagion amplification (non-linear at high CRI)
        # Below 0.3 CRI → barely affects markets
        # Above 0.7 CRI → panic selling dynamics
        cri_factor = max(0.1, cri ** 1.3)

        # Institutional pressure
        inst_mult = 1.0 + neg_inst_pct * 0.4
        ceo_mult = 1.0 + neg_ceo * 0.12

        intensity = eng_factor * wave_mult * cri_factor * inst_mult * ceo_mult

        # Polarization velocity adds urgency
        if polar_vel > 0:
            intensity *= (1.0 + polar_vel * 0.25)

        # ── PANIC MULTIPLIER (Fat Tails) ──────────────────────────────
        # When wave == 3 AND CRI > 0.8 → exponential amplification.
        # This models margin calls, forced selling, and herding.
        # math.exp(0.8 * 1.5) = 3.32x, math.exp(0.95 * 1.5) = 4.17x
        if wave >= 3 and cri > 0.8:
            # Clamp CRI to [0, 1] — corrupted values (e.g. from agent failures)
            # would cause math.exp overflow
            safe_cri = max(0.0, min(1.0, cri))
            panic_mult = math.exp(safe_cri * 1.5)
            intensity *= panic_mult
            logger.debug(
                f"PANIC MULTIPLIER active: wave={wave}, CRI={cri:.2f}, "
                f"mult={panic_mult:.2f}x → intensity={intensity:.2f}"
            )

        # Cap raised to 8.0 to allow fat-tailed moves
        return min(8.0, intensity)

    def _extract_agent_tickers(self, agents: list) -> dict[str, str]:
        """Extract ticker → source mapping from active agents.

        Checks affiliated_tickers first, then falls back to ORG_TICKER_MAP.
        Returns {ticker: "agent:agent_id"}.
        """
        tickers: dict[str, str] = {}
        for agent in agents:
            agent_id = getattr(agent, "id", "unknown")
            # Try affiliated_tickers attribute first
            affiliated = getattr(agent, "affiliated_tickers", None) or []
            if affiliated:
                for t in affiliated:
                    tickers[t] = f"agent:{agent_id}"
                continue

            # Fallback: resolve from party_or_org via this scorer's MarketContext
            org = getattr(agent, "party_or_org", "") or getattr(agent, "_party", "")
            resolved = _resolve_tickers_for_org(org, self.market)
            for t in resolved:
                tickers[t] = f"agent:{agent_id}"

        return tickers

    def _build_pair_trade(
        self,
        topic: str,
        intensity: float,
        agent_tickers: dict[str, str],
    ) -> Optional[PairTrade]:
        """Build a pair trade for a crisis topic."""
        pair_config = CRISIS_PAIR_TRADES.get(topic)
        if not pair_config:
            return None

        short_leg: list[TickerImpact] = []
        long_leg: list[TickerImpact] = []

        # SHORT leg: sectors that lose
        for sector_key in pair_config["short"]:
            beta_data = self.market.get_beta(sector_key)

            # Find tickers for this sector
            sector_tickers = self._tickers_for_sector(sector_key, agent_tickers)
            for ticker in sector_tickers[:3]:  # max 3 per sector
                source = agent_tickers.get(ticker, f"pair_trade:{topic}")
                impact = self._compute_ticker_impact(
                    ticker, sector_key, "short", intensity, beta_data, source,
                )
                short_leg.append(impact)

        # LONG leg: sectors that benefit
        for sector_key in pair_config["long"]:
            beta_data = self.market.get_beta(sector_key)

            sector_tickers = self._tickers_for_sector(sector_key, agent_tickers)
            for ticker in sector_tickers[:2]:  # max 2 per long sector
                source = agent_tickers.get(ticker, f"pair_trade:{topic}")
                impact = self._compute_ticker_impact(
                    ticker, sector_key, "long", intensity, beta_data, source,
                )
                long_leg.append(impact)

        if not short_leg and not long_leg:
            return None

        return PairTrade(
            short_leg=short_leg,
            long_leg=long_leg,
            topic=topic,
            rationale=pair_config["rationale"],
        )

    def _compute_ticker_impact(
        self,
        ticker: str,
        sector_key: str,
        direction: str,  # "short" or "long"
        intensity: float,
        beta: SectorBeta,
        source: str,
    ) -> TickerImpact:
        """Compute impact for a single ticker using its sector beta.

        Returns a 3-point temporal curve:
        - T+1 (Day 1): Panic / initial reaction. Maximum dislocation.
        - T+3 (Day 3): Partial resolution. Government walkbacks, ECB signals.
          Typically 40-60% recovery from T+1 peak for shorts (V-shape).
        - T+7 (Day 7): Structural outcome. Where the stock settles after
          the initial shock is digested. Includes crisis_alpha.

        Calibrated from Italian event studies:
        - 2023 Bank Tax: UCG -7% T+1, -4% T+3 (Meloni softened), -2% T+7
        - 2022 Draghi: UCG -5% T+1, -7% T+3 (no resolution), -8% T+7
        - 2018 Budget: UCG -10% T+1, -15% T+3 (escalation), -25% T+7
        """
        sign = -1.0 if direction == "short" else 1.0

        # Base market move at intensity=1 is ~-0.5% for FTSE MIB
        base_market_move = 0.5  # % per unit intensity

        # ── T+1: Day 1 Panic ───────────────────────────────────────────
        # Maximum dislocation. Pure beta × intensity.
        t1_pct = sign * beta.political_beta * base_market_move * intensity

        # ── T+3: Day 3 Resolution ─────────────────────────────────────
        # Two regimes:
        # a) Low-moderate intensity (<2): V-shape recovery (govt walkback typical)
        #    T+3 ≈ 50% of T+1 (half the panic fades)
        # b) High intensity (>=2): Continuation / escalation
        #    T+3 ≈ 120% of T+1 (crisis deepens, margin calls cascade)
        if intensity < 2.0:
            # V-shape: partial recovery
            recovery_factor = 0.5 + 0.1 * intensity  # 0.5 at low, 0.7 at intensity=2
            t3_pct = t1_pct * recovery_factor
        else:
            # Escalation: crisis deepens
            escalation_factor = 1.0 + 0.1 * (intensity - 2.0)  # 1.0→1.6 as intensity→8
            t3_pct = t1_pct * escalation_factor

        # ── T+7: Day 7 Structural Outcome ─────────────────────────────
        # Includes crisis_alpha (structural over/underperformance of the sector)
        # Plus spread_beta contribution for rate-sensitive sectors
        t7_pct = t3_pct * 1.3 + sign * beta.crisis_alpha * min(1.0, intensity)
        # Spread contribution for high-intensity political crises
        if intensity > 1.5:
            spread_contrib = beta.spread_beta * intensity * 2  # bps → rough % impact
            t7_pct += spread_contrib

        # Cap at reasonable bounds (raised for fat tails)
        t1_pct = max(-15.0, min(15.0, t1_pct))
        t3_pct = max(-25.0, min(25.0, t3_pct))
        t7_pct = max(-35.0, min(35.0, t7_pct))

        # Confidence: higher for agent-sourced tickers, lower for pair-trade inferred
        confidence = 0.8 if source.startswith("agent:") else 0.5
        # Scale with intensity (more confident when crisis is clearly severe)
        confidence = min(0.95, confidence * (0.5 + min(4.0, intensity) * 0.3))

        return TickerImpact(
            ticker=ticker,
            sector=sector_key,
            direction=direction,
            t1_pct=round(t1_pct, 2),
            t3_pct=round(t3_pct, 2),
            t7_pct=round(t7_pct, 2),
            beta=beta.political_beta,
            confidence=round(confidence, 2),
            source=source,
        )

    def _tickers_for_sector(
        self, sector_key: str, agent_tickers: dict[str, str],
    ) -> list[str]:
        """Get tickers for a sector, preferring agent-sourced ones."""
        # Agent tickers in this sector (highest priority)
        agent_in_sector = [
            t for t in agent_tickers
            if self.market.get_ticker_sector(t) == sector_key
        ]
        # All known tickers for sector (fallback)
        all_in_sector = [s["ticker"] for s in self.market.tickers_for_sector(sector_key)]
        # Agent tickers first, then fill with other known tickers
        result = list(agent_in_sector)
        for t in all_in_sector:
            if t not in result:
                result.append(t)
        return result

    def _compute_ftse_impact(
        self, ticker_impacts: list[TickerImpact], intensity: float,
    ) -> float:
        """Estimate broad local-index impact (FTSE MIB for IT, S&P 500 for US, …).

        Delegates to the MarketContext's `LocalIndexModel` so the aggregation
        (weighted shorts + long offset) uses the geography's calibrated cap.
        """
        if not ticker_impacts:
            return 0.0
        short_moves = [t.t1_pct for t in ticker_impacts if t.direction == "short"]
        long_moves = [t.t1_pct for t in ticker_impacts if t.direction == "long"]
        return self.market.local_index_impact_pct(short_moves, long_moves)

    def _compute_btp_spread(self, intensity: float) -> int:
        """Sovereign-spread impact in basis points for this geography.

        The method name is kept for backward compatibility with external
        callers; it now delegates to the MarketContext's SovereignSpreadModel
        (BTP-Bund for IT, UST for US, EM sovereign for EM, etc.).
        """
        return self.market.sovereign_spread_bps(intensity, self.detected_topics)

    def _classify_warning(self, intensity: float, cri: float) -> str:
        """Market volatility warning classification."""
        # Combine intensity and CRI for warning
        composite = intensity * 0.7 + cri * 3.0 * 0.3
        if composite < 0.3:
            return "LOW"
        elif composite < 1.0:
            return "MODERATE"
        elif composite < 2.0:
            return "HIGH"
        else:
            return "CRITICAL"

    def _build_fallback_headline(
        self,
        warning: str,
        pair_trades: list[PairTrade],
        ticker_impacts: list[TickerImpact],
        wave: int,
        btp_spread: int,
    ) -> str:
        """Sync fallback headline (used when LLM is not available).

        This is a structured summary, NOT the final output.
        The LLM Flash Note (async) replaces this when available.
        """
        if not ticker_impacts:
            return f"Market impact: {warning}. No directly exposed tickers identified."

        shorts = [t for t in ticker_impacts if t.direction == "short"]
        longs = [t for t in ticker_impacts if t.direction == "long"]

        parts = [f"[{warning}]"]

        if shorts:
            top_short = min(shorts, key=lambda t: t.t1_pct)
            parts.append(
                f"SHORT {top_short.ticker} ({top_short.sector}): "
                f"T+1={top_short.t1_pct:+.1f}% T+7={top_short.t7_pct:+.1f}% (β={top_short.beta:.1f})"
            )

        if longs:
            top_long = max(longs, key=lambda t: t.t1_pct)
            parts.append(
                f"LONG {top_long.ticker} ({top_long.sector}): "
                f"T+1={top_long.t1_pct:+.1f}%"
            )

        if btp_spread > 5:
            parts.append(f"BTP +{btp_spread}bps")

        wave_desc = {1: "local", 2: "national", 3: "institutional"}
        parts.append(f"Wave {wave} ({wave_desc.get(wave, 'active')})")

        return " | ".join(parts)

    # ── LLM Analyst Flash Note ────────────────────────────────────────────

    async def generate_flash_note(
        self,
        report: FinancialImpactReport,
        crisis_brief: str = "",
        round_num: int = 0,
    ) -> str:
        """Generate a Goldman Sachs-style Flash Note via LLM.

        Call this AFTER score_round() with the returned report.
        Falls back to report.headline if LLM is unavailable.
        """
        if not self.llm:
            return report.headline

        # Build the raw data payload for the LLM
        data_payload = {
            "round": round_num,
            "crisis_brief": crisis_brief[:300],
            "warning_level": report.market_volatility_warning,
            "engagement_score": report.engagement_score,
            "contagion_risk": report.contagion_risk,
            "crisis_wave": report.crisis_wave,
            "ftse_mib_impact_pct": report.ftse_mib_impact_pct,
            "btp_spread_impact_bps": report.btp_spread_impact_bps,
            "pair_trades": [
                {
                    "topic": pt.topic,
                    "short": [f"{t.ticker} {t.short_term_pct:+.1f}% (β={t.beta:.1f})" for t in pt.short_leg[:3]],
                    "long": [f"{t.ticker} {t.short_term_pct:+.1f}% (β={t.beta:.1f})" for t in pt.long_leg[:3]],
                }
                for pt in report.pair_trades[:3]
            ],
            "crisis_scope": report.crisis_scope,
            "top_shorts": [
                f"{t.ticker} ({t.sector}): T+1={t.t1_pct:+.1f}% T+3={t.t3_pct:+.1f}% T+7={t.t7_pct:+.1f}%"
                for t in sorted(report.ticker_impacts, key=lambda x: x.t1_pct)[:5]
            ],
            "top_longs": [
                f"{t.ticker} ({t.sector}): T+1={t.t1_pct:+.1f}%"
                for t in sorted(
                    [ti for ti in report.ticker_impacts if ti.direction == "long"],
                    key=lambda x: -x.t1_pct,
                )
            ][:3],
        }

        system_prompt = (
            "You are a sell-side equity strategist at a top-tier investment bank "
            "(Goldman Sachs / JP Morgan) covering Italian & European markets. "
            "You write Flash Notes for institutional traders and portfolio managers. "
            "Your tone is precise, data-driven, and actionable. "
            "Never use emojis. Use standard market terminology."
        )

        prompt = (
            f"Based on the following real-time crisis simulation data from our "
            f"Digital Twin platform, write a 3-line Flash Note.\n\n"
            f"DATA:\n```json\n{data_payload}\n```\n\n"
            f"FORMAT (exactly 3 lines):\n"
            f"Line 1: Market sentiment + key risk (1 sentence)\n"
            f"Line 2: Actionable trade recommendation with specific tickers\n"
            f"Line 3: Tail risk warning + catalyst to watch\n\n"
            f"Write in English. Be specific about tickers and basis points. "
            f"Include beta-adjusted expected moves where relevant."
        )

        try:
            note = await self.llm.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_output_tokens=300,
                component="financial_flash_note",
            )
            return note.strip()
        except Exception as e:
            logger.warning(f"LLM Flash Note generation failed: {e}")
            return report.headline

    def get_sector_summary(self) -> dict:
        """Quick summary for dashboard.

        Sector betas are resolved through this scorer's `MarketContext`,
        so the output reflects the active beta regime (e.g. IT betas for
        an Italian scenario, US betas for an American one) — not a
        process-wide static table.
        """
        sector_keys = list(self.market.provider.sectors().keys())
        return {
            "geography": self.market.geography,
            "beta_regime": self.market.beta_regime,
            "detected_topics": self.detected_topics,
            "detected_sectors": self.detected_sectors,
            "local_index": {
                "ticker": self.market.local_index.ticker,
                "label": self.market.local_index.label,
            },
            "sovereign_spread": {
                "name": self.market.sovereign.spread_name,
                "sensitivity_bps_per_unit": self.market.sovereign.sensitivity_bps_per_unit,
            },
            "sector_betas": {
                k: {"beta": (v := self.market.get_beta(k)).political_beta,
                    "spread_beta": v.spread_beta, "defensive": v.is_defensive}
                for k in sector_keys
            },
        }
