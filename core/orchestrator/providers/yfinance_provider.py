"""LiveMarketProvider — live market context overlay over StaticUniverseProvider.

The `SovereignSpreadRegression` trained in `sovereign_model/` takes two
market-state features at inference time:

  - `base_spread_bps`  — the current level of the relevant sovereign spread
  - `vix_level`        — the current CBOE volatility index close

The static JSON (`shared/stock_universe.json`) ships with geography-specific
priors for both, but those priors age badly: VIX can move 5-10 points
intraday, BTP-Bund 20-30 bps. Feeding the regression stale context
undercuts the whole point of a non-linear, market-aware model.

`YFinanceProvider` (kept as the class name for backward import-compat) wraps a
`StaticUniverseProvider` for all the stable fields (stocks, sectors,
org_aliases, beta regimes, structural macro config) and overlays `macro()`
with live values from TWO external feeds:

  yfinance (intra-day):
    - `^VIX`   → `macro.<geo>.sovereign.vix_level` (global volatility regime)
    - `^TNX`   → `macro.US.sovereign.base_spread_bps` (US 10Y yield × 100)
    - `^FVX`   → carried as metadata; not used by the current regression

  ECB Data Portal (SDMX, daily + monthly):
    - `YC B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y` (daily AAA euro-area 10Y spot)
       → used as a live "Bund proxy" benchmark yield
    - `IRS M.IT.L.L40.CI.0000.EUR.N.Z` (monthly Italy 10Y LTIR — published
       mid-month, ~2-4 week lag)
    - `IRS M.DE.L.L40.CI.0000.EUR.N.Z` (monthly Germany 10Y LTIR)
       → spread = (IT − DE) × 100 bps → `macro.IT.sovereign.base_spread_bps`

**Why ECB for IT/DE**: yfinance's free tier does NOT expose individual
EU sovereign yields (tried BTP=F, IT10Y=RR, ^GDBR10 — all return 404).
The ECB SDMX endpoint is the only free, authoritative, parsable source
for BTP/Bund yields. The IRS dataflow is MONTHLY (the honest trade-off
vs a fake daily value). For daily Bund-side moves we carry the AAA
euro-area yield curve as a supplementary signal.

Caching & failure modes:
  - SQLite cache at `cache/market_snapshot.db`, TTL 300s for the yfinance
    leg, 3600s for the ECB leg (monthly data doesn't move every 5 minutes)
  - Any fetch error → fall back silently to the static prior; callers
    always get a valid `macro()` dict, never a partial one
  - Thread-timeout wrappers: yfinance and ECB calls each capped at 8s

Every overlaid field is tagged with a `_<field>_source` key so the
frontend / logs / pitch deck can show exactly where each number came from
(e.g. `"ecb_irs_monthly:2026-03"` vs `"static_prior"`).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..market_data import StaticUniverseProvider, _DEFAULT_UNIVERSE_PATH

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "cache",
    "market_snapshot.db",
)
_CACHE_SCHEMA_VERSION = 2  # Bumped when snapshot shape changes
_DEFAULT_TTL_SEC = 300            # yfinance side — intraday data
_ECB_TTL_SEC = 3600               # ECB IRS side — monthly data, 1h cache is fine
_FETCH_TIMEOUT_SEC = 8.0

# yfinance tickers. One network round-trip, cheap to extend.
_LIVE_TICKERS = ("^VIX", "^TNX", "^FVX")

# ECB SDMX endpoints. Keep these URL-encoded exactly — the dataflow/key
# syntax is brittle. Each returns CSV via `?format=csvdata`.
_ECB_API_ROOT = "https://data-api.ecb.europa.eu/service/data"
_ECB_AAA_10Y_KEY = "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"   # daily AAA 10Y spot
_ECB_IT_10Y_LTIR_KEY = "IRS/M.IT.L.L40.CI.0000.EUR.N.Z"    # monthly IT 10Y
_ECB_DE_10Y_LTIR_KEY = "IRS/M.DE.L.L40.CI.0000.EUR.N.Z"    # monthly DE 10Y


@dataclass(frozen=True)
class LiveMarketSnapshot:
    """One-shot market snapshot — inspectable, serialisable, immutable.

    Every yield-like field is a percent (e.g. 4.25 means 4.25%). Spread
    fields are bps. `*_as_of` is the publication date of the underlying
    observation (not the fetch timestamp — that's `fetched_at`).
    """

    # ── yfinance leg (intraday) ────────────────────────────────────────
    vix: Optional[float] = None              # ^VIX close
    ust_10y_pct: Optional[float] = None      # ^TNX (percent)
    ust_5y_pct: Optional[float] = None       # ^FVX (percent)

    # ── ECB leg (daily AAA + monthly country LTIR) ─────────────────────
    ecb_aaa_10y_pct: Optional[float] = None  # Daily AAA euro-area 10Y spot
    ecb_aaa_as_of: Optional[str] = None      # ISO date of last ECB obs
    it_10y_pct: Optional[float] = None       # Monthly IT 10Y LTIR (percent)
    it_10y_as_of: Optional[str] = None       # e.g. "2026-03"
    de_10y_pct: Optional[float] = None       # Monthly DE 10Y LTIR (percent)
    de_10y_as_of: Optional[str] = None

    # ── Common metadata ────────────────────────────────────────────────
    fetched_at: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)

    # ── Derived ────────────────────────────────────────────────────────
    @property
    def btp_bund_spread_bps(self) -> Optional[float]:
        """IT − DE 10Y LTIR spread in basis points (live ECB data)."""
        if self.it_10y_pct is None or self.de_10y_pct is None:
            return None
        return round((self.it_10y_pct - self.de_10y_pct) * 100, 1)

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.fetched_at) < _DEFAULT_TTL_SEC

    @property
    def has_any_live(self) -> bool:
        return any(
            v is not None for v in (
                self.vix, self.ust_10y_pct, self.ust_5y_pct,
                self.ecb_aaa_10y_pct, self.it_10y_pct, self.de_10y_pct,
            )
        )


class _SnapshotCache:
    """SQLite-backed snapshot cache. Stores the whole snapshot as JSON so
    future fields land transparently — past versions of this file broke
    caches on every shape change."""

    def __init__(self, path: str = _DEFAULT_CACHE_PATH):
        self._path = path
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self._path) as c:
            # Drop any legacy v1 table that used columns instead of a blob.
            cur = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_snapshot'"
            )
            if cur.fetchone():
                cols = {
                    row[1]
                    for row in c.execute("PRAGMA table_info(market_snapshot)").fetchall()
                }
                if "payload" not in cols:
                    c.execute("DROP TABLE market_snapshot")
            c.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshot (
                    name TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    fetched_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
            """)

    def load(self, name: str = "default") -> Optional[LiveMarketSnapshot]:
        try:
            with sqlite3.connect(self._path) as c:
                row = c.execute(
                    "SELECT schema_version, fetched_at, payload FROM market_snapshot WHERE name = ?",
                    (name,),
                ).fetchone()
        except sqlite3.Error as e:
            logger.warning("Snapshot cache read failed (%s); skipping cache", e)
            return None
        if row is None or row[0] != _CACHE_SCHEMA_VERSION:
            return None
        try:
            payload = json.loads(row[2])
        except json.JSONDecodeError:
            return None
        return LiveMarketSnapshot(**payload)

    def save(self, snapshot: LiveMarketSnapshot, name: str = "default") -> None:
        # Pydantic would be overkill — build the payload by hand so we own
        # exactly which fields persist.
        payload = {
            "vix": snapshot.vix,
            "ust_10y_pct": snapshot.ust_10y_pct,
            "ust_5y_pct": snapshot.ust_5y_pct,
            "ecb_aaa_10y_pct": snapshot.ecb_aaa_10y_pct,
            "ecb_aaa_as_of": snapshot.ecb_aaa_as_of,
            "it_10y_pct": snapshot.it_10y_pct,
            "it_10y_as_of": snapshot.it_10y_as_of,
            "de_10y_pct": snapshot.de_10y_pct,
            "de_10y_as_of": snapshot.de_10y_as_of,
            "fetched_at": snapshot.fetched_at,
            "errors": snapshot.errors,
        }
        try:
            with sqlite3.connect(self._path) as c:
                c.execute(
                    "INSERT OR REPLACE INTO market_snapshot (name, schema_version, fetched_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (name, _CACHE_SCHEMA_VERSION, snapshot.fetched_at, json.dumps(payload)),
                )
        except sqlite3.Error as e:
            logger.warning("Snapshot cache write failed (%s)", e)


# ── ECB SDMX fetchers ────────────────────────────────────────────────────

def _ecb_fetch_csv(series_key: str, last_n: int = 1) -> list[tuple[str, float]]:
    """Fetch the last N observations from an ECB SDMX CSV series.

    Returns a list of (time_period, obs_value) tuples, chronologically.
    Raises RuntimeError on any parse/network failure — caller must catch.
    """
    import httpx

    url = f"{_ECB_API_ROOT}/{series_key}"
    resp = httpx.get(
        url,
        params={"format": "csvdata", "lastNObservations": last_n},
        headers={
            "Accept": "text/csv",
            "User-Agent": "DigitalTwinSim-MarketProvider/1.0",
        },
        timeout=_FETCH_TIMEOUT_SEC,
        follow_redirects=True,
    )
    resp.raise_for_status()
    body = resp.text

    # CSV: first line = header, rest = data rows
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"no observations returned for {series_key}")
    header = lines[0].split(",")
    try:
        tp_idx = header.index("TIME_PERIOD")
        val_idx = header.index("OBS_VALUE")
    except ValueError as e:
        raise RuntimeError(f"unexpected ECB CSV header: {e}") from e

    out: list[tuple[str, float]] = []
    for line in lines[1:]:
        # ECB CSV fields don't contain commas in obs rows but row parsing
        # must be comma-split (no quoting present for numeric rows).
        parts = line.split(",")
        if len(parts) <= max(tp_idx, val_idx):
            continue
        try:
            out.append((parts[tp_idx], float(parts[val_idx])))
        except ValueError:
            continue
    if not out:
        raise RuntimeError(f"parsed 0 rows for {series_key}")
    return out


def _fetch_ecb_snapshot() -> dict:
    """Pull ECB AAA daily + IT/DE monthly. Returns a dict of {field: value}
    with keys matching LiveMarketSnapshot. Missing fields left as None so
    partial failures don't poison the rest of the snapshot."""
    out = {
        "ecb_aaa_10y_pct": None, "ecb_aaa_as_of": None,
        "it_10y_pct": None, "it_10y_as_of": None,
        "de_10y_pct": None, "de_10y_as_of": None,
    }
    errors: dict[str, str] = {}

    def _pull(key: str, field_val: str, field_asof: str):
        try:
            rows = _ecb_fetch_csv(key, last_n=1)
            tp, val = rows[-1]
            out[field_val] = val
            out[field_asof] = tp
        except Exception as e:
            errors[key] = str(e)[:120]

    # Three serial fetches; small enough to not bother threading — each
    # returns in ~300-500ms, total ≤ 2s under normal network conditions.
    _pull(_ECB_AAA_10Y_KEY, "ecb_aaa_10y_pct", "ecb_aaa_as_of")
    _pull(_ECB_IT_10Y_LTIR_KEY, "it_10y_pct", "it_10y_as_of")
    _pull(_ECB_DE_10Y_LTIR_KEY, "de_10y_pct", "de_10y_as_of")

    return {"values": out, "errors": errors}


# ── yfinance fetcher (unchanged logic, isolated for clarity) ─────────────

def _fetch_yfinance_snapshot() -> dict:
    """One yfinance round-trip for VIX + UST 10Y/5Y. Returns dict of
    {field: value} + {errors}."""
    import yfinance as yf

    values: dict[str, Optional[float]] = {t: None for t in _LIVE_TICKERS}
    errors: dict[str, str] = {}

    try:
        data = yf.download(
            list(_LIVE_TICKERS),
            period="5d", interval="1d",
            progress=False, threads=True, auto_adjust=False,
        )
    except Exception as e:
        for t in _LIVE_TICKERS:
            errors[t] = f"batch_fetch_failed: {str(e)[:80]}"
        return {
            "values": {"vix": None, "ust_10y_pct": None, "ust_5y_pct": None},
            "errors": errors,
        }

    try:
        close = data["Close"]
        for t in _LIVE_TICKERS:
            if t in close.columns:
                series = close[t].dropna()
                if len(series) > 0:
                    values[t] = float(series.iloc[-1])
                else:
                    errors[t] = "empty_series"
            else:
                errors[t] = "ticker_missing_from_response"
    except Exception as e:
        errors["_parse"] = f"response_parse_failed: {str(e)[:80]}"

    return {
        "values": {
            "vix": values["^VIX"],
            "ust_10y_pct": values["^TNX"],
            "ust_5y_pct": values["^FVX"],
        },
        "errors": errors,
    }


def _fetch_snapshot_sync() -> LiveMarketSnapshot:
    """Combined fetch: yfinance + ECB, running them on parallel threads
    so total wall time ≈ max(yf_time, ecb_time) instead of sum."""
    yf_result: list[dict] = []
    ecb_result: list[dict] = []

    def _yf_worker():
        try:
            yf_result.append(_fetch_yfinance_snapshot())
        except Exception as e:
            yf_result.append({"values": {}, "errors": {"_yf_worker": str(e)[:80]}})

    def _ecb_worker():
        try:
            ecb_result.append(_fetch_ecb_snapshot())
        except Exception as e:
            ecb_result.append({"values": {}, "errors": {"_ecb_worker": str(e)[:80]}})

    t_yf = threading.Thread(target=_yf_worker, daemon=True)
    t_ecb = threading.Thread(target=_ecb_worker, daemon=True)
    t_yf.start()
    t_ecb.start()
    t_yf.join(timeout=_FETCH_TIMEOUT_SEC)
    t_ecb.join(timeout=_FETCH_TIMEOUT_SEC)

    yf_vals = (yf_result[0]["values"] if yf_result else {}) or {}
    yf_errs = (yf_result[0]["errors"] if yf_result else {"_yf_timeout": "no_result"}) or {}
    ecb_vals = (ecb_result[0]["values"] if ecb_result else {}) or {}
    ecb_errs = (ecb_result[0]["errors"] if ecb_result else {"_ecb_timeout": "no_result"}) or {}

    errors: dict[str, str] = {}
    errors.update(yf_errs)
    errors.update(ecb_errs)

    return LiveMarketSnapshot(
        vix=yf_vals.get("vix"),
        ust_10y_pct=yf_vals.get("ust_10y_pct"),
        ust_5y_pct=yf_vals.get("ust_5y_pct"),
        ecb_aaa_10y_pct=ecb_vals.get("ecb_aaa_10y_pct"),
        ecb_aaa_as_of=ecb_vals.get("ecb_aaa_as_of"),
        it_10y_pct=ecb_vals.get("it_10y_pct"),
        it_10y_as_of=ecb_vals.get("it_10y_as_of"),
        de_10y_pct=ecb_vals.get("de_10y_pct"),
        de_10y_as_of=ecb_vals.get("de_10y_as_of"),
        fetched_at=time.time(),
        errors=errors,
    )


class YFinanceProvider:
    """`MarketDataProvider` with a live-data overlay on the macro block.

    Name kept for backward import-compat; internally the class pulls from
    BOTH yfinance (intraday) AND ECB (daily AAA + monthly country LTIR).

    All structural fields (stocks, sectors, indices, org aliases) delegate
    to a wrapped `StaticUniverseProvider`. Only `macro()` is transformed —
    VIX globally, US base_spread_bps from live ^TNX, IT base_spread_bps
    from live ECB IT−DE yield differential.
    """

    def __init__(
        self,
        static: Optional[StaticUniverseProvider] = None,
        cache: Optional[_SnapshotCache] = None,
        ttl_sec: int = _DEFAULT_TTL_SEC,
    ):
        self._static = static or StaticUniverseProvider()
        self._cache = cache or _SnapshotCache()
        self._ttl_sec = ttl_sec
        self._snapshot: Optional[LiveMarketSnapshot] = None
        cached = self._cache.load()
        if cached and (time.time() - cached.fetched_at) < self._ttl_sec:
            self._snapshot = cached
            logger.info(
                "Loaded cached market snapshot (age=%.0fs): VIX=%s, UST10Y=%s, BTP-Bund=%s bps",
                time.time() - cached.fetched_at,
                cached.vix, cached.ust_10y_pct, cached.btp_bund_spread_bps,
            )

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, refresh: bool = True) -> "YFinanceProvider":
        """Construct and optionally fetch a fresh snapshot immediately."""
        provider = cls()
        if refresh:
            provider.refresh()
        return provider

    # ── Lifecycle ────────────────────────────────────────────────────────

    def refresh(self, force: bool = False) -> LiveMarketSnapshot:
        """Fetch a live snapshot unless the in-memory one is still fresh."""
        if not force and self._snapshot and self._snapshot.is_fresh:
            return self._snapshot
        logger.info("Refreshing market snapshot (yfinance + ECB)...")
        snapshot = _fetch_snapshot_sync()
        if snapshot.has_any_live:
            self._snapshot = snapshot
            self._cache.save(snapshot)
            logger.info(
                "Snapshot refreshed: VIX=%s, UST10Y=%s, BTP-Bund=%s bps "
                "(IT %s, DE %s), AAA euro-area 10Y=%s (%s); errors=%s",
                snapshot.vix, snapshot.ust_10y_pct,
                snapshot.btp_bund_spread_bps,
                snapshot.it_10y_as_of, snapshot.de_10y_as_of,
                snapshot.ecb_aaa_10y_pct, snapshot.ecb_aaa_as_of,
                list(snapshot.errors.keys()) or "none",
            )
        else:
            logger.warning(
                "Snapshot refresh returned no live values (errors: %s); "
                "keeping previous snapshot if any",
                snapshot.errors,
            )
            if self._snapshot is None:
                self._snapshot = snapshot
        return self._snapshot

    @property
    def snapshot(self) -> Optional[LiveMarketSnapshot]:
        return self._snapshot

    # ── MarketDataProvider protocol ──────────────────────────────────────

    def sectors(self) -> dict[str, dict]:
        return self._static.sectors()

    def stocks(self) -> list[dict]:
        return self._static.stocks()

    def indices(self) -> list[dict]:
        return self._static.indices()

    def org_aliases(self) -> dict[str, list[str]]:
        return self._static.org_aliases()

    def macro(self) -> dict[str, dict]:
        """Static macro config with live overlays.

        Applied overlays:
          - `vix_level` (all geos) ← yfinance ^VIX
          - `US.base_spread_bps` ← yfinance ^TNX × 100
          - `IT.base_spread_bps` ← ECB (IT 10Y LTIR − DE 10Y LTIR) × 100 bps

        Every overlaid field gets a `_<field>_source` marker and an
        `_<field>_as_of` date so downstream consumers (pitch deck,
        dashboard, regression audit log) know exactly how fresh the
        number is.
        """
        base = copy.deepcopy(self._static.macro())
        snap = self._snapshot

        for geo, cfg in base.items():
            sov = cfg.setdefault("sovereign", {})

            # ── VIX (global) ─────────────────────────────────────────
            if snap and snap.vix is not None:
                sov["vix_level"] = round(snap.vix, 2)
                sov["_vix_source"] = "yfinance:^VIX"
                sov["_vix_fetched_at"] = snap.fetched_at
            else:
                sov["_vix_source"] = "static_prior"

            # ── US 10Y from ^TNX ─────────────────────────────────────
            if geo == "US" and snap and snap.ust_10y_pct is not None:
                sov["base_spread_bps"] = round(snap.ust_10y_pct * 100, 1)
                sov["_spread_source"] = "yfinance:^TNX"
                sov["_spread_fetched_at"] = snap.fetched_at
                sov["_spread_as_of"] = "realtime"

            # ── IT BTP-Bund spread from ECB (IT − DE) ────────────────
            elif geo == "IT" and snap and snap.btp_bund_spread_bps is not None:
                sov["base_spread_bps"] = snap.btp_bund_spread_bps
                sov["_spread_source"] = (
                    f"ecb_irs_monthly:IT({snap.it_10y_as_of})-DE({snap.de_10y_as_of})"
                )
                sov["_spread_fetched_at"] = snap.fetched_at
                sov["_spread_as_of"] = snap.it_10y_as_of
                sov["_it_10y_pct"] = snap.it_10y_pct
                sov["_de_10y_pct"] = snap.de_10y_pct
                if snap.ecb_aaa_10y_pct is not None:
                    sov["_eu_aaa_10y_pct"] = snap.ecb_aaa_10y_pct
                    sov["_eu_aaa_as_of"] = snap.ecb_aaa_as_of

            # ── EU block: attach the ECB AAA daily yield as reference ─
            elif geo == "EU" and snap and snap.ecb_aaa_10y_pct is not None:
                # EU block carries AAA daily as its live benchmark; static
                # base_spread_bps prior stays as the portfolio-weighted
                # estimate (no free daily EA-wide spread index exists).
                sov["_spread_source"] = "static_prior"
                sov["_eu_aaa_10y_pct"] = snap.ecb_aaa_10y_pct
                sov["_eu_aaa_as_of"] = snap.ecb_aaa_as_of

            else:
                sov.setdefault("_spread_source", "static_prior")

        return base


# ── Module-level default + refresh hook for the API layer ────────────────

_LIVE_PROVIDER: Optional[YFinanceProvider] = None


def get_live_provider() -> YFinanceProvider:
    """Process-wide default live provider."""
    global _LIVE_PROVIDER
    if _LIVE_PROVIDER is None:
        _LIVE_PROVIDER = YFinanceProvider()
    return _LIVE_PROVIDER


def refresh_market_snapshot(force: bool = False) -> LiveMarketSnapshot:
    """Refresh hook — idempotent (TTL-gated), cheap, never raises."""
    return get_live_provider().refresh(force=force)
