"""Continuous self-calibration loop.

The architectural goal (Sella relazione §13.1): every day the system
fetches recent market news, runs a *shadow* forecast on a configurable
ticker watchlist, persists the prediction, and 7 trading days later
scores it against realised yfinance returns. The accumulated
prediction-vs-realised history is the data network effect — a competitor
who forks the code today still has to wait 12 months to accumulate
the equivalent calibration history.

This module is the **core**:
  - ``fetch_recent_news()`` — yfinance Ticker.news for a watchlist
  - ``predict_returns()`` — lightweight forecaster (correlation matrix
    + sector betas + impulse-response coefficients already in place)
  - ``record_forecast()`` — write to SQLite registry
  - ``evaluate_pending()`` — pick forecasts ≥ N trading days old, fetch
    realised returns, score MAE, append to drift log
  - ``running_summary()`` — aggregate metrics for the UI

The CLI entry point is ``scripts/continuous_calibration.py``.

Design choices:
  - **No LLM call** in the shadow forecast. We use the empirical pieces
    already shipped (correlation matrix, β per geography, impulse
    response per intensity bin). LLM-driven full simulation can be
    enabled via ``--llm`` flag in the CLI but is opt-in.
  - **SQLite single file** at ``outputs/continuous_calibration.db``.
    Append-only schema, no migrations, easy to backup / restore.
  - **No time-machine**: we score predictions against PAST realised
    returns, not future. ``--evaluate-on YYYY-MM-DD`` lets the operator
    backfill historical drift.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "outputs" / "continuous_calibration.db"
DEFAULT_DRIFT_LOG = REPO_ROOT / "outputs" / "continuous_calibration_drift.jsonl"

logger = logging.getLogger(__name__)


# ── Default watchlist (Italy-centric, demo-friendly for Sella) ───────────────
# Operator can override via the CLI --tickers flag.
DEFAULT_WATCHLIST: tuple[str, ...] = (
    "UCG.MI",   # UniCredit
    "ISP.MI",   # Intesa Sanpaolo
    "ENI.MI",   # ENI
    "ENEL.MI",  # Enel
    "STLAM.MI", # Stellantis
    "G.MI",     # Generali
)


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class NewsItem:
    ticker: str
    title: str
    url: Optional[str]
    published_ts: Optional[str]


@dataclass
class TickerForecast:
    ticker: str
    sector: Optional[str]
    country: Optional[str]
    t1_pct: float
    t3_pct: float
    t7_pct: float
    direction: str  # "long" / "short" / "flat"
    intensity_used: float
    confidence: float


@dataclass
class ForecastRun:
    """One nightly forecast record."""
    forecast_date: str  # YYYY-MM-DD
    brief: str
    headlines: list[NewsItem] = field(default_factory=list)
    forecasts: list[TickerForecast] = field(default_factory=list)
    inferred_intensity: float = 0.0
    inferred_cri: float = 0.0
    notes: Optional[str] = None


@dataclass
class EvaluationResult:
    forecast_date: str
    evaluated_at: str
    horizon_days: int  # 1, 3, 7
    ticker: str
    predicted_pct: float
    realized_pct: float
    abs_error_pp: float


@dataclass
class RunningSummary:
    n_forecasts: int
    n_evaluations: int
    last_forecast_date: Optional[str]
    last_evaluation_date: Optional[str]
    mae_t1_running: Optional[float]
    mae_t3_running: Optional[float]
    mae_t7_running: Optional[float]
    direction_acc_t1: Optional[float]
    by_ticker: dict[str, dict]


# ── SQLite schema + connection ──────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    sector TEXT,
    country TEXT,
    t1_pct REAL NOT NULL,
    t3_pct REAL NOT NULL,
    t7_pct REAL NOT NULL,
    direction TEXT NOT NULL,
    intensity REAL,
    confidence REAL,
    brief TEXT,
    headline_count INTEGER,
    inferred_cri REAL,
    inserted_at TEXT NOT NULL,
    PRIMARY KEY (forecast_date, ticker)
);
CREATE TABLE IF NOT EXISTS evaluations (
    forecast_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    predicted_pct REAL NOT NULL,
    realized_pct REAL NOT NULL,
    abs_error_pp REAL NOT NULL,
    evaluated_at TEXT NOT NULL,
    PRIMARY KEY (forecast_date, ticker, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_eval_date ON evaluations(evaluated_at);
"""


@contextmanager
def _connect(db_path: Path = DEFAULT_DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── News fetching ───────────────────────────────────────────────────────────


def fetch_recent_news(
    tickers: Iterable[str],
    max_per_ticker: int = 3,
) -> list[NewsItem]:
    """Pull recent headlines via yfinance Ticker.news. Best-effort:
    silently skips tickers that yfinance can't resolve."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; returning empty news list")
        return []
    out: list[NewsItem] = []
    for tk in tickers:
        try:
            news = yf.Ticker(tk).news or []
        except Exception as e:
            logger.debug("news fetch failed for %s: %s", tk, e)
            continue
        for item in news[:max_per_ticker]:
            content = item.get("content") or item
            title = content.get("title") or item.get("title")
            if not title:
                continue
            url = (
                content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else item.get("link")
            )
            published = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
            out.append(NewsItem(
                ticker=tk, title=title.strip(), url=url, published_ts=str(published) if published else None,
            ))
    return out


def headlines_to_brief(headlines: list[NewsItem]) -> str:
    """Concatenate top headlines into a one-paragraph brief. Lightweight:
    no LLM call. The shadow forecast doesn't need a polished narrative —
    it needs enough lexical signal to drive the intensity proxy."""
    if not headlines:
        return "(no recent headlines)"
    by_ticker: dict[str, list[str]] = {}
    for h in headlines:
        by_ticker.setdefault(h.ticker, []).append(h.title)
    lines = []
    for tk, titles in by_ticker.items():
        sample = "; ".join(titles[:3])
        lines.append(f"{tk}: {sample}")
    return " | ".join(lines)


# ── Intensity / CRI heuristic from headlines ────────────────────────────────


_NEG_KEYWORDS = (
    "crisis", "crash", "loss", "bankruptcy", "fraud", "fine", "lawsuit",
    "downgrade", "scandal", "outage", "breach", "default", "warning",
    "risk", "fall", "drop", "decline", "tank", "plunge", "slide",
    "layoff", "cut", "miss", "shrink", "fear", "panic",
    # Italian
    "crisi", "perdita", "fallimento", "frode", "multa", "rischio",
    "calo", "scende", "panico", "preoccupazione",
)
_POS_KEYWORDS = (
    "beat", "surge", "rally", "growth", "record", "profit", "gain",
    "rise", "jump", "boost", "deal", "merger", "acquisition", "win",
    "approval", "breakthrough",
    # Italian
    "guadagno", "balzo", "crescita", "record", "utile", "successo",
)


def infer_crisis_metrics(headlines: list[NewsItem]) -> tuple[float, float]:
    """Map headline negativity to (cri, intensity) in [0, 1] / [0, 8].

    Crude but serviceable: count negative vs positive keyword hits across
    all headline titles, scale to a CRI-like proxy. The point of the
    self-calibration loop is that this proxy gets *evaluated against
    realised returns* nightly — if it's miscalibrated, we measure that.
    """
    if not headlines:
        return 0.20, 0.5
    neg = pos = 0
    for h in headlines:
        t = h.title.lower()
        neg += sum(1 for kw in _NEG_KEYWORDS if kw in t)
        pos += sum(1 for kw in _POS_KEYWORDS if kw in t)
    total_signals = neg + pos
    if total_signals == 0:
        cri = 0.30  # mild background tension
    else:
        # negativity ratio in [0, 1] but clamped
        cri = max(0.10, min(0.95, neg / max(1, total_signals)))
        # boost if many headlines (signal density)
        cri += min(0.15, 0.02 * len(headlines))
        cri = min(0.95, cri)
    # Map cri (0-1) to intensity (0-8) via the simulator's heuristic:
    # high cri ≈ wave 3 + engagement 0.7 → intensity ~3-5
    intensity = max(0.5, min(8.0, cri * 6.0 + 0.5))
    return cri, intensity


# ── Shadow prediction ──────────────────────────────────────────────────────


def predict_returns(
    tickers: Iterable[str],
    cri: float,
    intensity: float,
    universe_meta: Optional[dict] = None,
) -> list[TickerForecast]:
    """Lightweight per-ticker prediction using the empirical pieces
    already shipped (sector β + impulse response + panic mult).

    Does NOT call the LLM. Reuses ``MarketContext.get_beta()`` (which
    consults the empirical-β JSON when present) and the empirical
    impulse-response / panic-mult tables in financial_impact.
    """
    from core.orchestrator.financial_impact import (
        _empirical_panic_mult, _empirical_ratios, _intensity_bin,
    )
    from core.orchestrator.market_context import MarketContext
    if universe_meta is None:
        universe_path = REPO_ROOT / "shared" / "stock_universe.json"
        try:
            data = json.loads(universe_path.read_text())
            universe_meta = {s["ticker"]: s for s in data.get("stocks", [])}
        except (FileNotFoundError, json.JSONDecodeError):
            universe_meta = {}

    out: list[TickerForecast] = []
    for tk in tickers:
        meta = universe_meta.get(tk, {})
        country = meta.get("country") or "IT"
        sector = meta.get("sector") or "banking"
        ctx = MarketContext(geography=country)
        beta = ctx.get_beta(sector)

        # Apply empirical panic multiplier on top of intensity
        emp_panic = _empirical_panic_mult(cri)
        effective_intensity = intensity
        if emp_panic is not None:
            effective_intensity = intensity * emp_panic[0]

        # Direction: negative CRI bias → short for high-β cyclicals
        sign = -1.0 if cri >= 0.5 and beta.political_beta > 0.8 else +1.0 if cri < 0.4 else -1.0

        # T+1: linear scaling vs effective intensity (matches simulator base)
        BASE = 0.5  # % per unit intensity
        t1_pct = sign * beta.political_beta * BASE * effective_intensity

        # T+3 / T+7: empirical ratios per (bin, sector), heuristic fallback
        emp = _empirical_ratios(sector, effective_intensity)
        if emp is not None:
            r3, r7, _, _ = emp
            t3_pct = t1_pct * r3
            t7_pct = t1_pct * r7
        else:
            t3_pct = t1_pct * (0.6 if effective_intensity < 2 else 1.2)
            t7_pct = t3_pct * 1.3

        # Cap at sane bounds (matches simulator)
        t1_pct = max(-15.0, min(15.0, t1_pct))
        t3_pct = max(-25.0, min(25.0, t3_pct))
        t7_pct = max(-35.0, min(35.0, t7_pct))

        direction = "short" if t1_pct < -0.05 else "long" if t1_pct > 0.05 else "flat"
        # Confidence: higher when empirical pieces fired
        confidence = 0.6
        if emp is not None:
            confidence += 0.15
        if emp_panic is not None:
            confidence += 0.10
        confidence = min(0.95, confidence)

        out.append(TickerForecast(
            ticker=tk, sector=sector, country=country,
            t1_pct=round(t1_pct, 3),
            t3_pct=round(t3_pct, 3),
            t7_pct=round(t7_pct, 3),
            direction=direction,
            intensity_used=round(float(effective_intensity), 3),
            confidence=round(confidence, 2),
        ))
    return out


# ── Persistence ─────────────────────────────────────────────────────────────


def record_forecast(run: ForecastRun, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Insert one forecast run (one row per ticker). Returns inserted count."""
    inserted = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        for fc in run.forecasts:
            try:
                conn.execute(
                    """INSERT INTO forecasts
                       (forecast_date, ticker, sector, country, t1_pct, t3_pct, t7_pct,
                        direction, intensity, confidence, brief, headline_count,
                        inferred_cri, inserted_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run.forecast_date, fc.ticker, fc.sector, fc.country,
                     fc.t1_pct, fc.t3_pct, fc.t7_pct, fc.direction,
                     fc.intensity_used, fc.confidence, run.brief[:1000],
                     len(run.headlines), run.inferred_cri, now),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # Duplicate (forecast_date, ticker) — overwrite
                conn.execute(
                    """UPDATE forecasts SET t1_pct=?, t3_pct=?, t7_pct=?,
                       direction=?, intensity=?, confidence=?, brief=?,
                       headline_count=?, inferred_cri=?, inserted_at=?
                       WHERE forecast_date=? AND ticker=?""",
                    (fc.t1_pct, fc.t3_pct, fc.t7_pct, fc.direction,
                     fc.intensity_used, fc.confidence, run.brief[:1000],
                     len(run.headlines), run.inferred_cri, now,
                     run.forecast_date, fc.ticker),
                )
                inserted += 1
    return inserted


# ── Evaluation against realised yfinance returns ────────────────────────────


def _fetch_realized(ticker: str, anchor_date: date) -> Optional[dict[str, float]]:
    """Pull realized log-return at T+1/T+3/T+7 from yfinance.
    Anchor = last trading day at or before ``anchor_date``."""
    try:
        import yfinance as yf
        import numpy as np
    except ImportError:
        return None
    try:
        df = yf.download(
            ticker,
            start=(anchor_date - timedelta(days=8)).isoformat(),
            end=(anchor_date + timedelta(days=15)).isoformat(),
            auto_adjust=True, progress=False, threads=False,
        )
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"]
        if hasattr(close, "iloc") and not hasattr(close, "name"):
            # multi-level columns (single-ticker yfinance quirk)
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return None
        before = close[close.index.date <= anchor_date]
        if before.empty:
            return None
        anchor_price = float(before.iloc[-1])
        after = close[close.index.date > anchor_date]
        if after.empty:
            return None
        out = {}
        for label, n in [("t1", 1), ("t3", 3), ("t7", 7)]:
            if len(after) >= n:
                # log-return × 100 → percent (matches simulator t*_pct units)
                out[label] = float(np.log(float(after.iloc[n - 1]) / anchor_price)) * 100
            else:
                out[label] = float("nan")
        return out
    except Exception as e:
        logger.debug("realised fetch failed for %s @ %s: %s", ticker, anchor_date, e)
        return None


def evaluate_pending(
    today: Optional[date] = None,
    horizon_days: int = 7,
    db_path: Path = DEFAULT_DB_PATH,
    drift_log_path: Path = DEFAULT_DRIFT_LOG,
) -> list[EvaluationResult]:
    """Score forecasts whose anchor-date is at least ``horizon_days``
    ago and that we haven't evaluated yet for that horizon."""
    today = today or date.today()
    cutoff = today - timedelta(days=horizon_days)
    cutoff_iso = cutoff.isoformat()

    results: list[EvaluationResult] = []
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT f.forecast_date, f.ticker, f.t1_pct, f.t3_pct, f.t7_pct
               FROM forecasts f
               LEFT JOIN evaluations e
                 ON e.forecast_date = f.forecast_date
                AND e.ticker = f.ticker
                AND e.horizon_days = ?
               WHERE f.forecast_date <= ?
                 AND e.forecast_date IS NULL""",
            (horizon_days, cutoff_iso),
        ).fetchall()

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in rows:
            anchor = date.fromisoformat(r["forecast_date"])
            realised = _fetch_realized(r["ticker"], anchor)
            if not realised:
                continue
            horizon_key = f"t{horizon_days}"
            real_val = realised.get(horizon_key)
            if real_val is None or real_val != real_val:  # NaN check
                continue
            pred_val = r[f"t{horizon_days}_pct"]
            err = abs(real_val - pred_val)
            conn.execute(
                """INSERT OR REPLACE INTO evaluations
                   (forecast_date, ticker, horizon_days, predicted_pct,
                    realized_pct, abs_error_pp, evaluated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (r["forecast_date"], r["ticker"], horizon_days,
                 pred_val, real_val, err, now),
            )
            results.append(EvaluationResult(
                forecast_date=r["forecast_date"],
                evaluated_at=now,
                horizon_days=horizon_days,
                ticker=r["ticker"],
                predicted_pct=pred_val,
                realized_pct=real_val,
                abs_error_pp=err,
            ))

    # Append to drift log (audit trail)
    if results:
        drift_log_path.parent.mkdir(parents=True, exist_ok=True)
        with drift_log_path.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
    return results


# ── Aggregation ─────────────────────────────────────────────────────────────


def running_summary(db_path: Path = DEFAULT_DB_PATH) -> RunningSummary:
    with _connect(db_path) as conn:
        n_fc = conn.execute("SELECT COUNT(*) AS c FROM forecasts").fetchone()["c"]
        n_ev = conn.execute("SELECT COUNT(*) AS c FROM evaluations").fetchone()["c"]
        last_fc = conn.execute(
            "SELECT MAX(forecast_date) AS d FROM forecasts"
        ).fetchone()["d"]
        last_ev = conn.execute(
            "SELECT MAX(evaluated_at) AS d FROM evaluations"
        ).fetchone()["d"]
        rows = conn.execute(
            "SELECT horizon_days, abs_error_pp, predicted_pct, realized_pct, ticker FROM evaluations"
        ).fetchall()

    by_h: dict[int, list[float]] = {1: [], 3: [], 7: []}
    by_ticker: dict[str, dict[int, list[float]]] = {}
    dir_correct = dir_total = 0
    for r in rows:
        h = r["horizon_days"]
        e = r["abs_error_pp"]
        by_h.setdefault(h, []).append(e)
        by_ticker.setdefault(r["ticker"], {}).setdefault(h, []).append(e)
        if h == 1:
            ps = 1 if r["predicted_pct"] > 0.05 else -1 if r["predicted_pct"] < -0.05 else 0
            rs = 1 if r["realized_pct"] > 0.05 else -1 if r["realized_pct"] < -0.05 else 0
            if ps != 0 and rs != 0:
                dir_total += 1
                if ps == rs:
                    dir_correct += 1

    def _mean(xs: list[float]) -> Optional[float]:
        return round(mean(xs), 3) if xs else None

    return RunningSummary(
        n_forecasts=n_fc,
        n_evaluations=n_ev,
        last_forecast_date=last_fc,
        last_evaluation_date=last_ev,
        mae_t1_running=_mean(by_h.get(1, [])),
        mae_t3_running=_mean(by_h.get(3, [])),
        mae_t7_running=_mean(by_h.get(7, [])),
        direction_acc_t1=round(dir_correct / dir_total, 3) if dir_total else None,
        by_ticker={
            tk: {f"mae_t{h}": round(mean(errs), 3) for h, errs in horizons.items() if errs}
            for tk, horizons in by_ticker.items()
        },
    )


def recent_evaluations(
    limit: int = 50,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Tail the evaluations table for UI display."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT forecast_date, ticker, horizon_days, predicted_pct,
                      realized_pct, abs_error_pp, evaluated_at
               FROM evaluations
               ORDER BY evaluated_at DESC, forecast_date DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
