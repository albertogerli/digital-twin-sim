"""Tests for the continuous self-calibration loop (Sella relazione §13.1).

Coverage:
  - Headline → CRI / intensity heuristic produces sane values on
    canonical positive / negative / mixed inputs.
  - Shadow predict() produces well-formed TickerForecast records using
    the empirical pieces (β, impulse response, panic mult).
  - SQLite registry round-trips: insert forecasts, evaluate against
    synthetic realised returns, aggregate summary.
  - Idempotency: re-running forecast for the same date overwrites; re-
    running evaluate for the same (date, ticker, horizon) is a no-op.
  - The drift log JSONL gets a row per scored evaluation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.continuous import (
    DEFAULT_WATCHLIST,
    ForecastRun,
    NewsItem,
    TickerForecast,
    evaluate_pending,
    headlines_to_brief,
    infer_crisis_metrics,
    predict_returns,
    record_forecast,
    recent_evaluations,
    running_summary,
)


# ── Heuristic intensity / CRI ───────────────────────────────────────────────


def test_infer_crisis_metrics_empty():
    cri, intensity = infer_crisis_metrics([])
    assert 0 <= cri <= 1
    assert 0 <= intensity <= 8
    # No headlines → low background tension
    assert cri < 0.4


def test_infer_crisis_metrics_negative_dominated():
    headlines = [
        NewsItem("UCG.MI", "Bank crisis deepens, default risk surges", None, None),
        NewsItem("UCG.MI", "Massive layoffs and capital loss reported", None, None),
        NewsItem("ISP.MI", "Spread widens after S&P downgrade scandal", None, None),
    ]
    cri, intensity = infer_crisis_metrics(headlines)
    assert cri >= 0.5
    assert intensity >= 3.0


def test_infer_crisis_metrics_positive_dominated():
    headlines = [
        NewsItem("ENI.MI", "Record profit beats expectations, dividend boost", None, None),
        NewsItem("ENI.MI", "Acquisition deal approved, growth surge expected", None, None),
    ]
    cri, intensity = infer_crisis_metrics(headlines)
    assert cri < 0.5


def test_infer_crisis_metrics_mixed():
    headlines = [
        NewsItem("X", "Profit gain announced", None, None),
        NewsItem("X", "Lawsuit filed alleging fraud", None, None),
        NewsItem("X", "Record growth in Q3", None, None),
    ]
    cri, intensity = infer_crisis_metrics(headlines)
    # 2 positive, 1 negative → CRI in mid range
    assert 0.2 < cri < 0.7


# ── Headlines → brief ────────────────────────────────────────────────────────


def test_headlines_to_brief_groups_by_ticker():
    h = [
        NewsItem("UCG.MI", "Title A", None, None),
        NewsItem("UCG.MI", "Title B", None, None),
        NewsItem("ISP.MI", "Title C", None, None),
    ]
    brief = headlines_to_brief(h)
    assert "UCG.MI:" in brief
    assert "ISP.MI:" in brief
    assert "Title A" in brief and "Title C" in brief


def test_headlines_to_brief_empty():
    assert "no recent headlines" in headlines_to_brief([])


# ── predict_returns: shadow forecast shape + invariants ─────────────────────


def test_predict_returns_well_formed():
    forecasts = predict_returns(
        ["UCG.MI", "ENI.MI"], cri=0.6, intensity=3.5,
    )
    assert len(forecasts) == 2
    for fc in forecasts:
        assert isinstance(fc, TickerForecast)
        assert -15.0 <= fc.t1_pct <= 15.0  # capped
        assert -25.0 <= fc.t3_pct <= 25.0
        assert -35.0 <= fc.t7_pct <= 35.0
        assert fc.direction in {"long", "short", "flat"}
        assert 0 < fc.confidence <= 1.0


def test_predict_returns_high_cri_drives_short_direction():
    """At CRI=0.85 most high-β tickers should land short."""
    forecasts = predict_returns(
        ["UCG.MI", "ISP.MI"], cri=0.85, intensity=5.0,
    )
    # At least one should land short under crisis
    assert any(fc.direction == "short" for fc in forecasts)
    # T+1 magnitudes should exceed a low-intensity baseline
    high_mags = [abs(fc.t1_pct) for fc in forecasts]
    low_forecasts = predict_returns(
        ["UCG.MI", "ISP.MI"], cri=0.20, intensity=0.5,
    )
    low_mags = [abs(fc.t1_pct) for fc in low_forecasts]
    assert sum(high_mags) > sum(low_mags)


# ── SQLite registry round-trip ──────────────────────────────────────────────


def test_record_forecast_insert_and_overwrite(tmp_path):
    db = tmp_path / "calib.db"
    run = ForecastRun(
        forecast_date="2026-05-01",
        brief="test",
        forecasts=[
            TickerForecast("UCG.MI", "banking", "IT", -2.5, -3.1, -3.6,
                           "short", 4.0, 0.75),
        ],
    )
    n = record_forecast(run, db_path=db)
    assert n == 1
    # Re-record same date+ticker → overwrite (still 1 row total)
    run.forecasts[0] = TickerForecast(
        "UCG.MI", "banking", "IT", -3.5, -4.2, -5.0, "short", 5.0, 0.80,
    )
    record_forecast(run, db_path=db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT t1_pct, intensity FROM forecasts WHERE forecast_date='2026-05-01'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == -3.5
    assert rows[0][1] == 5.0


def test_evaluate_pending_scores_against_realised(tmp_path):
    """Evaluate uses a mocked _fetch_realized so the test doesn't hit
    yfinance. We seed a forecast for a date 10 days ago and verify that
    evaluate_pending(horizon=7) scores it against the synthetic realised."""
    db = tmp_path / "calib.db"
    drift = tmp_path / "drift.jsonl"

    forecast_date = "2026-04-20"  # well in the past
    run = ForecastRun(
        forecast_date=forecast_date,
        brief="test",
        forecasts=[
            TickerForecast("UCG.MI", "banking", "IT", -2.0, -2.8, -3.5,
                           "short", 4.0, 0.75),
        ],
    )
    record_forecast(run, db_path=db)

    fake_realised = {"t1": -1.5, "t3": -2.3, "t7": -2.0}

    from core.calibration import continuous as mod
    with patch.object(mod, "_fetch_realized", return_value=fake_realised):
        results = mod.evaluate_pending(
            today=date(2026, 5, 1),
            horizon_days=7,
            db_path=db,
            drift_log_path=drift,
        )

    assert len(results) == 1
    r = results[0]
    assert r.ticker == "UCG.MI"
    assert r.predicted_pct == -3.5
    assert r.realized_pct == -2.0
    assert abs(r.abs_error_pp - 1.5) < 1e-6
    # Drift log row appended
    rows = drift.read_text().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["ticker"] == "UCG.MI"
    assert parsed["horizon_days"] == 7


def test_evaluate_pending_idempotent(tmp_path):
    """Running evaluate twice should NOT double-count."""
    db = tmp_path / "calib.db"
    run = ForecastRun(
        forecast_date="2026-04-20",
        brief="t",
        forecasts=[TickerForecast("UCG.MI", "banking", "IT", -2.0, -2.8, -3.5,
                                   "short", 4.0, 0.75)],
    )
    record_forecast(run, db_path=db)
    from core.calibration import continuous as mod
    with patch.object(mod, "_fetch_realized",
                      return_value={"t1": -1.5, "t3": -2.3, "t7": -2.0}):
        first = mod.evaluate_pending(today=date(2026, 5, 1), horizon_days=7,
                                     db_path=db, drift_log_path=tmp_path / "d.jsonl")
        second = mod.evaluate_pending(today=date(2026, 5, 1), horizon_days=7,
                                      db_path=db, drift_log_path=tmp_path / "d.jsonl")
    assert len(first) == 1
    assert len(second) == 0  # nothing new to score


# ── Aggregation ─────────────────────────────────────────────────────────────


def test_running_summary_aggregates(tmp_path):
    db = tmp_path / "calib.db"
    run = ForecastRun(
        forecast_date="2026-04-20",
        brief="t",
        forecasts=[
            TickerForecast("UCG.MI", "banking", "IT", -2.0, -2.8, -3.5,
                           "short", 4.0, 0.75),
            TickerForecast("ISP.MI", "banking", "IT", -1.5, -2.0, -2.5,
                           "short", 3.5, 0.70),
        ],
    )
    record_forecast(run, db_path=db)

    from core.calibration import continuous as mod
    with patch.object(mod, "_fetch_realized",
                      return_value={"t1": -1.5, "t3": -2.3, "t7": -2.0}):
        for h in (1, 3, 7):
            mod.evaluate_pending(today=date(2026, 5, 1), horizon_days=h,
                                 db_path=db, drift_log_path=tmp_path / "d.jsonl")

    summary = running_summary(db_path=db)
    assert summary.n_forecasts == 2
    assert summary.n_evaluations == 6  # 2 tickers × 3 horizons
    assert summary.mae_t1_running is not None and summary.mae_t1_running >= 0
    assert summary.last_forecast_date == "2026-04-20"
    assert "UCG.MI" in summary.by_ticker
    assert "mae_t1" in summary.by_ticker["UCG.MI"]


def test_recent_evaluations_returns_rows(tmp_path):
    db = tmp_path / "calib.db"
    run = ForecastRun(
        forecast_date="2026-04-20", brief="t",
        forecasts=[TickerForecast("UCG.MI", "banking", "IT", -2.0, -2.8, -3.5,
                                   "short", 4.0, 0.75)],
    )
    record_forecast(run, db_path=db)
    from core.calibration import continuous as mod
    with patch.object(mod, "_fetch_realized",
                      return_value={"t1": -1.5, "t3": -2.3, "t7": -2.0}):
        mod.evaluate_pending(today=date(2026, 5, 1), horizon_days=1,
                             db_path=db, drift_log_path=tmp_path / "d.jsonl")
    rows = recent_evaluations(limit=5, db_path=db)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "UCG.MI"


# ── Default watchlist sanity ────────────────────────────────────────────────


def test_default_watchlist_non_empty_italian_centric():
    assert len(DEFAULT_WATCHLIST) >= 4
    assert any(t.endswith(".MI") for t in DEFAULT_WATCHLIST)
