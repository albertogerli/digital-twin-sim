"""Continuous self-calibration CLI — entry point for the daily loop.

Sub-commands:

  forecast   — fetch today's headlines for the watchlist, run a shadow
               prediction per ticker, persist to SQLite.
  evaluate   — for forecasts ≥ N trading days old, fetch the realised
               returns and score (T+1 / T+3 / T+7).
  report     — print the running summary (n forecasts, MAE per horizon,
               direction accuracy, per-ticker breakdown).
  daemon     — loop: forecast every 24h, evaluate every 6h, exit on SIGINT.

The script is idempotent — running ``forecast`` twice on the same day
overwrites the previous record. ``evaluate`` only scores rows it hasn't
scored before for that horizon.

Examples::

    # Manually run the day's pipeline
    python scripts/continuous_calibration.py forecast
    python scripts/continuous_calibration.py evaluate --horizon 1
    python scripts/continuous_calibration.py evaluate --horizon 7
    python scripts/continuous_calibration.py report

    # Long-running daemon (cron-equivalent inside one process)
    python scripts/continuous_calibration.py daemon

    # cron-friendly one-shot pipeline (recommended for production)
    python scripts/continuous_calibration.py forecast \\
      && python scripts/continuous_calibration.py evaluate --horizon 1 \\
      && python scripts/continuous_calibration.py evaluate --horizon 3 \\
      && python scripts/continuous_calibration.py evaluate --horizon 7
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.calibration.continuous import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_DRIFT_LOG,
    DEFAULT_WATCHLIST,
    ForecastRun,
    evaluate_pending,
    fetch_recent_news,
    headlines_to_brief,
    infer_crisis_metrics,
    predict_returns,
    record_forecast,
    recent_evaluations,
    running_summary,
)


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )


def cmd_forecast(args) -> int:
    tickers = args.tickers.split(",") if args.tickers else list(DEFAULT_WATCHLIST)
    logging.info("Fetching news for %d tickers...", len(tickers))
    news = fetch_recent_news(tickers, max_per_ticker=args.max_news)
    logging.info("  got %d headlines", len(news))
    cri, intensity = infer_crisis_metrics(news)
    logging.info("  inferred CRI=%.2f, intensity=%.2f", cri, intensity)
    brief = headlines_to_brief(news)
    forecasts = predict_returns(tickers, cri=cri, intensity=intensity)
    run = ForecastRun(
        forecast_date=(args.date or date.today().isoformat()),
        brief=brief,
        headlines=news,
        forecasts=forecasts,
        inferred_intensity=intensity,
        inferred_cri=cri,
    )
    n = record_forecast(run, db_path=Path(args.db))
    logging.info("Recorded %d ticker forecasts for %s", n, run.forecast_date)
    if args.print:
        print(f"\nForecast for {run.forecast_date} (cri={cri:.2f}, intensity={intensity:.2f}):")
        print(f"  brief: {brief[:200]}")
        for fc in forecasts:
            print(f"  {fc.ticker:<10} {fc.country:<3} {fc.sector:<18} "
                  f"T+1={fc.t1_pct:+.2f}%  T+3={fc.t3_pct:+.2f}%  T+7={fc.t7_pct:+.2f}%  "
                  f"({fc.direction})")
    return 0


def cmd_evaluate(args) -> int:
    today = date.fromisoformat(args.today) if args.today else date.today()
    horizon = args.horizon
    logging.info("Evaluating pending forecasts (horizon=%d days, today=%s)...",
                 horizon, today)
    results = evaluate_pending(
        today=today,
        horizon_days=horizon,
        db_path=Path(args.db),
        drift_log_path=Path(args.drift_log),
    )
    logging.info("  scored %d (forecast_date, ticker) cells", len(results))
    if args.print:
        for r in results[:30]:
            print(f"  {r.forecast_date}  {r.ticker:<10} T+{r.horizon_days}  "
                  f"pred={r.predicted_pct:+6.2f}pp  real={r.realized_pct:+6.2f}pp  "
                  f"|err|={r.abs_error_pp:.2f}pp")
    return 0


def cmd_report(args) -> int:
    summary = running_summary(db_path=Path(args.db))
    print()
    print("─" * 60)
    print("Continuous self-calibration — running summary")
    print("─" * 60)
    print(f"  n forecasts       : {summary.n_forecasts}")
    print(f"  n evaluations     : {summary.n_evaluations}")
    print(f"  last forecast     : {summary.last_forecast_date or '—'}")
    print(f"  last evaluation   : {summary.last_evaluation_date or '—'}")
    print()
    print(f"  MAE T+1 (running) : {_fmt(summary.mae_t1_running)} pp")
    print(f"  MAE T+3 (running) : {_fmt(summary.mae_t3_running)} pp")
    print(f"  MAE T+7 (running) : {_fmt(summary.mae_t7_running)} pp")
    print(f"  Dir acc T+1       : {_fmt_pct(summary.direction_acc_t1)}")
    print()
    if summary.by_ticker:
        print("By ticker (MAE):")
        print(f"  {'TICKER':<10}{'T+1':>8}{'T+3':>8}{'T+7':>8}")
        for tk in sorted(summary.by_ticker.keys()):
            row = summary.by_ticker[tk]
            t1 = row.get("mae_t1")
            t3 = row.get("mae_t3")
            t7 = row.get("mae_t7")
            print(f"  {tk:<10}"
                  f"{(_fmt(t1) + 'pp') if t1 is not None else '—':>8}"
                  f"{(_fmt(t3) + 'pp') if t3 is not None else '—':>8}"
                  f"{(_fmt(t7) + 'pp') if t7 is not None else '—':>8}")
    if args.json:
        print()
        print(json.dumps({
            "n_forecasts": summary.n_forecasts,
            "n_evaluations": summary.n_evaluations,
            "last_forecast_date": summary.last_forecast_date,
            "last_evaluation_date": summary.last_evaluation_date,
            "mae_t1_running": summary.mae_t1_running,
            "mae_t3_running": summary.mae_t3_running,
            "mae_t7_running": summary.mae_t7_running,
            "direction_acc_t1": summary.direction_acc_t1,
            "by_ticker": summary.by_ticker,
        }, indent=2))
    return 0


def _fmt(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "—"


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:.0f}%" if x is not None else "—"


_stop_requested = False


def _on_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True
    logging.info("SIGINT received — daemon will stop after current iteration")


def cmd_daemon(args) -> int:
    """Long-running daemon. Forecast every 24h, evaluate every 6h."""
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)
    last_forecast = 0.0
    last_eval = 0.0
    forecast_interval = args.forecast_interval_hours * 3600
    eval_interval = args.eval_interval_hours * 3600
    logging.info("Daemon started — forecast every %dh, evaluate every %dh",
                 args.forecast_interval_hours, args.eval_interval_hours)
    while not _stop_requested:
        now = time.time()
        if now - last_forecast >= forecast_interval:
            try:
                cmd_forecast(args)
            except Exception as e:
                logging.error("forecast failed: %s", e)
            last_forecast = time.time()
        if now - last_eval >= eval_interval:
            for h in (1, 3, 7):
                try:
                    a = argparse.Namespace(**vars(args))
                    a.horizon = h
                    a.today = None
                    cmd_evaluate(a)
                except Exception as e:
                    logging.error("evaluate horizon=%d failed: %s", h, e)
            last_eval = time.time()
        time.sleep(min(60, args.eval_interval_hours * 3600 // 60))
    logging.info("Daemon stopped cleanly.")
    return 0


def cmd_recent(args) -> int:
    rows = recent_evaluations(limit=args.limit, db_path=Path(args.db))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\nLast {len(rows)} evaluations:")
        print(f"  {'forecast_date':<12}{'ticker':<10}{'h':>3}{'pred':>8}{'real':>8}{'|err|':>8}")
        for r in rows:
            print(f"  {r['forecast_date']:<12}{r['ticker']:<10}T+{r['horizon_days']:<2}"
                  f"{r['predicted_pct']:+8.2f}{r['realized_pct']:+8.2f}{r['abs_error_pp']:>8.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite registry path")
    ap.add_argument("--drift-log", default=str(DEFAULT_DRIFT_LOG))
    ap.add_argument("--verbose", "-v", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast", help="Run today's shadow forecast")
    f.add_argument("--tickers", help="Comma-separated ticker override")
    f.add_argument("--max-news", type=int, default=3)
    f.add_argument("--date", help="Override forecast date (YYYY-MM-DD)")
    f.add_argument("--print", action="store_true")
    f.set_defaults(func=cmd_forecast)

    e = sub.add_parser("evaluate", help="Score pending forecasts")
    e.add_argument("--horizon", type=int, choices=[1, 3, 7], default=7)
    e.add_argument("--today", help="Override 'today' (YYYY-MM-DD)")
    e.add_argument("--print", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="Running aggregate summary")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("daemon", help="Long-running scheduler (forecast 24h, eval 6h)")
    d.add_argument("--tickers")
    d.add_argument("--max-news", type=int, default=3)
    d.add_argument("--date", default=None)
    d.add_argument("--print", action="store_true")
    d.add_argument("--forecast-interval-hours", type=int, default=24)
    d.add_argument("--eval-interval-hours", type=int, default=6)
    d.set_defaults(func=cmd_daemon)

    rec = sub.add_parser("recent", help="Tail recent evaluations")
    rec.add_argument("--limit", type=int, default=20)
    rec.add_argument("--json", action="store_true")
    rec.set_defaults(func=cmd_recent)

    args = ap.parse_args()
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
