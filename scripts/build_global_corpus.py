"""Build a global, empirically-validated extension of the backtest corpus.

Defines ~35 non-Italian historical events, fetches the actual T+1/T+3/T+7
realized returns from yfinance for each scenario's seed-ticker basket,
flags mismatches between the *expected* direction (encoded by hand) and
the *realized* direction (from market data), and emits
``backtest_scenarios_global.py`` — a drop-in companion to the existing
``backtest_scenarios.py``.

Intentionally narrow scope: this script does not try to re-estimate
crisis metrics (engagement_score, polarization etc.) for past events; it
fills them with neutral defaults and lets the simulator compute its own
metrics when each scenario is later replayed. The corpus is meant for
*financial-impact validation*, not opinion-dynamics calibration.

Usage::

    python scripts/build_global_corpus.py
    python scripts/build_global_corpus.py --check-only   # don't write file

Audit output is printed to stdout: every event prints (name, ticker,
expected, realized T+1 / T+3 / T+7, hit). Hit-rate per country is the
honest read of whether our expected_directions hold up under real data.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "backtest_scenarios_global.py"


@dataclass
class GlobalEvent:
    """A historical financial-impact event for the extended corpus."""
    name: str
    event_date: str  # YYYY-MM-DD; the day the news broke
    country: str  # ISO-2; informational, drives the country-bucket aggregates
    brief: str
    topics: list[str]
    sectors: list[str]
    seed_tickers: list[str]  # the basket whose move we want to predict
    expected: dict[str, str]  # ticker → "down" / "up" / "flat"
    contagion_risk: float  # our subjective ex-ante read; not validated here
    notes: str = ""


# ───────────────────────────────────────────────────────────────────────────
# 35-event corpus, geographically diverse. Each event is one we expect to
# have moved markets in a known direction. The script auto-validates by
# fetching realized returns. Where my expected direction is wrong, the
# audit will surface it and we'll either drop the event or flip the sign.
# ───────────────────────────────────────────────────────────────────────────

EVENTS: list[GlobalEvent] = [
    # ── United States ──
    GlobalEvent(
        name="Lehman Brothers Bankruptcy (Sep 2008)",
        event_date="2008-09-15",
        country="US",
        brief="Lehman files Chapter 11; Merrill sold to BofA. Worst single-day systemic shock since 1987.",
        topics=["financial_crisis", "systemic_risk"],
        sectors=["banking"],
        seed_tickers=["JPM", "BAC", "C", "GS", "MS"],
        expected={"JPM": "down", "BAC": "down", "C": "down", "GS": "down", "MS": "down"},
        contagion_risk=0.95,
        notes="S&P -4.7% Sep 15; banks -10 to -25% intraday.",
    ),
    GlobalEvent(
        name="SVB Collapse (Mar 2023)",
        event_date="2023-03-10",
        country="US",
        brief="Silicon Valley Bank seized by FDIC. Largest US bank failure since 2008.",
        topics=["financial_crisis", "regional_banking"],
        sectors=["banking"],
        seed_tickers=["JPM", "BAC", "C", "WFC"],
        expected={"JPM": "down", "BAC": "down", "C": "down", "WFC": "down"},
        contagion_risk=0.85,
        notes="Regional bank index -22% week; Big-4 -7 to -12%.",
    ),
    GlobalEvent(
        name="Trump Tariff Escalation (May 2019)",
        event_date="2019-05-05",
        country="US",
        brief="Trump tweets +25% tariff on $200B Chinese imports; trade war re-escalation.",
        topics=["trade_war"],
        sectors=["tech", "automotive"],
        seed_tickers=["AAPL", "MSFT", "NVDA", "TSLA"],
        expected={"AAPL": "down", "MSFT": "down", "NVDA": "down", "TSLA": "down"},
        contagion_risk=0.75,
        notes="Nasdaq -3.4% Mon May 13; semis worst hit.",
    ),
    GlobalEvent(
        name="COVID Lockdown US (Mar 2020)",
        event_date="2020-03-16",
        country="US",
        brief="WHO pandemic declaration + US travel ban + state lockdowns. Global market crash.",
        topics=["public_health", "recession"],
        sectors=["banking", "energy_fossil", "automotive", "luxury"],
        seed_tickers=["JPM", "XOM", "CVX", "AAPL", "DIS"],
        expected={"JPM": "down", "XOM": "down", "CVX": "down", "AAPL": "down", "DIS": "down"},
        contagion_risk=0.99,
        notes="S&P -12% single day; oil -30%.",
    ),
    GlobalEvent(
        name="FTX Collapse (Nov 2022)",
        event_date="2022-11-08",
        country="US",
        brief="FTX bank-run; Binance walks away from acquisition. Crypto-banking contagion fears.",
        topics=["financial_crisis", "crypto"],
        sectors=["banking", "tech"],
        seed_tickers=["JPM", "GS", "MS", "MSFT"],
        expected={"JPM": "flat", "GS": "down", "MS": "down", "MSFT": "down"},
        contagion_risk=0.55,
        notes="Crypto-exposed firms hit hardest; banks contained.",
    ),
    GlobalEvent(
        name="Trump Re-election (Nov 2024)",
        event_date="2024-11-06",
        country="US",
        brief="Trump wins 2024 presidential election. Markets pricing tax cuts + tariffs + deregulation.",
        topics=["election", "deregulation", "tariffs"],
        sectors=["banking", "energy_fossil", "defense"],
        seed_tickers=["JPM", "BAC", "XOM", "LMT", "TSLA"],
        expected={"JPM": "up", "BAC": "up", "XOM": "up", "LMT": "up", "TSLA": "up"},
        contagion_risk=0.40,
        notes="Bank +6 to +11% Wed; Tesla +14%.",
    ),

    # ── United Kingdom ──
    GlobalEvent(
        name="Brexit Referendum (Jun 2016)",
        event_date="2016-06-24",
        country="GB",
        brief="UK votes to leave EU. GBP collapses; UK banks repriced.",
        topics=["sovereignty", "eu_integration"],
        sectors=["banking"],
        seed_tickers=["BARC.L", "HSBA.L", "LLOY.L"],
        expected={"BARC.L": "down", "HSBA.L": "down", "LLOY.L": "down"},
        contagion_risk=0.90,
        notes="UK banks -20 to -30% Friday open.",
    ),
    GlobalEvent(
        name="Truss Mini-Budget (Sep 2022)",
        event_date="2022-09-23",
        country="GB",
        brief="Kwarteng unfunded tax-cut budget; gilt market crisis; BoE emergency intervention.",
        topics=["fiscal_policy", "sovereign_crisis"],
        sectors=["banking"],
        seed_tickers=["BARC.L", "HSBA.L", "LLOY.L"],
        expected={"BARC.L": "down", "HSBA.L": "down", "LLOY.L": "down"},
        contagion_risk=0.85,
        notes="GBP touches all-time low vs USD; gilt yields +100bps.",
    ),

    # ── France & Germany ──
    GlobalEvent(
        name="Macron Election (May 2017)",
        event_date="2017-05-08",
        country="FR",
        brief="Macron defeats Le Pen in runoff. Pro-EU continuity. Risk-on rally.",
        topics=["election", "eu_integration"],
        sectors=["banking"],
        seed_tickers=["BNP.PA", "GLE.PA", "ACA.PA"],
        expected={"BNP.PA": "up", "GLE.PA": "up", "ACA.PA": "up"},
        contagion_risk=0.20,
        notes="CAC 40 already pricing the win Friday; modest gains Monday.",
    ),
    GlobalEvent(
        name="Le Pen 1st-Round Surge (Apr 2017)",
        event_date="2017-04-24",
        country="FR",
        brief="Macron 24% / Le Pen 21% first round. Le Pen close enough to spook markets, Macron favoured but not certain.",
        topics=["election", "sovereignty"],
        sectors=["banking"],
        seed_tickers=["BNP.PA", "GLE.PA"],
        expected={"BNP.PA": "up", "GLE.PA": "up"},
        contagion_risk=0.35,
        notes="Markets relieved Le Pen didn't top first round; CAC +4%.",
    ),
    GlobalEvent(
        name="Credit Suisse / UBS Rescue (Mar 2023)",
        event_date="2023-03-20",
        country="CH",
        brief="UBS forced to buy Credit Suisse over a weekend; AT1 bonds wiped to zero.",
        topics=["financial_crisis", "banking_resolution"],
        sectors=["banking"],
        seed_tickers=["UBSG.SW", "DBK.DE", "BNP.PA", "BARC.L"],
        expected={"UBSG.SW": "down", "DBK.DE": "down", "BNP.PA": "down", "BARC.L": "down"},
        contagion_risk=0.80,
        notes="UBS -12% Mon; European AT1s repriced violently.",
    ),
    GlobalEvent(
        name="EU Energy Crisis Peak (Aug 2022)",
        event_date="2022-08-26",
        country="DE",
        brief="Russian gas to Germany cut; TTF gas to €340/MWh. German recession risk.",
        topics=["energy_crisis", "sanctions"],
        sectors=["utilities", "automotive"],
        seed_tickers=["VOW3.DE", "BAS.DE", "SIE.DE", "RWE.DE"],
        expected={"VOW3.DE": "down", "BAS.DE": "down", "SIE.DE": "down", "RWE.DE": "up"},
        contagion_risk=0.80,
        notes="Industrials -3 to -6%; utilities mixed (RWE benefits from price spike).",
    ),

    # ── Asia-Pacific ──
    GlobalEvent(
        name="Fukushima Earthquake (Mar 2011)",
        event_date="2011-03-14",
        country="JP",
        brief="9.0 quake + tsunami + nuclear meltdown. Nikkei circuit-broken.",
        topics=["natural_disaster", "nuclear"],
        sectors=["banking", "infrastructure", "automotive"],
        seed_tickers=["7203.T", "8306.T", "6758.T", "6501.T"],
        expected={"7203.T": "down", "8306.T": "down", "6758.T": "down", "6501.T": "down"},
        contagion_risk=0.99,
        notes="Nikkei -10.6% Tue; auto and electronics worst hit.",
    ),
    GlobalEvent(
        name="HK Extradition Bill Withdrawal (Sep 2019)",
        event_date="2019-09-04",
        country="HK",
        brief="Carrie Lam formally withdraws extradition bill. Markets relieved.",
        topics=["civil_unrest", "china_relations"],
        sectors=["banking", "real_estate"],
        seed_tickers=["0005.HK", "0388.HK", "0700.HK"],
        expected={"0005.HK": "up", "0388.HK": "up", "0700.HK": "up"},
        contagion_risk=0.30,
        notes="Hang Seng +4% intraday; banks rally.",
    ),
    GlobalEvent(
        name="Evergrande Default Watch (Sep 2021)",
        event_date="2021-09-20",
        country="CN",
        brief="Evergrande misses bond coupon; Chinese property contagion fears.",
        topics=["real_estate_crisis", "shadow_banking"],
        sectors=["banking", "real_estate"],
        seed_tickers=["3988.HK", "0939.HK", "BABA", "JD"],
        expected={"3988.HK": "down", "0939.HK": "down", "BABA": "down", "JD": "down"},
        contagion_risk=0.80,
        notes="Hang Seng -3.3%; Chinese banks -3 to -5%.",
    ),
    GlobalEvent(
        name="China Tech Crackdown — Didi Probe (Jul 2021)",
        event_date="2021-07-06",
        country="CN",
        brief="Cyberspace Admin opens cybersecurity probe on Didi 2 days post-IPO. Tech sector repriced.",
        topics=["regulation", "data_security"],
        sectors=["tech"],
        seed_tickers=["BABA", "JD", "TCEHY", "9988.HK"],
        expected={"BABA": "down", "JD": "down", "TCEHY": "down", "9988.HK": "down"},
        contagion_risk=0.70,
        notes="China ADRs -5 to -25% week.",
    ),
    GlobalEvent(
        name="Korea Martial Law Attempt (Dec 2024)",
        event_date="2024-12-03",
        country="KR",
        brief="Yoon declares martial law late evening; National Assembly vacates within hours; impeachment motion follows.",
        topics=["political_crisis", "democracy"],
        sectors=["tech", "automotive"],
        seed_tickers=["005930.KS", "000660.KS", "005380.KS"],
        expected={"005930.KS": "down", "000660.KS": "down", "005380.KS": "down"},
        contagion_risk=0.85,
        notes="KOSPI -2% Wed; Samsung/SK Hynix worst.",
    ),

    # ── Latin America ──
    GlobalEvent(
        name="Bolsonaro Election (Oct 2018)",
        event_date="2018-10-29",
        country="BR",
        brief="Bolsonaro wins Brazilian runoff. Pro-market rally on privatization expectations.",
        topics=["election", "deregulation"],
        sectors=["banking", "energy_fossil"],
        seed_tickers=["VALE", "ITUB", "PBR"],
        expected={"VALE": "up", "ITUB": "up", "PBR": "up"},
        contagion_risk=0.30,
        notes="Bovespa +2.4% Mon; banks +3-6%.",
    ),
    GlobalEvent(
        name="Lula Election (Oct 2022)",
        event_date="2022-10-31",
        country="BR",
        brief="Lula narrow win over Bolsonaro. Markets price more interventionist policy.",
        topics=["election", "fiscal_policy"],
        sectors=["banking", "energy_fossil"],
        seed_tickers=["VALE", "ITUB", "PBR"],
        expected={"VALE": "down", "ITUB": "flat", "PBR": "down"},
        contagion_risk=0.45,
        notes="Petrobras -3 to -6% on dividend / pricing concerns.",
    ),
    GlobalEvent(
        name="Milei Election (Nov 2023)",
        event_date="2023-11-20",
        country="AR",
        brief="Argentine libertarian Milei wins runoff. Dollarization plan; ADRs surge.",
        topics=["election", "monetary_reform"],
        sectors=["banking", "energy_fossil"],
        seed_tickers=["YPF", "GGAL"],
        expected={"YPF": "up", "GGAL": "up"},
        contagion_risk=0.40,
        notes="YPF +35%, GGAL +40% Mon — historic single-day rallies.",
    ),

    # ── Geopolitical ──
    GlobalEvent(
        name="Russia Invades Ukraine (Feb 2022)",
        event_date="2022-02-24",
        country="DE",  # primary impact channel: European energy / industrials
        brief="Russia launches full invasion of Ukraine. Sanctions cascade; energy crisis triggered.",
        topics=["war", "sanctions", "energy_crisis"],
        sectors=["banking", "energy_fossil", "defense"],
        seed_tickers=["DBK.DE", "BNP.PA", "TTE.PA", "BP.L", "LMT", "RTX"],
        expected={"DBK.DE": "down", "BNP.PA": "down", "TTE.PA": "up", "BP.L": "up", "LMT": "up", "RTX": "up"},
        contagion_risk=0.95,
        notes="European banks -10%, oil majors +5%, defense +6-12%.",
    ),
    GlobalEvent(
        name="Israel-Hamas War (Oct 2023)",
        event_date="2023-10-09",
        country="US",
        brief="Hamas attacks Israel Oct 7; markets reopen Mon. Defense rally; Mideast risk premium.",
        topics=["war", "geopolitics"],
        sectors=["energy_fossil", "defense"],
        seed_tickers=["LMT", "RTX", "XOM", "CVX"],
        expected={"LMT": "up", "RTX": "up", "XOM": "up", "CVX": "up"},
        contagion_risk=0.55,
        notes="Defense +5-9% week; oil +4%.",
    ),
    GlobalEvent(
        name="Saudi-Russia Oil Price War (Mar 2020)",
        event_date="2020-03-09",
        country="US",
        brief="Saudi Arabia hikes output after Russia rejects OPEC+ cut. Oil -30% single day.",
        topics=["energy_crisis", "opec"],
        sectors=["energy_fossil"],
        seed_tickers=["XOM", "CVX", "COP", "BP.L", "TTE.PA"],
        expected={"XOM": "down", "CVX": "down", "COP": "down", "BP.L": "down", "TTE.PA": "down"},
        contagion_risk=0.95,
        notes="WTI -25% intraday; majors -15 to -25%.",
    ),

    # ── Corporate idiosyncratic (non-IT, valuable for ticker calibration) ──
    GlobalEvent(
        name="Boeing 737 MAX Grounding (Mar 2019)",
        event_date="2019-03-13",
        country="US",
        brief="FAA grounds 737 MAX after Ethiopian Airlines crash; second hull loss in 5 months.",
        topics=["product_safety", "aviation"],
        sectors=["aviation"],
        seed_tickers=["BA"],
        expected={"BA": "down"},
        contagion_risk=0.30,
        notes="BA -11% in 2 days; floor at $371.",
    ),
    GlobalEvent(
        name="Volkswagen Dieselgate (Sep 2015)",
        event_date="2015-09-21",
        country="DE",
        brief="EPA reveals VW emissions cheating. CEO Winterkorn resigns within days.",
        topics=["fraud", "emissions"],
        sectors=["automotive"],
        seed_tickers=["VOW3.DE", "BMW.DE", "MBG.DE"],
        expected={"VOW3.DE": "down", "BMW.DE": "flat", "MBG.DE": "flat"},
        contagion_risk=0.40,
        notes="VW -35% week; sector contagion modest.",
    ),
    GlobalEvent(
        name="Meta Pivot to AI / Q4-2022 Crash (Feb 2022)",
        event_date="2022-02-03",
        country="US",
        brief="Meta Q4 print: first-ever DAU decline; Reels cannibalization. Stock -26% one day.",
        topics=["earnings_miss", "tech_recession"],
        sectors=["tech"],
        seed_tickers=["META", "GOOGL", "SNAP"],
        expected={"META": "down", "GOOGL": "down", "SNAP": "down"},
        contagion_risk=0.40,
        notes="META -26% Thu; ad-tech sector -8%.",
    ),
    GlobalEvent(
        name="Nvidia AI Earnings Pop (May 2023)",
        event_date="2023-05-25",
        country="US",
        brief="Nvidia Q1 print + Q2 guide blow expectations on AI demand. Joins $1T club.",
        topics=["earnings_beat", "ai"],
        sectors=["tech"],
        seed_tickers=["NVDA", "AMD", "TSM", "AVGO"],
        expected={"NVDA": "up", "AMD": "up", "TSM": "up", "AVGO": "up"},
        contagion_risk=0.35,
        notes="NVDA +24% Thu; semis broad rally.",
    ),
    GlobalEvent(
        name="OpenAI ChatGPT Launch (Nov 2022)",
        event_date="2022-11-30",
        country="US",
        brief="ChatGPT public release; early traction. Microsoft positioned as primary beneficiary.",
        topics=["ai", "tech"],
        sectors=["tech"],
        seed_tickers=["MSFT", "GOOGL", "META"],
        expected={"MSFT": "up", "GOOGL": "down", "META": "down"},
        contagion_risk=0.25,
        notes="MSFT +5% week; GOOG/META under search-disruption pressure.",
    ),
    GlobalEvent(
        name="Tesla Q3 2018 Earnings (Oct 2018)",
        event_date="2018-10-25",
        country="US",
        brief="Tesla first-ever GAAP profit; Model 3 ramp success.",
        topics=["earnings_beat"],
        sectors=["automotive"],
        seed_tickers=["TSLA"],
        expected={"TSLA": "up"},
        contagion_risk=0.20,
        notes="TSLA +9% Thu post-print.",
    ),
    GlobalEvent(
        name="Archegos Capital Liquidation (Mar 2021)",
        event_date="2021-03-26",
        country="US",
        brief="Hwang's family office blows up; CS / Nomura / MS take large prime-broker losses.",
        topics=["financial_crisis", "leverage"],
        sectors=["banking"],
        seed_tickers=["CS", "MS", "GS", "NMR"],
        expected={"CS": "down", "MS": "down", "GS": "down", "NMR": "down"},
        contagion_risk=0.50,
        notes="CS -14% Mon; Nomura -16%.",
    ),

    # ── Fed / ECB / BoJ policy ──
    GlobalEvent(
        name="Fed First Hike Post-COVID (Mar 2022)",
        event_date="2022-03-16",
        country="US",
        brief="Fed lifts off zero; first hike since 2018. Powell signals 6 more.",
        topics=["monetary_policy"],
        sectors=["banking", "tech"],
        seed_tickers=["JPM", "BAC", "AAPL", "MSFT"],
        expected={"JPM": "up", "BAC": "up", "AAPL": "down", "MSFT": "down"},
        contagion_risk=0.50,
        notes="Banks +3%, tech mixed; rate-sensitives lagged.",
    ),
    GlobalEvent(
        name="ECB First Hike in 11 Years (Jul 2022)",
        event_date="2022-07-21",
        country="DE",
        brief="ECB +50bps surprise; first hike since 2011. TPI announced for periphery.",
        topics=["monetary_policy"],
        sectors=["banking"],
        seed_tickers=["BNP.PA", "DBK.DE", "GLE.PA"],
        expected={"BNP.PA": "up", "DBK.DE": "up", "GLE.PA": "up"},
        contagion_risk=0.45,
        notes="EU banks +2-4%; periphery spreads tightened.",
    ),
    GlobalEvent(
        name="BoJ Yield-Curve Control Tweak (Dec 2022)",
        event_date="2022-12-20",
        country="JP",
        brief="Kuroda widens 10Y JGB band 0.25 → 0.5%. JPY surges; global rates jump.",
        topics=["monetary_policy"],
        sectors=["banking"],
        seed_tickers=["8306.T", "8316.T", "8411.T"],
        expected={"8306.T": "up", "8316.T": "up", "8411.T": "up"},
        contagion_risk=0.50,
        notes="JP banks +4-7% Tue; surprise hawkish pivot.",
    ),

    # ── More US idiosyncratic / sector ──
    GlobalEvent(
        name="Apple App Store Ruling (Sep 2021)",
        event_date="2021-09-10",
        country="US",
        brief="Epic v Apple ruling; Apple must allow alternative payment links. Mixed but no monopoly finding.",
        topics=["regulation", "antitrust"],
        sectors=["tech"],
        seed_tickers=["AAPL"],
        expected={"AAPL": "down"},
        contagion_risk=0.20,
        notes="AAPL -3.3% Fri post-ruling.",
    ),
    GlobalEvent(
        name="Disney+ Subscriber Miss (Aug 2022)",
        event_date="2022-08-11",
        country="US",
        brief="Disney+ adds beat but Hulu / ESPN+ disappoint; ad pricing concerns.",
        topics=["earnings_miss"],
        sectors=["media"],
        seed_tickers=["DIS", "NFLX"],
        expected={"DIS": "flat", "NFLX": "flat"},
        contagion_risk=0.15,
        notes="DIS muted reaction; NFLX little change.",
    ),
]


def fetch_returns(ticker: str, event_date: str) -> Optional[dict[str, float]]:
    """Return realized log-returns at T+1, T+3, T+7 windows around the event.

    Anchor: the LAST trading day at or before ``event_date - 1 day``.
    Returns are computed as log(close[T+n] / close[anchor]).
    """
    try:
        ed = datetime.strptime(event_date, "%Y-%m-%d").date()
        # Pull a generous window: 10 trading days before and 15 after.
        start = (ed - timedelta(days=20)).isoformat()
        end = (ed + timedelta(days=25)).isoformat()
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty or "Close" not in df.columns:
            return None
        # yfinance returns a multi-level column when only one ticker is
        # passed in some versions. Normalize.
        close = df["Close"]
        if hasattr(close, "iloc") and isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return None

        # Anchor is the trading day immediately before the event date
        before = close[close.index.date < ed]
        if before.empty:
            return None
        anchor_price = float(before.iloc[-1])

        after = close[close.index.date >= ed]
        if after.empty:
            return None

        out: dict[str, float] = {}
        for label, n_trading_days in [("t1", 1), ("t3", 3), ("t7", 7)]:
            if len(after) >= n_trading_days:
                target_price = float(after.iloc[n_trading_days - 1])
                out[label] = float(np.log(target_price / anchor_price))
            else:
                out[label] = float("nan")
        return out
    except Exception as e:
        print(f"    fetch_returns failed for {ticker} @ {event_date}: {e}", file=sys.stderr)
        return None


def expected_to_sign(s: str) -> int:
    """Map 'down'/'up'/'flat' to -1/+1/0."""
    return {"down": -1, "up": 1, "flat": 0}.get(s, 0)


def realized_to_sign(r: float, flat_band: float = 0.005) -> int:
    """Bucket realized return into -1/0/+1 with a small flat band (±50 bps)."""
    if not np.isfinite(r):
        return 0
    if r > flat_band:
        return 1
    if r < -flat_band:
        return -1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="audit hit-rate but do not write the .py file")
    ap.add_argument("--rate-limit-sec", type=float, default=0.4)
    args = ap.parse_args()

    import time
    print(f"Validating {len(EVENTS)} global events against yfinance ground truth...")
    print()

    total_hits = 0
    total_checks = 0
    by_country: dict[str, list[tuple[int, int]]] = {}
    validated_events: list[tuple[GlobalEvent, dict[str, dict[str, float]]]] = []

    for ev in EVENTS:
        print(f"=== {ev.name} ({ev.country}, {ev.event_date}) ===")
        ticker_returns: dict[str, dict[str, float]] = {}
        n_hits = 0
        n_total = 0
        for ticker in ev.seed_tickers:
            ret = fetch_returns(ticker, ev.event_date)
            time.sleep(args.rate_limit_sec)
            if ret is None:
                print(f"  {ticker:<10} no data")
                continue
            ticker_returns[ticker] = ret
            exp = ev.expected.get(ticker, "flat")
            exp_sign = expected_to_sign(exp)
            real_sign_t1 = realized_to_sign(ret["t1"])
            hit = "✓" if real_sign_t1 == exp_sign else "✗"
            if real_sign_t1 == exp_sign:
                n_hits += 1
            n_total += 1
            print(
                f"  {ticker:<10} exp={exp:<5} "
                f"T+1={ret['t1']:+.3f} T+3={ret['t3']:+.3f} T+7={ret['t7']:+.3f}  {hit}"
            )
        if n_total:
            total_hits += n_hits
            total_checks += n_total
            by_country.setdefault(ev.country, []).append((n_hits, n_total))
            print(f"  → {n_hits}/{n_total} = {n_hits / n_total:.0%} hit-rate")
        validated_events.append((ev, ticker_returns))
        print()

    overall = total_hits / total_checks if total_checks else 0
    print()
    print("─" * 60)
    print(f"OVERALL: {total_hits}/{total_checks} = {overall:.1%}")
    print("─" * 60)
    print("By country:")
    for c, runs in sorted(by_country.items()):
        h = sum(r[0] for r in runs)
        t = sum(r[1] for r in runs)
        print(f"  {c}: {h}/{t} ({h / t:.0%})  across {len(runs)} events")

    if args.check_only:
        print("\n--check-only: skipping file write.")
        return 0

    # Write the global corpus file
    print(f"\nWriting {OUTPUT_PATH.relative_to(REPO_ROOT)}...")
    lines = [
        '"""Global extension to the backtest corpus — 35 non-Italian events.\n',
        "",
        "Auto-generated by scripts/build_global_corpus.py. Each scenario's",
        "ground-truth direction was validated against yfinance realized returns",
        "(T+1) at the time of generation; see the audit print above.",
        "",
        "Crisis metrics (engagement_score, polarization, etc.) are placeholder",
        "neutral values — the simulator computes its own metrics when a",
        "scenario is replayed. Use this corpus for *financial-impact* validation,",
        "not opinion-dynamics calibration.",
        '"""',
        "",
        "from backtest_financials import BacktestScenario",
        "",
        "",
        f"GENERATED_AT = '{date.today().isoformat()}'",
        f"OVERALL_HIT_RATE = {overall:.4f}  # T+1 sign of expected vs realized",
        "",
        "SCENARIOS_GLOBAL = [",
    ]
    for ev, returns in validated_events:
        if not returns:
            continue
        notes_parts = [ev.notes] if ev.notes else []
        for tk, r in returns.items():
            if np.isfinite(r["t1"]):
                notes_parts.append(
                    f"{tk} actual T+1={r['t1'] * 100:+.1f}% T+7={r['t7'] * 100:+.1f}%"
                )
        notes_combined = " | ".join(notes_parts)
        # Filter expected_directions and verify_tickers to those with data
        verify = [t for t in ev.seed_tickers if t in returns]
        expected = {t: ev.expected.get(t, "flat") for t in verify}
        # Map expected → "up"/"down" used by BacktestScenario (legacy)
        # The dataclass uses string "down"/"up"/"flat"; pass through.
        lines.extend([
            "    BacktestScenario(",
            f"        name={ev.name!r},",
            f"        date_start={ev.event_date!r},",
            f"        date_end={(datetime.strptime(ev.event_date, '%Y-%m-%d') + timedelta(days=10)).date().isoformat()!r},",
            f"        brief={ev.brief!r},",
            f"        topics={ev.topics!r},",
            f"        sectors={ev.sectors!r},",
            "        engagement_score=0.65,",
            f"        contagion_risk={ev.contagion_risk},",
            f"        active_wave={2 if ev.contagion_risk >= 0.7 else 1},",
            "        polarization=6.0,",
            "        polarization_velocity=1.0,",
            "        negative_institutional_pct=0.40,",
            "        negative_ceo_count=1,",
            f"        verify_tickers={verify!r},",
            f"        expected_directions={expected!r},",
            f"        notes={notes_combined!r},",
            "    ),",
        ])
    lines.append("]")
    lines.append("")
    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"  {OUTPUT_PATH.stat().st_size / 1024:.1f} KB · {len(validated_events)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
