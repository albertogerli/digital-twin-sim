"use client";

import { useEffect, useState } from "react";
import {
  BacktestScenario,
  AggregateStats,
  computeAggregates,
  computeCategoryStats,
  computeTickerStats,
} from "@/lib/backtest-types";
import { HeroStats } from "./HeroStats";
import { TemporalAccuracyChart } from "./TemporalAccuracyChart";
import { CategoryBreakdown } from "./CategoryBreakdown";
import { PredictedVsActualScatter } from "./PredictedVsActualScatter";
import { TickerLeaderboard } from "./TickerLeaderboard";
import { ScenarioTable } from "./ScenarioTable";
import { WaveAnalysis } from "./WaveAnalysis";

export default function BacktestDashboard() {
  const [data, setData] = useState<BacktestScenario[]>([]);
  const [loading, setLoading] = useState(true);
  const [agg, setAgg] = useState<AggregateStats | null>(null);

  useEffect(() => {
    fetch("/data/backtest_results.json")
      .then((r) => r.json())
      .then((d: BacktestScenario[]) => {
        setData(d);
        setAgg(computeAggregates(d));
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-ki-surface flex items-center justify-center">
        <span className="font-data text-sm text-ki-on-surface-muted">
          LOADING BACKTEST DATA<span className="cursor-blink">_</span>
        </span>
      </div>
    );
  }

  const categoryStats = computeCategoryStats(data);
  const tickerStats = computeTickerStats(data);

  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface">
      {/* Top bar */}
      <header className="border-b border-ki-border h-8 flex items-center px-3 justify-between">
        <div className="flex items-center gap-2 font-data text-[10px]">
          <span className="text-ki-on-surface-muted">DIGITALTWIN</span>
          <span className="text-ki-on-surface-muted">/</span>
          <span className="text-ki-on-surface-muted">BACKTEST</span>
          <span className="text-ki-on-surface-muted">|</span>
          <span className="text-[#00d26a]">{data.length} SCENARIOS</span>
          <span className="text-ki-on-surface-muted">|</span>
          <span className="text-ki-on-surface-muted">{agg?.totalTickers} TICKERS</span>
        </div>
        <div className="flex items-center gap-2 font-data text-[10px]">
          <span className="text-ki-on-surface-muted">v3.0 SECTOR BETAS + PANIC MULT</span>
          <span className="w-1.5 h-1.5 rounded-full bg-[#00d26a]" />
        </div>
      </header>

      {/* KPI Strip */}
      {agg && <HeroStats agg={agg} />}

      {/* Main grid — dense, no gaps */}
      <div className="grid grid-cols-1 lg:grid-cols-12 border-t border-ki-border">
        {/* Left panel: Category + Wave */}
        <div className="lg:col-span-4 border-r border-ki-border">
          <CategoryBreakdown stats={categoryStats} />
          <WaveAnalysis data={data} />
        </div>

        {/* Right panel: Charts */}
        <div className="lg:col-span-8">
          <TemporalAccuracyChart data={data} />
          <PredictedVsActualScatter data={data} />
        </div>
      </div>

      {/* Ticker leaderboard */}
      <TickerLeaderboard stats={tickerStats} />

      {/* Full scenario table */}
      <ScenarioTable data={data} />

      {/* Footer */}
      <div className="border-t border-ki-border h-6 flex items-center px-3">
        <span className="font-data text-[9px] text-ki-on-surface-muted">
          FINANCIAL IMPACT SCORER v3 — CALIBRATED 2011-2025 — 16 SECTOR BETAS — 72 ORG TICKER MAP — T+1/T+3/T+7 TEMPORAL CURVE
        </span>
      </div>
    </div>
  );
}
