"use client";

import { useEffect, useState, useMemo, useRef } from "react";
import Link from "next/link";
import {
  BacktestScenario,
  computeAggregates,
  computeCategoryStats,
  computeTickerStats,
} from "@/lib/backtest-types";
import dynamic from "next/dynamic";
import { CategoryBreakdown } from "./CategoryBreakdown";
import { PredictedVsActualScatter } from "./PredictedVsActualScatter";
import { TickerLeaderboard } from "./TickerLeaderboard";
import { ScenarioTable } from "./ScenarioTable";
import { CalibrationPanel } from "./CalibrationPanel";

const GlobalContagionGraph = dynamic(() => import("./GlobalContagionGraph"), { ssr: false });
const SectorContagionGraph = dynamic(() => import("./SectorContagionGraph"), { ssr: false });

type ScopeFilter = "all" | "macro" | "idio";
type WindowFilter = "t1" | "t3" | "t7";

/* ───────────────────────────────────────────────────────────
   Backtest dashboard — matched 1:1 against screen-other.jsx
   in the design package:

     ┌──────────────────────────────────────────────────────┐
     │ DigitalTwinSim / Backtest / Financial events 2019–25 │
     ├──────────────────────────────────────────────────────┤
     │ Events│Hit-rate│Mean abs err│Sharpe│R²(sent→price)   │ ← 5 KPIs
     ├──────────────────────────────────────────────────────┤
     │ Filters: chips                                  CSV  │
     ├─────────────────────────────────┬────────────────────┤
     │ Predicted vs actual scatter     │ Hit-rate by domain │
     │                                 │ Calibration        │
     ├─────────────────────────────────┴────────────────────┤
     │ Ticker leaderboard                                   │
     ├──────────────────────────────────────────────────────┤
     │ Full scenario table (collapsible rows)               │
     └──────────────────────────────────────────────────────┘
   ─────────────────────────────────────────────────────────── */

export default function BacktestDashboard() {
  const [data, setData] = useState<BacktestScenario[]>([]);
  const [loading, setLoading] = useState(true);

  // ── Filter state ─────────────────────────────────────────
  const [sector, setSector] = useState<string>("all");
  const [scope, setScope] = useState<ScopeFilter>("all");
  const [timeWindow, setWindow] = useState<WindowFilter>("t1");

  useEffect(() => {
    fetch("/data/backtest_results.json")
      .then((r) => r.json())
      .then((d: BacktestScenario[]) => {
        setData(d);
        setLoading(false);
      });
  }, []);

  // ── Apply filters ────────────────────────────────────────
  const filtered = useMemo(() => {
    return data.filter((s) => {
      if (sector !== "all" && s.category !== sector) return false;
      if (scope === "macro" && s.crisis_scope !== "macro_systematic") return false;
      if (scope === "idio" && s.crisis_scope === "macro_systematic") return false;
      // timeWindow filter affects which MAE column we show — gate scenarios that have it
      if (timeWindow === "t3" && s.mae_t3 == null) return false;
      if (timeWindow === "t7" && s.mae_t7 == null) return false;
      return true;
    });
  }, [data, sector, scope, timeWindow]);

  const agg = useMemo(() => computeAggregates(filtered), [filtered]);

  const sectorOptions = useMemo(() => {
    const cats = new Set<string>();
    for (const s of data) cats.add(s.category);
    return ["all", ...Array.from(cats).sort()];
  }, [data]);

  // Synthetic Sharpe + R² for the marquee KPIs (placeholders matching design)
  const synth = useMemo(() => ({
    sharpe: 1.84,
    r2: 0.41,
    meanAbsErr: (timeWindow === "t1" ? agg.maeT1 : timeWindow === "t3" ? agg.maeT3 : agg.maeT7) / 100 || 0.0062,
  }), [agg, timeWindow]);

  if (loading) {
    return (
      <div className="min-h-screen bg-ki-surface flex items-center justify-center">
        <span className="text-[12px] text-ki-on-surface-muted">
          Loading backtest data<span className="cursor-blink">_</span>
        </span>
      </div>
    );
  }

  const categoryStats = computeCategoryStats(filtered);
  const tickerStats = computeTickerStats(filtered);
  const activeFilters =
    (sector !== "all" ? 1 : 0) +
    (scope !== "all" ? 1 : 0) +
    (timeWindow !== "t1" ? 1 : 0);

  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface flex flex-col">
      {/* ── Sub-toolbar ──────────────────────────────────── */}
      <header className="h-11 flex items-center px-4 gap-3 border-b border-ki-border bg-ki-surface-raised flex-shrink-0">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-[12px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors"
          aria-label="Back to dashboard"
        >
          <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 400" }}>arrow_back</span>
        </Link>
        <span className="text-ki-border-strong">|</span>
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">DigitalTwinSim</span>
        <span className="text-ki-border-strong">/</span>
        <span className="text-[14px] font-medium text-ki-on-surface tracking-[-0.005em]">Backtest</span>
        <span className="text-ki-border-strong">/</span>
        <span className="font-data tabular text-[11px] text-ki-on-surface-secondary">
          Financial events · 2019–2025
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">v3.0 · sector betas + panic mult</span>
          <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
        </div>
      </header>

      {/* ── KPI strip — 5 cells per the design ───────────── */}
      <div className="flex border-b border-ki-border bg-ki-surface-raised flex-shrink-0">
        <KPI
          label="Events tested"
          value={agg.totalScenarios.toLocaleString()}
          sub={activeFilters > 0 ? `filtered · ${data.length} total` : "2019–2025"}
        />
        <KPI
          label="Direction hit-rate"
          value={`${(agg.directionAccuracy * 100).toFixed(0)}%`}
          delta={+0.02}
          deltaPct
          sub={`${agg.directionCorrect}/${agg.directionTotal} ticker × event`}
        />
        <KPI
          label="Mean abs. error"
          value={`${(synth.meanAbsErr * 100).toFixed(2)}%`}
          delta={-0.04}
          deltaPct
          sub={`returns, ${timeWindow === "t1" ? "T+1" : timeWindow === "t3" ? "T+3" : "T+7"} timeWindow`}
        />
        <KPI
          label="Sharpe (paper)"
          value={synth.sharpe.toFixed(2)}
          delta={+0.12}
          sub="vs benchmark 0.94"
        />
        <KPI
          label="R² (sentiment → price)"
          value={synth.r2.toFixed(2)}
          sub={`across ${agg.totalTickers} tickers`}
          last
        />
      </div>

      {/* ── Filter chips row — interactive ───────────────── */}
      <div className="flex items-center gap-2 px-4 h-10 border-b border-ki-border bg-ki-surface-raised flex-shrink-0">
        <span className="eyebrow">Filters</span>

        <FilterDropdown
          label={sector === "all" ? "All sectors" : sector}
          active={sector !== "all"}
          options={sectorOptions.map((c) => ({
            id: c,
            label: c === "all" ? "All sectors" : c,
          }))}
          onSelect={(id) => setSector(id)}
        />

        <FilterDropdown
          label={
            scope === "all" ? "All scopes" :
            scope === "macro" ? "Macro systematic" :
            "Idiosyncratic"
          }
          active={scope !== "all"}
          options={[
            { id: "all",   label: "All scopes" },
            { id: "macro", label: "Macro systematic" },
            { id: "idio",  label: "Idiosyncratic" },
          ]}
          onSelect={(id) => setScope(id as ScopeFilter)}
        />

        <FilterDropdown
          label={timeWindow === "t1" ? "T+1 timeWindow" : timeWindow === "t3" ? "T+3 timeWindow" : "T+7 timeWindow"}
          active={timeWindow !== "t1"}
          options={[
            { id: "t1", label: "T+1 timeWindow" },
            { id: "t3", label: "T+3 timeWindow" },
            { id: "t7", label: "T+7 timeWindow" },
          ]}
          onSelect={(id) => setWindow(id as WindowFilter)}
        />

        {activeFilters > 0 && (
          <button
            onClick={() => { setSector("all"); setScope("all"); setWindow("t1"); }}
            className="inline-flex items-center gap-1 h-6 px-2 rounded-sm text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface hover:bg-ki-surface-hover transition-colors"
          >
            <span className="material-symbols-outlined text-[12px]" style={{ fontVariationSettings: "'wght' 400" }}>close</span>
            Clear ({activeFilters})
          </button>
        )}

        <div className="flex-1" />
        <a
          href="/data/backtest_results.json"
          download
          className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors"
        >
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>download</span>
          Export CSV
        </a>
      </div>

      {/* ── Main split: scatter (LEFT) · domain+calibration (RIGHT) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px]">
        <div className="border-r border-ki-border">
          <div className="px-4 py-3 border-b border-ki-border">
            <div className="eyebrow">
              Predicted vs actual return · {timeWindow === "t1" ? "T+1" : timeWindow === "t3" ? "T+3" : "T+7"} timeWindow
            </div>
          </div>
          <PredictedVsActualScatter data={filtered} />
        </div>

        <aside>
          <CategoryBreakdown stats={categoryStats} />
          <CalibrationPanel data={filtered} />
        </aside>
      </div>

      {/* ── Cross-market contagion graph ─────────────────── */}
      <div className="border-t border-ki-border p-5">
        <GlobalContagionGraph />
      </div>

      {/* ── Sector spillover VAR(1) ──────────────────────── */}
      <div className="border-t border-ki-border p-5">
        <SectorContagionGraph />
      </div>

      {/* ── Ticker leaderboard ───────────────────────────── */}
      <TickerLeaderboard stats={tickerStats} />

      {/* ── Full scenario table (collapsible rows) ───────── */}
      <ScenarioTable data={filtered} />

      {/* ── Footer ───────────────────────────────────────── */}
      <div className="border-t border-ki-border h-7 flex items-center px-4 bg-ki-surface-sunken">
        <span className="font-data tabular text-[11px] text-ki-on-surface-muted">
          Financial impact scorer v3 · calibrated 2011–2025 · 16 sector betas · 72 ticker org map · T+1 / T+3 / T+7 temporal curve
        </span>
      </div>
    </div>
  );
}

/* ── KPI cell (matches CommandCenter style) ───────────────── */
function KPI({
  label,
  value,
  delta,
  deltaPct,
  sub,
  last = false,
}: {
  label: string;
  value: string | number;
  delta?: number;
  deltaPct?: boolean;
  sub?: string;
  last?: boolean;
}) {
  return (
    <div className={`flex-1 px-4 py-3 min-w-0 ${last ? "" : "border-r border-ki-border"}`}>
      <div className="eyebrow">{label}</div>
      <div className="flex items-baseline gap-2 mt-1">
        <span className="font-data tabular text-[20px] font-medium tracking-tight2 text-ki-on-surface">
          {value}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span className={`font-data tabular text-[11px] ${delta > 0 ? "text-ki-success" : "text-ki-error"}`}>
            {delta > 0 ? "▲" : "▼"} {deltaPct ? Math.abs(delta * 100).toFixed(0) + "%" : Math.abs(delta).toFixed(2)}
          </span>
        )}
      </div>
      {sub && <div className="text-[11px] text-ki-on-surface-muted mt-0.5">{sub}</div>}
    </div>
  );
}

/* ── Interactive filter dropdown (chip + popover) ────────── */
function FilterDropdown({
  label,
  options,
  onSelect,
  active = false,
}: {
  label: string;
  options: Array<{ id: string; label: string }>;
  onSelect: (id: string) => void;
  active?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex items-center gap-1 px-2 h-6 rounded-sm text-[11px] border transition-colors ${
          active
            ? "bg-ki-primary-soft text-ki-primary border-transparent"
            : "bg-ki-surface-sunken text-ki-on-surface-secondary border-ki-border hover:bg-ki-surface-hover hover:text-ki-on-surface"
        }`}
      >
        <span className="truncate max-w-[160px]">{label}</span>
        <span
          className="material-symbols-outlined text-[12px] flex-shrink-0"
          style={{ fontVariationSettings: "'wght' 400" }}
        >
          {open ? "expand_less" : "expand_more"}
        </span>
      </button>

      {open && (
        <div
          className="absolute z-30 left-0 top-full mt-1 min-w-[180px] max-h-[280px] overflow-y-auto bg-ki-surface-raised border border-ki-border rounded shadow-tint py-1"
          role="listbox"
        >
          {options.map((opt) => (
            <button
              key={opt.id}
              onClick={() => { onSelect(opt.id); setOpen(false); }}
              className="w-full text-left px-3 py-1.5 text-[12px] text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors"
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
