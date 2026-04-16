"use client";

import { useMemo, useState } from "react";
import { BacktestScenario } from "@/lib/backtest-types";

type SortKey = "scenario" | "cat" | "wave" | "warn" | "dir" | "mae" | "btp";

export function ScenarioTable({ data }: { data: BacktestScenario[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("mae");
  const [sortAsc, setSortAsc] = useState(true);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [filter, setFilter] = useState("");

  const filtered = useMemo(() => {
    let result = [...data];
    if (filter) {
      const f = filter.toLowerCase();
      result = result.filter(
        (s) => s.scenario.toLowerCase().includes(f) || s.category.toLowerCase().includes(f),
      );
    }
    result.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "scenario": cmp = a.scenario.localeCompare(b.scenario); break;
        case "cat": cmp = a.category.localeCompare(b.category); break;
        case "wave": cmp = a.wave - b.wave; break;
        case "warn": {
          const o = { LOW: 0, MODERATE: 1, HIGH: 2, CRITICAL: 3 };
          cmp = (o[a.warning] || 0) - (o[b.warning] || 0); break;
        }
        case "dir": {
          const [ac, at] = a.direction_accuracy.split("/").map(Number);
          const [bc, bt] = b.direction_accuracy.split("/").map(Number);
          cmp = (at > 0 ? ac / at : 0) - (bt > 0 ? bc / bt : 0); break;
        }
        case "mae": cmp = (a.mae_t1 || 99) - (b.mae_t1 || 99); break;
        case "btp": cmp = a.btp_spread - b.btp_spread; break;
      }
      return sortAsc ? cmp : -cmp;
    });
    return result;
  }, [data, sortKey, sortAsc, filter]);

  const sort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((p) => !p);
    else { setSortKey(key); setSortAsc(key === "mae"); }
  };

  const SH = ({ label, field, className = "" }: { label: string; field: SortKey; className?: string }) => (
    <button
      onClick={() => sort(field)}
      className={`text-[8px] font-data uppercase tracking-wider transition-colors ${
        sortKey === field ? "text-ki-on-surface" : "text-ki-on-surface-muted hover:text-ki-on-surface-muted"
      } ${className}`}
    >
      {label}{sortKey === field && (sortAsc ? "↑" : "↓")}
    </button>
  );

  const WARN_COLORS: Record<string, string> = {
    LOW: "#00d26a", MODERATE: "#ffaa00", HIGH: "#ff7700", CRITICAL: "#ff3b3b",
  };

  return (
    <div className="border-t border-ki-border">
      {/* Header */}
      <div className="h-6 flex items-center px-2 justify-between border-b border-ki-border-strong bg-ki-surface-sunken">
        <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
          Scenario Detail — {filtered.length}/{data.length}
        </span>
        <input
          type="text"
          placeholder="FILTER"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-transparent border border-ki-border px-2 py-0 h-4 text-[9px] font-data text-ki-on-surface-muted placeholder:text-ki-on-surface-muted w-32 focus:outline-none focus:border-ki-border-strong"
        />
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-[1fr_110px_28px_60px_44px_44px_40px] gap-0 px-2 h-5 items-center border-b border-ki-border-strong">
        <SH label="Scenario " field="scenario" />
        <SH label="Cat " field="cat" />
        <SH label="W " field="wave" />
        <SH label="Warn " field="warn" />
        <SH label="Dir " field="dir" className="text-right" />
        <SH label="MAE " field="mae" className="text-right" />
        <SH label="BTP " field="btp" className="text-right" />
      </div>

      {/* Rows */}
      <div className="max-h-[480px] overflow-y-auto">
        {filtered.map((s, i) => {
          const [dc, dt] = s.direction_accuracy.split("/").map(Number);
          const dirPct = dt > 0 ? (dc / dt) * 100 : 0;
          const dirColor = dirPct >= 70 ? "#00d26a" : dirPct >= 50 ? "#ffaa00" : "#ff3b3b";
          const isExp = expandedIdx === i;

          return (
            <div key={i}>
              <button
                onClick={() => setExpandedIdx(isExp ? null : i)}
                className="w-full grid grid-cols-[1fr_110px_28px_60px_44px_44px_40px] gap-0 px-2 h-6 items-center border-b border-ki-border-strong hover:bg-ki-surface-hover transition-colors text-left"
              >
                <span className="font-data text-[10px] text-ki-on-surface-muted truncate pr-1">
                  <span className="text-ki-on-surface-muted mr-1">{isExp ? "▾" : "▸"}</span>
                  {s.scenario}
                </span>
                <span className="font-data text-[9px] text-ki-on-surface-muted truncate">{s.category}</span>
                <span className="font-data text-[10px] text-center" style={{
                  color: s.wave === 3 ? "#ff3b3b" : s.wave === 2 ? "#ffaa00" : "#00d26a"
                }}>{s.wave}</span>
                <span className="font-data text-[9px] text-center" style={{ color: WARN_COLORS[s.warning] }}>
                  {s.warning}
                </span>
                <span className="font-data text-[10px] text-right" style={{ color: dirColor }}>
                  {dc}/{dt}
                </span>
                <span className="font-data text-[10px] text-right" style={{
                  color: (s.mae_t1 || 0) > 5 ? "#ff3b3b" : (s.mae_t1 || 0) > 2 ? "#ffaa00" : "#00d26a"
                }}>
                  {s.mae_t1 != null ? s.mae_t1.toFixed(1) : "—"}
                </span>
                <span className="font-data text-[9px] text-ki-on-surface-muted text-right">
                  {s.btp_spread > 0 ? `+${s.btp_spread}` : "0"}
                </span>
              </button>

              {/* Expanded ticker detail */}
              {isExp && (
                <div className="bg-ki-surface-sunken border-b border-ki-border">
                  <div className="grid grid-cols-[70px_70px_70px_50px] gap-0 px-4 h-4 items-center text-[8px] font-data text-ki-on-surface-muted uppercase">
                    <span>Ticker</span>
                    <span className="text-right">Pred T+1</span>
                    <span className="text-right">Act T+1</span>
                    <span className="text-right">Err</span>
                  </div>
                  {Object.entries(s.tickers).map(([ticker, tr]) => {
                    const err = Math.abs(tr.actual_t1 - tr.predicted_t1);
                    const dirOk = (tr.predicted_t1 < 0 && tr.actual_t1 < -0.5) || (tr.predicted_t1 >= 0 && tr.actual_t1 > -0.5);
                    return (
                      <div
                        key={ticker}
                        className="grid grid-cols-[70px_70px_70px_50px] gap-0 px-4 h-5 items-center border-t border-ki-border-strong"
                      >
                        <span className="font-data text-[10px] text-ki-on-surface-muted">
                          <span className={`inline-block w-1 h-1 rounded-full mr-1 ${dirOk ? "bg-[#00d26a]" : "bg-[#ff3b3b]"}`} />
                          {ticker.replace(".MI", "")}
                        </span>
                        <span className="font-data text-[10px] text-right" style={{ color: tr.predicted_t1 < 0 ? "#ff3b3b" : "#00d26a" }}>
                          {tr.predicted_t1 > 0 ? "+" : ""}{tr.predicted_t1.toFixed(2)}%
                        </span>
                        <span className="font-data text-[10px] text-right" style={{ color: tr.actual_t1 < 0 ? "#ff3b3b" : "#00d26a" }}>
                          {tr.actual_t1 > 0 ? "+" : ""}{tr.actual_t1.toFixed(2)}%
                        </span>
                        <span className="font-data text-[10px] text-right" style={{ color: err > 5 ? "#ff3b3b" : err > 2 ? "#ffaa00" : "#8a8a8a" }}>
                          {err.toFixed(1)}
                        </span>
                      </div>
                    );
                  })}
                  <div className="flex gap-3 px-4 h-5 items-center text-[9px] font-data text-ki-on-surface-muted border-t border-ki-border-strong">
                    <span>FTSE: <span style={{ color: s.ftse_impact < 0 ? "#ff3b3b" : "#00d26a" }}>{s.ftse_impact > 0 ? "+" : ""}{s.ftse_impact.toFixed(2)}%</span></span>
                    <span>BTP: <span className="text-ki-on-surface-muted">+{s.btp_spread}bps</span></span>
                    <span>SCOPE: <span className={s.crisis_scope === "macro_systematic" ? "text-[#00d26a]" : "text-[#ffaa00]"}>
                      {s.crisis_scope === "macro_systematic" ? "MACRO" : "IDIO"}
                    </span></span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
