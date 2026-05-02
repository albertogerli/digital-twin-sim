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
      className={`eyebrow font-medium transition-colors ${
        sortKey === field ? "text-ki-on-surface" : "text-ki-on-surface-muted hover:text-ki-on-surface-secondary"
      } ${className}`}
    >
      {label}{sortKey === field && (sortAsc ? " ↑" : " ↓")}
    </button>
  );

  const warnTone = (w: string) =>
    w === "CRITICAL" ? "var(--neg)" :
    w === "HIGH"     ? "var(--neg)" :
    w === "MODERATE" ? "var(--warn)" :
    "var(--pos)";

  return (
    <div className="border-t border-ki-border">
      {/* Header */}
      <div className="h-8 flex items-center px-3 justify-between border-b border-ki-border bg-ki-surface-sunken">
        <span className="eyebrow">
          Scenario detail · {filtered.length}/{data.length}
        </span>
        <input
          type="text"
          placeholder="Filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-ki-surface-raised border border-ki-border rounded-sm px-2 h-6 text-[11px] text-ki-on-surface placeholder:text-ki-on-surface-muted w-40 focus:outline-none focus:border-ki-primary"
        />
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-[1fr_110px_28px_72px_52px_52px_48px] gap-0 px-3 h-7 items-center border-b border-ki-border bg-ki-surface-raised">
        <SH label="Scenario" field="scenario" />
        <SH label="Cat" field="cat" />
        <SH label="W" field="wave" />
        <SH label="Warn" field="warn" />
        <SH label="Dir" field="dir" className="text-right" />
        <SH label="MAE" field="mae" className="text-right" />
        <SH label="BTP" field="btp" className="text-right" />
      </div>

      {/* Rows */}
      <div className="max-h-[480px] overflow-y-auto bg-ki-surface-raised">
        {filtered.map((s, i) => {
          const [dc, dt] = s.direction_accuracy.split("/").map(Number);
          const dirPct = dt > 0 ? (dc / dt) * 100 : 0;
          const dirColor = dirPct >= 70 ? "var(--pos)" : dirPct >= 50 ? "var(--warn)" : "var(--neg)";
          const isExp = expandedIdx === i;

          return (
            <div key={i}>
              <button
                onClick={() => setExpandedIdx(isExp ? null : i)}
                className="w-full grid grid-cols-[1fr_110px_28px_72px_52px_52px_48px] gap-0 px-3 h-7 items-center border-b border-ki-border-faint hover:bg-ki-surface-hover transition-colors text-left"
              >
                <span className="text-[12px] text-ki-on-surface truncate pr-2 inline-flex items-center gap-1.5">
                  <span className="text-ki-on-surface-muted text-[10px]">{isExp ? "▾" : "▸"}</span>
                  {s.scenario}
                </span>
                <span className="font-data text-[11px] text-ki-on-surface-secondary truncate">{s.category}</span>
                <span className="font-data tabular text-[11px] text-center" style={{
                  color: s.wave === 3 ? "var(--neg)" : s.wave === 2 ? "var(--warn)" : "var(--pos)"
                }}>{s.wave}</span>
                <span className="font-data tabular text-[10px] text-center uppercase tracking-[0.04em]" style={{ color: warnTone(s.warning) }}>
                  {s.warning}
                </span>
                <span className="font-data tabular text-[11px] text-right" style={{ color: dirColor }}>
                  {dc}/{dt}
                </span>
                <span className="font-data tabular text-[11px] text-right" style={{
                  color: (s.mae_t1 || 0) > 5 ? "var(--neg)" : (s.mae_t1 || 0) > 2 ? "var(--warn)" : "var(--pos)"
                }}>
                  {s.mae_t1 != null ? s.mae_t1.toFixed(1) : "—"}
                </span>
                <span className="font-data tabular text-[11px] text-ki-on-surface-secondary text-right">
                  {s.btp_spread > 0 ? `+${s.btp_spread}` : "0"}
                </span>
              </button>

              {/* Expanded ticker detail */}
              {isExp && (
                <div className="bg-ki-surface-sunken border-b border-ki-border">
                  <div className="grid grid-cols-[80px_80px_80px_56px] gap-0 px-5 h-6 items-center eyebrow">
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
                        className="grid grid-cols-[80px_80px_80px_56px] gap-0 px-5 h-6 items-center border-t border-ki-border-faint"
                      >
                        <span className="font-data text-[11px] text-ki-on-surface inline-flex items-center gap-1.5">
                          <span className="inline-block w-1 h-1 rounded-full" style={{ background: dirOk ? "var(--pos)" : "var(--neg)" }} />
                          {ticker.replace(".MI", "")}
                        </span>
                        <span className="font-data tabular text-[11px] text-right" style={{ color: tr.predicted_t1 < 0 ? "var(--neg)" : "var(--pos)" }}>
                          {tr.predicted_t1 > 0 ? "+" : ""}{tr.predicted_t1.toFixed(2)}%
                        </span>
                        <span className="font-data tabular text-[11px] text-right" style={{ color: tr.actual_t1 < 0 ? "var(--neg)" : "var(--pos)" }}>
                          {tr.actual_t1 > 0 ? "+" : ""}{tr.actual_t1.toFixed(2)}%
                        </span>
                        <span className="font-data tabular text-[11px] text-right" style={{ color: err > 5 ? "var(--neg)" : err > 2 ? "var(--warn)" : "var(--ink-3)" }}>
                          {err.toFixed(1)}
                        </span>
                      </div>
                    );
                  })}
                  <div className="flex gap-4 px-5 h-6 items-center font-data tabular text-[11px] text-ki-on-surface-secondary border-t border-ki-border-faint">
                    <span>FTSE <span style={{ color: s.ftse_impact < 0 ? "var(--neg)" : "var(--pos)" }}>{s.ftse_impact > 0 ? "+" : ""}{s.ftse_impact.toFixed(2)}%</span></span>
                    <span>BTP <span className="text-ki-on-surface">+{s.btp_spread}bps</span></span>
                    <span>SCOPE <span style={{ color: s.crisis_scope === "macro_systematic" ? "var(--pos)" : "var(--warn)" }}>
                      {s.crisis_scope === "macro_systematic" ? "macro" : "idio"}
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
