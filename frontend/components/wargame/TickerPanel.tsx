"use client";

import { useState, useMemo } from "react";
import { type WgTicker, type WgRoundState } from "@/lib/wargame-types";

function SectorHeader({ sector, bg }: { sector: string; bg: string }) {
  return (
    <div
      className="h-4 flex items-center px-2 border-b border-ki-border-strong sticky top-0 z-[5]"
      style={{ backgroundColor: bg }}
    >
      <span className="font-data text-[8px] text-ki-on-surface-muted uppercase tracking-wider">{sector}</span>
    </div>
  );
}

function groupBySector(tickers: WgTicker[]): Map<string, WgTicker[]> {
  const map = new Map<string, WgTicker[]>();
  tickers.forEach((t) => {
    const list = map.get(t.sector);
    if (list) list.push(t);
    else map.set(t.sector, [t]);
  });
  return map;
}

function TickerRow({ t }: { t: WgTicker }) {
  const flashClass = t.flash === "red" ? "flash-red" : t.flash === "green" ? "flash-green" : "";

  return (
    <div className={`grid grid-cols-[70px_50px_50px_50px] gap-0 px-2 h-7 items-center border-b border-ki-border-strong hover:bg-ki-surface-sunken transition-colors ${flashClass}`}>
      <div>
        <div className="font-data text-[11px] text-ki-on-surface font-semibold">
          {t.ticker.replace(".MI", "")}
        </div>
        <div className="font-data text-[8px] text-ki-on-surface-muted">{t.sector}</div>
      </div>
      <div className="text-right">
        <div className="font-data text-[11px] font-semibold" style={{ color: t.t1 < 0 ? "#ff3b3b" : "#00d26a" }}>
          {t.t1 > 0 ? "+" : ""}{t.t1.toFixed(1)}%
        </div>
        <div className="font-data text-[7px] text-ki-on-surface-muted">T+1</div>
      </div>
      <div className="text-right">
        <div className="font-data text-[10px]" style={{ color: t.t3 < 0 ? "#ff3b3b" : "#00d26a", opacity: 0.7 }}>
          {t.t3 > 0 ? "+" : ""}{t.t3.toFixed(1)}%
        </div>
        <div className="font-data text-[7px] text-ki-on-surface-muted">T+3</div>
      </div>
      <div className="text-right">
        <div className="font-data text-[10px]" style={{ color: t.t7 < 0 ? "#ff3b3b" : "#00d26a", opacity: 0.5 }}>
          {t.t7 > 0 ? "+" : ""}{t.t7.toFixed(1)}%
        </div>
        <div className="font-data text-[7px] text-ki-on-surface-muted">T+7</div>
      </div>
    </div>
  );
}

export function TickerPanel({ tickers, state }: { tickers: WgTicker[]; state: WgRoundState }) {
  const [search, setSearch] = useState("");
  const [outlierOnly, setOutlierOnly] = useState(false);

  const filtered = useMemo(() => {
    let list = tickers;
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((t) => t.ticker.toLowerCase().includes(q) || t.sector.toLowerCase().includes(q));
    }
    if (outlierOnly) {
      const values = list.map((t) => Math.abs(t.t1));
      const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
      const std = Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / (values.length || 1));
      const threshold = mean + 1.5 * std;
      list = list.filter((t) => Math.abs(t.t1) >= threshold);
    }
    return list;
  }, [tickers, search, outlierOnly]);

  const shorts = filtered.filter((t) => t.direction === "short").sort((a, b) => a.t1 - b.t1);
  const longs = filtered.filter((t) => t.direction === "long").sort((a, b) => b.t1 - a.t1);

  const shortSectors = useMemo(() => groupBySector(shorts), [shorts]);
  const longSectors = useMemo(() => groupBySector(longs), [longs]);

  const pairTrade = useMemo(() => {
    const worstShort = shorts[0];
    const bestLong = longs[0];
    return {
      short: { sector: worstShort?.sector ?? "—", beta: worstShort ? Math.abs(worstShort.t1 / 1.5) : 0 },
      long: { sector: bestLong?.sector ?? "—", beta: bestLong ? Math.abs(bestLong.t1 / 1.5) : 0 },
    };
  }, [shorts, longs]);

  const btpBps = Math.min(120, Math.round(35 * state.contagionRisk * state.wave));

  return (
    <div className="flex-1 overflow-y-auto min-h-0 flex flex-col">
      {/* Search & filter bar */}
      <div className="flex items-center gap-1.5 px-2 py-1 border-b border-ki-border-strong bg-ki-surface shrink-0">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter..."
          className="flex-1 min-w-0 h-5 px-1.5 rounded text-[10px] font-data bg-ki-surface-sunken text-ki-on-surface border border-ki-border-strong outline-none focus:border-ki-accent placeholder:text-ki-on-surface-muted"
        />
        <button
          onClick={() => setOutlierOnly((v) => !v)}
          className={`shrink-0 h-5 px-2 rounded text-[9px] font-data uppercase tracking-wider border transition-colors ${outlierOnly
              ? "bg-[#ff3b3b]/20 border-[#ff3b3b]/50 text-[#ff3b3b]"
              : "bg-ki-surface-sunken border-ki-border-strong text-ki-on-surface-muted hover:text-ki-on-surface"
            }`}
        >
          Outliers
        </button>
      </div>

      {/* SHORT section */}
      <div className="h-5 flex justify-between items-center px-2 bg-[#1a0a0a] border-b border-ki-border-strong sticky top-0 z-10">
        <span className="font-data text-[9px] text-[#ff3b3b] uppercase tracking-wider">Short Leg ({shorts.length})</span>
      </div>
      <div className="max-h-[220px] overflow-y-auto">
        {Array.from(shortSectors.entries()).map(([sector, items]) => (
          <div key={sector}>
            <SectorHeader sector={sector} bg="#1a0a0a" />
            {items.map((t) => (
              <TickerRow key={t.ticker} t={t} />
            ))}
          </div>
        ))}
      </div>

      {/* LONG section */}
      <div className="h-5 flex justify-between items-center px-2 bg-[#0a1a0a] border-b border-t border-ki-border-strong sticky top-0 z-10">
        <span className="font-data text-[9px] text-[#00d26a] uppercase tracking-wider">Long Leg ({longs.length})</span>
      </div>
      <div className="max-h-[220px] overflow-y-auto">
        {Array.from(longSectors.entries()).map(([sector, items]) => (
          <div key={sector}>
            <SectorHeader sector={sector} bg="#0a1a0a" />
            {items.map((t) => (
              <TickerRow key={t.ticker} t={t} />
            ))}
          </div>
        ))}
      </div>

      {/* Aggregate row */}
      <div className="border-t border-ki-border px-2 py-2 shrink-0">
        <div className="grid grid-cols-2 gap-1">
          <div>
            <div className="font-data text-[8px] text-ki-on-surface-muted uppercase">FTSE MIB Est.</div>
            <div className="font-data text-[13px] text-[#ff3b3b] font-bold">
              {(state.contagionRisk * -2.5).toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="font-data text-[8px] text-ki-on-surface-muted uppercase">BTP Spread</div>
            <div className="font-data text-[13px] text-[#ffaa00] font-bold">
              +{btpBps}bps
            </div>
          </div>
        </div>

        {/* Pair trade summary — dynamic */}
        <div className="mt-2 pt-2 border-t border-ki-border-strong">
          <div className="font-data text-[8px] text-ki-on-surface-muted uppercase mb-1">Pair Trade</div>
          <div className="font-data text-[9px] text-ki-on-surface-muted">
            SHORT <span className="text-[#ff3b3b]">{pairTrade.short.sector} &beta;{pairTrade.short.beta.toFixed(2)}</span> / LONG <span className="text-[#00d26a]">{pairTrade.long.sector} &beta;{pairTrade.long.beta.toFixed(2)}</span>
          </div>
        </div>

        {/* Confidence */}
        <div className="mt-2 pt-2 border-t border-ki-border-strong">
          <div className="font-data text-[8px] text-ki-on-surface-muted uppercase mb-1">Model Confidence</div>
          <div className="w-full h-1 bg-ki-surface-sunken overflow-hidden">
            <div
              className="h-full"
              style={{
                width: `${state.contagionRisk * 100}%`,
                background: state.contagionRisk > 0.7 ? "#ff3b3b" : "#ffaa00",
              }}
            />
          </div>
          <div className="font-data text-[8px] text-ki-on-surface-muted mt-0.5">
            SCOPE: MACRO_SYSTEMATIC
          </div>
        </div>
      </div>
    </div>
  );
}
