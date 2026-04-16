"use client";

import { useState } from "react";
import { TickerStats } from "@/lib/backtest-types";

const TICKER_SECTORS: Record<string, string> = {
  "UCG.MI": "BANK", "ISP.MI": "BANK", "BAMI.MI": "BANK", "BMPS.MI": "BANK",
  "MB.MI": "BANK", "BPE.MI": "BANK", "FBK.MI": "BANK",
  "G.MI": "INS", "UNI.MI": "INS", "PST.MI": "INS",
  "ENEL.MI": "REN", "ERG.MI": "REN",
  "ENI.MI": "O&G", "SRS.MI": "O&G",
  "A2A.MI": "UTL", "SNAM.MI": "UTL", "TRN.MI": "UTL", "HER.MI": "UTL", "IG.MI": "UTL",
  "STLAM.MI": "AUTO", "RACE.MI": "AUTO", "CNHI.MI": "AUTO", "PIRC.MI": "AUTO", "PIA.MI": "AUTO", "IVG.MI": "AUTO",
  "LDO.MI": "DEF", "FCT.MI": "DEF",
  "TLIT.MI": "TEL", "STM.MI": "TECH", "REY.MI": "TECH",
  "REC.MI": "HC", "DIA.MI": "HC", "AMP.MI": "HC",
  "MONC.MI": "LUX", "BC.MI": "LUX", "SFER.MI": "LUX",
  "CPR.MI": "F&B", "MFEB.MI": "MED", "ENAV.MI": "INFRA",
};

export function TickerLeaderboard({ stats }: { stats: TickerStats[] }) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? stats : stats.slice(0, 15);

  return (
    <div className="border-t border-ki-border">
      <div className="h-6 flex items-center px-2 justify-between border-b border-ki-border-strong bg-ki-surface-sunken">
        <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
          Ticker Leaderboard — {stats.length} instruments
        </span>
        {stats.length > 15 && (
          <button
            onClick={() => setShowAll((p) => !p)}
            className="font-data text-[9px] text-ki-on-surface-muted hover:text-ki-on-surface-muted transition-colors"
          >
            {showAll ? "COLLAPSE" : `SHOW ALL ${stats.length}`}
          </button>
        )}
      </div>

      {/* Header */}
      <div className="grid grid-cols-[80px_40px_40px_50px_50px_60px_60px] gap-0 px-2 h-5 items-center border-b border-ki-border-strong text-[8px] font-data text-ki-on-surface-muted uppercase">
        <span>Ticker</span>
        <span className="text-right">Sec</span>
        <span className="text-right">N</span>
        <span className="text-right">Dir%</span>
        <span className="text-right">MAE</span>
        <span className="text-right">Avg Pred</span>
        <span className="text-right">Avg Real</span>
      </div>

      {visible.map((t) => {
        const sector = TICKER_SECTORS[t.ticker] || "—";
        const dirPct = t.dirTotal > 0 ? (t.dirCorrect / t.dirTotal) * 100 : 0;
        const dirColor = dirPct >= 65 ? "#00d26a" : dirPct >= 50 ? "#ffaa00" : "#ff3b3b";

        return (
          <div
            key={t.ticker}
            className="grid grid-cols-[80px_40px_40px_50px_50px_60px_60px] gap-0 px-2 h-5 items-center border-b border-ki-border-strong hover:bg-ki-surface-hover transition-colors"
          >
            <span className="font-data text-[11px] text-ki-on-surface font-medium">
              {t.ticker.replace(".MI", "")}
            </span>
            <span className="font-data text-[9px] text-ki-on-surface-muted text-right">{sector}</span>
            <span className="font-data text-[10px] text-ki-on-surface-muted text-right">{t.appearances}</span>
            <span className="font-data text-[10px] text-right" style={{ color: dirColor }}>
              {dirPct.toFixed(0)}%
            </span>
            <span className="font-data text-[10px] text-right" style={{ color: t.maeT1 > 5 ? "#ff3b3b" : "#8a8a8a" }}>
              {t.maeT1.toFixed(1)}
            </span>
            <span className="font-data text-[10px] text-right" style={{ color: t.avgPredicted < 0 ? "#ff3b3b" : "#00d26a" }}>
              {t.avgPredicted > 0 ? "+" : ""}{t.avgPredicted.toFixed(2)}
            </span>
            <span className="font-data text-[10px] text-right" style={{ color: t.avgActual < 0 ? "#ff3b3b" : "#00d26a" }}>
              {t.avgActual > 0 ? "+" : ""}{t.avgActual.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
