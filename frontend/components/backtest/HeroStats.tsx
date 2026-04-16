"use client";

import { AggregateStats } from "@/lib/backtest-types";

function Stat({ label, value, unit, color }: { label: string; value: string; unit?: string; color?: string }) {
  return (
    <div className="flex items-baseline gap-1.5 px-3">
      <span className="font-data text-[10px] text-ki-on-surface-muted uppercase">{label}</span>
      <span className="font-data text-sm font-semibold" style={{ color: color || "#1a1a1a" }}>
        {value}
      </span>
      {unit && <span className="font-data text-[10px] text-ki-on-surface-muted">{unit}</span>}
    </div>
  );
}

export function HeroStats({ agg }: { agg: AggregateStats }) {
  const dirPct = (agg.directionAccuracy * 100).toFixed(0);
  const dirColor = agg.directionAccuracy >= 0.6 ? "#00d26a" : agg.directionAccuracy >= 0.5 ? "#ffaa00" : "#ff3b3b";

  return (
    <div className="flex items-center h-10 border-b border-ki-border overflow-x-auto">
      <Stat label="DIR%" value={`${dirPct}%`} color={dirColor} />
      <span className="text-ki-border">|</span>
      <Stat label="DIR" value={`${agg.directionCorrect}/${agg.directionTotal}`} />
      <span className="text-ki-border">|</span>
      <Stat label="MAE T+1" value={agg.maeT1.toFixed(2)} unit="pp" color="#1a1a1a" />
      <span className="text-ki-border">|</span>
      <Stat label="MAE T+3" value={agg.maeT3.toFixed(2)} unit="pp" />
      <span className="text-ki-border">|</span>
      <Stat label="MAE T+7" value={agg.maeT7.toFixed(2)} unit="pp" />
      <span className="text-ki-border">|</span>
      <Stat label="MACRO" value={String(agg.macroCount)} color="#00d26a" />
      <span className="text-ki-border">|</span>
      <Stat label="IDIO" value={String(agg.idioCount)} color="#ffaa00" />
      <span className="text-ki-border">|</span>
      <Stat label="TICKERS" value={String(agg.totalTickers)} />
    </div>
  );
}
