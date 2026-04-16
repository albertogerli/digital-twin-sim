"use client";

import { BacktestScenario } from "@/lib/backtest-types";

export function WaveAnalysis({ data }: { data: BacktestScenario[] }) {
  const waveMap: Record<number, { count: number; dc: number; dt: number; maes: number[] }> = {};
  for (const s of data) {
    if (!waveMap[s.wave]) waveMap[s.wave] = { count: 0, dc: 0, dt: 0, maes: [] };
    const w = waveMap[s.wave];
    w.count++;
    const [dc, dt] = s.direction_accuracy.split("/").map(Number);
    w.dc += dc; w.dt += dt;
    if (s.mae_t1 != null) w.maes.push(s.mae_t1);
  }

  const LABELS: Record<number, string> = { 1: "LOCAL", 2: "NATIONAL", 3: "INSTITUTIONAL" };
  const COLORS: Record<number, string> = { 1: "#00d26a", 2: "#ffaa00", 3: "#ff3b3b" };

  return (
    <div>
      <div className="h-6 flex items-center px-2 border-b border-ki-border-strong bg-ki-surface-sunken">
        <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
          Wave / Severity
        </span>
      </div>

      <div className="grid grid-cols-[60px_1fr_50px_50px_50px] gap-0 px-2 h-5 items-center border-b border-ki-border-strong text-[9px] font-data text-ki-on-surface-muted uppercase">
        <span>Wave</span>
        <span>Level</span>
        <span className="text-right">N</span>
        <span className="text-right">Dir%</span>
        <span className="text-right">MAE</span>
      </div>

      {[1, 2, 3].map((wave) => {
        const w = waveMap[wave];
        if (!w) return null;
        const dirPct = w.dt > 0 ? (w.dc / w.dt) * 100 : 0;
        const mae = w.maes.length > 0 ? w.maes.reduce((a, b) => a + b, 0) / w.maes.length : 0;
        const color = COLORS[wave];
        const dirColor = dirPct >= 60 ? "#00d26a" : dirPct >= 45 ? "#ffaa00" : "#ff3b3b";

        return (
          <div
            key={wave}
            className="grid grid-cols-[60px_1fr_50px_50px_50px] gap-0 px-2 h-7 items-center border-b border-ki-border-strong hover:bg-ki-surface-hover"
          >
            <div className="flex items-center gap-1.5">
              <span className="font-data text-xs font-bold" style={{ color }}>W{wave}</span>
            </div>
            <span className="font-data text-[10px] text-ki-on-surface-muted">{LABELS[wave]}</span>
            <span className="font-data text-[10px] text-ki-on-surface-muted text-right">{w.count}</span>
            <span className="font-data text-[10px] text-right" style={{ color: dirColor }}>
              {dirPct.toFixed(0)}%
            </span>
            <span className="font-data text-[10px] text-right" style={{ color: mae > 3 ? "#ff3b3b" : "#8a8a8a" }}>
              {mae.toFixed(1)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
