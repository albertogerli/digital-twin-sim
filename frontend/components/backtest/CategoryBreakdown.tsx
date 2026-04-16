"use client";

import { CategoryStats } from "@/lib/backtest-types";

export function CategoryBreakdown({ stats }: { stats: CategoryStats[] }) {
  return (
    <div className="border-b border-ki-border">
      {/* Header */}
      <div className="h-6 flex items-center px-2 border-b border-ki-border-strong bg-ki-surface-sunken">
        <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
          Category Breakdown
        </span>
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-[1fr_30px_50px_50px_50px] gap-0 px-2 h-5 items-center border-b border-ki-border-strong text-[9px] font-data text-ki-on-surface-muted uppercase">
        <span>Category</span>
        <span className="text-right">N</span>
        <span className="text-right">Dir%</span>
        <span className="text-right">MAE</span>
        <span className="text-right">BTP</span>
      </div>

      {/* Rows */}
      {stats.map((cat) => {
        const dirPct = cat.dirTotal > 0 ? (cat.dirCorrect / cat.dirTotal) * 100 : 0;
        const dirColor = dirPct >= 70 ? "#00d26a" : dirPct >= 50 ? "#ffaa00" : "#ff3b3b";
        const maeColor = cat.maeT1 > 5 ? "#ff3b3b" : cat.maeT1 > 2 ? "#ffaa00" : "#00d26a";

        return (
          <div
            key={cat.category}
            className="grid grid-cols-[1fr_30px_50px_50px_50px] gap-0 px-2 h-6 items-center border-b border-ki-border-strong hover:bg-ki-surface-hover transition-colors"
          >
            <span className="text-[11px] text-ki-on-surface-muted truncate">{cat.category}</span>
            <span className="font-data text-[10px] text-ki-on-surface-muted text-right">{cat.count}</span>
            <span className="font-data text-[10px] text-right" style={{ color: dirColor }}>
              {dirPct.toFixed(0)}%
            </span>
            <span className="font-data text-[10px] text-right" style={{ color: maeColor }}>
              {cat.maeT1.toFixed(1)}
            </span>
            <span className="font-data text-[10px] text-ki-on-surface-muted text-right">
              {cat.avgBtp > 0 ? `+${cat.avgBtp.toFixed(0)}` : "0"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
