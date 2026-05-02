"use client";

import { CategoryStats } from "@/lib/backtest-types";

/* ───────────────────────────────────────────────────────────
   Hit-rate by domain — bars + monospace metrics, eyebrow
   labels. Pattern from screen-other.jsx (design v2).
   ─────────────────────────────────────────────────────────── */

export function CategoryBreakdown({ stats }: { stats: CategoryStats[] }) {
  const sorted = [...stats].sort((a, b) => b.count - a.count);

  return (
    <div className="border-b border-ki-border bg-ki-surface-raised">
      <div className="h-8 flex items-center px-3 border-b border-ki-border bg-ki-surface-sunken">
        <span className="eyebrow">Hit-rate by domain</span>
      </div>

      <div className="px-3 py-3 flex flex-col gap-3">
        {sorted.map((cat) => {
          const dirPct = cat.dirTotal > 0 ? (cat.dirCorrect / cat.dirTotal) * 100 : 0;
          const dirColor =
            dirPct >= 70 ? "var(--pos)" :
            dirPct >= 50 ? "var(--warn)" :
            "var(--neg)";
          const maeColor =
            cat.maeT1 > 5 ? "var(--neg)" :
            cat.maeT1 > 2 ? "var(--warn)" :
            "var(--pos)";

          return (
            <div key={cat.category} className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between">
                <span className="text-[12px] text-ki-on-surface truncate flex-1 mr-2">{cat.category}</span>
                <span className="font-data tabular text-[11px] text-ki-on-surface">
                  {dirPct.toFixed(0)}%
                </span>
                <span className="font-data tabular text-[10px] text-ki-on-surface-muted ml-2">
                  N={cat.count}
                </span>
              </div>
              <div className="h-1 bg-ki-surface-sunken rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${Math.max(2, dirPct)}%`, background: dirColor }}
                />
              </div>
              <div className="flex justify-between font-data tabular text-[10px] text-ki-on-surface-muted">
                <span>MAE <span style={{ color: maeColor }}>{cat.maeT1.toFixed(1)}</span></span>
                <span>BTP {cat.avgBtp > 0 ? `+${cat.avgBtp.toFixed(0)}bps` : "0"}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
