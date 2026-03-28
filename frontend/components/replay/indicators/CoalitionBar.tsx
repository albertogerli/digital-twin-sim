"use client";

import { motion, AnimatePresence } from "framer-motion";

interface CoalitionSize {
  label: string;
  size: number;
  color: string;
}

interface Props {
  coalitions: CoalitionSize[];
}

export default function CoalitionBar({ coalitions }: Props) {
  const total = coalitions.reduce((sum, c) => sum + c.size, 0);
  if (total === 0) {
    return (
      <div className="text-[10px] font-mono text-gray-400 text-center py-2">
        Coalitions forming...
      </div>
    );
  }

  const sorted = [...coalitions].sort((a, b) => b.size - a.size);

  return (
    <div className="space-y-2">
      {/* Stacked bar */}
      <div className="h-6 rounded-lg overflow-hidden flex bg-gray-100 shadow-inner">
        <AnimatePresence mode="popLayout">
          {sorted.map((c) => {
            const pct = (c.size / total) * 100;
            if (pct < 1) return null;
            return (
              <motion.div
                key={c.label}
                layout
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
                className="h-full relative group first:rounded-l-lg last:rounded-r-lg"
                style={{ backgroundColor: c.color }}
              >
                {pct >= 12 && (
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-bold text-white/90 drop-shadow-sm">
                    {Math.round(pct)}%
                  </span>
                )}
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 rounded bg-gray-700 text-white text-[9px] font-mono whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-lg">
                  {c.label}: {c.size} members ({Math.round(pct)}%)
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Vertical legend */}
      <div className="space-y-1">
        {sorted.map((c) => {
          const pct = total > 0 ? Math.round((c.size / total) * 100) : 0;
          if (pct < 1) return null;
          return (
            <div key={c.label} className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                style={{ backgroundColor: c.color }}
              />
              <span className="text-[10px] text-gray-700 flex-1 truncate">
                {c.label}
              </span>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <span className="text-[9px] font-mono text-gray-400">
                  {c.size}
                </span>
                <span className="text-[10px] font-mono text-gray-700 font-bold tabular-nums w-7 text-right">
                  {pct}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
