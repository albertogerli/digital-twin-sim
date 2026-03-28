"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface Coalition {
  label: string;
  size: number;
  color?: string;
  avg_position?: number;
}

interface CoalitionEvolutionProps {
  rounds: { round: number; coalitions: Coalition[] }[];
}

const PALETTE = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#06b6d4", "#ec4899", "#14b8a6",
];

export default function CoalitionEvolution({ rounds }: CoalitionEvolutionProps) {
  if (rounds.length === 0) return null;

  // Collect all coalition labels across rounds
  const allLabels = [...new Set(rounds.flatMap((r) => r.coalitions.map((c) => c.label)))];
  const labelColor = Object.fromEntries(allLabels.map((l, i) => [l, PALETTE[i % PALETTE.length]]));

  // Build data for stacked bar
  const data = rounds.map((r) => {
    const entry: Record<string, number | string> = { round: r.round };
    for (const c of r.coalitions) {
      entry[c.label] = c.size;
    }
    return entry;
  });

  // Only show last 3 labels in legend if too many
  const legendLabels = allLabels.slice(0, 6);

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-3">
        Coalizioni
      </p>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <XAxis
              dataKey="round"
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e5e7eb" }}
              labelFormatter={(l) => `Round ${l}`}
            />
            {allLabels.map((label) => (
              <Bar
                key={label}
                dataKey={label}
                stackId="a"
                fill={labelColor[label]}
                animationDuration={600}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-2 text-[10px]">
        {legendLabels.map((l) => (
          <span key={l} className="flex items-center gap-1 text-gray-500" title={l}>
            <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: labelColor[l] }} />
            <span className="truncate max-w-[120px]">{l}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
