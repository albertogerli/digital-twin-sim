"use client";

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface PolarizationChartProps {
  rounds: { round: number; polarization: number }[];
}

export default function PolarizationChart({ rounds }: PolarizationChartProps) {
  if (rounds.length === 0) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-3">
        Polarizzazione
      </p>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rounds} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="polGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="round"
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 10]}
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e5e7eb" }}
              formatter={(v) => [Number(v).toFixed(1), "Polarizzazione"]}
              labelFormatter={(l) => `Round ${l}`}
            />
            <Area
              type="monotone"
              dataKey="polarization"
              stroke="#ef4444"
              strokeWidth={2}
              fill="url(#polGrad)"
              animationDuration={600}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
