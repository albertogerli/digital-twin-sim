"use client";

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface PolarizationChartProps {
  rounds: { round: number; polarization: number }[];
}

export default function PolarizationChart({ rounds }: PolarizationChartProps) {
  if (rounds.length === 0) return null;

  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-2" role="img" aria-label={`Grafico polarizzazione: ${rounds.length} round, ultimo valore ${rounds[rounds.length - 1]?.polarization.toFixed(1)}`}>
      <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-2">
        Polarizzazione
      </p>
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rounds} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="polGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#dc2626" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#dc2626" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="round"
              tick={{ fontSize: 10, fill: "#8a8a8a" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 10]}
              tick={{ fontSize: 10, fill: "#8a8a8a" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, borderRadius: 4, border: "1px solid #d4d4d4" }}
              formatter={(v) => [Number(v).toFixed(1), "Polarizzazione"]}
              labelFormatter={(l) => `Round ${l}`}
            />
            <Area
              type="monotone"
              dataKey="polarization"
              stroke="#dc2626"
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
