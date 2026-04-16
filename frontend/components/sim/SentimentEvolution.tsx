"use client";

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface SentimentEvolutionProps {
  rounds: { round: number; positive: number; neutral: number; negative: number }[];
}

export default function SentimentEvolution({ rounds }: SentimentEvolutionProps) {
  if (rounds.length === 0) return null;

  const data = rounds.map((r) => ({
    round: r.round,
    positive: Math.round(r.positive * 100),
    neutral: Math.round(r.neutral * 100),
    negative: Math.round(r.negative * 100),
  }));

  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-2" role="img" aria-label={`Grafico sentiment: ${data.length} round`}>
      <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-2">
        Sentiment
      </p>
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }} stackOffset="expand">
            <XAxis
              dataKey="round"
              tick={{ fontSize: 10, fill: "#8a8a8a" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#8a8a8a" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, borderRadius: 4, border: "1px solid #d4d4d4" }}
              formatter={(v) => [`${v}%`]}
              labelFormatter={(l) => `Round ${l}`}
            />
            <Area type="monotone" dataKey="positive" stackId="1" stroke="#16a34a" fill="#16a34a" fillOpacity={0.6} animationDuration={600} />
            <Area type="monotone" dataKey="neutral" stackId="1" stroke="#8a8a8a" fill="#8a8a8a" fillOpacity={0.4} animationDuration={600} />
            <Area type="monotone" dataKey="negative" stackId="1" stroke="#dc2626" fill="#dc2626" fillOpacity={0.6} animationDuration={600} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-center gap-4 mt-1.5 text-[10px]">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-ki-success" />Positivo</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-ki-on-surface-muted" />Neutrale</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-ki-error" />Negativo</span>
      </div>
    </div>
  );
}
